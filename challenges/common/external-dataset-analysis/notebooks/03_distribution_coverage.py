#!/usr/bin/env python3
"""
Analysis 03 — Embedding-Based Distribution Coverage
Extracts RADIO-H embeddings for all external images, runs t-SNE, computes OOD coverage.
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/03_distribution_coverage.py
Prereq: Analysis 00 must be complete (dedup_kept_index.json must exist)
"""
import os, sys, json, sqlite3, time, logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import normalize as sk_normalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
REPO     = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
LARS_ANA = Path('/home/farms/macvi2026-farms-team/challenges/common/lars-analysis')
DATA     = Path('/home/farms/data/datasets')
SHARED   = REPO / 'shared_data'
FIGS     = REPO / 'figures'
LOGS_DIR = REPO / 'logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_03_run.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("Analysis 03 — Distribution Coverage")
log.info("=" * 60)

LARS_DB   = LARS_ANA / 'shared_data' / 'embeddings.db'
EXT_DB    = SHARED / 'external_embeddings.db'
TSNE_PATH = SHARED / 'tsne_combined.npy'
LABELS_PATH = SHARED / 'tsne_labels.json'
COVERAGE_PATH = SHARED / 'coverage_table.csv'
FRAMES_CSV = Path('/home/farms/macvi2026-farms-team/challenges/embedded-semantic-segmentation/submissions/16537_pidnets_sgd/results/frames.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log.info(f"  GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB")

# ──────────────────────────────────────────────
# Load dedup index from Analysis 00
# ──────────────────────────────────────────────
DEDUP_INDEX = SHARED / 'dedup_kept_index.json'
assert DEDUP_INDEX.exists(), f"Run Analysis 00 first: {DEDUP_INDEX} not found"
with open(DEDUP_INDEX) as f:
    dedup_kept = json.load(f)

log.info("Loaded dedup index:")
for ds, paths in dedup_kept.items():
    log.info(f"  {ds}: {len(paths)} images")

# ──────────────────────────────────────────────
# RADIO-H Model Loading
# ──────────────────────────────────────────────
MODEL_NAME = 'nvidia/C-RADIOv3-H'
COMMIT_EVERY = 200
BATCH_SIZE = 8        # images per GPU forward pass (same-resolution batch)
MAX_LONG_SIDE = 896   # cap large images (MULTIAQUA is 2208px — too slow at full res)


def load_radio():
    from transformers import AutoModel
    log.info(f"Loading RADIO model: {MODEL_NAME} ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE).eval()
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Warmup at capped resolution
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        probe = torch.zeros(BATCH_SIZE, 3, 512, 512, device=DEVICE)
        _ = model(probe)
    del probe
    torch.cuda.empty_cache()
    return model


def prep_image(model, img_pil):
    """Resize + RADIO resolution snap + normalize. Returns (H, W, tensor_CHW_float)."""
    H, W = img_pil.height, img_pil.width
    # Cap long side to MAX_LONG_SIDE to keep RADIO fast
    long = max(H, W)
    if long > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long
        H2, W2 = int(H * scale), int(W * scale)
        img_pil = img_pil.resize((W2, H2), Image.BICUBIC)
        H, W = H2, W2
    th, tw = model.get_nearest_supported_resolution(H, W)
    img_resized = img_pil.resize((tw, th), Image.BICUBIC)
    return th, tw, np.array(img_resized).astype(np.float32) / 255.0


def extract_batch(model, imgs_np):
    """imgs_np: list of (H, W, 3) float32 arrays — all same shape.
    Returns numpy array (N, embed_dim)."""
    batch_t = torch.stack([
        torch.from_numpy(a).permute(2, 0, 1) for a in imgs_np
    ]).to(DEVICE)
    batch_t = model.input_conditioner(batch_t)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        output = model(batch_t)
    if hasattr(output, 'summary'):
        summary_t = output.summary
    elif isinstance(output, (list, tuple)):
        summary_t = output[0]
    else:
        summary_t = output
    return summary_t.float().cpu().numpy()  # (N, 3840)


# ──────────────────────────────────────────────
# SQLite DB for external embeddings
# ──────────────────────────────────────────────
def init_ext_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT NOT NULL,
            dataset   TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dataset ON embeddings(dataset);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_filename_ds ON embeddings(filename, dataset);
    """)
    conn.commit()

def get_done_keys(conn):
    rows = conn.execute('SELECT filename, dataset FROM embeddings').fetchall()
    return {(r[0], r[1]) for r in rows}

def blob(arr):
    return arr.astype(np.float32).tobytes()

# ──────────────────────────────────────────────
# Embedding Extraction
# ──────────────────────────────────────────────
def count_ext_rows():
    if not EXT_DB.exists():
        return 0
    with sqlite3.connect(EXT_DB) as c:
        return c.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]

# Build full task list: (filepath, dataset_name)
# For WaterScenes: use ALL deduplicated for coverage; subsample to 10k for t-SNE
all_tasks = []
for ds, paths in dedup_kept.items():
    for p in paths:
        all_tasks.append((p, ds))

total_tasks = len(all_tasks)
log.info(f"\nTotal external images to embed: {total_tasks}")

# Check if we can skip
EXISTING_COUNT = count_ext_rows()
log.info(f"Already in DB: {EXISTING_COUNT}/{total_tasks}")

if EXISTING_COUNT < total_tasks:
    log.info("Starting embedding extraction...")
    radio_model = load_radio()

    conn = sqlite3.connect(EXT_DB)
    init_ext_db(conn)
    done_keys = get_done_keys(conn)
    log.info(f"  {len(done_keys)} already done, {total_tasks - len(done_keys)} remaining")

    pending = [(fp, ds) for fp, ds in all_tasks if (fp, ds) not in done_keys]
    db_batch = []
    errors = 0
    img_buffer = []   # list of (fp, ds, arr_HWC)
    prev_shape = None

    def flush_gpu_batch():
        """Run GPU inference on img_buffer (all same shape), return results."""
        if not img_buffer:
            return []
        arrs = [a for _, _, a in img_buffer]
        try:
            embs = extract_batch(radio_model, arrs)
            return [(img_buffer[k][0], img_buffer[k][1], blob(embs[k])) for k in range(len(img_buffer))]
        except Exception as e:
            log.warning(f"  Batch GPU error (shape={arrs[0].shape}): {e}")
            return []

    t0 = time.time()
    for i, (fp, ds) in enumerate(pending):
        try:
            img_pil = Image.open(fp).convert('RGB')
            th, tw, arr = prep_image(radio_model, img_pil)
            curr_shape = (th, tw)

            # Flush if resolution changes or buffer full
            if (prev_shape is not None and curr_shape != prev_shape) or len(img_buffer) >= BATCH_SIZE:
                results = flush_gpu_batch()
                db_batch.extend(results)
                errors += len(img_buffer) - len(results)
                img_buffer = []

            img_buffer.append((fp, ds, arr))
            prev_shape = curr_shape

        except Exception as e:
            errors += 1
            if errors <= 20:
                log.warning(f"  LOAD ERROR {fp}: {e}")

        # Commit DB periodically
        if len(db_batch) >= COMMIT_EVERY:
            conn.executemany('INSERT OR IGNORE INTO embeddings (filename, dataset, embedding) VALUES (?,?,?)', db_batch)
            conn.commit()
            db_batch = []

        if i % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(pending) - i - 1) / max(rate, 0.01)
            log.info(f"  [{i+1}/{len(pending)}] rate={rate:.1f} img/s  ETA={eta/60:.1f} min  errors={errors}")

    # Flush remaining buffer
    results = flush_gpu_batch()
    db_batch.extend(results)
    errors += len(img_buffer) - len(results)

    if db_batch:
        conn.executemany('INSERT OR IGNORE INTO embeddings (filename, dataset, embedding) VALUES (?,?,?)', db_batch)
        conn.commit()

    conn.close()
    del radio_model
    torch.cuda.empty_cache()
    log.info(f"Extraction complete. Total in DB: {count_ext_rows()}, errors: {errors}")
else:
    log.info("Embedding extraction already complete — skipping")

# ──────────────────────────────────────────────
# Load external embeddings
# ──────────────────────────────────────────────
log.info("\nLoading external embeddings from DB...")
with sqlite3.connect(EXT_DB) as conn:
    rows = conn.execute('SELECT filename, dataset, embedding FROM embeddings ORDER BY dataset, filename').fetchall()

ext_filenames = [r[0] for r in rows]
ext_datasets  = [r[1] for r in rows]
ext_embs = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
embed_dim = ext_embs.shape[1]
log.info(f"External embeddings: {len(ext_embs)} × {embed_dim}")

unique_ds = sorted(set(ext_datasets))
for ds in unique_ds:
    n = ext_datasets.count(ds)
    log.info(f"  {ds}: {n}")

# ──────────────────────────────────────────────
# Load LaRS embeddings (summary_embedding)
# ──────────────────────────────────────────────
log.info("\nLoading LaRS embeddings...")
with sqlite3.connect(LARS_DB) as conn:
    # Use summary_embedding (CLS token) for consistency
    lars_rows = conn.execute(
        'SELECT img_name, split, summary_embedding FROM embeddings ORDER BY id'
    ).fetchall()

lars_names  = [r[0] for r in lars_rows]
lars_splits = [r[1] for r in lars_rows]
lars_embs   = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in lars_rows])
log.info(f"LaRS: {len(lars_embs)} embeddings × {lars_embs.shape[1]} dim")

# Align dimensions if needed (trim to min)
common_dim = min(embed_dim, lars_embs.shape[1])
if common_dim < embed_dim:
    log.warning(f"Dimension mismatch: ext={embed_dim}, lars={lars_embs.shape[1]}. Trimming to {common_dim}.")
    ext_embs  = ext_embs[:, :common_dim]
    lars_embs = lars_embs[:, :common_dim]

# ──────────────────────────────────────────────
# t-SNE on combined set (LaRS + external)
# WaterScenes subsampled to 10k for t-SNE
# ──────────────────────────────────────────────
log.info("\n--- t-SNE ---")

# Build combined array for t-SNE (WaterScenes subsampled)
WS_TSNE_SUBSAMPLE = 10000
tsne_embs = []
tsne_labels = []  # 'lars_train', 'lars_val', 'lars_test', or dataset name

# LaRS
for i, split in enumerate(lars_splits):
    tsne_embs.append(lars_embs[i])
    tsne_labels.append(f'lars_{split}')

# External (WaterScenes subsampled)
ws_indices = [i for i, ds in enumerate(ext_datasets) if 'WaterScenes' in ds]
non_ws_indices = [i for i, ds in enumerate(ext_datasets) if 'WaterScenes' not in ds]

# Subsample WaterScenes
if len(ws_indices) > WS_TSNE_SUBSAMPLE:
    step = len(ws_indices) // WS_TSNE_SUBSAMPLE
    ws_tsne_indices = ws_indices[::step][:WS_TSNE_SUBSAMPLE]
    log.info(f"WaterScenes subsampled: {len(ws_indices)} → {len(ws_tsne_indices)} for t-SNE")
else:
    ws_tsne_indices = ws_indices

for i in non_ws_indices + ws_tsne_indices:
    tsne_embs.append(ext_embs[i])
    tsne_labels.append(ext_datasets[i])

tsne_arr = np.stack(tsne_embs)
log.info(f"t-SNE input: {tsne_arr.shape} (LaRS={len(lars_embs)}, ext={len(non_ws_indices)+len(ws_tsne_indices)})")

if TSNE_PATH.exists() and LABELS_PATH.exists():
    tsne_result = np.load(TSNE_PATH)
    with open(LABELS_PATH) as f:
        tsne_labels_loaded = json.load(f)
    log.info(f"Loaded t-SNE from {TSNE_PATH}")
    # Verify size matches
    if len(tsne_result) != len(tsne_arr):
        log.warning(f"t-SNE size mismatch ({len(tsne_result)} vs {len(tsne_arr)}), recomputing")
        os.remove(TSNE_PATH)
        os.remove(LABELS_PATH)

if not TSNE_PATH.exists():
    log.info("Computing t-SNE (PCA → openTSNE)...")
    from sklearn.decomposition import PCA

    n_pca = min(50, tsne_arr.shape[1], tsne_arr.shape[0] - 1)
    log.info(f"PCA: {tsne_arr.shape[1]} → {n_pca} dims")
    pca = PCA(n_components=n_pca, random_state=42)
    tsne_pca = pca.fit_transform(tsne_arr)
    log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    try:
        import openTSNE
        log.info("Using openTSNE...")
        t0 = time.time()
        tsne = openTSNE.TSNE(
            n_components=2, perplexity=30, n_iter=1000,
            random_state=42, n_jobs=-1, verbose=True,
            initialization='pca',
        )
        tsne_result = tsne.fit(tsne_pca.astype(np.float32))
        tsne_result = np.array(tsne_result)
        log.info(f"openTSNE done in {(time.time()-t0)/60:.1f} min")
    except Exception as e:
        log.warning(f"openTSNE failed ({e}), falling back to sklearn TSNE")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1, verbose=2)
        tsne_result = tsne.fit_transform(tsne_pca)

    np.save(TSNE_PATH, tsne_result)
    with open(LABELS_PATH, 'w') as f:
        json.dump(tsne_labels, f)
    log.info(f"t-SNE saved: {TSNE_PATH}")

# ── t-SNE Plot ──
log.info("Plotting t-SNE...")
DATASET_COLORS = {
    'lars_train': '#1f77b4',
    'lars_val':   '#ff7f0e',
    'lars_test':  '#2ca02c',
    'MULTIAQUA_night': '#d62728',
    'OrcaTech_1280x640': '#9467bd',
    'OrcaTech_640x320':  '#8c564b',
    'Tampere-WaterSeg':  '#e377c2',
    'WaterScenes':       '#7f7f7f',
    'ROSEBUD':           '#bcbd22',
}
DATASET_SIZES = {
    'lars_train': 4, 'lars_val': 4, 'lars_test': 4,
    'MULTIAQUA_night': 5, 'OrcaTech_1280x640': 6, 'OrcaTech_640x320': 6,
    'Tampere-WaterSeg': 6, 'WaterScenes': 2, 'ROSEBUD': 6,
}
DATASET_ALPHA = {'WaterScenes': 0.2, 'lars_train': 0.5}

fig, ax = plt.subplots(figsize=(14, 10))
unique_labels = sorted(set(tsne_labels), key=lambda x: (0 if x.startswith('lars') else 1, x))
for label in unique_labels:
    idxs = [i for i, l in enumerate(tsne_labels) if l == label]
    x = tsne_result[idxs, 0]
    y = tsne_result[idxs, 1]
    color = DATASET_COLORS.get(label, '#aaaaaa')
    size  = DATASET_SIZES.get(label, 4)
    alpha = DATASET_ALPHA.get(label, 0.7)
    ax.scatter(x, y, s=size, c=color, alpha=alpha, label=f"{label} ({len(idxs)})", rasterized=True)

ax.legend(fontsize=8, markerscale=3, loc='upper right')
ax.set_xlabel('t-SNE dim 0')
ax.set_ylabel('t-SNE dim 1')
ax.set_title('t-SNE: LaRS + External Datasets (RADIO-H CLS embeddings)')
plt.tight_layout()
tsne_fig_path = FIGS / 'tsne_all_datasets.png'
plt.savefig(tsne_fig_path, dpi=120)
plt.close()
log.info(f"Saved: {tsne_fig_path}")

# ──────────────────────────────────────────────
# OOD Coverage Metric
# ──────────────────────────────────────────────
log.info("\n--- OOD Coverage Metric ---")

# LaRS source parsing (from filename prefix)
def parse_lars_source(img_name):
    n = img_name.lower()
    if n.startswith('yt'):        return 'YouTube'
    if n.startswith('davimar'):   return 'Davimar'
    if n.startswith('orca'):      return 'ORCA'
    if n.startswith('inhouse'):   return 'InHouse'
    if n.startswith('smd'):       return 'SMD'
    if n.startswith('mit'):       return 'MIT'
    return 'Other'

# Load frames.csv for per-image metrics (used to identify "hard" test images)
import csv as csv_mod
lars_sources = {}
if FRAMES_CSV.exists():
    with open(FRAMES_CSV) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            img_name = Path(row.get('image', row.get('filename', ''))).stem
            lars_sources[img_name] = parse_lars_source(img_name)
    log.info(f"Loaded source labels for {len(lars_sources)} images from frames.csv")
else:
    log.warning(f"frames.csv not found: {FRAMES_CSV}")

# Normalize all embeddings for cosine similarity
lars_norm = sk_normalize(lars_embs, norm='l2')
ext_norm  = sk_normalize(ext_embs,  norm='l2')

train_mask = np.array([s == 'train' for s in lars_splits])
test_mask  = np.array([s == 'test'  for s in lars_splits])

train_emb = lars_norm[train_mask]
test_emb  = lars_norm[test_mask]
test_names = [lars_names[i] for i in range(len(lars_names)) if lars_splits[i] == 'test']

log.info(f"Computing max cosine sim: {len(test_emb)} test vs {len(train_emb)} train...")
CHUNK = 100
max_sim_to_train = []
for i in range(0, len(test_emb), CHUNK):
    batch = test_emb[i:i+CHUNK]
    sims = batch @ train_emb.T
    max_sim_to_train.append(sims.max(axis=1))
    if i % 500 == 0:
        log.info(f"  cosine sim test-train: {i}/{len(test_emb)}")
max_sim_to_train = np.concatenate(max_sim_to_train)

# Thresholds for "hard" test images
thresholds = [0.85, 0.88, 0.90]
log.info(f"\nOOD coverage at thresholds: {thresholds}")

# Per external dataset: compute coverage
ext_datasets_unique = sorted(set(ext_datasets))
coverage_rows = []

for thr in thresholds:
    hard_mask = max_sim_to_train < thr
    hard_indices = np.where(hard_mask)[0]
    hard_names = [test_names[i] for i in hard_indices]
    hard_embs  = test_emb[hard_indices]
    n_hard = len(hard_indices)
    log.info(f"\nThreshold {thr}: {n_hard} hard test images")

    # Per LaRS source breakdown
    hard_sources = {}
    for name in hard_names:
        src = lars_sources.get(name, parse_lars_source(name))
        hard_sources[src] = hard_sources.get(src, 0) + 1
    log.info(f"  Hard images by source: {hard_sources}")

    if n_hard == 0:
        continue

    for ds in ext_datasets_unique:
        ds_indices = [i for i, d in enumerate(ext_datasets) if d == ds]
        ds_embs = ext_norm[ds_indices]

        # Coverage: fraction of hard test images where sim to any ext image > sim_to_nearest_train
        covered = 0
        for hi, (h_emb, h_sim_train) in enumerate(zip(hard_embs, max_sim_to_train[hard_indices])):
            sim_ext = (h_emb @ ds_embs.T).max()
            if sim_ext > h_sim_train:
                covered += 1

        cov_pct = 100 * covered / n_hard
        log.info(f"  [{ds}] coverage={cov_pct:.1f}% ({covered}/{n_hard} hard images covered)")

        # Per source breakdown
        source_cov = {}
        for src in set(hard_sources.keys()):
            src_hi = [i for i, name in enumerate(hard_names)
                      if lars_sources.get(name, parse_lars_source(name)) == src]
            if not src_hi:
                source_cov[src] = None
                continue
            src_embs = hard_embs[src_hi]
            src_train_sims = max_sim_to_train[hard_indices][src_hi]
            src_covered = sum(1 for h_emb, h_sim in zip(src_embs, src_train_sims)
                              if (h_emb @ ds_embs.T).max() > h_sim)
            source_cov[src] = round(100 * src_covered / len(src_hi), 1)

        row = {'threshold': thr, 'dataset': ds, 'n_hard_test': n_hard, 'coverage_pct': round(cov_pct, 1)}
        row.update({f'cov_{src}': v for src, v in source_cov.items()})
        coverage_rows.append(row)

# Save coverage table
import csv as csv_mod
if coverage_rows:
    all_keys = set()
    for r in coverage_rows:
        all_keys.update(r.keys())
    all_keys = ['threshold', 'dataset', 'n_hard_test', 'coverage_pct'] + \
               sorted([k for k in all_keys if k.startswith('cov_')])

    with open(COVERAGE_PATH, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(coverage_rows)
    log.info(f"\nSaved coverage table: {COVERAGE_PATH}")

# ── Coverage Table Figure ──
log.info("Generating coverage table figure...")
if coverage_rows:
    # Pivot: rows=datasets, cols=thresholds
    fig, ax = plt.subplots(figsize=(10, max(4, len(ext_datasets_unique) * 0.6 + 2)))
    ax.axis('off')

    col_labels = ['Dataset'] + [f'thr={t}' for t in thresholds] + ['n_hard (0.85)']
    rows_data = []
    for ds in ext_datasets_unique:
        row_data = [ds]
        n_hard_row = None
        for thr in thresholds:
            found = [r for r in coverage_rows if r['dataset'] == ds and r['threshold'] == thr]
            if found:
                row_data.append(f"{found[0]['coverage_pct']:.1f}%")
                if n_hard_row is None:
                    n_hard_row = found[0]['n_hard_test']
            else:
                row_data.append('—')
        row_data.append(str(n_hard_row) if n_hard_row else '—')
        rows_data.append(row_data)

    table = ax.table(cellText=rows_data, colLabels=col_labels,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title('OOD Coverage: fraction of hard LaRS test images better covered by external dataset\n(coverage > sim_to_nearest_train)', fontsize=10)
    plt.tight_layout()
    cov_fig = FIGS / 'coverage_table.png'
    plt.savefig(cov_fig, dpi=100, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {cov_fig}")

log.info("\n" + "=" * 60)
log.info("Analysis 03 COMPLETE")
log.info("=" * 60)
