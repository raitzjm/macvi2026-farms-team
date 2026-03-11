#!/usr/bin/env python3
"""
Analysis 02 — Cross-Dataset Duplicate Detection (LaRS vs external)
Step 1: pHash screen (Hamming ≤ 8)
Step 2: RADIO-H cosine confirmation for flagged pairs (cosine ≥ 0.97 = near-dup)
Prereq: Analysis 00 (dedup_kept_index.json) and Analysis 03 embeddings.
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/02_duplicate_detection.py
"""
import json, sys, sqlite3, logging
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO   = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
LARS_DB = Path('/home/farms/macvi2026-farms-team/challenges/common/lars-analysis/shared_data/embeddings.db')
SHARED = REPO / 'shared_data'
FIGS   = REPO / 'figures'
LOGS_DIR = REPO / 'logs'
LARS_IMG = Path('/home/farms/data/lars_dataset/img_dir')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_02_run.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=== Analysis 02: Cross-Dataset Duplicate Detection ===")

with open(SHARED / 'dedup_kept_index.json') as f:
    dedup_kept = json.load(f)

# ── Step 1: pHash for all LaRS images ──
log.info("Computing pHash for LaRS images...")
lars_hashes = {}
for split in ['train', 'val', 'test']:
    img_dir = LARS_IMG / split / 'images'
    if not img_dir.exists():
        continue
    for p in sorted(img_dir.iterdir()):
        try:
            h = imagehash.phash(Image.open(p).convert('RGB'), hash_size=8)
            lars_hashes[p.stem] = (h, split, p)
        except:
            pass
log.info(f"LaRS: {len(lars_hashes)} images hashed")

# ── pHash for external images ──
log.info("Computing pHash for external images...")
ext_hashes = {}  # stem -> (hash, dataset, path)
for ds, paths in dedup_kept.items():
    for p_str in paths:
        p = Path(p_str)
        try:
            h = imagehash.phash(Image.open(p).convert('RGB'), hash_size=8)
            ext_hashes[p.stem] = (h, ds, p)
        except:
            pass
    log.info(f"  {ds}: {len([p for p in dedup_kept[ds]])} → hashed")
log.info(f"External: {len(ext_hashes)} images hashed")

# ── Cross-dataset pHash comparison ──
log.info("Cross-comparing hashes (LaRS vs external, Hamming ≤ 8)...")
HASH_THR = 8
flagged = []
lars_hashes_list = list(lars_hashes.items())
ext_hashes_list  = list(ext_hashes.items())

for lars_stem, (lh, ls, lp) in lars_hashes_list:
    for ext_stem, (eh, ds, ep) in ext_hashes_list:
        dist = lh - eh
        if dist <= HASH_THR:
            flagged.append({
                'lars_stem': lars_stem, 'lars_split': ls, 'lars_path': str(lp),
                'ext_stem': ext_stem, 'ext_dataset': ds, 'ext_path': str(ep),
                'hamming': dist,
            })

log.info(f"pHash screen: {len(flagged)} pairs flagged (Hamming ≤ {HASH_THR})")

# ── Step 2: RADIO-H cosine confirmation (if embeddings available) ──
EXT_DB = SHARED / 'external_embeddings.db'
confirmed_dups = []

if EXT_DB.exists() and len(flagged) > 0:
    log.info("Loading embeddings for cosine confirmation...")
    from sklearn.preprocessing import normalize as sk_normalize

    # Load LaRS embeddings
    with sqlite3.connect(LARS_DB) as c:
        lars_emb_rows = {r[0]: np.frombuffer(r[1], dtype=np.float32)
                         for r in c.execute('SELECT img_name, summary_embedding FROM embeddings')}

    # Load external embeddings
    with sqlite3.connect(EXT_DB) as c:
        ext_emb_rows = {(r[0], r[1]): np.frombuffer(r[2], dtype=np.float32)
                        for r in c.execute('SELECT filename, dataset, embedding FROM embeddings')}

    for pair in flagged:
        lars_emb = lars_emb_rows.get(pair['lars_stem'])
        ext_key = (pair['ext_path'], pair['ext_dataset'])
        # Try matching by filename
        ext_emb = None
        for (fn, ds), emb in ext_emb_rows.items():
            if Path(fn).stem == pair['ext_stem'] and ds == pair['ext_dataset']:
                ext_emb = emb
                break

        if lars_emb is None or ext_emb is None:
            pair['cosine'] = None
            continue

        # Normalize and compute cosine
        ln = lars_emb / (np.linalg.norm(lars_emb) + 1e-8)
        en = ext_emb  / (np.linalg.norm(ext_emb)  + 1e-8)
        cosine = float(np.dot(ln, en))
        pair['cosine'] = round(cosine, 4)

        if cosine >= 0.97:
            pair['status'] = 'near-duplicate' if cosine < 0.99 else 'exact-duplicate'
            confirmed_dups.append(pair)
        else:
            pair['status'] = 'false-positive'

    log.info(f"Cosine confirmation: {len(confirmed_dups)} confirmed duplicates "
             f"({len([p for p in flagged if p.get('cosine') is not None])} pairs checked)")
else:
    log.info("Skipping cosine step (embeddings not yet available or no pairs flagged)")
    for pair in flagged:
        pair['status'] = 'pHash-flagged-unconfirmed'
    confirmed_dups = flagged  # treat all pHash hits as potential dups

# ── Summary ──
by_dataset = {}
for pair in confirmed_dups:
    ds = pair['ext_dataset']
    by_dataset[ds] = by_dataset.get(ds, 0) + 1

log.info("\nDuplicate summary by external dataset:")
for ds, n in sorted(by_dataset.items()):
    log.info(f"  {ds}: {n} duplicates with LaRS")

# ── Visualization grid (if any duplicates found) ──
if confirmed_dups:
    n_show = min(16, len(confirmed_dups))
    samples = confirmed_dups[:n_show]
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 2 * 3, nrows * 2.5))
    axes = np.array(axes).reshape(-1)

    for idx, pair in enumerate(samples):
        try:
            img_lars = Image.open(pair['lars_path']).convert('RGB')
            img_ext  = Image.open(pair['ext_path']).convert('RGB')
            ax_l = axes[idx * 2]
            ax_e = axes[idx * 2 + 1]
            ax_l.imshow(img_lars)
            ax_l.set_title(f"LaRS {pair['lars_split']}\n{pair['lars_stem'][:15]}", fontsize=6)
            ax_l.axis('off')
            ax_e.imshow(img_ext)
            cosine_str = f", cos={pair.get('cosine','?')}" if pair.get('cosine') else ""
            ax_e.set_title(f"{pair['ext_dataset']}\nH={pair['hamming']}{cosine_str}", fontsize=6)
            ax_e.axis('off')
        except:
            pass

    plt.suptitle(f"Cross-dataset duplicates: {len(confirmed_dups)} pairs found", fontsize=10)
    plt.tight_layout()
    out = FIGS / 'duplicate_pairs_grid.png'
    plt.savefig(out, dpi=80)
    plt.close()
    log.info(f"Saved: {out}")
else:
    log.info("No confirmed duplicates — no grid figure needed")

# Save results
import csv
dup_csv = SHARED / 'duplicate_pairs.csv'
if confirmed_dups:
    keys = ['lars_stem', 'lars_split', 'ext_stem', 'ext_dataset', 'hamming', 'cosine', 'status']
    with open(dup_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(confirmed_dups)
    log.info(f"Saved: {dup_csv}")

log.info("=== Analysis 02 COMPLETE ===")
