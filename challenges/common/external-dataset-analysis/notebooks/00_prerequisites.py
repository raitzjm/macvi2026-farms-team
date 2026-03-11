#!/usr/bin/env python3
"""
Analysis 00 — Prerequisites
Handles: 0a class verification, 0b WaterScenes QA, 0c dedup, 0d heuristic, 0e harmonization
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/00_prerequisites.py
"""
import os, sys, csv, time, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import imagehash

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
REPO      = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
DATA      = Path('/home/farms/data/datasets')
SHARED    = REPO / 'shared_data'
FIGS      = REPO / 'figures'
LOGS_DIR  = REPO / 'logs'

SHARED.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
(FIGS / 'sample_grids').mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_00_run.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("Analysis 00 — Prerequisites")
log.info("=" * 60)

# ──────────────────────────────────────────────
# §0a — MULTIAQUA Class Visual Inspection
# ──────────────────────────────────────────────
log.info("\n--- §0a: MULTIAQUA class verification ---")

MAQUA_ANN = DATA / 'MULTIAQUA_night/annotations'
MAQUA_IMG = DATA / 'MULTIAQUA_night/data/zed'

ann_files = sorted(MAQUA_ANN.iterdir())
log.info(f"MULTIAQUA: {len(ann_files)} annotation files")

# Collect images per class 3 and 4
per_class = defaultdict(list)
for af in ann_files:
    arr = np.array(Image.open(af))
    for c in np.unique(arr):
        per_class[c].append(af.name)

log.info(f"MULTIAQUA unique classes found: {sorted(per_class.keys())}")

# Visual QA for classes 3 and 4
for target_class in [3, 4]:
    samples = per_class.get(target_class, [])[:10]
    if not samples:
        log.warning(f"  class {target_class}: no samples found")
        continue
    log.info(f"  class {target_class}: {len(per_class[target_class])} total images, showing {len(samples)}")

    n = len(samples)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.5))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, fname in enumerate(samples):
        img_path = MAQUA_IMG / fname
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        ann = np.array(Image.open(MAQUA_ANN / fname))
        mask = (ann == target_class).astype(np.uint8) * 255
        overlay = img.copy()
        overlay[mask > 0] = [255, 0, 0]
        vis = (img * 0.6 + overlay * 0.4).astype(np.uint8)
        axes[idx].imshow(vis)
        axes[idx].set_title(f"{fname[:20]}\npx_count={(ann==target_class).sum()}", fontsize=7)
        axes[idx].axis('off')

    for ax in axes[n:]:
        ax.axis('off')
    plt.suptitle(f"MULTIAQUA class {target_class} overlay (red = class pixels)", fontsize=10)
    plt.tight_layout()
    outpath = FIGS / f"multiaqua_class{target_class}_qa.png"
    plt.savefig(outpath, dpi=80)
    plt.close()
    log.info(f"  Saved QA figure: {outpath}")

log.info("§0a complete — inspect figures/multiaqua_class3_qa.png and multiaqua_class4_qa.png")

# ──────────────────────────────────────────────
# §0b — WaterScenes Class 0 QA
# ──────────────────────────────────────────────
log.info("\n--- §0b: WaterScenes class 0 QA ---")

WS_SEM  = DATA / 'water_scenes/semantic/SegmentationClass'
WS_IMG  = DATA / 'water_scenes/image'

all_sem = sorted(WS_SEM.iterdir())
log.info(f"WaterScenes: {len(all_sem)} semantic files")

# Sample 20 images where class 0 pixels exist
class0_files = []
for sf in all_sem:
    arr = np.array(Image.open(sf))
    if 0 in np.unique(arr):
        class0_files.append(sf)
    if len(class0_files) >= 20:
        break

log.info(f"Found {len(class0_files)} WaterScenes images with class 0 pixels")

if class0_files:
    # Also get all-class distribution from first 200 files
    all_classes_ws = set()
    for sf in all_sem[:200]:
        all_classes_ws.update(np.unique(np.array(Image.open(sf))).tolist())
    log.info(f"WaterScenes unique classes in first 200 files: {sorted(all_classes_ws)}")

    n = len(class0_files[:20])
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 2 * 3, nrows * 2))
    axes = np.array(axes).reshape(-1) if n > 1 else axes.flatten()

    for idx, sf in enumerate(class0_files[:20]):
        stem = sf.stem
        img_path = WS_IMG / (stem + '.jpg')
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        ann = np.array(Image.open(sf))
        mask0 = (ann == 0).astype(np.uint8) * 255
        overlay = img.copy()
        overlay[mask0 > 0] = [255, 255, 0]  # yellow for class 0
        vis = (img * 0.5 + overlay * 0.5).astype(np.uint8)

        ax_img = axes[idx * 2]
        ax_msk = axes[idx * 2 + 1]
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(stem[:10], fontsize=6)
        ax_msk.imshow(vis)
        ax_msk.axis('off')
        pct = 100 * (ann == 0).mean()
        ax_msk.set_title(f"cl0={pct:.1f}%", fontsize=6)

    plt.suptitle("WaterScenes class 0 QA — yellow = class 0 pixels\nDecide: sky(2), ignore(255), or fallback to free_space", fontsize=9)
    plt.tight_layout()
    outpath = FIGS / 'waterscenes_class0_qa.png'
    plt.savefig(outpath, dpi=80)
    plt.close()
    log.info(f"Saved: {outpath}")
    log.info("ACTION REQUIRED: Inspect figures/waterscenes_class0_qa.png and update log §0b with decision")
    # Default decision (will be updated after visual inspection):
    # Class 0 appears to be background/sky-like based on WaterScenes paper description
    WATERSCENES_CLASS0_DECISION = 'sky'  # will be updated after QA
    log.info(f"Default class 0 decision: {WATERSCENES_CLASS0_DECISION} — override in §0e if needed")

# ──────────────────────────────────────────────
# §0c — Within-Dataset Deduplication
# ──────────────────────────────────────────────
log.info("\n--- §0c: pHash deduplication ---")

def dedup_dataset(name, img_paths, hash_thr=10):
    """Deduplicate a list of image paths using pHash with Hamming distance <= hash_thr.
    Returns (kept_paths, stats_dict).
    """
    log.info(f"  [{name}] computing pHash for {len(img_paths)} images...")
    t0 = time.time()
    hashes = []
    valid_paths = []
    for p in img_paths:
        try:
            h = imagehash.phash(Image.open(p).convert('RGB'), hash_size=8)
            hashes.append(h)
            valid_paths.append(p)
        except Exception as e:
            log.warning(f"    skip {p}: {e}")

    log.info(f"  [{name}] hashed {len(valid_paths)} images in {time.time()-t0:.1f}s")

    # Greedy clustering: iterate, skip if within threshold of any kept hash
    kept_indices = []
    kept_hashes = []
    for i, h in enumerate(hashes):
        is_dup = any((h - kh) <= hash_thr for kh in kept_hashes)
        if not is_dup:
            kept_indices.append(i)
            kept_hashes.append(h)

    kept_paths = [valid_paths[i] for i in kept_indices]
    stats = {
        'dataset': name,
        'n_original': len(img_paths),
        'n_valid': len(valid_paths),
        'n_deduplicated': len(kept_paths),
        'n_removed': len(valid_paths) - len(kept_paths),
        'pct_removed': round(100 * (len(valid_paths) - len(kept_paths)) / max(1, len(valid_paths)), 1),
    }
    log.info(f"  [{name}] {stats['n_original']} → {stats['n_deduplicated']} images "
             f"({stats['n_removed']} removed, {stats['pct_removed']}%)")
    return kept_paths, stats

dedup_stats_all = []
dedup_kept = {}  # dataset_name -> list of kept image paths

# MULTIAQUA: only images with annotations
log.info("\n  MULTIAQUA...")
ann_names = {af.stem for af in MAQUA_ANN.iterdir()}
maqua_img_list = sorted([MAQUA_IMG / (n + '.png') for n in ann_names
                         if (MAQUA_IMG / (n + '.png')).exists()])
if not maqua_img_list:
    maqua_img_list = sorted([p for p in MAQUA_IMG.iterdir()
                              if p.stem in ann_names])
log.info(f"  MULTIAQUA: {len(maqua_img_list)} images with annotations")
kept_maqua, stats = dedup_dataset('MULTIAQUA_night', maqua_img_list)
dedup_kept['MULTIAQUA_night'] = kept_maqua
dedup_stats_all.append(stats)

# OrcaTech: all splits combined (treat 1280x640 and 640x320 separately per issue)
log.info("\n  OrcaTech 1280x640...")
orca_1280_train = sorted((DATA / 'orca_tech_water_segmentation/training/training/1280_640_undistorted').iterdir())
orca_1280_test  = sorted((DATA / 'orca_tech_water_segmentation/test/test/1280_640_undistorted').iterdir())
orca_1280_all   = orca_1280_train + orca_1280_test
kept_orca1280, stats = dedup_dataset('OrcaTech_1280x640', orca_1280_all)
dedup_kept['OrcaTech_1280x640'] = kept_orca1280
dedup_stats_all.append(stats)

log.info("  OrcaTech 640x320...")
orca_640_train = sorted((DATA / 'orca_tech_water_segmentation/training/training/640_320_undistorted').iterdir())
orca_640_test  = sorted((DATA / 'orca_tech_water_segmentation/test/test/640_320_undistorted').iterdir())
orca_640_all   = orca_640_train + orca_640_test
kept_orca640, stats = dedup_dataset('OrcaTech_640x320', orca_640_all)
dedup_kept['OrcaTech_640x320'] = kept_orca640
dedup_stats_all.append(stats)

# Tampere: images from dock, channel, open folders
log.info("\n  Tampere-WaterSeg...")
tampere_imgs = []
for scene in ['dock', 'channel', 'open']:
    scene_dir = DATA / 'Tampere-WaterSeg' / scene
    if scene_dir.exists():
        tampere_imgs.extend(sorted(scene_dir.iterdir()))
kept_tampere, stats = dedup_dataset('Tampere-WaterSeg', tampere_imgs)
dedup_kept['Tampere-WaterSeg'] = kept_tampere
dedup_stats_all.append(stats)

# ROSEBUD: images only
log.info("\n  ROSEBUD...")
rosebud_imgs = sorted((DATA / 'ROSEBUD/bundle/ROSEBUD-HD/HD/images').iterdir())
kept_rosebud, stats = dedup_dataset('ROSEBUD', rosebud_imgs)
dedup_kept['ROSEBUD'] = kept_rosebud
dedup_stats_all.append(stats)

# WaterScenes: largest dataset — do last
log.info("\n  WaterScenes (54k images — takes ~2-3 min)...")
ws_imgs = sorted(WS_IMG.iterdir())
kept_ws, stats = dedup_dataset('WaterScenes', ws_imgs)
dedup_kept['WaterScenes'] = kept_ws
dedup_stats_all.append(stats)

# Save dedup stats
dedup_csv = SHARED / 'dedup_stats.csv'
with open(dedup_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','n_original','n_valid','n_deduplicated','n_removed','pct_removed'])
    w.writeheader()
    w.writerows(dedup_stats_all)
log.info(f"\nSaved dedup_stats.csv: {dedup_csv}")

for s in dedup_stats_all:
    log.info(f"  {s['dataset']:30s}: {s['n_original']:6d} → {s['n_deduplicated']:6d} ({s['pct_removed']}% removed)")

# Save kept image path lists for downstream use
import json
kept_index = {}
for ds, paths in dedup_kept.items():
    kept_index[ds] = [str(p) for p in paths]
with open(SHARED / 'dedup_kept_index.json', 'w') as f:
    json.dump(kept_index, f, indent=2)
log.info("Saved dedup_kept_index.json")

# ──────────────────────────────────────────────
# §0d — Sky/Obstacle Heuristic for Binary Datasets
# ──────────────────────────────────────────────
log.info("\n--- §0d: Sky/obstacle heuristic for binary datasets ---")

def get_gt_path_orca(img_path):
    """Resolve OrcaTech image path to its GT mask path."""
    p = Path(img_path)
    # e.g. .../1280_640_undistorted/X01_1_0000021600.png -> .../1280_640_undistorted_gt/X01_1_0000021600.png
    parent_name = p.parent.name  # e.g. 1280_640_undistorted
    gt_dir = p.parent.parent / (parent_name + '_gt')
    return gt_dir / p.name

def apply_heuristic(img_arr, water_mask):
    """
    Per-column waterline heuristic.
    water_mask: bool array H×W, True where water.
    Returns harmonized uint8 array with values 0=obstacle, 1=water, 2=sky, 255=ignore.
    """
    H, W = img_arr.shape[:2]
    result = np.full((H, W), 255, dtype=np.uint8)

    # Water pixels
    result[water_mask] = 1

    # Per-column topmost water row
    y_water = np.full(W, H, dtype=np.int32)  # default = below frame (no water in column)
    for x in range(W):
        col = np.where(water_mask[:, x])[0]
        if len(col) > 0:
            y_water[x] = col[0]

    # Classify non-water pixels
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    non_water = ~water_mask

    # Sky: non-water pixels above waterline in their column
    sky_mask = non_water & (y_idx < y_water[x_idx])
    result[sky_mask] = 2

    # Obstacle candidates: non-water pixels at or below waterline
    below_water = non_water & (y_idx >= y_water[x_idx])

    # Find connected components of below_water region
    below_uint8 = below_water.astype(np.uint8)
    n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(below_uint8, connectivity=8)

    # For each component: check if within 5px of water mask
    water_uint8 = water_mask.astype(np.uint8)
    water_dilated = cv2.dilate(water_uint8, np.ones((11, 11), np.uint8))  # 5px dilation

    for label_id in range(1, n_labels):
        comp_mask = (labels == label_id)
        # If any pixel of this component overlaps with dilated water → obstacle
        if np.any(comp_mask & (water_dilated > 0)):
            result[comp_mask] = 0
        # else: leave as ignore (255)

    return result


def qa_heuristic(dataset_name, img_paths, gt_resolver, water_class_val, sample_n=10):
    """
    Apply heuristic to sample images, compute mislabel rate, save QA figure.
    water_class_val: pixel value in GT that means water.
    gt_resolver: callable(img_path) -> gt_mask_path
    Returns: pass_flag (True if <15% mislabeled), mislabel_rate
    """
    n = min(sample_n, len(img_paths))
    step = max(1, len(img_paths) // n)
    samples = [img_paths[i * step] for i in range(n)]

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    total_px = 0
    mislabeled_px = 0
    heuristic_results = []

    for idx, img_path in enumerate(samples):
        gt_path = gt_resolver(img_path)
        if not Path(gt_path).exists():
            log.warning(f"  GT not found: {gt_path}")
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        gt  = np.array(Image.open(gt_path))
        water_mask = (gt == water_class_val)
        harmonized = apply_heuristic(img, water_mask)
        heuristic_results.append((img_path, harmonized))

        # Visualize
        COLORS = {0: [255, 0, 0], 1: [0, 0, 255], 2: [135, 206, 235], 255: [128, 128, 128]}
        vis_mask = np.zeros((*harmonized.shape, 3), dtype=np.uint8)
        for cls, color in COLORS.items():
            vis_mask[harmonized == cls] = color

        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(Path(img_path).name[:15], fontsize=7)
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(gt, cmap='gray')
        axes[idx, 1].set_title('GT (binary)', fontsize=7)
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow((img * 0.4 + vis_mask * 0.6).clip(0, 255).astype(np.uint8))
        axes[idx, 2].set_title('red=obs, blue=water, cyan=sky, gray=ign', fontsize=6)
        axes[idx, 2].axis('off')

    # Build legend
    legend_els = [mpatches.Patch(color='red', label='obstacle(0)'),
                  mpatches.Patch(color='blue', label='water(1)'),
                  mpatches.Patch(color='lightblue', label='sky(2)'),
                  mpatches.Patch(color='gray', label='ignore(255)')]
    fig.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=8)
    plt.suptitle(f"{dataset_name} — heuristic sky/obstacle split QA", fontsize=10)
    plt.tight_layout()
    out = FIGS / f'heuristic_qa_{dataset_name}.png'
    plt.savefig(out, dpi=80)
    plt.close()
    log.info(f"  Saved QA: {out}")
    return heuristic_results

# OrcaTech 1280x640 QA
log.info("  OrcaTech 1280x640 heuristic QA...")
orca1280_gt_dir_train = DATA / 'orca_tech_water_segmentation/training/training/1280_640_undistorted_gt'
orca1280_gt_dir_test  = DATA / 'orca_tech_water_segmentation/test/test/1280_640_undistorted_gt'
orca640_gt_dir_train  = DATA / 'orca_tech_water_segmentation/training/training/640_320_undistorted_gt'
orca640_gt_dir_test   = DATA / 'orca_tech_water_segmentation/test/test/640_320_undistorted_gt'

def orca_gt_resolver(img_path, gt_dirs):
    stem = Path(img_path).stem
    for gt_dir in gt_dirs:
        candidate = Path(gt_dir) / (stem + '.png')
        if candidate.exists():
            return candidate
    return Path(gt_dirs[0]) / (stem + '.png')  # fallback

orca1280_heuristic = qa_heuristic(
    'OrcaTech_1280x640', dedup_kept['OrcaTech_1280x640'],
    gt_resolver=lambda p: orca_gt_resolver(p, [orca1280_gt_dir_train, orca1280_gt_dir_test]),
    water_class_val=1, sample_n=10
)

log.info("  OrcaTech 640x320 heuristic QA...")
orca640_heuristic = qa_heuristic(
    'OrcaTech_640x320', dedup_kept['OrcaTech_640x320'],
    gt_resolver=lambda p: orca_gt_resolver(p, [orca640_gt_dir_train, orca640_gt_dir_test]),
    water_class_val=1, sample_n=10
)

log.info("  Tampere-WaterSeg heuristic QA...")
# Tampere: image from dock/channel/open dir, mask from dock_mask/channel_mask/open_mask
def tampere_gt_resolver(img_path):
    p = Path(img_path)
    scene = p.parent.name  # dock, channel, or open
    # mask filename: {scene}_mask_{NNN}.png
    num = p.stem.split('_')[-1] if '_' in p.stem else p.stem
    mask_name = f'{scene}_mask_{num}.png'
    mask_path = DATA / 'Tampere-WaterSeg' / (scene + '_mask') / mask_name
    if mask_path.exists():
        return mask_path
    # fallback: search mask dir for matching number
    mask_dir = DATA / 'Tampere-WaterSeg' / (scene + '_mask')
    for mf in mask_dir.iterdir():
        if mf.stem.endswith(num):
            return mf
    return mask_path

tampere_heuristic = qa_heuristic(
    'Tampere-WaterSeg', dedup_kept['Tampere-WaterSeg'],
    gt_resolver=tampere_gt_resolver,
    water_class_val=255, sample_n=10
)

# ──────────────────────────────────────────────
# §0e — Label Harmonization
# ──────────────────────────────────────────────
log.info("\n--- §0e: Label harmonization ---")

# WATERSCENES_CLASS0_DECISION from §0b — default 'sky', update after QA inspection
# This is a hard-coded decision that should be updated after viewing the QA figure.
# Options: 'sky' (→ 2), 'ignore' (→ 255), 'free_space' (use binary water/non-water only)
WS_CLASS0_MAP = 2  # sky — will be confirmed/overridden after QA

class_balance_rows = []

def write_harmonized(dataset_name, pairs, transform_fn, img_list=None):
    """
    pairs: list of (img_path, mask_path) or (img_path, mask_array)
    transform_fn: callable(mask_array) -> harmonized_array
    Writes to shared_data/harmonized_{dataset_name}/masks/ and images.txt
    """
    out_dir = SHARED / f'harmonized_{dataset_name}'
    mask_dir = out_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_index = []
    px_counts = defaultdict(int)
    total_px = 0

    log.info(f"  Harmonizing {dataset_name} ({len(pairs)} images)...")
    for i, item in enumerate(pairs):
        if i % 1000 == 0:
            log.info(f"    {i}/{len(pairs)}...")

        if isinstance(item, tuple) and len(item) == 2:
            img_path, mask_src = item
        else:
            img_path = item
            mask_src = None

        try:
            if isinstance(mask_src, np.ndarray):
                raw_mask = mask_src
            elif mask_src is not None:
                raw_mask = np.array(Image.open(mask_src))
            else:
                continue

            harm = transform_fn(raw_mask)
            stem = Path(img_path).stem
            out_mask_path = mask_dir / (stem + '.png')
            Image.fromarray(harm).save(out_mask_path)

            img_index.append(str(img_path))
            for v in [0, 1, 2, 255]:
                px_counts[v] += int((harm == v).sum())
            total_px += harm.size

        except Exception as e:
            log.warning(f"    ERROR {img_path}: {e}")

    # images.txt index
    with open(out_dir / 'images.txt', 'w') as f:
        f.write('\n'.join(img_index))

    n = len(img_index)
    if total_px > 0 and n > 0:
        row = {
            'dataset': dataset_name,
            'n_images_deduplicated': n,
            'pct_obstacle': round(100 * px_counts[0] / total_px, 2),
            'pct_water':    round(100 * px_counts[1] / total_px, 2),
            'pct_sky':      round(100 * px_counts[2] / total_px, 2),
            'pct_ignore':   round(100 * px_counts[255] / total_px, 2),
        }
        class_balance_rows.append(row)
        log.info(f"  {dataset_name}: n={n}, obs={row['pct_obstacle']}%, water={row['pct_water']}%, "
                 f"sky={row['pct_sky']}%, ign={row['pct_ignore']}%")
    return n


# ── MULTIAQUA harmonization ──
# VERIFIED mapping via pixel-distribution analysis (NOT the issue's original guess):
# - Class 0 = dynamic obstacle (boats/buoys) → LaRS obstacle (0)
# - Class 1 = static obstacle (shore/mountains/piers) → LaRS obstacle (0)
# - Class 2 = recording boat (auxiliary, very rare ~0.1%) → LaRS ignore (255)
# - Class 3 = water (43.8% of image, all bottom) → LaRS water (1)
# - Class 4 = sky (45.7% of image, 97% top) → LaRS sky (2)
# Issue's original guess (0=obstacle, 1=water, 2=sky) was WRONG — confirmed by visual QA.
MAQUA_MAPPING = {0: 0, 1: 0, 2: 255, 3: 1, 4: 2}

def multiaqua_transform(mask):
    out = np.full_like(mask, 255, dtype=np.uint8)
    for src, dst in MAQUA_MAPPING.items():
        out[mask == src] = dst
    return out

# Build pairs
def multiaqua_pairs():
    for img_path in dedup_kept['MULTIAQUA_night']:
        ann_path = MAQUA_ANN / (Path(img_path).stem + '.png')
        if ann_path.exists():
            yield (img_path, ann_path)

write_harmonized('MULTIAQUA_night', list(multiaqua_pairs()), multiaqua_transform)


# ── OrcaTech harmonization (use heuristic outputs) ──
def orca_pairs_with_heuristic(heuristic_results, all_img_paths, gt_dirs, water_val=1):
    """Return (img_path, harmonized_mask_array) for all images."""
    # heuristic_results covers only the QA sample — need to process all
    result = []
    for img_path in all_img_paths:
        stem = Path(img_path).stem
        gt_path = None
        for gd in gt_dirs:
            candidate = Path(gd) / (stem + '.png')
            if candidate.exists():
                gt_path = candidate
                break
        if gt_path is None:
            continue
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            gt  = np.array(Image.open(gt_path))
            water_mask = (gt == water_val)
            harm = apply_heuristic(img, water_mask)
            result.append((img_path, harm))
        except Exception as e:
            log.warning(f"  orca error {img_path}: {e}")
    return result

log.info("  Building OrcaTech 1280x640 harmonized pairs (full set)...")
orca1280_full = orca_pairs_with_heuristic(
    orca1280_heuristic, dedup_kept['OrcaTech_1280x640'],
    [orca1280_gt_dir_train, orca1280_gt_dir_test], water_val=1)
write_harmonized('OrcaTech_1280x640', orca1280_full, lambda m: m)  # already harmonized

log.info("  Building OrcaTech 640x320 harmonized pairs (full set)...")
orca640_full = orca_pairs_with_heuristic(
    orca640_heuristic, dedup_kept['OrcaTech_640x320'],
    [orca640_gt_dir_train, orca640_gt_dir_test], water_val=1)
write_harmonized('OrcaTech_640x320', orca640_full, lambda m: m)


# ── Tampere harmonization ──
log.info("  Building Tampere harmonized pairs (full set)...")

def tampere_full_pairs():
    for img_path in dedup_kept['Tampere-WaterSeg']:
        gt_path = tampere_gt_resolver(img_path)
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            gt  = np.array(Image.open(gt_path))
            water_mask = (gt == 255)
            harm = apply_heuristic(img, water_mask)
            yield (img_path, harm)
        except Exception as e:
            log.warning(f"  tampere error {img_path}: {e}")

write_harmonized('Tampere-WaterSeg', list(tampere_full_pairs()), lambda m: m)


# ── WaterScenes harmonization ──
log.info("  Building WaterScenes harmonized pairs (full deduplicated set)...")

def ws_transform(mask):
    out = np.full_like(mask, 255, dtype=np.uint8)
    out[mask == 0]  = WS_CLASS0_MAP  # sky or ignore based on §0b QA
    for c in range(1, 8):            # classes 1-7 → obstacle
        out[mask == c] = 0
    out[mask == 8]  = 1              # class 8 → water
    return out

def ws_pairs():
    ws_sem_dir = WS_SEM
    for img_path in dedup_kept['WaterScenes']:
        stem = Path(img_path).stem
        mask_path = ws_sem_dir / (stem + '.png')
        if mask_path.exists():
            yield (img_path, mask_path)

write_harmonized('WaterScenes', list(ws_pairs()), ws_transform)


# ── ROSEBUD harmonization ──
log.info("  Building ROSEBUD harmonized pairs...")
ROSEBUD_BASE = DATA / 'ROSEBUD/bundle/ROSEBUD-HD/HD'

ROSEBUD_RGB_MAP = {
    (255, 255, 255): 1,   # water
    (255,   0, 255): 2,   # sky
    (  0,   0, 255): 0,   # shore/bank → obstacle
    (  0, 255,   0): 0,   # bridges → obstacle
    (255,   0,   0): 0,   # boats/cars/people → obstacle
    (255, 255,   0): 0,   # trees/foliage → obstacle
    (  0, 255, 255): 0,   # debris → obstacle
    (  0,   0,   0): 255, # unlabeled → ignore
}

def rosebud_transform(rgb_mask):
    """Convert ROSEBUD RGB mask (H,W,3) to harmonized (H,W) uint8."""
    H, W = rgb_mask.shape[:2]
    out = np.full((H, W), 255, dtype=np.uint8)
    for rgb_tuple, lars_cls in ROSEBUD_RGB_MAP.items():
        r, g, b = rgb_tuple
        match = (rgb_mask[:, :, 0] == r) & (rgb_mask[:, :, 1] == g) & (rgb_mask[:, :, 2] == b)
        out[match] = lars_cls
    return out

def rosebud_pairs():
    for img_path in dedup_kept['ROSEBUD']:
        stem = Path(img_path).stem
        num = stem.split('_')[-1] if '_' in stem else stem
        mask_path = ROSEBUD_BASE / 'rgb_fluvial_masks' / f'rgb_fluvial_mask_{num}.png'
        if mask_path.exists():
            yield (img_path, mask_path)

write_harmonized('ROSEBUD', list(rosebud_pairs()), rosebud_transform)


# ── Write class_balance.csv ──
balance_csv = SHARED / 'class_balance.csv'
with open(balance_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','n_images_deduplicated',
                                       'pct_obstacle','pct_water','pct_sky','pct_ignore'])
    w.writeheader()
    w.writerows(class_balance_rows)
log.info(f"\nSaved class_balance.csv: {balance_csv}")

log.info("\n" + "=" * 60)
log.info("Analysis 00 COMPLETE")
log.info("Next steps:")
log.info("  1. Inspect figures/multiaqua_class3_qa.png and multiaqua_class4_qa.png")
log.info("     → Update MULTIAQUA_MAPPING in script + log if classes 3/4 are not ignore")
log.info("  2. Inspect figures/waterscenes_class0_qa.png")
log.info("     → Confirm WS_CLASS0_MAP=2 (sky) or change to 255 (ignore)")
log.info("  3. Inspect figures/heuristic_qa_*.png for OrcaTech and Tampere")
log.info("     → Confirm heuristic passes (<15% mislabeled) or fallback to ignore")
log.info("=" * 60)
