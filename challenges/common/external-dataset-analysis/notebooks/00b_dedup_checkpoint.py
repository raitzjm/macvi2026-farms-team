#!/usr/bin/env python3
"""
Analysis 00b — Within-Dataset Deduplication with Per-Dataset Checkpointing
Saves a JSON checkpoint after each dataset completes so restarts are cheap.
On completion, writes shared_data/dedup_kept_index.json and dedup_stats.csv.
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/00b_dedup_checkpoint.py
"""
import json, sys, csv, time, logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
import imagehash

N_WORKERS = 14  # leave 2 cores free


def _hash_one(path_str):
    """Worker: open image, compute pHash, return (path_str, hash_flat_bytes) or (path_str, None)."""
    try:
        h = imagehash.phash(Image.open(path_str).convert('RGB'), hash_size=8)
        return path_str, h.hash.flatten().tobytes()
    except Exception as e:
        return path_str, None

REPO   = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
DATA   = Path('/home/farms/data/datasets')
SHARED = REPO / 'shared_data'
LOGS_DIR = REPO / 'logs'
CKPT_DIR = SHARED / 'dedup_checkpoints'

SHARED.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_00b_dedup.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=== Analysis 00b: Deduplication with checkpointing ===")


def dedup_dataset(name, img_paths, hash_thr=10):
    """pHash dedup with multiprocessing + numpy vectorized greedy dedup."""
    # Fast-path: load checkpoint if exists
    ckpt = CKPT_DIR / f'{name}.json'
    if ckpt.exists():
        data = json.loads(ckpt.read_text())
        log.info(f"  [{name}] CHECKPOINT found — skipping ({data['stats']['n_deduplicated']} kept)")
        return data['kept_paths'], data['stats']

    img_paths_str = [str(p) for p in img_paths]
    n = len(img_paths_str)
    workers = min(N_WORKERS, n, 14)
    log.info(f"  [{name}] computing pHash for {n} images (workers={workers})...")
    t0 = time.time()

    # Parallel hash computation — preserve original order
    hash_bytes = [None] * n
    path_to_idx = {p: i for i, p in enumerate(img_paths_str)}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_hash_one, p): i for i, p in enumerate(img_paths_str)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            path_str, hb = fut.result()
            if hb is not None:
                hash_bytes[idx] = hb
            done += 1
            if done % 5000 == 0:
                log.info(f"    [{name}] {done}/{n} hashed...")

    valid_indices = [i for i, hb in enumerate(hash_bytes) if hb is not None]
    valid_paths = [img_paths_str[i] for i in valid_indices]
    # Stack hash bits into numpy array for vectorized Hamming
    hash_mat = np.stack([
        np.frombuffer(hash_bytes[i], dtype=np.bool_) for i in valid_indices
    ])  # shape (n_valid, 64)
    log.info(f"  [{name}] hashed {len(valid_paths)}/{n} images in {time.time()-t0:.1f}s")

    # Vectorized greedy dedup
    t1 = time.time()
    kept_indices_local = []
    kept_rows = []  # list of 1-D bool arrays

    for i in range(len(valid_paths)):
        h = hash_mat[i]
        if kept_rows:
            kept_arr = np.stack(kept_rows)            # shape (k, 64)
            dists = (kept_arr != h[np.newaxis, :]).sum(axis=1)
            if dists.min() <= hash_thr:
                continue
        kept_indices_local.append(i)
        kept_rows.append(h)
        if len(kept_indices_local) % 2000 == 0:
            log.info(f"    [{name}] greedy: {i+1}/{len(valid_paths)} checked, {len(kept_indices_local)} kept")

    log.info(f"  [{name}] greedy dedup done in {time.time()-t1:.1f}s")
    kept_paths = [valid_paths[i] for i in kept_indices_local]
    stats = {
        'dataset': name,
        'n_original': len(img_paths),
        'n_valid': len(valid_paths),
        'n_deduplicated': len(kept_paths),
        'n_removed': len(valid_paths) - len(kept_paths),
        'pct_removed': round(100 * (len(valid_paths) - len(kept_paths)) / max(1, len(valid_paths)), 1),
    }
    log.info(f"  [{name}] {stats['n_original']} → {stats['n_deduplicated']} "
             f"({stats['n_removed']} removed, {stats['pct_removed']}%)")

    # Save checkpoint
    ckpt.write_text(json.dumps({'kept_paths': kept_paths, 'stats': stats}, indent=2))
    log.info(f"  [{name}] checkpoint saved: {ckpt}")
    return kept_paths, stats


dedup_stats_all = []
dedup_kept = {}

# MULTIAQUA
log.info("\n  MULTIAQUA...")
MAQUA_ANN = DATA / 'MULTIAQUA_night/annotations'
MAQUA_IMG = DATA / 'MULTIAQUA_night/data/zed'
ann_names = {af.stem for af in MAQUA_ANN.iterdir()}
maqua_imgs = sorted([MAQUA_IMG / (n + '.png') for n in ann_names if (MAQUA_IMG / (n + '.png')).exists()])
if not maqua_imgs:
    maqua_imgs = sorted([p for p in MAQUA_IMG.iterdir() if p.stem in ann_names])
log.info(f"  MULTIAQUA: {len(maqua_imgs)} images with annotations")
kept, stats = dedup_dataset('MULTIAQUA_night', maqua_imgs)
dedup_kept['MULTIAQUA_night'] = kept
dedup_stats_all.append(stats)

# OrcaTech 1280x640
log.info("\n  OrcaTech 1280x640...")
orca_1280 = (sorted((DATA / 'orca_tech_water_segmentation/training/training/1280_640_undistorted').iterdir()) +
             sorted((DATA / 'orca_tech_water_segmentation/test/test/1280_640_undistorted').iterdir()))
kept, stats = dedup_dataset('OrcaTech_1280x640', orca_1280)
dedup_kept['OrcaTech_1280x640'] = kept
dedup_stats_all.append(stats)

# OrcaTech 640x320
log.info("  OrcaTech 640x320...")
orca_640 = (sorted((DATA / 'orca_tech_water_segmentation/training/training/640_320_undistorted').iterdir()) +
            sorted((DATA / 'orca_tech_water_segmentation/test/test/640_320_undistorted').iterdir()))
kept, stats = dedup_dataset('OrcaTech_640x320', orca_640)
dedup_kept['OrcaTech_640x320'] = kept
dedup_stats_all.append(stats)

# Tampere
log.info("\n  Tampere-WaterSeg...")
tampere_imgs = []
for scene in ['dock', 'channel', 'open']:
    d = DATA / 'Tampere-WaterSeg' / scene
    if d.exists():
        tampere_imgs.extend(sorted(d.iterdir()))
kept, stats = dedup_dataset('Tampere-WaterSeg', tampere_imgs)
dedup_kept['Tampere-WaterSeg'] = kept
dedup_stats_all.append(stats)

# ROSEBUD
log.info("\n  ROSEBUD...")
rosebud_imgs = sorted((DATA / 'ROSEBUD/bundle/ROSEBUD-HD/HD/images').iterdir())
kept, stats = dedup_dataset('ROSEBUD', rosebud_imgs)
dedup_kept['ROSEBUD'] = kept
dedup_stats_all.append(stats)

# WaterScenes (bottleneck ~54k images)
log.info("\n  WaterScenes (54k images — longest step)...")
ws_imgs = sorted((DATA / 'water_scenes/image').iterdir())
log.info(f"  WaterScenes: {len(ws_imgs)} images found")
kept, stats = dedup_dataset('WaterScenes', ws_imgs)
dedup_kept['WaterScenes'] = kept
dedup_stats_all.append(stats)

# Write final outputs
dedup_csv = SHARED / 'dedup_stats.csv'
with open(dedup_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','n_original','n_valid','n_deduplicated','n_removed','pct_removed'])
    w.writeheader()
    w.writerows(dedup_stats_all)
log.info(f"\nSaved: {dedup_csv}")

with open(SHARED / 'dedup_kept_index.json', 'w') as f:
    json.dump(dedup_kept, f, indent=2)
log.info("Saved: dedup_kept_index.json")

log.info("\nDedup summary:")
for s in dedup_stats_all:
    log.info(f"  {s['dataset']:30s}: {s['n_original']:6d} → {s['n_deduplicated']:6d} ({s['pct_removed']}% removed)")

log.info("=== Analysis 00b COMPLETE ===")
