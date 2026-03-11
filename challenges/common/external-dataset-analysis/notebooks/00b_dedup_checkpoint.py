#!/usr/bin/env python3
"""
Analysis 00b — Within-Dataset Deduplication with Per-Dataset Checkpointing
Saves a JSON checkpoint after each dataset completes so restarts are cheap.
On completion, writes shared_data/dedup_kept_index.json and dedup_stats.csv.
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/00b_dedup_checkpoint.py
"""
import json, sys, csv, time, logging
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

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
    """pHash dedup; returns (kept_paths_str, stats_dict)."""
    # Fast-path: load checkpoint if exists
    ckpt = CKPT_DIR / f'{name}.json'
    if ckpt.exists():
        data = json.loads(ckpt.read_text())
        log.info(f"  [{name}] CHECKPOINT found — skipping ({data['stats']['n_deduplicated']} kept)")
        return data['kept_paths'], data['stats']

    log.info(f"  [{name}] computing pHash for {len(img_paths)} images...")
    t0 = time.time()
    hashes = []
    valid_paths = []
    for i, p in enumerate(img_paths):
        try:
            h = imagehash.phash(Image.open(p).convert('RGB'), hash_size=8)
            hashes.append(h)
            valid_paths.append(str(p))
        except Exception as e:
            log.warning(f"    skip {p}: {e}")
        if (i + 1) % 5000 == 0:
            log.info(f"    [{name}] {i+1}/{len(img_paths)} hashed...")

    log.info(f"  [{name}] hashed {len(valid_paths)} images in {time.time()-t0:.1f}s")

    # Greedy dedup
    kept_indices = []
    kept_hashes = []
    for i, h in enumerate(hashes):
        if not any((h - kh) <= hash_thr for kh in kept_hashes):
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
