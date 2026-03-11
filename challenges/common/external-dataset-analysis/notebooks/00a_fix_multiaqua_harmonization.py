#!/usr/bin/env python3
"""
Fixup: re-do MULTIAQUA harmonization with the correct class mapping.
Run AFTER 00_prerequisites.py has completed (dedup_kept_index.json must exist).
"""
import json, logging, sys, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

REPO   = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
SHARED = REPO / 'shared_data'
MAQUA_ANN = Path('/home/farms/data/datasets/MULTIAQUA_night/annotations')
LOGS_DIR = REPO / 'logs'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler(LOGS_DIR/'analysis_00a_fix.log'),
                               logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# Correct mapping (verified by visual inspection)
MAQUA_MAPPING = {0: 0, 1: 0, 2: 255, 3: 1, 4: 2}

with open(SHARED / 'dedup_kept_index.json') as f:
    dedup_kept = json.load(f)

kept_paths = [Path(p) for p in dedup_kept.get('MULTIAQUA_night', [])]
log.info(f"MULTIAQUA: {len(kept_paths)} deduplicated images to re-harmonize")

# Remove old harmonized dir
import shutil
old_dir = SHARED / 'harmonized_MULTIAQUA_night'
if old_dir.exists():
    shutil.rmtree(old_dir)
    log.info(f"Removed old harmonized dir: {old_dir}")

mask_dir = old_dir / 'masks'
mask_dir.mkdir(parents=True)

px_counts = defaultdict(int)
total_px = 0
img_index = []

for i, img_path in enumerate(kept_paths):
    ann_path = MAQUA_ANN / (img_path.stem + '.png')
    if not ann_path.exists():
        continue
    try:
        raw = np.array(Image.open(ann_path))
        out = np.full_like(raw, 255, dtype=np.uint8)
        for src, dst in MAQUA_MAPPING.items():
            out[raw == src] = dst
        out_path = mask_dir / (img_path.stem + '.png')
        Image.fromarray(out).save(out_path)
        img_index.append(str(img_path))
        for v in [0, 1, 2, 255]:
            px_counts[v] += int((out == v).sum())
        total_px += out.size
        if i % 500 == 0:
            log.info(f"  {i}/{len(kept_paths)}")
    except Exception as e:
        log.warning(f"  {img_path}: {e}")

(old_dir / 'images.txt').write_text('\n'.join(img_index))
n = len(img_index)
log.info(f"Done: {n} images harmonized")
if total_px > 0:
    log.info(f"  obs={100*px_counts[0]/total_px:.1f}%  water={100*px_counts[1]/total_px:.1f}%  "
             f"sky={100*px_counts[2]/total_px:.1f}%  ign={100*px_counts[255]/total_px:.1f}%")

# Update class_balance.csv
balance_csv = SHARED / 'class_balance.csv'
rows = []
if balance_csv.exists():
    with open(balance_csv) as f:
        rows = list(csv.DictReader(f))
# Remove old MULTIAQUA row
rows = [r for r in rows if r['dataset'] != 'MULTIAQUA_night']
# Add correct row
rows.append({
    'dataset': 'MULTIAQUA_night',
    'n_images_deduplicated': n,
    'pct_obstacle': round(100*px_counts[0]/max(total_px,1), 2),
    'pct_water':    round(100*px_counts[1]/max(total_px,1), 2),
    'pct_sky':      round(100*px_counts[2]/max(total_px,1), 2),
    'pct_ignore':   round(100*px_counts[255]/max(total_px,1), 2),
})
with open(balance_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','n_images_deduplicated','pct_obstacle','pct_water','pct_sky','pct_ignore'])
    w.writeheader()
    w.writerows(rows)
log.info(f"Updated class_balance.csv: {balance_csv}")
