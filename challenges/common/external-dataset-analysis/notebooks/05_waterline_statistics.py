#!/usr/bin/env python3
"""
Analysis 05 — Waterline Statistics
Prereq: Analysis 00 complete
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/05_waterline_statistics.py
"""
import sys, logging
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO   = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
DATA   = Path('/home/farms/data/datasets')
LARS   = Path('/home/farms/data/lars_dataset')
SHARED = REPO / 'shared_data'
FIGS   = REPO / 'figures'
LOGS_DIR = REPO / 'logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_05_run.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=== Analysis 05: Waterline Statistics ===")

def extract_waterline_y_positions(mask_arr, water_class=1, sample_cols=20):
    """
    For each image, find the top boundary of the water region.
    Returns a list of normalized y positions (0=top, 1=bottom) from sampled columns.
    """
    H, W = mask_arr.shape
    water_mask = (mask_arr == water_class)
    positions = []
    step = max(1, W // sample_cols)
    for x in range(0, W, step):
        col_water = np.where(water_mask[:, x])[0]
        if len(col_water) > 0:
            y_top = col_water[0]
            positions.append(y_top / H)
    return positions

# ──────────────────────────────────────────────
# Collect waterline positions per dataset
# ──────────────────────────────────────────────
all_positions = {}

# LaRS train (ground truth)
log.info("LaRS train waterline...")
lars_train_dir = LARS / 'ann_dir' / 'train' / 'semantic_masks'
lars_test_dir  = LARS / 'ann_dir' / 'val'   / 'semantic_masks'
for split, gt_dir in [('LaRS train', lars_train_dir), ('LaRS val', lars_test_dir)]:
    if not gt_dir.exists():
        continue
    positions = []
    for mf in sorted(gt_dir.iterdir())[:300]:
        try:
            ann = np.array(Image.open(mf))
            positions.extend(extract_waterline_y_positions(ann, water_class=1))
        except:
            pass
    all_positions[split] = positions
    log.info(f"  {split}: {len(positions)} waterline y-positions from {min(300, len(list(gt_dir.iterdir())))} masks")

# External harmonized datasets
for ds_name in ['MULTIAQUA_night', 'OrcaTech_1280x640', 'OrcaTech_640x320',
                'Tampere-WaterSeg', 'WaterScenes', 'ROSEBUD']:
    harm_dir = SHARED / f'harmonized_{ds_name}' / 'masks'
    if not harm_dir.exists():
        log.warning(f"  {ds_name}: harmonized masks not found")
        continue

    mask_files = sorted(harm_dir.iterdir())
    # Subsample large datasets
    if len(mask_files) > 500:
        step = len(mask_files) // 500
        mask_files = mask_files[::step][:500]

    positions = []
    for mf in mask_files:
        try:
            harm = np.array(Image.open(mf))
            positions.extend(extract_waterline_y_positions(harm, water_class=1))
        except:
            pass
    all_positions[ds_name] = positions
    log.info(f"  {ds_name}: {len(positions)} waterline y-positions")

# WaterScenes: also try waterline annotations directly
ws_waterline_dir = DATA / 'water_scenes' / 'waterline' / 'SegmentationClass'
if ws_waterline_dir.exists():
    log.info("  WaterScenes: also extracting from native waterline annotations...")
    ws_wl_positions = []
    for wf in sorted(ws_waterline_dir.iterdir())[:500]:
        try:
            ann = np.array(Image.open(wf))
            # Waterline annotation: pixel=255 means waterline
            wl_rows = np.where(ann == 255)
            if len(wl_rows[0]) > 0:
                ws_wl_positions.append(wl_rows[0].mean() / ann.shape[0])
        except:
            pass
    if ws_wl_positions:
        all_positions['WaterScenes_native_wl'] = ws_wl_positions
        log.info(f"    {len(ws_wl_positions)} waterline positions from native annotations")

# ──────────────────────────────────────────────
# Violin Plot
# ──────────────────────────────────────────────
log.info("Generating waterline violin plot...")

# Filter datasets with enough data
plot_data = {k: v for k, v in all_positions.items() if len(v) >= 20}
labels = sorted(plot_data.keys(), key=lambda x: (0 if 'LaRS' in x else 1, x))
data_arrays = [np.array(plot_data[l]) for l in labels]

fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 7))

parts = ax.violinplot(data_arrays, positions=range(len(labels)),
                      showmeans=True, showmedians=True, showextrema=True)

# Styling
for i, pc in enumerate(parts['bodies']):
    if 'LaRS' in labels[i]:
        pc.set_facecolor('#1f77b4')
    else:
        pc.set_facecolor('#ff7f0e')
    pc.set_alpha(0.7)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Waterline y-position (0=top, 1=bottom of frame)')
ax.set_title('Waterline y-position distribution per dataset\n'
             '(WE_IoU=42.2% in best submission — waterline quality is a key weakness)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='frame midpoint')
ax.invert_yaxis()  # 0=top visually
ax.legend(fontsize=8)
plt.tight_layout()

out = FIGS / 'waterline_distribution.png'
plt.savefig(out, dpi=120)
plt.close()
log.info(f"Saved: {out}")

# Summary stats
log.info("\nWaterline statistics summary:")
for label, arr in zip(labels, data_arrays):
    log.info(f"  {label:30s}: mean={arr.mean():.3f}  std={arr.std():.3f}  n={len(arr)}")

log.info("=== Analysis 05 COMPLETE ===")
