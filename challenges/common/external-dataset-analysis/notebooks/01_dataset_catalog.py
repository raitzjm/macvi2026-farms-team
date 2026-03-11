#!/usr/bin/env python3
"""
Analysis 01 — Dataset Catalog & Sample Gallery
Prereq: Analysis 00 complete (dedup_kept_index.json)
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/01_dataset_catalog.py
"""
import json, sys, logging, math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

REPO   = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
LARS   = Path('/home/farms/data/lars_dataset')
DATA   = Path('/home/farms/data/datasets')
SHARED = REPO / 'shared_data'
FIGS   = REPO / 'figures'
FIGS_GRIDS = FIGS / 'sample_grids'
LOGS_DIR = REPO / 'logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_01_run.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=== Analysis 01: Dataset Catalog ===")

with open(SHARED / 'dedup_kept_index.json') as f:
    dedup_kept = json.load(f)

# ──────────────────────────────────────────────
# Helper: LAB brightness
# ──────────────────────────────────────────────
def image_stats(img_pil):
    """Returns dict: mean_r, mean_g, mean_b, std_r, std_g, std_b, mean_L_lab"""
    arr = np.array(img_pil.convert('RGB')).astype(np.float32)
    mean_r, mean_g, mean_b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    std_r,  std_g,  std_b  = arr[:,:,0].std(),  arr[:,:,1].std(),  arr[:,:,2].std()
    # LAB L* channel
    arr_u8 = arr.astype(np.uint8)
    lab = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2LAB)
    mean_L = float(lab[:,:,0].mean())
    return dict(mean_r=mean_r, mean_g=mean_g, mean_b=mean_b,
                std_r=std_r, std_g=std_g, std_b=std_b, mean_L=mean_L)

# Dataset metadata
DATASET_META = {
    'MULTIAQUA_night': {
        'paper': 'MULTIAQUA (arXiv:2512.17450)', 'license': 'unknown',
        'sensor': 'stereo ZED camera', 'location': 'Mediterranean coast (night)',
        'native_classes': '7 (0-6)', 'notes': 'Night-time; only classes 0-4 observed in this copy',
    },
    'OrcaTech_1280x640': {
        'paper': 'USVInland (OrcaTech)', 'license': 'unknown',
        'sensor': 'OrcaTech USV camera', 'location': 'Inland waterways (Wabash River area)',
        'native_classes': '2 (binary)', 'notes': '1280×640 undistorted; X-prefix sequences',
    },
    'OrcaTech_640x320': {
        'paper': 'USVInland (OrcaTech)', 'license': 'unknown',
        'sensor': 'OrcaTech USV camera (half-res)', 'location': 'Inland waterways (H/N sequences)',
        'native_classes': '2 (binary)', 'notes': '640×320 undistorted; different scenes from 1280×640',
    },
    'Tampere-WaterSeg': {
        'paper': 'Tampere-WaterSeg (TUT)', 'license': 'unknown',
        'sensor': 'various cameras', 'location': 'Tampere region, Finland (harbor/docks)',
        'native_classes': '2 (binary: 0/255)', 'notes': '3 scene types: dock, channel, open',
    },
    'WaterScenes': {
        'paper': 'WaterScenes (RAL 2024)', 'license': 'CC BY-NC 4.0',
        'sensor': 'RGB + radar + lidar', 'location': 'Various waterways (China)',
        'native_classes': '9 (0-8) + free_space + waterline', 'notes': '54k frames; video sequences; heavy dedup expected',
    },
    'ROSEBUD': {
        'paper': 'ROSEBUD (Purdue, doi:10.4231/MMJ2-NH88)', 'license': 'CC BY-NC 4.0',
        'sensor': 'RGB camera on USV', 'location': 'Wabash River, Indiana, USA',
        'native_classes': '7-class fluvial (RGB coded)', 'notes': 'HD 1920×1440; fluvial (river) scenes',
    },
}

# ──────────────────────────────────────────────
# Gallery + stats per dataset
# ──────────────────────────────────────────────
brightness_stats = {}

for ds_name, paths_str in dedup_kept.items():
    paths = [Path(p) for p in paths_str]
    n = len(paths)
    log.info(f"\n{ds_name}: {n} deduplicated images")

    # Uniform sample of 12 images
    n_sample = 12
    step = max(1, n // n_sample)
    samples = paths[::step][:n_sample]

    # Compute stats on sample
    stats_list = []
    for p in samples:
        try:
            img = Image.open(p).convert('RGB')
            stats_list.append(image_stats(img))
        except Exception as e:
            log.warning(f"  {p}: {e}")

    if stats_list:
        mean_L  = np.mean([s['mean_L']  for s in stats_list])
        mean_r  = np.mean([s['mean_r']  for s in stats_list])
        mean_g  = np.mean([s['mean_g']  for s in stats_list])
        mean_b  = np.mean([s['mean_b']  for s in stats_list])
        std_r   = np.mean([s['std_r']   for s in stats_list])
        std_g   = np.mean([s['std_g']   for s in stats_list])
        std_b   = np.mean([s['std_b']   for s in stats_list])
        brightness_stats[ds_name] = dict(
            mean_L=mean_L, mean_r=mean_r, mean_g=mean_g, mean_b=mean_b,
            std_r=std_r, std_g=std_g, std_b=std_b, n=n
        )
        log.info(f"  mean L*={mean_L:.1f}, mean RGB=({mean_r:.0f},{mean_g:.0f},{mean_b:.0f})")

    # Gallery: 3×4 grid
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.5))
    axes = axes.flatten()

    meta = DATASET_META.get(ds_name, {})
    # Resolution from first image
    try:
        w, h = Image.open(samples[0]).size
        res_str = f"{w}×{h}"
    except:
        res_str = "unknown"

    for idx, ax in enumerate(axes):
        if idx < len(samples):
            try:
                img = Image.open(samples[idx]).convert('RGB')
                ax.imshow(img)
                ax.set_title(samples[idx].name[:20], fontsize=6)
            except:
                ax.set_title("error", fontsize=6)
        ax.axis('off')

    title = (f"{ds_name}\n"
             f"n={n} (post-dedup) | res={res_str}\n"
             f"{meta.get('paper','?')} | {meta.get('location','?')}")
    plt.suptitle(title, fontsize=8)
    plt.tight_layout()
    out = FIGS_GRIDS / f'{ds_name}_gallery.png'
    plt.savefig(out, dpi=80)
    plt.close()
    log.info(f"  Gallery saved: {out}")

# ── LaRS brightness stats (for comparison) ──
log.info("\nComputing LaRS brightness stats (sample of 200)...")
lars_img_paths = []
for split in ['train', 'val', 'test']:
    split_dir = LARS / 'img_dir' / split / 'images'
    if split_dir.exists():
        lars_img_paths.extend(list(split_dir.iterdir())[:70])

lars_stats_list = []
for p in lars_img_paths[:200]:
    try:
        img = Image.open(p).convert('RGB')
        lars_stats_list.append(image_stats(img))
    except:
        pass

if lars_stats_list:
    brightness_stats['LaRS (reference)'] = dict(
        mean_L=np.mean([s['mean_L'] for s in lars_stats_list]),
        mean_r=np.mean([s['mean_r'] for s in lars_stats_list]),
        mean_g=np.mean([s['mean_g'] for s in lars_stats_list]),
        mean_b=np.mean([s['mean_b'] for s in lars_stats_list]),
        std_r=np.mean([s['std_r'] for s in lars_stats_list]),
        std_g=np.mean([s['std_g'] for s in lars_stats_list]),
        std_b=np.mean([s['std_b'] for s in lars_stats_list]),
        n=4006
    )
    log.info(f"  LaRS: mean L*={brightness_stats['LaRS (reference)']['mean_L']:.1f}")

# ── Brightness comparison chart ──
log.info("Generating brightness comparison chart...")
ds_names = list(brightness_stats.keys())
L_values = [brightness_stats[ds]['mean_L'] for ds in ds_names]
r_values = [brightness_stats[ds]['mean_r'] for ds in ds_names]
g_values = [brightness_stats[ds]['mean_g'] for ds in ds_names]
b_values = [brightness_stats[ds]['mean_b'] for ds in ds_names]

x = np.arange(len(ds_names))
width = 0.2
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax1 = axes[0]
bars = ax1.bar(x, L_values, color='steelblue', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(ds_names, rotation=30, ha='right', fontsize=8)
ax1.set_ylabel('Mean L* (LAB)')
ax1.set_title('Mean LAB L* brightness per dataset (lower = darker)')
ax1.axhline(y=128, color='gray', linestyle='--', alpha=0.5, label='mid-gray (128)')
ax1.legend(fontsize=8)
for bar, v in zip(bars, L_values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{v:.0f}', ha='center', va='bottom', fontsize=7)

ax2 = axes[1]
ax2.bar(x - width, r_values, width, label='R', color='red', alpha=0.7)
ax2.bar(x,         g_values, width, label='G', color='green', alpha=0.7)
ax2.bar(x + width, b_values, width, label='B', color='blue', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(ds_names, rotation=30, ha='right', fontsize=8)
ax2.set_ylabel('Mean channel value (0-255)')
ax2.set_title('Mean RGB channel values per dataset')
ax2.legend(fontsize=8)

plt.tight_layout()
out = FIGS / 'brightness_comparison.png'
plt.savefig(out, dpi=100)
plt.close()
log.info(f"Saved: {out}")

# Save brightness stats as CSV
import csv
with open(SHARED / 'brightness_stats.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','n','mean_L','mean_r','mean_g','mean_b','std_r','std_g','std_b'])
    w.writeheader()
    for ds, stats in brightness_stats.items():
        w.writerow({'dataset': ds, **{k: round(v, 2) for k, v in stats.items()}})

log.info("=== Analysis 01 COMPLETE ===")
