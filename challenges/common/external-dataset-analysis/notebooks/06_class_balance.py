#!/usr/bin/env python3
"""
Analysis 06 — Class Balance & Mixing Strategy
Prereq: Analysis 00 complete (class_balance.csv)
Run: /home/farms/miniforge3/envs/macvi/bin/python -u notebooks/06_class_balance.py
"""
import sys, csv, logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO   = Path('/home/farms/macvi2026-farms-team/challenges/common/external-dataset-analysis')
SHARED = REPO / 'shared_data'
FIGS   = REPO / 'figures'
LOGS_DIR = REPO / 'logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis_06_run.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("=== Analysis 06: Class Balance ===")

# Load class balance CSV
balance_csv = SHARED / 'class_balance.csv'
rows = []
with open(balance_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            'dataset': row['dataset'],
            'n': int(row['n_images_deduplicated']),
            'obstacle': float(row['pct_obstacle']),
            'water':    float(row['pct_water']),
            'sky':      float(row['pct_sky']),
            'ignore':   float(row['pct_ignore']),
        })

# Add LaRS reference stats
LARS_GT = Path('/home/farms/data/lars_dataset/ann_dir/train/semantic_masks')
if LARS_GT.exists():
    from PIL import Image
    total_px = 0
    class_px = {0: 0, 1: 0, 2: 0}
    files = sorted(LARS_GT.iterdir())[:500]
    for mf in files:
        try:
            ann = np.array(Image.open(mf))
            total_px += ann.size
            for c in [0, 1, 2]:
                class_px[c] += int((ann == c).sum())
        except:
            pass
    if total_px > 0:
        rows.insert(0, {
            'dataset': 'LaRS train (ref)',
            'n': 2605,
            'obstacle': round(100 * class_px[0] / total_px, 2),
            'water':    round(100 * class_px[1] / total_px, 2),
            'sky':      round(100 * class_px[2] / total_px, 2),
            'ignore':   0.0,
        })
        log.info(f"LaRS train: obs={rows[0]['obstacle']}%, water={rows[0]['water']}%, sky={rows[0]['sky']}%")

log.info("\nClass balance table:")
for r in rows:
    flag = " ← LOW OBJ" if r['obstacle'] < 2.0 else ""
    log.info(f"  {r['dataset']:30s}: obs={r['obstacle']:5.1f}%  water={r['water']:5.1f}%  "
             f"sky={r['sky']:5.1f}%  ign={r['ignore']:5.1f}%  n={r['n']}{flag}")

# ──────────────────────────────────────────────
# Stacked Bar Chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.5), 7))

x = np.arange(len(rows))
labels = [r['dataset'] for r in rows]

obs_vals  = [r['obstacle'] for r in rows]
water_vals = [r['water']   for r in rows]
sky_vals  = [r['sky']      for r in rows]
ign_vals  = [r['ignore']   for r in rows]

b1 = ax.bar(x, obs_vals,   label='Obstacle (0)', color='#d62728', alpha=0.85)
b2 = ax.bar(x, water_vals, bottom=obs_vals, label='Water (1)', color='#1f77b4', alpha=0.85)
b3 = ax.bar(x, sky_vals,   bottom=[o+w for o,w in zip(obs_vals, water_vals)],
            label='Sky (2)', color='#7ecef4', alpha=0.85)
b4 = ax.bar(x, ign_vals,
            bottom=[o+w+s for o,w,s in zip(obs_vals, water_vals, sky_vals)],
            label='Ignore (255)', color='#aaaaaa', alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Pixel percentage (%)')
ax.set_title('Class balance per dataset (harmonized to LaRS classes)\nDatasets with obstacle% < 2% provide weak obstacle training signal')
ax.legend(loc='upper right', fontsize=8)
ax.axhline(y=0, color='black', lw=0.5)

# Mark low-obstacle datasets
for i, r in enumerate(rows):
    if r['obstacle'] < 2.0:
        ax.annotate('LOW', xy=(i, 1), xytext=(i, 2), ha='center', fontsize=7,
                    color='red', fontweight='bold')

plt.tight_layout()
out = FIGS / 'class_balance.png'
plt.savefig(out, dpi=120)
plt.close()
log.info(f"Saved: {out}")

# ──────────────────────────────────────────────
# Mixing Projection Table
# ──────────────────────────────────────────────
log.info("\nMixing projection (LaRS 2605 + N external):")
LARS_N = 2605
LARS_OBS  = next((r['obstacle'] for r in rows if 'LaRS' in r['dataset']), 3.0)
LARS_WATER = next((r['water']   for r in rows if 'LaRS' in r['dataset']), 50.0)

for mix_n in [500, 1000, 2605]:
    log.info(f"\n  Mix N={mix_n} external images with {LARS_N} LaRS:")
    total_n = LARS_N + mix_n
    for r in rows:
        if 'LaRS' in r['dataset']:
            continue
        # Weighted average (by image count, assuming similar resolution)
        mixed_obs = (LARS_N * LARS_OBS + mix_n * r['obstacle']) / total_n
        mixed_water = (LARS_N * LARS_WATER + mix_n * r['water']) / total_n
        flag = " ← <2% obs" if mixed_obs < 2.0 else ""
        log.info(f"    {r['dataset']:30s}: obs={mixed_obs:.1f}%  water={mixed_water:.1f}%{flag}")

log.info("=== Analysis 06 COMPLETE ===")
