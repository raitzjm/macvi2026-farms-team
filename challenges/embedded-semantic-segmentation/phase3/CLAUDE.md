# Phase 3: Augmentation-Focused Training

Issue: #37

## Overview
Two training runs using CopyPasteObstacles + EnhancedPhotoMetricDistortion +
GenerateEdgeMasked (PIDNet-S only). See issue #37 for full rationale.

## Conda Env
`macvi2` — same as Phase 2/2.5.

## Experiments

| EXP | Model | Status | Config | Iters | WandB |
|-----|-------|--------|--------|-------|-------|
| EXP-019 | PIDNet-S | Pending | `configs/pidnets_768x384_lars_phase3.py` | 8000 | `phase3_pidnets` |
| EXP-020 | STDC2 | Pending (after EXP-019) | `configs/stdc2_768x384_lars_phase3.py` | 5000 | `phase3_stdc2` |

## Baselines to Beat
- PIDNet-S: EXP-008 local val Q=71.03% / server Q=49.5
- STDC2: EXP-017 local val Q=64.49% / server Q=45.8

## Key Transforms (pipelines.py)

- **CopyPasteObstacles**: Pastes external obstacle crops onto water regions.
  DB: `challenges/common/external-dataset-analysis/shared_data/copypaste_candidates.db`
  DB schema: `dataset, filename (mask path), x1, y1, x2, y2, area_native, area_normalized, area_pct, size_tier`
  Images resolved from `images.txt` in parent of the masks/ dir (stem-to-path 1:1 mapping).
  GT key: `gt_seg_map` (mmseg 1.x).

- **GenerateEdgeMasked**: Drop-in replacement for `GenerateEdge` (PIDNet-S only).
  Zeros edge targets in reflect-padded border regions (reads `img_padding_size`).
  This is Option 1 from issue #28. ReflectPad MUST run before GenerateEdgeMasked.

- **EnhancedPhotoMetricDistortion**: Wider bounded ranges + optional RandomGamma.
  Do not increase brightness_delta beyond 77 or saturation/contrast beyond 1.5.

## Pre-Training Gate (MANDATORY)
Run and pass ALL 5 checks in `scripts/visualize_pipeline.py` before launching training.
Exit code 0 = safe to launch. Exit code 1 = fix before launching.

```bash
cd /home/farms/macvi2026-farms-team
conda run -n macvi2 python \
    challenges/embedded-semantic-segmentation/phase3/scripts/visualize_pipeline.py \
    --config challenges/embedded-semantic-segmentation/phase3/configs/pidnets_768x384_lars_phase3.py \
    --n 50 --out /tmp/aug_vis_pidnets/
```

## Training Launch

```bash
# From repo root, inside container
source challenges/embedded-semantic-segmentation/phase3/scripts/setup_wandb.sh

# Step 1: PIDNet-S (EXP-019) — ~5h on 2× RTX A6000
bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh pidnets

# Step 2: STDC2 (EXP-020) — ONLY after EXP-019 fully completes
bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh stdc2
```

## Evaluation
Use `scripts/export_and_eval.py --quant` (copied from phase2_5).

## WandB Notes
- WANDB_API_KEY loaded via `setup_wandb.sh` (key at `~/data/wandb_key.txt`)
- WANDB_DIR set to `outputs/phase3` (NOT on PYTHONPATH)
- Delete failed runs from WandB before restarting

## mmengine patch reminder (reapply after container restart)
- `vis_backend.py`: `allow_val_change` default False→True
- `setuptools<70` installed in macvi2 env
