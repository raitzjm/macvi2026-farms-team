# Phase 2 — mmseg 1.2.2 Migration + Full Baseline Training

## Environment
- Conda env: `macvi2` at `/home/farms/miniforge3/envs/macvi2/`
- Python 3.11, PyTorch 2.1.2+cu121, mmcv 2.1.0, mmseg 1.2.2 (stock pip install)
- Additional deps vs `macvi`: `ftfy`, `regex` (required by mmseg tokenizer at import time)
- Tools at: `/home/farms/miniforge3/envs/macvi2/lib/python3.11/site-packages/mmseg/.mim/tools/`
- Run with `conda run --no-capture-output -n macvi2 torchrun ...` or activate env directly

## Required Patches (apply after reinstall)
- `wandb/sdk/lib/wbauth/validation.py`: early-return for `wandb_` prefix before 40-char check
  (same as macvi env — new wandb API key format)

## Phase 2 Directory Structure
```
phase2/
├── CLAUDE.md             ← this file
├── datasets/
│   ├── __init__.py
│   └── lars_dataset.py   ← LaRSDataset for mmseg 1.x BaseSegDataset API
├── pipelines.py          ← ReflectPad (mmseg 1.x TRANSFORMS registry)
├── configs/
│   ├── _base_/
│   │   ├── lars_dataset.py    ← shared dataset config
│   │   └── runtime.py         ← WandB vis backend, hooks, env_cfg
│   ├── segformer_mit-b2_768x384_lars_phase2.py  (3k iters, AdamW)
│   ├── segnext_t_768x384_lars_phase2.py          (5k iters, AdamW)
│   ├── segnext_s_768x384_lars_phase2.py          (5k iters, AdamW) [added late to Phase 2]
│   ├── stdc2_768x384_lars_phase2.py              (3k iters, SGD)
│   ├── pidnets_768x384_lars_phase2.py            (5k iters, SGD)
│   ├── bisenetv2_768x384_lars_phase2.py          (6k iters, SGD)
│   ├── unet_s5d16_768x384_lars_phase2.py        (5k iters, AdamW) [added late to Phase 2]
│   ├── stdc2_adamw_768x384_lars_phase2.py        (5k iters, AdamW) [additional AdamW variant]
│   ├── bisenetv2_adamw_768x384_lars_phase2.py    (5k iters, AdamW) [additional AdamW variant]
│   └── pidnets_adamw_768x384_lars_phase2.py      (5k iters, AdamW) [additional AdamW variant]
└── scripts/
    ├── run_phase2.sh       ← DDP launcher (torchrun, 2 GPUs, macvi2 env)
    ├── setup_wandb.sh      ← load WANDB_API_KEY from ~/data/wandb_key.txt
    └── export_and_eval.py  ← ONNX export + PyTorch/ONNX eval (mmseg 1.x API)
```

## Custom Imports
All Phase 2 configs use:
```python
custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2.pipelines',
    ],
    allow_failed_imports=False)
```
PYTHONPATH must include the repo root (handled by run_phase2.sh).

## Training Launch
```bash
# From repo root, inside container
source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
bash challenges/embedded-semantic-segmentation/phase2/scripts/run_phase2.sh <model>
# SGD models:   stdc2 | pidnets | bisenetv2
# AdamW models: segformer_mit-b2 | segnext_t | segnext_s | unet
#               stdc2_adamw | bisenetv2_adamw | pidnets_adamw
```

The run script rebuilds the LaRS symlink tree if not present.

## mmseg 1.x vs 0.30.0 Config API Differences
| Component | mmseg 0.30.0 (Phase 1) | mmseg 1.x (Phase 2) |
|---|---|---|
| Runner | `runner = dict(type='IterBasedRunner')` | `train_cfg = dict(type='IterBasedTrainLoop')` |
| Optimizer | `optimizer = dict(...)` + `optimizer_config` | `optim_wrapper = dict(type='OptimWrapper', optimizer=...)` |
| LR schedule | `lr_config = dict(policy='poly', ...)` | `param_scheduler = [dict(type='PolyLR', ...)]` |
| Data loaders | `data = dict(samples_per_gpu=N, train={})` | `train_dataloader = dict(batch_size=N, ...)` |
| Normalize | `dict(type='Normalize', ...)` in pipeline | `data_preprocessor` inside model dict |
| Logging | `log_config = dict(hooks=[WandbLoggerHook])` | `vis_backends = [WandbVisBackend]` + `visualizer` |
| Evaluation | `evaluation = dict(interval=N, metric='mIoU')` | `train_cfg.val_interval` + `val_evaluator` |
| Pipeline end | `DefaultFormatBundle` + `Collect` | `PackSegInputs` |

## Preprocessing (CRITICAL — same as Phase 1)
Train pipeline: `RandomResize (keep_ratio=False)` → `RandomCrop` → `ReflectPad`
Test pipeline:  `Resize (keep_ratio=True)` → `ReflectPad` → `PackSegInputs`
- Normalization is in `data_preprocessor` (NOT in pipeline — mmseg 1.x change)
- `ReflectPad` registered via `@TRANSFORMS.register_module()` from `mmseg.registry`

## PIDNet Batch Size Calibration — DONE (2026-03-04)
Calibrated with single-GPU fwd+bwd at 768×384 (A6000 48 GB):
  bs=80→17.8 GB, bs=120→26.7 GB, bs=160→35.5 GB, bs=200→44.4 GB, bs=220→48.9 GB (OOM boundary)
OOM boundary ≈ 220/GPU. 20% reduction → 176. Rounded to **bs=160** (74% utilization).
LR = 0.01 × (320/12) = **0.267** (linear scaling, SGD).
Config updated. Issue #14 notified.

## WandB Prediction Image Logging
Phase 2 uses `WandbVisBackend` + `SegVisualizationHook` for prediction overlay images.
This requires `WANDB_API_KEY` to be set AND the wandb validation.py patch for `wandb_` prefix.
Images appear in WandB "Media" tab per run at every val interval (every 250 iters).

## Pinned Batch Sizes (from Phase 1)
| Model | samples_per_gpu | total_bs | LR | Optimizer |
|---|---|---|---|---|
| SegFormer MiT-B2 | 36 | 72 | 6e-5 | AdamW |
| SegNeXt-T | 56 | 112 | 6e-5 | AdamW |
| STDC2 | 160 | 320 | 0.267 | SGD |
| BiSeNetV2 | 128 | 256 | 0.80 (CORRECTED — Phase 1 used 3.2, wrong) | SGD |
| PIDNet-S | 160 | 320 | 0.133 (0.01×160/12; REVISED 2026-03-04; 0.267 caused 9% oscillations) | SGD |

## New Models (added late to Phase 2)
| Model | samples_per_gpu | total_bs | LR | Optimizer | Notes |
|---|---|---|---|---|---|
| SegNeXt-S | 40 | 80 | 6e-5 | AdamW | OOM@56 → 40/GPU (20% reduction from ~50 boundary); CALIBRATED 2026-03-04 |
| UNet-s5-d16 | 16 | 32 | 6e-5 (fixed; AdamW) | AdamW | Trains from scratch; CALIBRATED 2026-03-04: OOM@~19 → bs=16 (82%); bs=18→44.4GB |

Ref for SegNeXt-S: upstream mscan-s_1xb16-adamw-160k_ade20k (same backbone, larger dims).
Ref for UNet: switched to AdamW (user request 2026-03-04). Starting estimate bs=4/GPU; calibrate before first run.
Update configs + issue after calibration.

## AdamW Variants of SGD Models (additional Phase 2 experiments)
Same architecture as corresponding SGD runs; separate config files (do NOT override SGD configs).

| Model | Config suffix | samples_per_gpu | total_bs | LR | WandB run |
|---|---|---|---|---|---|
| STDC2 AdamW | stdc2_adamw | 64 (est; calibrate) | 128 | 6e-5 | phase2_stdc2_adamw |
| BiSeNetV2 AdamW | bisenetv2_adamw | 64 (est; calibrate) | 128 | 6e-5 | phase2_bisenetv2_adamw |
| PIDNet-S AdamW | pidnets_adamw | 64 (est; calibrate) | 128 | 6e-5 | phase2_pidnets_adamw |

All AdamW variants: 5k iters, LinearLR (start_factor=1e-6, 500 iters) + PolyLR (power=1.0).
Calibrate BS before each run. Update table after calibration.

## Phase 2 Cross-check vs Phase 1
At iter 1500, record mIoU and compare to Phase 1 reference:
| Model | Phase 1 mIoU@1500 (training metric) | Acceptable Phase 2 delta |
|---|---|---|
| SegFormer | 97.00% (converged at 1000) | ±1.0% |
| SegNeXt-T | 95.61% | ±1.0% |
| STDC2 | 95.51% | ±1.0% |
| BiSeNetV2 | skip — Phase 1 unstable | N/A |
| PIDNet-S | N/A (new model) | N/A |

If 2/3 stable models match within ±1%, migration is confirmed lossless.

## Output Directory Convention
```
challenges/embedded-semantic-segmentation/outputs/phase2/{model_name}/
```
Set as `--work-dir` via run_phase2.sh. Gitignored — curated artifacts only.
