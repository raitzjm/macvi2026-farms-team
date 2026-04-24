# Phase 2 — mmseg 1.2.2 Migration + Full Baseline Training (COMPLETE)

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
├── AGENTS.md             ← this file (canonical context)
├── CLAUDE.md             ← one-line stub: `@AGENTS.md` (for Claude Code)
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
│   ├── pidnets_adamw_768x384_lars_phase2.py      (5k iters, AdamW) [additional AdamW variant]
│   ├── segnext_t_uperhead_768x384_lars_phase2.py (5k iters, AdamW) [EXP-015: MSCAN-T+UPerHead+BoundaryLoss]
│   └── pidnetm_768x384_lars_phase2.py            (5k iters, SGD)  [EXP-016: PIDNet-M scale-up]
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

For mmseg 0.30.0 vs 1.x config API migration reference, see `phase1/phase1_report.md` section 13.

## Preprocessing (CRITICAL — same as Phase 1)
Train pipeline: `RandomResize (keep_ratio=False)` → `RandomCrop` → `ReflectPad`
Test pipeline:  `Resize (keep_ratio=True)` → `ReflectPad` → `PackSegInputs`
- Normalization is in `data_preprocessor` (NOT in pipeline — mmseg 1.x change)
- `ReflectPad` registered via `@TRANSFORMS.register_module()` from `mmseg.registry`

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

## Additional Models (added during Phase 2)
| Model | samples_per_gpu | total_bs | LR | Optimizer |
|---|---|---|---|---|
| SegNeXt-S | 40 | 80 | 6e-5 | AdamW |
| UNet-s5-d16 | 16 | 32 | 6e-5 | AdamW |
| PIDNet-M | 80 | 160 | 0.0667 | SGD |

AdamW variants of SGD models also trained (`stdc2_adamw`, `bisenetv2_adamw`, `pidnets_adamw`):
5k iters, LR=6e-5, LinearLR warmup + PolyLR. Separate config files.

See `../eval_metrics.md` for all Phase 2 results.

