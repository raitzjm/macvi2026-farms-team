# Phase 2.5 — Pipeline Fix Verification (COMPLETE)

## Purpose
Controlled re-training of 2 INT8-stable models (PIDNet-S, STDC2) with all
preprocessing pipeline fixes applied, to measure their effect vs Phase 2.
The comparison uses `export_and_eval.py` (authoritative FP32 eval).

See GitHub issue #28 for full methodology and results.

## Environment
- Same as Phase 2: conda env `macvi2`, Python 3.11, PyTorch 2.1.2+cu121,
  mmcv 2.1.0, mmseg 1.2.2 (stock pip install)
- Run all training from within the container

## Directory Structure
```
phase2_5/
├── CLAUDE.md             ← this file
├── datasets/
│   ├── __init__.py
│   └── lars_dataset.py   ← LaRSDataset (identical to phase2)
├── pipelines.py          ← ReflectPad (Fix 2+3) + MacViSegVisualizer (Fix 4)
├── configs/
│   ├── _base_/
│   │   ├── lars_dataset.py    ← shared dataset config (Fix 1: keep_ratio=True)
│   │   └── runtime.py         ← WandB vis backend, MacViSegVisualizer
│   ├── stdc2_768x384_lars_phase2_5.py    (3k iters, SGD — identical to EXP-007)
│   └── pidnets_768x384_lars_phase2_5.py  (5k iters, SGD — identical to EXP-008)
└── scripts/
    ├── setup_wandb.sh      ← load WANDB_API_KEY (same as phase2)
    ├── run_phase2_5.sh     ← DDP launcher (torchrun, 2 GPUs, macvi2 env)
    └── export_and_eval.py  ← ONNX export + PyTorch/ONNX eval (same as phase2)
```

## Pipeline Fixes Applied (vs Phase 2)

### Fix 1: keep_ratio=True in train RandomResize
- File: `configs/_base_/lars_dataset.py` and `configs/pidnets_*` (PIDNet has
  its own train pipeline with GenerateEdge, so the fix is replicated there)
- Preserves image AR during training, matching challenge server preprocessing
- Phase 2 used `keep_ratio=False` (undocumented regression)

### Fix 2: img_padding_size written to results
- File: `pipelines.py` → `ReflectPad._pad_img`
- Writes `results['img_padding_size'] = [left, right, top, bottom]` so the
  mmengine runner can de-pad predictions before metric computation
- Fixes training val mIoU underestimation from Phase 2

### Fix 3: Centred val/test padding
- File: `pipelines.py` → `ReflectPad` gains `center=True` parameter
- Test pipeline in `configs/_base_/lars_dataset.py` uses `center=True`
- Splits padding evenly on both sides, matching challenge server centering
- Training pipeline keeps `center=False` (bottom-right only)

### Fix 4: Ignore-area overlay on WandB predicted images
- File: `pipelines.py` → `MacViSegVisualizer`
- Subclasses `SegLocalVisualizer`; marks GT==255 pixels grey in prediction
  overlay, matching how GT overlays already show ignore areas
- Used in `configs/_base_/runtime.py` and all model configs

## Custom Imports
All Phase 2.5 configs use:
```python
custom_imports = dict(
    imports=[
        'challenges.embedded-semantic-segmentation.phase2_5.datasets.lars_dataset',
        'challenges.embedded-semantic-segmentation.phase2_5.pipelines',
    ],
    allow_failed_imports=False)
```
PYTHONPATH must include the repo root (handled by run_phase2_5.sh).

## Training Launch
```bash
# From repo root, inside container
source challenges/embedded-semantic-segmentation/phase2_5/scripts/setup_wandb.sh
bash challenges/embedded-semantic-segmentation/phase2_5/scripts/run_phase2_5.sh stdc2
bash challenges/embedded-semantic-segmentation/phase2_5/scripts/run_phase2_5.sh pidnets
```

## WandB Run Names
- STDC2:   `phase2_5_stdc2`
- PIDNet-S: `phase2_5_pidnets`

## Output Directories
- `challenges/embedded-semantic-segmentation/outputs/phase2_5/stdc2/`
- `challenges/embedded-semantic-segmentation/outputs/phase2_5/pidnets/`

## What NOT to change vs Phase 2
- Model architecture, optimizer type, LR, weight decay, batch size, iters,
  loss functions, scheduler, warmup settings, eval interval, resolution
- The ONLY changes are in the preprocessing pipeline (Fixes 1-4 above)
