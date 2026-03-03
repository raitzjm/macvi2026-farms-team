# Experiment Log — FARMS Team MaCVi 2026

Tracks training experiments only. Install/refactor/script work is tracked via git commits.

---

## Phase 1 Baselines — Issue #9

**Branch:** `exp/phase1-baselines`
**Started:** 2026-03-03 08:22 UTC
**Git commit:** `8ffd3aa`
**Hardware:** 2× RTX A6000 48 GB, DDP (torchrun --standalone --nproc_per_node=2)
**Dataset:** LaRS (train: 2607, val: 200) @ 768×384
**Schedule:** 1500 iters, eval every 250 (6 points), checkpoint every 750
**WandB:** little_farms/embedded_segmentation_macvi2026

### EXP-001 — STDC2

- **Config:** `phase1/configs/stdc2_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/stdc2/`
- **Optimizer:** SGD, lr=0.267, momentum=0.9, wd=0.0005
- **Batch size:** samples_per_gpu=160 × 2 GPUs = 320 total
- **LR scaling:** linear (ref bs=12, ref lr=0.01 → scaled lr=0.01×320/12=0.267)
- **Norm:** SyncBN (DDP)
- **Started:** 2026-03-03 08:38 UTC (restarted at 1.5k iters)
- **Status:** RUNNING
- **Results:** TBD after completion

### EXP-002 — BiSeNetV2

- **Config:** `phase1/configs/bisenetv2_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/bisenetv2/`
- **Optimizer:** SGD, lr=3.8, momentum=0.9, wd=0.0005
- **Batch size:** samples_per_gpu=152 × 2 GPUs = 304 total
- **LR scaling:** linear (ref bs=4, ref lr=0.05 → scaled lr=0.05×304/4=3.8)
- **Norm:** SyncBN (DDP, already set)
- **Started:** after EXP-001 completes
- **Status:** PENDING
- **Results:** TBD after completion

### EXP-003 — SegNeXt-T

- **Config:** `phase1/configs/segnext_t_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/segnext_t/`
- **Optimizer:** AdamW, lr=6e-5 (fixed — no linear scaling for AdamW)
- **Batch size:** samples_per_gpu=75 × 2 GPUs = 150 total
- **Norm:** LayerNorm (unaffected by DDP)
- **Started:** after EXP-002 completes
- **Status:** PENDING
- **Results:** TBD after completion

### EXP-004 — SegFormer MiT-B2

- **Config:** `phase1/configs/segformer_mit-b2_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/segformer_mit-b2/`
- **Optimizer:** AdamW, lr=6e-5 (fixed — no linear scaling for AdamW)
- **Batch size:** samples_per_gpu=45 × 2 GPUs = 90 total
- **Norm:** LayerNorm (unaffected by DDP)
- **Started:** after EXP-003 completes
- **Status:** PENDING
- **Results:** TBD after completion
