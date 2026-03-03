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
- **Started:** 2026-03-03 08:58 UTC (restarted after DDP eval patch)
- **Finished:** 2026-03-03 10:06 UTC (~1h08m)
- **Status:** COMPLETED ✓
- **Best checkpoint:** `best_mIoU_iter_1250.pth`
- **Val mIoU progression:**

| Iter | mIoU  | aAcc  | IoU obstacle | IoU water | IoU sky |
|------|-------|-------|-------------|-----------|---------|
| 250  | 95.00 | 97.66 | 90.91       | 96.67     | 97.43   |
| 500  | 95.44 | 97.90 | 91.71       | 97.11     | 97.50   |
| 750  | 95.54 | 97.93 | 91.96       | 97.14     | 97.53   |
| 1000 | 94.56 | 97.51 | 90.32       | 96.80     | 96.57   |
| 1250 | **95.76** | 97.08 | **92.41** | **97.60** | 97.26 |
| 1500 | 95.51 | 97.98 | 92.06       | 97.62     | 96.83   |

### EXP-002 — BiSeNetV2

- **Config:** `phase1/configs/bisenetv2_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/bisenetv2/`
- **Optimizer:** SGD, lr=3.2, momentum=0.9, wd=0.0005
- **Batch size:** samples_per_gpu=128 × 2 GPUs = 256 total (reduced from 152 after OOM — 5 aux heads)
- **LR scaling:** linear (ref bs=4, ref lr=0.05 → scaled lr=0.05×256/4=3.2)
- **Norm:** SyncBN (DDP, already set)
- **Started:** 2026-03-03 10:12 UTC (restarted after OOM fix)
- **Finished:** 2026-03-03 11:14 UTC (~1h02m)
- **Status:** COMPLETED ✓
- **Best checkpoint:** `best_mIoU_iter_1500.pth` (still climbing at end)
- **Val mIoU progression:**

| Iter | mIoU  | aAcc  | IoU obstacle | IoU water | IoU sky |
|------|-------|-------|-------------|-----------|---------|
| 250  | 48.28 | 71.92 | 14.86       | 76.51     | 53.48   |
| 500  | 67.93 | 83.15 | 49.55       | 83.84     | 70.42   |
| 750  | 37.25 | 63.52 | 0.12        | 61.33     | 50.29   |
| 1000 | 57.24 | 76.56 | 52.12       | 70.60     | 49.00   |
| 1250 | 86.03 | 93.20 | 75.89       | 91.45     | 90.77   |
| 1500 | **92.55** | 96.46 | **86.41** | **94.88** | **96.38** |

Note: Near-catastrophic obstacle collapse at iter 750 (IoU=0.12%) due to large SGD LR=3.2 causing
instability; full recovery by iter 1000. Model still climbing at iter 1500 — would benefit from more iters.

### EXP-003 — SegNeXt-T

- **Config:** `phase1/configs/segnext_t_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/segnext_t/`
- **Optimizer:** AdamW, lr=6e-5 (fixed — no linear scaling for AdamW)
- **Batch size:** samples_per_gpu=75 × 2 GPUs = 150 total
- **Norm:** LayerNorm (unaffected by DDP)
- **Started:** 2026-03-03 11:20 UTC (restarted at bs=56 after OOM)
- **Finished:** 2026-03-03 11:54 UTC (~34 min)
- **Status:** COMPLETED ✓
- **Best checkpoint:** `best_mIoU_iter_1500.pth` (still climbing at end)
- **Val mIoU progression:**

| Iter | mIoU  | aAcc  | IoU obstacle | IoU water | IoU sky |
|------|-------|-------|-------------|-----------|---------|
| 250  | 90.51 | 95.56 | 84.19       | 94.44     | 92.91   |
| 500  | 94.26 | 97.35 | 90.12       | 96.50     | 96.16   |
| 750  | 94.99 | 97.68 | 91.19       | 96.86     | 96.91   |
| 1000 | 95.39 | 97.87 | 91.83       | 97.13     | 97.20   |
| 1250 | 95.42 | 97.90 | 91.86       | 97.19     | 97.21   |
| 1500 | **95.61** | 97.99 | **92.19** | **97.34** | **97.31** |

Note: Monotonic convergence throughout. Still climbing at iter 1500. Very fast run (~34 min).

### EXP-004 — SegFormer MiT-B2

- **Config:** `phase1/configs/segformer_mit-b2_768x384_lars_phase1.py`
- **Work dir:** `outputs/phase1/segformer_mit-b2/`
- **Optimizer:** AdamW, lr=6e-5 (fixed — no linear scaling for AdamW)
- **Batch size:** samples_per_gpu=36 × 2 GPUs = 72 total (pre-emptively reduced from 45)
- **Norm:** LayerNorm (unaffected by DDP)
- **Started:** 2026-03-03 11:55 UTC
- **Status:** RUNNING
- **Results:** TBD after completion
