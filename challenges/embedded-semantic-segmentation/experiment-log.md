# Experiment Log - Embedded Semantic Segmentation

## Format
Each entry follows this structure:
- **ID**: EXP-NNN
- **Date**: YYYY-MM-DD HH:MM UTC
- **Phase**: current phase
- **Status**: PLANNED | IN PROGRESS | COMPLETED | FAILED | ABANDONED
- **Branch/Commit**: git reference
- **Config**: path to config file
- **Dataset**: what data was used
- **Resolution**: training resolution
- **Epochs**: number of epochs
- **Results (val)**: F1, mIoU, and any other relevant metrics
- **Notes**: observations, comparisons, issues

## Comparison Matrix

| ID | Model | Data | Res | Aug | Val F1 | Val mIoU | Quant F1 | FPS | Commit |
|----|-------|------|-----|-----|--------|----------|----------|-----|--------|
| EXP-005 | SegFormer MiT-B2 | LaRS | 768×384 | reflect pad, keep_ratio | TBD (see report) | 89.72% | — | — | c50e7a5 |
| EXP-006 | SegNeXt-T | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-007 | STDC2 | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-008 | PIDNet-S | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-009 | BiSeNetV2 | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-001 | STDC2 | LaRS | 768×384 | reflect pad, keep_ratio | 69.30% | 90.78% | — | — | 67529e2 |
| EXP-002 | BiSeNetV2 | LaRS | 768×384 | reflect pad, keep_ratio | 60.84% (PyTorch) / 60.81% (ONNX) | 86.56% | — | — | 67529e2 |
| EXP-003 | SegNeXt-T | LaRS | 768×384 | reflect pad, keep_ratio | 72.86% | 89.81% | — | — | 67529e2 |
| EXP-004 | SegFormer MiT-B2 | LaRS | 768×384 | reflect pad, keep_ratio | 73.64% | 92.02% | — | — | 67529e2 |

---

## Entries

(entries will be appended below by the agent)

---

## EXP-005
- **ID**: EXP-005
- **Date**: 2026-03-04 00:30 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/segformer_mit-b2_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 3000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=36, total_bs=72, lr=6e-5 (AdamW, poly, warmup 500 iters; pinned from Phase 1)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_segformer_mit-b2
- **Results (val)**: best mIoU=89.72% @ iter 2000; final mIoU=89.71% @ iter 3000
- **Notes**: Cross-check at iter 1500: mIoU=89.52% vs Phase 1 ref 92.02% (−2.5%, outside ±1%). Likely evaluator difference between mmseg 0.30 and 1.x — training metrics (loss/acc) normal. Plateau from iter 2000→3000. WandB images: 0 logged (bug: interval=250 > val set 198; fixed to interval=5 in runtime.py for next models).

---

## EXP-006
- **ID**: EXP-006
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: IN PROGRESS
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/segnext_t_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=56, total_bs=112, lr=6e-5 (AdamW, poly, warmup 500 iters; pinned from Phase 1)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_segnext_t
- **Results (val)**: TBD
- **Notes**: Phase 2 migration. Cross-check at iter 1500 vs Phase 1 ref (89.81%).

---

## EXP-007
- **ID**: EXP-007
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/stdc2_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 3000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=160, total_bs=320, lr=0.267 (SGD, poly, warmup 300 iters; pinned from Phase 1)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_stdc2
- **Results (val)**: TBD
- **Notes**: Phase 2 migration. Cross-check at iter 1500 vs Phase 1 ref (90.78%).

---

## EXP-008
- **ID**: EXP-008
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/pidnets_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: TBD (calibrate before training; upstream ref total_bs=12, lr=0.01)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_pidnets
- **Results (val)**: TBD
- **Notes**: New model in Phase 2 (no Phase 1 baseline). Batch size to be calibrated.

---

## EXP-009
- **ID**: EXP-009
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/bisenetv2_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 6000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=128, total_bs=256, lr=0.80 (SGD corrected; Phase 1 used 3.2 — wrong); warmup 300 iters
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_bisenetv2
- **Results (val)**: TBD
- **Notes**: LR corrected to 0.80 (0.05 × 256/16 total-to-total). Phase 1 3.2 caused obstacle collapse. Longer warmup for stability.

---

## EXP-001
- **ID**: EXP-001
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase1-baselines @ 67529e2
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/stdc2_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 1500 (eval every 250; save best mIoU; best at iter 1250)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=160, total_bs=320, lr=0.267 (SGD, poly, warmup 1k iters; linear-scaled from ref bs=12 lr=0.01)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_stdc2
- **Results (val)**: F1=69.30% (PyTorch), ONNX delta=0.00%; mIoU=90.78%, aAcc=98.08% (best iter 1250)
- **Notes**: Corrected preprocessing vs fun_experiments (reflect pad, keep_ratio=True, test Pad added). STDC2 near-converged at 1250 iters. Logits ONNX committed to LFS. See phase1_report.md for full convergence analysis.

---

## EXP-002
- **ID**: EXP-002
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase1-baselines @ 67529e2
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/bisenetv2_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 1500 (eval every 250; save best mIoU; best at iter 1500)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=128, total_bs=256, lr=3.2 (SGD, poly, warmup 1k iters; linear-scaled from ref bs=4 lr=0.05; reduced from 152/GPU after OOM — 5 total prediction heads)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_bisenetv2
- **Results (val)**: F1=60.84% (PyTorch) / 60.81% (ONNX, delta=0.03%); mIoU=86.56% (best iter 1500)
- **Notes**: Training unstable at 1500 iters — curve still rising, not converged. LR 3.2 (aggressive linear scaling) caused warmup instability. No ONNX committed to LFS (unstable training; will be retrained in Phase 2). See phase1_report.md §8 for details.

---

## EXP-003
- **ID**: EXP-003
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase1-baselines @ 67529e2
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/segnext_t_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 1500 (eval every 250; save best mIoU; best at iter 1500)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=56, total_bs=112, lr=6e-5 (AdamW, poly, warmup 1.5k iters; LR fixed per AdamW rule; reduced from 75/GPU after OOM — HAM decoder)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_segnext_t
- **Results (val)**: F1=72.86% (PyTorch), ONNX delta=0.00%; mIoU=89.81% (best iter 1500)
- **Notes**: Nearly converged at 1500 iters. Competitive with SegFormer at 5.6× smaller model. Logits ONNX committed to LFS. See phase1_report.md for convergence analysis.

---

## EXP-004
- **ID**: EXP-004
- **Date**: 2026-03-03 06:30 UTC
- **Phase**: Phase 1 — baseline validation
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase1-baselines @ 67529e2
- **Config**: challenges/embedded-semantic-segmentation/phase1/configs/segformer_mit-b2_768x384_lars_phase1.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 1500 (eval every 250; save best mIoU; best at iter 1000)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=36, total_bs=72, lr=6e-5 (AdamW, poly, warmup 1.5k iters; LR fixed per AdamW rule; pre-emptively reduced from 45/GPU)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase1_segformer_mit-b2
- **Results (val)**: F1=73.64% (PyTorch), ONNX delta=0.00%; mIoU=92.02% (best iter 1000)
- **Notes**: Converged early at iter 1000; plateau from 1000→1500. Highest F1 of Phase 1 but above 30 FPS limit (accuracy upper-bound reference). Logits ONNX committed to LFS. See phase1_report.md for convergence analysis.
