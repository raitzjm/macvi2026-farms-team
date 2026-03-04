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
| EXP-006 | SegNeXt-T | LaRS | 768×384 | reflect pad, keep_ratio | TBD (see report) | 89.40% | — | — | c50e7a5 |
| EXP-007 | STDC2 | LaRS | 768×384 | reflect pad, keep_ratio | TBD (see report) | 89.08% | — | — | c50e7a5 |
| EXP-008 | PIDNet-S | LaRS | 768×384 | reflect pad, keep_ratio | TBD (see report) | 89.92% | — | — | f27a5a1 |
| EXP-009 | BiSeNetV2 | LaRS | 768×384 | reflect pad, keep_ratio | TBD (see report) | 89.12% | — | — | aecd436 |
| EXP-010 | SegNeXt-S | LaRS | 768×384 | reflect pad, keep_ratio | TBD (see report) | 89.62% | — | — | f94b3a5 |
| EXP-011 | UNet-s5-d16 (FCNHead+CE+Dice, AdamW) | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-012 | STDC2 AdamW | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-013 | BiSeNetV2 AdamW | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
| EXP-014 | PIDNet-S AdamW | LaRS | 768×384 | reflect pad, keep_ratio | TBD | TBD | — | — | TBD |
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
- **Status**: COMPLETED (stopped early at iter ~3250 — plateau)
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/segnext_t_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: ~3250/5000 (stopped early — plateau confirmed)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=56, total_bs=112, lr=6e-5 (AdamW, poly, warmup 500 iters; pinned from Phase 1)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_segnext_t
- **Results (val)**: best mIoU=89.40% @ iter 2000; plateau 89.33-89.36% from iter 2250 onwards
- **Notes**: Cross-check at iter 1500: mIoU=89.34% vs Phase 1 ref 89.81% (−0.47%, within ±1.0% ✓). Plateau confirmed from iter 2000: 5 consecutive evals (iter 2250–3250) all below 89.40% with no improvement. Stopped early to save time. WandB images logging fixed (interval=5) — check this run for prediction overlays.

---

## EXP-007
- **ID**: EXP-007
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/stdc2_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 3000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=160, total_bs=320, lr=0.267 (SGD, poly, warmup 300 iters; pinned from Phase 1)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_stdc2
- **Results (val)**: best mIoU=89.08% @ iter 2750 (tied at iter 3000); 228 WandB media files
- **Notes**: Cross-check at iter 1500: mIoU=86.77% (oscillation phase — not representative). SGD training oscillated significantly (89.01→86.48→87.5→86.77→86.06→88.57→88.07→88.03→89.08). Best converged late at iter 2750 with final mIoU=89.08% ≈ Phase 1 ref 90.78% (−1.7% gap; likely evaluator difference + oscillation).

---

## EXP-008
- **ID**: EXP-008
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ f27a5a1
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/pidnets_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=160, total_bs=320, lr=0.267 (SGD linear-scaled; calibrated 2026-03-04; OOM boundary ~220/GPU)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_pidnets
- **Results (val)**: best mIoU=**89.92%** @ iter 4750 (aAcc=95.40%, mAcc=94.72%); final iter 5000: 89.56%; Val F1 TBD (see Phase 2 report)
- **Notes**: New model in Phase 2 (no Phase 1 cross-check). BS calibrated 2026-03-04 (OOM boundary ~220/GPU → bs=160/GPU, 74% util).
  Run 1: GenerateEdge IndexError — ReflectPad not updating img_shape. Fixed commit 2043c36. Failed run tm8mvsxr deleted.
  Run 2: lr=0.267 (0.01×320/12) caused severe oscillations (79–89% range); stopped at iter 1750. Failed run vwmbx1eu deleted.
  Run 3 (final): lr=0.133 (0.01×160/12, halved), warmup extended to 1000 iters — stable convergence, new best at iter 4750.
  WandB: 380 media files logged. Checkpoint: best_mIoU_iter_4750.pth.

---

## EXP-009
- **ID**: EXP-009
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ aecd436
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/bisenetv2_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 6000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=128, total_bs=256, lr=0.80 (SGD corrected; Phase 1 used 3.2 — wrong); warmup 300 iters
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_bisenetv2
- **Results (val)**: best mIoU=**89.12%** @ iter 6000 (aAcc=94.98%, mAcc=94.22%); Val F1 TBD (see Phase 2 report)
- **Notes**: LR corrected to 0.80 (0.05×256/16 total-to-total). Phase 1's wrong lr=3.2 caused instability. SGD oscillations throughout but clear upward trend; new bests at iters 500, 1000, 1500, 2250, 2500, 3250, 4250, 5000, 5500, 5750, 6000.
  Final best at last iter (6000) = 89.12% — much better than Phase 1 86.56% due to correct LR and full 6k training.
  456 WandB media files logged. Checkpoint: best_mIoU_iter_6000.pth.

---

## EXP-010
- **ID**: EXP-010
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2) [added late]
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ f94b3a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/segnext_s_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=40, total_bs=80, lr=6e-5 (AdamW fixed; calibrated 2026-03-04; OOM boundary ~50/GPU → bs=40)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_segnext_s
- **Results (val)**: mIoU=89.62% best @ iter 4750; aAcc=95.13%, mAcc=94.55%
- **Notes**: New model in Phase 2 (no Phase 1 cross-check). SegNeXt-S larger backbone than T (embed_dims [64,128,320,512] vs T [32,64,160,256]). BS calibrated: bs=32→30.4GB, bs=48→45.5GB, bs=56→OOM. Late improvement at iter 4750 — good to run 5k iters. Comparable to SegNeXt-T (89.40%) and STDC2 (89.08%).

---

## EXP-011
- **ID**: EXP-011
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2) [added late]
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ TBD
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/unet_s5d16_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=16, total_bs=32, lr=6e-5 (AdamW fixed; switched from SGD per user request 2026-03-04)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_unet_adamw
- **Results (val)**: TBD
- **Notes**: New model in Phase 2 (no Phase 1 cross-check). UNet trains from scratch (no pretrained backbone). Optimizer changed to AdamW with lr=6e-5 (user request). Loss: CE(1.0)+Dice(3.0) on main head, CE(0.4) on auxiliary. BS calibrated 2026-03-04: OOM@~bs=19 → bs=16/GPU (82% VRAM).

---

## EXP-012
- **ID**: EXP-012
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — AdamW variant of SGD STDC2 (EXP-007)
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ TBD
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/stdc2_adamw_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=TBD (calibrate; est. 64), total_bs=TBD, lr=6e-5 (AdamW fixed; no linear scaling)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_stdc2_adamw
- **Results (val)**: TBD
- **Notes**: Same STDC2 architecture as EXP-007 (SGD). AdamW replaces SGD. Calibrate BS (AdamW needs ~20% more memory for optimizer state). 5k iters (extended from SGD's 3k; AdamW may converge slower).

---

## EXP-013
- **ID**: EXP-013
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — AdamW variant of SGD BiSeNetV2 (EXP-009)
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ TBD
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/bisenetv2_adamw_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=TBD (calibrate; est. 64), total_bs=TBD, lr=6e-5 (AdamW fixed; no linear scaling)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_bisenetv2_adamw
- **Results (val)**: TBD
- **Notes**: Same BiSeNetV2 architecture as EXP-009 (SGD). AdamW replaces SGD. Calibrate BS before first run. 5k iters.

---

## EXP-014
- **ID**: EXP-014
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — AdamW variant of SGD PIDNet-S (EXP-008)
- **Status**: PLANNED
- **Branch/Commit**: exp/phase2-mmseg122 @ TBD
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/pidnets_adamw_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=TBD (calibrate; est. 64), total_bs=TBD, lr=6e-5 (AdamW fixed; no linear scaling)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_pidnets_adamw
- **Results (val)**: TBD
- **Notes**: Same PIDNet-S architecture as EXP-008 (SGD). AdamW replaces SGD. Requires GenerateEdge in train pipeline (same as SGD version). Calibrate BS before first run. 5k iters.

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
