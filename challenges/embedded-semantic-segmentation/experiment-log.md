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
| EXP-005 | SegFormer MiT-B2 | LaRS | 768×384 | reflect pad, keep_ratio | 77.51% | 89.72% | — | — | c50e7a5 |
| EXP-006 | SegNeXt-T | LaRS | 768×384 | reflect pad, keep_ratio | 71.74% | 89.60% | — | — | c50e7a5 |
| EXP-007 | STDC2 | LaRS | 768×384 | reflect pad, keep_ratio | 68.92% | 89.08% | — | — | c50e7a5 |
| EXP-008 | PIDNet-S | LaRS | 768×384 | reflect pad, keep_ratio | 76.17% | 89.92% | — | — | f27a5a1 |
| EXP-009 | BiSeNetV2 | LaRS | 768×384 | reflect pad, keep_ratio | 69.06% | 89.12% | — | — | aecd436 |
| EXP-010 | SegNeXt-S | LaRS | 768×384 | reflect pad, keep_ratio | 72.33% | 89.62% | — | — | f94b3a5 |
| EXP-011 | UNet-s5-d16 (FCNHead+CE+Dice, AdamW) | LaRS | 768×384 | reflect pad, keep_ratio | — (not converged) | 85.42% | — | — | f0ad3e5 |
| EXP-012 | STDC2 AdamW | LaRS | 768×384 | reflect pad, keep_ratio | — (not evaluated) | 89.18% | — | — | ef35985 |
| EXP-013 | BiSeNetV2 AdamW | LaRS | 768×384 | reflect pad, keep_ratio | — (not evaluated) | 86.45% | — | — | ef35985 |
| EXP-014 | PIDNet-S AdamW | LaRS | 768×384 | reflect pad, keep_ratio | — (not evaluated) | 89.10% | — | — | ef35985 |
| EXP-015 | MSCAN-T + UPerHead + BoundaryLoss | LaRS | 768×384 | reflect pad, keep_ratio + GenerateEdge | IN PROGRESS | — | — | — | feat/segnext-t-uperhead-boundary |
| EXP-016 | PIDNet-M | LaRS | 768×384 | reflect pad, keep_ratio + GenerateEdge | IN PROGRESS | — | — | — | feat/pidnetm |
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
- **Date**: 2026-03-04 UTC (resumed + completed 2026-03-05 20:51 UTC)
- **Phase**: Phase 2 — full training baselines (mmseg 1.2.2)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ c50e7a5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/segnext_t_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (resumed from iter_2500.pth; was stopped at ~3250)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=56, total_bs=112, lr=6e-5 (AdamW, poly, warmup 500 iters; pinned from Phase 1)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_segnext_t (run ID: v0j5q1ik)
- **Results (val)**: best mIoU=**89.60%** @ iter 4250 (aAcc=95.22%, mAcc=94.56%); final iter 5000: 89.54%; Val F1 TBD (see Phase 2 report)
- **Notes**: Cross-check at iter 1500: mIoU=89.34% vs Phase 1 ref 89.81% (−0.47%, within ±1.0% ✓).
  Originally stopped at iter ~3250 (apparent plateau at 89.40%); resumed 2026-03-05 to full 5k for consistency.
  Full resumed trajectory: 2750→89.40%, 3000→89.40%, 3250→89.37%, 3500→89.48%, 3750→89.53%, 4000→89.58%, 4250→89.60% (best), 4500→89.58%, 4750→89.54%, 5000→89.54%.
  Early stop at iter 3250 cost +0.20% mIoU. Model peaked at 4250 then gently declined.
  Checkpoints: best_mIoU_iter_4250.pth (89.60%, use for submission) + iter_5000.pth (89.54%, fully decayed final state) — both kept.
  WandB note: new data points from resume not visible in UI (step counter backwards from 2500; silently dropped). Use .log files for full trajectory.

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
- **bs / lr**: samples_per_gpu=160, total_bs=320, lr=0.133 (SGD linear-scaled; calibrated 2026-03-04; OOM boundary ~220/GPU; initial 0.267 caused oscillations, halved to 0.133 for Run 3)
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
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ f0ad3e5
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/unet_s5d16_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=16, total_bs=32, lr=6e-5 (AdamW fixed; switched from SGD per user request 2026-03-04)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_unet_adamw
- **Results (val)**: mIoU=85.42% best @ iter 2750; aAcc=92.94%, mAcc=92.18%; val mIoU still oscillating ±2% at end of run — NOT converged
- **Notes**: New model in Phase 2 (no Phase 1 cross-check). UNet trains from scratch (no pretrained backbone). Lower mIoU than pretrained models (85.42% vs ~89%) expected — no ImageNet init. Loss: CE(1.0)+Dice(3.0) on main head, CE(0.4) on auxiliary. BS calibrated: OOM@~bs=19 → bs=16/GPU (82% VRAM).
  **⚠ NOT USABLE FOR ANALYSIS**: The model is not converged. Training loss at iter 5000 was ~0.137 and still declining; val mIoU oscillated between 78–85% throughout (characteristic of scratch training). PolyLR schedule decayed to lr=0 at iter 5000, so resuming this run is not practical.
  **Why not converged**: UNet-s5-d16 has no pretrained backbone — it must learn all features from scratch. 5000 iters (~62 epochs) is insufficient; scratch-trained CNNs typically need 150–200 epochs (~12–16k iters) to stabilize.
  **Future work**: Retrain properly with either (a) a pretrained ResNet encoder (ResUNet via segmentation-models-pytorch) which would bring it to ~88–91% territory, or (b) fresh run with SGD + 15k iters on a faster server (H100). Do not use EXP-011 checkpoint for Phase 2 analysis.

---

## EXP-012
- **ID**: EXP-012
- **Date**: 2026-03-04 UTC
- **Phase**: Phase 2 — AdamW variant of SGD STDC2 (EXP-007)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ ef35985
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/stdc2_adamw_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=128, total_bs=256, lr=6e-5 (AdamW fixed; no linear scaling; calibrated 2026-03-04: OOM@~192/GPU → bs=128, 72.8% VRAM)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_stdc2_adamw
- **Results (val)**: best mIoU=**89.18%** @ iter 2750 (aAcc=94.98%, mAcc=94.39%); final iter 5000: 89.06%; Val F1 TBD (see Phase 2 report)
- **Notes**: Same STDC2 architecture as EXP-007 (SGD). AdamW replaces SGD. Very smooth convergence — no SGD oscillations. Best at iter 2750, slight plateau afterwards with minor fluctuations (88.86–89.18%). +0.10% vs EXP-007 SGD (89.08%). WandB: 380 media files logged. Checkpoint: best_mIoU_iter_2750.pth.

---

## EXP-013
- **ID**: EXP-013
- **Date**: 2026-03-04 UTC (completed 2026-03-05 02:37 UTC)
- **Phase**: Phase 2 — AdamW variant of SGD BiSeNetV2 (EXP-009)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ ef35985
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/bisenetv2_adamw_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=112, total_bs=224, lr=6e-5 (AdamW fixed; no linear scaling; calibrated 2026-03-04: OOM between 128-160/GPU → bs=112, 73.9% VRAM)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_bisenetv2_adamw
- **Results (val)**: best mIoU=**86.45%** @ iter 4500 (tied at iter 4750; aAcc=93.57%, mAcc=92.67%); final iter 5000: 86.23%; Val F1 TBD (see Phase 2 report)
- **Notes**: Same BiSeNetV2 architecture as EXP-009 (SGD). AdamW replaces SGD. Smooth convergence (no oscillations), late-peaking at iter 4500.
  Val mIoU trajectory: 250→77.65%, 500→82.58%, 1000→83.92%, 1500→84.42%, 1750→84.56%, 2500→85.22%, 3000→85.89%, 3250→85.96%, 4500→86.45% (best), 5000→86.23%.
  −2.67% vs EXP-009 SGD (89.12%) — BiSeNetV2 benefits more from its tuned SGD LR than from AdamW.
  WandB: 380 media files logged. Checkpoint: best_mIoU_iter_4500.pth.

---

## EXP-014
- **ID**: EXP-014
- **Date**: 2026-03-05 UTC (completed 2026-03-05 19:22 UTC)
- **Phase**: Phase 2 — AdamW variant of SGD PIDNet-S (EXP-008)
- **Status**: COMPLETED
- **Branch/Commit**: exp/phase2-mmseg122 @ ef35985
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/pidnets_adamw_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=160, total_bs=320, lr=6e-5 (AdamW fixed; no linear scaling; calibrated 2026-03-05: OOM boundary ~232/GPU → bs=160, ~69.6% VRAM single-GPU, matches SGD version)
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_pidnets_adamw (run ID: 2l8jpdgk)
- **Results (val)**: best mIoU=**89.10%** @ iter 4500 (aAcc=94.94%, mAcc=94.30%); final iter 5000: 89.00%; Val F1 TBD (see Phase 2 report)
- **Notes**: Same PIDNet-S architecture as EXP-008 (SGD). AdamW replaces SGD. Requires GenerateEdge in train pipeline (same as SGD version). BS calibrated 2026-03-05: OOM between 224-240/GPU; selected bs=160 for DDP headroom (matches SGD).
  Val mIoU trajectory (resumed from iter 4000 @ 89.07%): 4250→89.09%, 4500→89.10% (best), 4750→89.08%, 5000→89.00%.
  Resumed run: WandB env var approach (WANDB_RUN_ID=2l8jpdgk WANDB_RESUME=must). Fast-forward with GenerateEdge took ~47 min (vs ~3 min for models without edge maps).
  +0.18% vs EXP-008 SGD (89.92% best — but SGD EXP-008 is the full 5k run so this is a fair comparison: SGD slightly better for PIDNet). Checkpoint: best_mIoU_iter_4500.pth.

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

---

## EXP-015
- **ID**: EXP-015
- **Date**: 2026-03-07 (started)
- **Phase**: Phase 2 — architecture experiment (Issue #18)
- **Status**: IN PROGRESS
- **Branch/Commit**: feat/segnext-t-uperhead-boundary
- **Config**: challenges/embedded-semantic-segmentation/phase2/configs/segnext_t_uperhead_768x384_lars_phase2.py
- **Dataset**: LaRS (train: 2605, val: 198)
- **Resolution**: 768×384
- **Iters**: 5000 (eval every 250 iters; save best mIoU)
- **Hardware**: 2× RTX A6000 48 GB (DDP, torchrun --standalone --nproc_per_node=2)
- **bs / lr**: samples_per_gpu=48, total_bs=96, lr=6e-5 (AdamW, fixed — no linear scaling)
  BS calibrated 2026-03-07: OOM boundary ~54/GPU; 85% target → rounded UP to 48 (88.9% of OOM).
  UPerHead (4 feat maps + PPM + BoundaryHead) heavier than HAM; boundary was ~75 in Phase 1.
  SyncBN in decode heads (UPerHead + BoundaryHead); BN in MSCAN backbone.
- **WandB run**: little_farms/embedded_segmentation_macvi2026/phase2_segnext_t_uperhead
- **Results (val)**: see phase report (F1 via export_and_eval.py post-training)
- **Notes**: Replaces LightHamHead (GN+NMF) with UPerHead (SyncBN+PPM) + MSCANBoundaryHead.
  Goal: INT8 stability (BN throughout) + improved F1 (BoundaryLoss). Reference: EXP-006 Q=65.85%.
  MSCANBoundaryHead: custom aux head in pipelines.py; takes stride-4 MSCAN feature (32ch, in_index=0).
  Scheduler: LinearLR (500 iters) + PolyLR (power=1.0) — same as EXP-006. 5k total iters.
