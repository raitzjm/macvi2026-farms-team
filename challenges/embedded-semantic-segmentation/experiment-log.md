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
| EXP-001 | STDC2 | LaRS | 768×384 | reflect pad, keep_ratio | 69.30% | 90.78% | — | — | 67529e2 |
| EXP-002 | BiSeNetV2 | LaRS | 768×384 | reflect pad, keep_ratio | 60.84% (PyTorch) / 60.81% (ONNX) | 86.56% | — | — | 67529e2 |
| EXP-003 | SegNeXt-T | LaRS | 768×384 | reflect pad, keep_ratio | 72.86% | 89.81% | — | — | 67529e2 |
| EXP-004 | SegFormer MiT-B2 | LaRS | 768×384 | reflect pad, keep_ratio | 73.64% | 92.02% | — | — | 67529e2 |

---

## Entries

(entries will be appended below by the agent)

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
