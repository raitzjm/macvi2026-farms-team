# Local Eval Metrics — Embedded Semantic Segmentation

Preprocessing: keep-ratio bilinear resize → center reflect-pad to 768×384 → ImageNet normalize
Post-processing: unpad → nearest-neighbour resize to original resolution (argmax ONNX, server-matching)
Dataset: LaRS val (198 images)
Script: `phase2/scripts/export_and_eval.py`

**These are proxy numbers — server uses Qualcomm SNPE INT8 on RVC4.**
Cross-reference against `submissions.md` to calibrate local vs server gap.

---

## Phase 2 — FP32 ONNX (2026-03-05)

Platform: NVIDIA RTX A6000, CUDA 12.1, ORT 1.24.2 (CPUExecutionProvider — FPS is CPU inference, not GPU)

| Model | EXP | Checkpoint | mIoU | F1 | Q | Pr | Re | IoU-obs | IoU-wat | IoU-sky | FPS |
|-------|-----|------------|------|----|---|----|----|---------|---------|---------|-----|
| segformer_mit-b2 | EXP-005 | best_mIoU_iter_2000 | 92.70% | 77.51% | 71.85% | 84.90% | 71.31% | 90.40% | 98.17% | 89.52% | 3.3 |
| segnext_s | EXP-010 | best_mIoU_iter_4750 | 92.40% | 72.33% | 66.83% | 83.35% | 63.88% | 89.48% | 97.77% | 89.94% | 4.4 |
| segnext_t | EXP-006 | best_mIoU_iter_4250 | 91.79% | 71.74% | 65.85% | 82.58% | 63.42% | 88.59% | 97.72% | 89.05% | 6.3 |
| bisenetv2 | EXP-009 | best_mIoU_iter_6000 | 91.09% | 69.06% | 62.91% | 83.70% | 58.78% | 86.45% | 97.30% | 89.53% | 12.1 |
| stdc2 | EXP-007 | best_mIoU_iter_2750 | 90.69% | 68.92% | 62.50% | 87.69% | 56.77% | 85.77% | 96.70% | 89.59% | 10.2 |
| pidnets | EXP-008 | best_mIoU_iter_4750 | 93.25% | 76.17% | 71.03% | 86.60% | 67.98% | 90.38% | 98.15% | 91.22% | 16.4 |

<!-- Q = mIoU × F1 (both as fractions, result ×100); matches challenge server formula -->
<!-- IoU-obs = obstacle class; hardest class, main lever for leaderboard improvement -->

PyTorch upper bound (logits + bilinear upsample, GPU):

| Model | PT mIoU | PT F1 | PT Q | ΔmIoU vs ONNX | ΔF1 vs ONNX |
|-------|---------|-------|------|---------------|-------------|
| segformer_mit-b2 | 92.78% | 77.47% | 71.87% | -0.08% | +0.04% |
| segnext_s | 92.46% | 72.38% | 66.92% | -0.07% | -0.05% |
| segnext_t | 91.84% | 71.81% | 65.95% | -0.06% | -0.07% |
| bisenetv2 | 91.14% | 69.15% | 63.02% | -0.05% | -0.09% |
| stdc2 | 90.69% | 68.77% | 62.37% | -0.00% | -0.15% |
| pidnets | 93.33% | 76.09% | 71.01% | -0.08% | +0.08% |

ONNX delta is ≤0.08% mIoU for all models — export is accurate.

---

## Phase 2 — INT8 ONNX (onnxruntime static quantization, QUInt8, 50-image calibration)

**WARNING: SegNeXt models catastrophically fail under ORT INT8.**
The HAM (Hamburger attention) decoder uses iterative matrix decompositions (NMF/VQ)
that are highly sensitive to quantization noise. GroupNorm layers also degrade sharply.
These INT8 numbers are NOT predictive of server performance; they are included only to
flag the quantization sensitivity issue for the submission decision.

| Model | INT8 mIoU | INT8 F1 | INT8 Q | ΔmIoU | ΔF1 | INT8 FPS | INT8-stable? |
|-------|-----------|---------|--------|-------|-----|----------|--------------|
| segformer_mit-b2 | 70.92% | 50.19% | 35.60% | -21.78% | -27.32% | 3.2 | ✗ (sensitive) |
| segnext_s | 10.70% | 92.47% | 9.89% | -81.70% | +20.14% | 4.6 | ✗ (catastrophic) |
| segnext_t | 24.91% | 1.92% | 0.48% | -66.88% | -69.82% | 5.8 | ✗ (catastrophic) |
| bisenetv2 | 91.17% | 70.44% | 64.22% | +0.08% | +1.38% | 9.8 | ✓ |
| stdc2 | 90.82% | 69.31% | 62.95% | +0.13% | +0.39% | 11.4 | ✓ |
| pidnets | 93.43% | 75.66% | 70.69% | +0.18% | -0.51% | 15.1 | ✓ |

**Key insight:** CNN backbones with BatchNorm (PIDNet, STDC2, BiSeNetV2) are highly
INT8-stable (±0.5% mIoU). Transformer/attention models (SegFormer, SegNeXt) degrade
significantly. This matters for the server submission since the challenge server applies
Qualcomm SNPE INT8 — SegNeXt and SegFormer may underperform their FP32 numbers on device.

**Submission recommendation:** Prefer PIDNet-S or STDC2 for robustness to server INT8.
SegFormer and SegNeXt-S FP32 numbers look competitive but server INT8 may negate their advantage.

---

## Summary Ranking (FP32 ONNX, local proxy)

| Rank | Model | Q | mIoU | F1 | FPS (CPU ORT) |
|------|-------|---|------|----|---------------|
| 1 | pidnets | 71.03% | 93.25% | 76.17% | 16.4 |
| 2 | segformer_mit-b2 | 71.85% | 92.70% | 77.51% | 3.3 |
| 3 | segnext_s | 66.83% | 92.40% | 72.33% | 4.4 |
| 4 | segnext_t | 65.85% | 91.79% | 71.74% | 6.3 |
| 5 | bisenetv2 | 62.91% | 91.09% | 69.06% | 12.1 |
| 6 | stdc2 | 62.50% | 90.69% | 68.92% | 10.2 |

Note: FPS above is CPU ORT (CPUExecutionProvider); challenge server is Qualcomm SNPE on
RVC4, measured at 67.6 FPS for PIDNet-S SGD (Sub #16537). All models must achieve ≥30 FPS
on device. FPS constraint likely rules out SegFormer (3.3 CPU FPS suggests very low on device).
