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
Note: FPS includes image preprocessing (PIL resize + numpy pad + normalize). Pure inference FPS would be higher.

| Model | EXP | Checkpoint | mIoU | F1 | Q | Pr | Re | IoU-obs | IoU-wat | IoU-sky | FPS |
|-------|-----|------------|------|----|---|----|----|---------|---------|---------|-----|
| segformer_mit-b2 | EXP-005 | best_mIoU_iter_2000 | 92.70% | 77.51% | 71.85% | 84.90% | 71.31% | 90.40% | 98.17% | 89.52% | 3.3 |
| segnext_t_uperhead | EXP-015 | best_mIoU_iter_4250 | 92.48% | 80.21% | 74.18% | 83.94% | 76.80% | 89.88% | 97.80% | 89.76% | 3.7 |
| segnext_s | EXP-010 | best_mIoU_iter_4750 | 92.40% | 72.33% | 66.83% | 83.35% | 63.88% | 89.48% | 97.77% | 89.94% | 4.4 |
| pidnetm | EXP-016 | best_mIoU_iter_3250 | 92.33% | 72.82% | 67.23% | 85.91% | 63.19% | 89.39% | 97.65% | 89.95% | 7.8 |
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
| segnext_t_uperhead | 92.57% | 80.49% | 74.50% | -0.09% | -0.28% |
| segnext_s | 92.46% | 72.38% | 66.92% | -0.07% | -0.05% |
| pidnetm | 92.39% | 73.09% | 67.53% | -0.06% | -0.28% |
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
| segnext_t_uperhead | 14.32% | 90.58% | 12.97% | -78.16% | +10.37% | 3.9 | ✗ (catastrophic) |
| segnext_s | 10.70% | 92.47% | 9.89% | -81.70% | +20.14% | 4.6 | ✗ (catastrophic) |
| segnext_t | 24.91% | 1.92% | 0.48% | -66.88% | -69.82% | 5.8 | ✗ (catastrophic) |
| pidnets | 93.43% | 75.66% | 70.69% | +0.18% | -0.51% | 15.1 | ✓ |
| pidnetm | 92.37% | 74.24% | 68.57% | +0.04% | +1.42% | 7.9 | ✓ |
| bisenetv2 | 91.17% | 70.44% | 64.22% | +0.08% | +1.38% | 9.8 | ✓ |
| stdc2 | 90.82% | 69.31% | 62.95% | +0.13% | +0.39% | 11.4 | ✓ |

**Key insight:** CNN backbones with BatchNorm (PIDNet, STDC2, BiSeNetV2) are highly
INT8-stable (±0.5% mIoU). Transformer/attention models (SegFormer, SegNeXt) degrade
significantly. This matters for the server submission since the challenge server applies
Qualcomm SNPE INT8 — SegNeXt and SegFormer may underperform their FP32 numbers on device.

**Submission recommendation:** Prefer PIDNet-S or STDC2 for robustness to server INT8.
SegFormer and SegNeXt-S FP32 numbers look competitive but server INT8 may negate their advantage.

---

## Summary Ranking (FP32 ONNX, local proxy)

Sorted by descending FP32 Q-score. Note: this ranking does not account for INT8 stability —
see the INT8 table above for deployment recommendation (CNN models preferred for SNPE INT8).

| Rank | Model | Q | mIoU | F1 | FPS (CPU ORT) | INT8-stable? |
|------|-------|---|------|----|---------------|-------------|
| 1 | segnext_t_uperhead | 74.18% | 92.48% | 80.21% | 3.7 | ✗ (−78%) |
| 2 | segformer_mit-b2 | 71.85% | 92.70% | 77.51% | 3.3 | ✗ (−22%) |
| 3 | pidnets | 71.03% | 93.25% | 76.17% | 16.4 | ✓ |
| 4 | pidnetm | 67.23% | 92.33% | 72.82% | 7.8 | ✓ |
| 5 | segnext_s | 66.83% | 92.40% | 72.33% | 4.4 | ✗ (−82%) |
| 6 | segnext_t | 65.85% | 91.79% | 71.74% | 6.3 | ✗ (−67%) |
| 7 | bisenetv2 | 62.91% | 91.09% | 69.06% | 12.1 | ✓ |
| 8 | stdc2 | 62.50% | 90.69% | 68.92% | 10.2 | ✓ |

Note: FPS above is CPU ORT (CPUExecutionProvider); challenge server is Qualcomm SNPE on
RVC4, measured at 67.6 FPS for PIDNet-S SGD (Sub #16537). All models must achieve ≥30 FPS
on device. FPS constraint likely rules out SegFormer (3.3 CPU FPS suggests very low on device).

---

## Phase 2.5 — FP32 ONNX (2026-03-09)

Pipeline fixes vs Phase 2: (1) keep_ratio=True in train RandomResize, (2) img_padding_size
written to results for correct val de-padding, (3) centred reflect-pad in val pipeline,
(4) MacViSegVisualizer ignore-area overlay in WandB.
Script: `phase2_5/scripts/export_and_eval.py`
Platform: NVIDIA RTX A6000, CUDA 12.1, ORT 1.24.2

| Model | EXP | Checkpoint | mIoU | F1 | Q | Pr | Re | IoU-obs | IoU-wat | IoU-sky | FPS |
|-------|-----|------------|------|----|---|----|----|---------|---------|---------|-----|
| stdc2 | EXP-017 | best_mIoU_iter_3000 | 91.81% | 70.25% | 64.49% | 86.44% | 59.16% | 87.42% | 97.25% | 90.76% | 10.7* |
| pidnets | EXP-018 | best_mIoU_iter_4750 | 89.98% | 71.94% | 64.73% | 79.89% | 65.43% | 85.71% | 96.58% | 87.65% | 15.4 |

*STDC2 FPS measured while PIDNet-S training was running in parallel — slight underestimate.

Phase 2 vs Phase 2.5 delta (ONNX):

| Model | Phase 2 Q | Phase 2.5 Q | ΔQ | Phase 2 mIoU | Phase 2.5 mIoU | ΔmIoU | Phase 2 F1 | Phase 2.5 F1 | ΔF1 |
|-------|-----------|-------------|-----|-------------|----------------|-------|------------|--------------|-----|
| stdc2 | 62.50% | 64.49% | **+1.99%** | 90.69% | 91.81% | +1.12% | 68.92% | 70.25% | +1.33% |
| pidnets | 71.03% | 64.73% | **−6.30%** | 93.25% | 89.98% | −3.27% | 76.17% | 71.94% | −4.23% |

PyTorch upper bound (Phase 2.5):

| Model | PT mIoU | PT F1 | PT Q | ΔmIoU vs ONNX | ΔF1 vs ONNX |
|-------|---------|-------|------|---------------|-------------|
| stdc2 | 91.82% | 70.29% | 64.54% | −0.01% | −0.04% |
| pidnets | 90.04% | 71.92% | 64.75% | −0.06% | +0.02% |

## Phase 2.5 — INT8 ONNX (onnxruntime static quantization, QUInt8, 50-image calibration)

| Model | INT8 mIoU | INT8 F1 | INT8 Q | ΔmIoU | ΔF1 | ΔQ | INT8 FPS | INT8-stable? |
|-------|-----------|---------|--------|-------|-----|----|----------|--------------|
| stdc2 | 91.76% | 70.20% | 64.42% | −0.05% | −0.05% | −0.08% | 11.5 | ✓ |
| pidnets | 89.84% | 71.90% | 64.60% | −0.14% | −0.04% | −0.13% | 14.0 | ✓ |

Both Phase 2.5 models are INT8-stable (≤0.13% Q drop). STDC2 gains +1.99% Q from pipeline fixes; PIDNet-S loses −6.30% Q — pipeline fixes had an adverse effect on PIDNet-S (see issue #28 for analysis).

---

## Phase 3 — FP32 ONNX (2026-03-14)

Platform: NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.1, ORT 1.24.2 (CPUExecutionProvider — FPS is CPU inference, not GPU)
Script: `phase3/scripts/export_and_eval.py`

| Model | EXP | Checkpoint | mIoU | F1 | Q | Pr | Re | IoU-obs | IoU-wat | IoU-sky | FPS |
|-------|-----|------------|------|----|---|----|----|---------|---------|---------|-----|
| pidnets | EXP-020 | pidnets_020.pth | 92.74% | 75.78% | 70.28% | 85.00% | 68.37% | 89.64% | 97.81% | 90.77% | 17.8 |
| pidchad_s_021 | EXP-021 | pidchad_s_021.pth | 91.64% | 75.94% | 69.60% | 83.77% | 69.45% | 89.00% | 97.68% | 88.25% | 18.3 |

PyTorch upper bound (Phase 3):

| Model | PT mIoU | PT F1 | PT Q | ΔmIoU vs ONNX | ΔF1 vs ONNX |
|-------|---------|-------|------|---------------|-------------|
| pidnets | 92.82% | 75.97% | 70.51% | −0.08% | −0.18% |
| pidchad_s_021 | 91.70% | 76.30% | 69.97% | −0.06% | −0.36% |

ONNX delta is ≤0.08% mIoU — export is accurate.

## Phase 3 — INT8 ONNX (onnxruntime static quantization, QUInt8, 50-image calibration)

| Model | INT8 mIoU | INT8 F1 | INT8 Q | ΔmIoU | ΔF1 | ΔQ | INT8 FPS | INT8-stable? |
|-------|-----------|---------|--------|-------|-----|----|----------|--------------|
| pidnets | 92.33% | 76.35% | 70.49% | −0.41% | +0.56% | +0.21% | 21.5 | ✓ |
| pidchad_s_021 | 91.65% | 75.73% | 69.40% | +0.00% | −0.22% | −0.19% | 23.6 | ✓ |

Both Phase 3 models are INT8-stable. PIDNet-S (EXP-020) leads with Q=70.28% FP32 / 70.49% INT8.
PIDChad-S (EXP-021) AFFM+FPPM structural changes regressed mIoU by −1.10% and sky IoU by −2.55%
vs EXP-020, without meaningful F1 improvement. More aggressive LSJ (0.1–2.0) may also contribute.
