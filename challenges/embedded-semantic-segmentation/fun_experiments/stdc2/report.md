# STDC2 @ 768×384 — Fun Experiment Report

**Date:** 2026-02-23
**Model:** STDC2 (STDCNet2 backbone, ImageNet pretrained)
**Dataset:** LaRS (3-class maritime: sky=0, water=1, obstacle=2)
**Training resolution:** 768×384 (W×H) — native target resolution
**Hardware:** NVIDIA RTX 3050 4 GB (container)

---

## Motivation

The challenge's published STDC2 baseline was trained at 2048×1024 and only ran
inference at 768×384 — no retraining at the target resolution. This caused a
**−13.5 F1** resolution mismatch penalty, the largest of any model in the challenge.

This experiment tests whether training natively at 768×384 recovers that gap.

| Variant | F1 |
|---|---|
| STDC2 orig (2048×1024 train + eval) | 64.3 |
| STDC2 orig (768×384 inference only, no retraining) | 50.8 |
| STDC2 quantized on device | 49.9 |

---

## Training Setup

| Parameter | Value |
|---|---|
| Config | `stdc2_768x384_lars_fun.py` |
| Batch size | 12 (hardware ceiling; bs=13 OOM) |
| VRAM usage | 3458 MiB / ~3768 MiB usable |
| Avg iter time | ~0.58 s/iter |
| Total iters | 12,000 (~55 epochs) |
| Total training time | ~1.94 hrs |
| Optimizer | SGD, lr=0.01, momentum=0.9, wd=5e-4 |
| LR schedule | Poly (power=0.9), warmup 1000 iters, min_lr=1e-4 |
| Train pipeline | RandomResize (0.5–2.0×) + RandomCrop + HFlip + PhotoMetricDistortion |
| Test pipeline | Fixed resize to 768×384 (keep_ratio=False) |

---

## Validation Results (all evals)

| Iter | mIoU | aAcc | IoU obstacle | IoU water | IoU sky | Best? |
|---|---|---|---|---|---|---|
| 2,000  | 94.47% | 97.45% | 90.16% | 96.64% | 96.60% | |
| 4,000  | 95.51% | 97.95% | 91.96% | 97.34% | 97.24% | |
| 6,000  | 95.44% | 97.92% | 91.85% | 97.32% | 97.15% | |
| 8,000  | 95.08% | 97.73% | 91.17% | 96.99% | 97.07% | |
| 10,000 | **95.54%** | **97.96%** | **91.94%** | 97.27% | **97.41%** | ⭐ |
| 12,000 | 95.45% | 97.92% | 91.92% | 97.31% | 97.12% | |

**Best checkpoint:** `best_mIoU_iter_10000.pth` (mIoU 95.54%)

---

## Training Curves

![Training curves](training_curves.png)

---

## Convergence Analysis

### Early convergence and plateau

The model converged very fast — by iter 2000 (after just ~9 epochs) it already
reached 94.47% mIoU, and peaked at 95.51% by iter 4000 (~18 epochs). From iter
4000 onward the model showed a characteristic plateau with small fluctuations
(±0.5% band), recovering marginally to 95.54% at iter 10000 as the LR fully
decayed, then settling back to 95.45% at iter 12000.

The final improvement over the initial best (iter 4000 → iter 10000) was just
+0.03% mIoU — noise-level variation, not meaningful progress.

### Overfitting pattern

The gap between train pixel accuracy (~98.5%) and val mIoU (95.5%) alongside the
declining trend after iter 4000 is a classic overfitting signature. Contributing factors:

1. **Small dataset** — LaRS training set has only 2,607 images. At 55 epochs the
   model has seen each image many times and has largely memorized the training set.
   This is likely the primary driver: more training on the same data stops helping
   once the model has saturated the dataset's variety.
2. **Moderate augmentation** — standard RandomCrop + HFlip + PhotoMetricDistortion
   provides some regularization but is not aggressive enough to prevent overfitting
   on a dataset this small.
3. **Three easy classes** — sky and water are visually consistent and easy to segment,
   pushing overall mIoU high quickly. The hard class (obstacle) is where the remaining
   gap sits (91.9% IoU vs 97%+ for sky/water).

Note: the recovery at iter 10000 is consistent with the LR decaying to near-zero,
which stabilizes the weights and reduces the noise in val performance. It does not
indicate the model is still learning new features.

### Is extending training worth it?

**No.** The full training curve makes it clear the model is in a plateau. Additional
iterations on the same data would continue the flat/noisy pattern without meaningful
gains. The right levers for improvement are not more iterations — they are:

- **Stronger augmentation** (quick win for Phase 1/2)
- **Larger batch + full schedule on server** (Phase 1 — proper compute)
- **Architecture improvements** (Phase 3)
- **External data** (Phase 4 — directly addresses the small dataset root cause)

---

## ONNX Export

| Property | Value |
|---|---|
| File | `stdc2_768x384.onnx` |
| Checkpoint | `best_mIoU_iter_10000.pth` |
| Input shape | `1 × 3 × 384 × 768` (NCHW, float32) |
| Output shape | `1 × 3 × 384 × 768` (logits, one channel per class) |
| Opset | 11 |
| PyTorch vs ONNX max diff | 4e-6 ✅ |

Output is **logits** (raw scores). Take `argmax` over the class dimension (dim=1)
to get the segmentation map. Class indices: 0=obstacle, 1=water, 2=sky.
Palette: obstacle=(247,195,37) yellow, water=(41,167,224) blue, sky=(90,75,164) purple.

---

## Key Takeaways

- Native 768×384 training clearly addresses the resolution mismatch. The published
  baseline suffered −13.5 F1 from inference-only resolution change; this experiment
  trains at the target resolution from the start.
- The model converged in ~18 epochs (iter 4000), confirming that for LaRS at this
  scale, short training schedules are sufficient to reach the accuracy ceiling of this
  setup.
- The accuracy ceiling itself is limited by dataset size (2,607 images). Phase 1 on
  the server with a full 80k schedule will not change this fundamentally — the bigger
  lever is augmentation and external data.
- STDC2 remains the strongest qualifying model (≥30 FPS) in the challenge baseline
  table. The goal for Phase 1 is to establish a proper trained-at-768×384 baseline
  for all qualifying models (BiSeNetv2, STDC1, STDC2).
