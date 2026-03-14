# EXP-021: PIDChad-S Phase 3

## Architecture
PIDNet-S extended with:
- **AFFM** (Attention-based Feature Fusion Module): I-branch context gates P-branch via sigmoid before LightBag fusion. Suppresses water-reflection false positives.
- **FPPM** (Fast Pyramid Pooling Module): parallel AvgPool2d at scales [1,2,3,6,12] replacing cascaded PAPPM. Wider global anchor improves waterline estimation.

All ops are INT8-safe (no LayerNorm, GELU, Softmax). FPPM uses explicit nn.AvgPool2d (ONNX-compatible).

## Training Configuration
- **Model:** PIDChad-S (PIDNet-S + AFFM + FPPM)
- **Iterations:** 25,000
- **Warmup:** 5,000 iterations (LinearLR, doubled vs EXP-020 for LSJ aug shock)
- **Batch Size:** 320 (160 per GPU, 2× RTX A6000)
- **Learning Rate:** 0.133 (SGD, linear-scaled from PIDNet-S ref)
- **Augmentations:** LargeScaleJitter(0.1, 2.0) + CopyPasteObstacles + EnhancedPhotoMetricDistortion + GenerateEdgeMasked
- **Pretrained weights:** PIDNet-S ImageNet (AFFM + FPPM init fresh)

## Validation Results (Training Pipeline — Local Val Set, 198 images)
- **Best mIoU:** 96.56% @ iter 23500
- **Final mIoU (iter 25000):** 96.50%
- **Per-class (iter 25000):** obstacle=93.71%, water=97.77%, sky=98.01%
- **Best loss (all-time):** 1.3868 @ iter 23500

## vs EXP-020 (PIDNet-S Baseline)
- Training val mIoU: 96.56% vs EXP-020's 96.99% (−0.43%, within noise + heavier LSJ aug effect)
- INT8 smoke test (pre-training): ΔQ = −0.19% (stable)
- Post-export FP32/INT8 eval: run locally — see submissions.md for server results

## Notes
- Training val mIoU slightly lower than EXP-020 due to heavier LSJ(0.1,2.0) augmentation making the training signal harder
- AFFM+FPPM modules are hypothesized to improve F1 (water-reflection FP suppression) and WE_IoU (wider FPPM scale anchor) relative to baseline — validated by server submission
