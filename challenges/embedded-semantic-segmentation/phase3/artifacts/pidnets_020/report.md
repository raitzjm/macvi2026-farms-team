# EXP-020: PIDNet-S Phase 3 Definitive Run

## Training Configuration
- **Model:** PIDNet-S
- **Iterations:** 15,000
- **Warmup:** 2,500 iterations (LinearLR)
- **Batch Size:** 320 (160 per GPU, 2 GPUs)
- **Learning Rate:** 0.133 (SGD)
- **Augmentations:** CopyPasteObstacles (shared RAM optimization), EnhancedPhotoMetricDistortion, GenerateEdgeMasked

## Validation Results (Local Val Set)
- **Best mIoU:** 92.82% (PyTorch FP32)
- **F1 Score:** 75.97%
- **Q Score:** 70.51%
- **Obstacle IoU:** 89.82%

## ONNX Export & Quantization
- **ONNX mIoU:** 92.74% (Δ -0.08% vs PyTorch)
- **INT8 Approx mIoU:** 92.32% (Δ -0.42% vs ONNX)
- **INT8 Approx F1:** 76.29% (Δ +0.51% vs ONNX)
- **Stability:** High INT8 stability observed.

## Notes
Implemented shared RAM cache for CopyPasteObstacles, reducing iteration data_time from ~0.85s to ~0.53s.
