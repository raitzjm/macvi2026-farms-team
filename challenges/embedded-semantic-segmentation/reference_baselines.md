# Published Baselines (from challenge website)

Three variants per model:
- orig: trained AND evaluated at original 2048x1024 resolution
- orig (768x384): same weights, inference-only at 768x384 (no retraining)
- quantized: INT8 quantized via Qualcomm SNPE, runs on device (Luxonis RVC4) at 768x384

The challenge platform does quantization — we submit ONNX, they quantize.
FPS requirement: >=30 FPS. Models below threshold cannot win (shown for reference).

| Model                   | orig F1 | orig (768x384) F1 | delta res  | quantized F1 | FPS  | >=30 FPS? |
|-------------------------|---------|-------------------|--------|--------------|------|----------|
| FCN (ResNet-50)         | 57.9    | 52.8              | -5.2   | 54.0         | 19.7 | no       |
| FCN (ResNet-101)        | 63.4    | 52.7              | -10.6  | 53.8         | 16.8 | no       |
| DeepLabv3+ (ResNet-101) | 64.0    | 58.0              | -6.0   | 57.4         | 16.6 | no       |
| BiSeNetv1 (ResNet-50)   | 42.8    | 45.1              | +2.3   | 45.6         | 28.7 | no       |
| BiSeNetv2               | 54.7    | 42.9              | -11.8  | 46.0         | 44.6 | yes      |
| STDC1                   | 61.8    | 48.5              | -13.3  | 47.9         | 45.8 | yes      |
| STDC2                   | 64.3    | 50.8              | -13.5  | 49.9         | 38.3 | yes      |
| SegFormer (MiT-B2)      | 70.0    | 61.8              | -8.2   | 58.0         | 15.0 | no       |

Key observation: STDC1 and STDC2 suffer the largest resolution penalty (-13 F1) because they
were not trained at the target resolution. Training natively at 768x384 should recover much
of this gap and is our primary advantage over these published baselines.
