# Embedded Semantic Segmentation Challenge

## Task
3-class maritime segmentation (sky=0, water=1, obstacle=2) on LaRS dataset.
We train and export an ONNX model. The challenge platform handles INT8
quantization (Qualcomm SNPE) and deployment (Luxonis RVC4) - we do NOT do this.
Our job: train model -> export ONNX -> submit ONNX to challenge server.
Winner: highest F1 detection score (FPS tiebreaker). Must achieve >=30 FPS at 768x384.
Challenge page: https://macvi.org/workshop/cvpr/challenges/usv_seg_emb

## Dataset
- LaRS: /home/farms/data/lars_dataset/ (train: 2607, val: 200, test: 1205 images)
- Test set has NO segmentation masks
- External data allowed but must be disclosed
- img_dir: {split}/images/, ann_dir: {split}/semantic_masks/
- Symlink tree at /home/farms/lars_linked/ must be rebuilt each container session (not persistent)

## Workspace
- mmsegmentation-macvi/: copied from github.com/lojzezust/mmsegmentation-macvi
- Training resolution: 768x384 (CRITICAL - always train at target resolution)

## Key Dependencies
- Conda env `macvi` at /home/farms/miniforge3/envs/macvi/ (on PATH by default)
- Python 3.11 (conda-forge), CUDA 12.1
- PyTorch 2.1.2+cu121, mmcv-full 1.7.2 (prebuilt wheel — no source build)
- mmsegmentation 0.30.0 installed non-editable at /opt/macvi-env/mmsegmentation-macvi/
- All baked into the Docker image — no post-start setup needed
- Required patches baked in: setuptools<70, numpy<2, mmcv scatter patch (_functions.py)

## Experiment Phases
Phase 1: Baselines - default mmseg-macvi configs, LaRS-only, no changes
Phase 2: Augmentation - same models, vary augmentation strategies
Phase 3: Training tricks - 4-class training, loss tuning, architecture experiments
Phase 4: External data - conditional, only if distribution analysis shows gaps

Phases can be modified, reordered, added, or skipped as the project evolves.

## Current Status: READY FOR TRAINING
Container environment validated (issue #2). Dependency stack baked into image.
Next milestone: Phase 1 baselines (DeepLabv3+ ResNet-101, BiSeNetv2, STDC1/2).

## Published Baselines (from challenge website)
Three variants per model:
- orig: trained AND evaluated at original 2048×1024 resolution
- orig (768×384): same weights, inference-only at 768×384 (no retraining)
- quantized: INT8 quantized via Qualcomm SNPE, runs on device (Luxonis RVC4) at 768×384

The challenge platform does quantization — we submit ONNX, they quantize.
FPS requirement: ≥30 FPS. Models below threshold cannot win (shown for reference).

| Model                   | orig F1 | orig (768×384) F1 | Δ res  | quantized F1 | FPS  | ≥30 FPS? |
|-------------------------|---------|-------------------|--------|--------------|------|----------|
| FCN (ResNet-50)         | 57.9    | 52.8              | -5.2   | 54.0         | 19.7 | ✗        |
| FCN (ResNet-101)        | 63.4    | 52.7              | -10.6  | 53.8         | 16.8 | ✗        |
| DeepLabv3+ (ResNet-101) | 64.0    | 58.0              | -6.0   | 57.4         | 16.6 | ✗        |
| BiSeNetv1 (ResNet-50)   | 42.8    | 45.1              | +2.3   | 45.6         | 28.7 | ✗        |
| BiSeNetv2               | 54.7    | 42.9              | -11.8  | 46.0         | 44.6 | ✓        |
| STDC1                   | 61.8    | 48.5              | -13.3  | 47.9         | 45.8 | ✓        |
| STDC2                   | 64.3    | 50.8              | -13.5  | 49.9         | 38.3 | ✓        |
| SegFormer (MiT-B2)      | 70.0    | 61.8              | -8.2   | 58.0         | 15.0 | ✗        |

Key observation: STDC1 and STDC2 suffer the largest resolution penalty (-13 F1) because they
were not trained at the target resolution. Training natively at 768×384 should recover much
of this gap and is our primary advantage over these published baselines.

## Key References
- Challenge: macvi.org/workshop/cvpr/challenges/usv_seg_emb
- Starter repo: github.com/lojzezust/mmsegmentation-macvi
- Strategy report: docs/MaCVi2026_Preliminary_Report.pdf
