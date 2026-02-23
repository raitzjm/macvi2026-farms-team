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

## Published Baselines (from challenge website, for validation)
| Model                      | Quant F1 | FPS  |
|----------------------------|----------|------|
| DeepLabv3+ (ResNet-101)    | 57.4     | 16.6 |
| BiSeNetv2                  | 46.0     | 44.6 |
| STDC1                      | 47.9     | 45.8 |
| STDC2                      | 49.9     | 38.3 |
| SegFormer (MiT-B2)         | 58.0     | 15.0 |

## Key References
- Challenge: macvi.org/workshop/cvpr/challenges/usv_seg_emb
- Starter repo: github.com/lojzezust/mmsegmentation-macvi
- Strategy report: docs/MaCVi2026_Preliminary_Report.pdf
