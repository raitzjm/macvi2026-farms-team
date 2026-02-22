# Embedded Semantic Segmentation Challenge

## Task
3-class maritime segmentation (sky=0, water=1, obstacle=2) on LaRS dataset.
We train and export an ONNX model. The challenge platform handles INT8
quantization (Qualcomm SNPE) and deployment (Luxonis RVC4) - we do NOT do this.
Our job: train model -> export ONNX -> submit ONNX to challenge server.
Winner: highest F1 detection score (FPS tiebreaker). Must achieve >=30 FPS at 768x384.
Challenge page: https://macvi.org/workshop/cvpr/challenges/usv_seg_emb

## Dataset
- LaRS: /home/farms/data/lars/ (train: 2607, val: 200, test: 1205 images)
- Test set has NO segmentation masks
- External data allowed but must be disclosed

## Workspace
- mmsegmentation-macvi/: copied from github.com/lojzezust/mmsegmentation-macvi
- Training resolution: 768x384 (CRITICAL - always train at target resolution)

## Key Dependencies
- Python 3.12, CUDA 12.1, PyTorch 2.4.x, mmcv 2.2.0
- Install mmcv prebuilt: pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

## Experiment Phases
Phase 1: Baselines - default mmseg-macvi configs, LaRS-only, no changes
Phase 2: Augmentation - same models, vary augmentation strategies
Phase 3: Training tricks - 4-class training, loss tuning, architecture experiments
Phase 4: External data - conditional, only if distribution analysis shows gaps

Phases can be modified, reordered, added, or skipped as the project evolves.

## Current Status: SETUP
Setting up container, environment, and validating mmsegmentation-macvi installation.
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
