# Embedded Semantic Segmentation Challenge

## Task
3-class maritime segmentation (sky=0, water=1, obstacle=2) on LaRS dataset.
We train and export an ONNX model. The challenge platform handles INT8
quantization (Qualcomm SNPE) and deployment (Luxonis RVC4) - we do NOT do this.
Our job: train model -> export ONNX -> submit ONNX to challenge server.
Winner: highest F1 detection score (FPS tiebreaker). Must achieve >=30 FPS at 768x384.
Challenge page: https://macvi.org/workshop/cvpr/challenges/usv_seg_emb

## Dataset
- LaRS: /home/farms/data/lars_dataset/ (train: 2605, val: 198, test images)
  Note: images and masks are split into separate img_dir/ and ann_dir/ trees.
  A symlink tree at /home/farms/lars_linked/ reconciles this for LaRSDataset.
- Test set has NO segmentation masks
- External data allowed but must be disclosed

## Workspace
- mmsegmentation-macvi/: copied from https://github.com/lojzezust/mmsegmentation-macvi
  (fork of open-mmlab/mmsegmentation 0.x, adds LaRSDataset, MaSTrDataset, and 25 LaRS configs)
  We use the ORIGINAL version (not any ported/updated fork) to ensure a clean baseline.
  Do NOT commit any changes to this directory — it must stay at upstream state.
- Training resolution: 768x384 (CRITICAL - always train at target resolution)

## Key Dependencies (baked into Docker image — Option C: Pixi)
- Python 3.11 (conda-forge via Pixi)
- CUDA 12.1 (container base image: nvidia/cuda:12.1.1-devel-ubuntu22.04)
- PyTorch 2.1.2+cu121 (prebuilt wheel)
- mmcv-full 1.7.2 (prebuilt wheel for cu121+torch2.1 — NO source build needed)
  mmseg/__init__.py asserts: MMCV_MIN='1.3.13', MMCV_MAX='1.8.0'. mmcv 2.x will NOT work.
- numpy<2 (mmcv-full 1.7.2 compiled with NumPy 1.x ABI)
- setuptools<70 (restores pkg_resources removed in setuptools 82.x)
- System libs (apt): libxcb1 libxcb-xinerama0 libgl1 libglib2.0-0

## Environment Setup
The Docker image pre-installs the complete stack — Python 3.11, PyTorch, mmcv-full,
and mmsegmentation-macvi (non-editable, baked at build time). No post-start setup
is needed. `python -c "import mmseg, torch, mmcv"` works immediately on container start.

mmsegmentation-macvi is never modified (upstream-only per project rules), so the
non-editable baked install is correct. If the upstream is updated, rebuild the image.

## mmcv scatter patch (already applied in Docker image)
PyTorch 2.1.x passes GPU device IDs as plain int but mmcv's _get_stream()
expects torch.device. A one-line patch in mmcv/parallel/_functions.py line 75
fixes this (applied automatically during image build).

## Dataset symlink tree (must be rebuilt on each fresh container)
LaRSDataset expects: data_root/images/ and data_root/semantic_masks/
The LaRS dataset stores images and masks in separate trees. Build the symlink
tree once per container session:

    mkdir -p /home/farms/lars_linked/train /home/farms/lars_linked/val
    ln -sf /home/farms/data/lars_dataset/img_dir/train/images    /home/farms/lars_linked/train/images
    ln -sf /home/farms/data/lars_dataset/ann_dir/train/semantic_masks /home/farms/lars_linked/train/semantic_masks
    ln -sf /home/farms/data/lars_dataset/img_dir/train/image_list.txt /home/farms/lars_linked/train/image_list.txt
    ln -sf /home/farms/data/lars_dataset/img_dir/val/images      /home/farms/lars_linked/val/images
    ln -sf /home/farms/data/lars_dataset/ann_dir/val/semantic_masks   /home/farms/lars_linked/val/semantic_masks
    ln -sf /home/farms/data/lars_dataset/img_dir/val/image_list.txt   /home/farms/lars_linked/val/image_list.txt

Then use data_root = '/home/farms/lars_linked/' in training configs.

## Building from source
If any package must be built from source (e.g. mmcv-full with CUDA ops), always use all
available CPU cores to avoid multi-hour compile times:
    MAX_JOBS=$(nproc) MMCV_WITH_OPS=1 pip install mmcv-full==1.7.2 --no-build-isolation
MAX_JOBS controls the number of parallel ninja/make jobs for CUDA extension compilation.

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
