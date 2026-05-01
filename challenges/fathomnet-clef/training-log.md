# FathomNet-CLEF 2026 — Training Log

Metric: **mAP@[0.50:0.95]** (COCO standard, 10 IoU thresholds). Higher is better.
Micro-val: TBD — 200-image hand-audited stratified split (to be defined before first training run).

> **Format note:** This format will be adapted as the challenge progresses. If a new training
> dependency pattern requires changes, update the format and note it here with a date.
> Backfill of historical entries is not required — note the effective date of each format change.
> Last updated: 2026-04-30

> **2026-04-30 artifact policy:** every new tracked training run must have a durable bundle at
> `artifacts/training-runs/T-NNN/` with `metadata.json`, raw `stdout.log`, a resolved config
> snapshot, and the framework-native metrics artifact (`results.csv`, `metrics.jsonl`, or
> equivalent). The selected checkpoint remains under `artifacts/checkpoints/` and is referenced
> from the bundle and from the log entry. Canonical model configs stay in one path per model
> family; do not copy a new config file per experiment unless the model family truly forks.

---

## Training Index

| ID | Date | Track | Description | Seed | μval mAP |
|----|------|-------|-------------|------|----------|
| T-001 | 2026-04-28 | YOLO26l | YOLO26l-local-v1: 50 epochs, rect 1152px, batch≈4 (auto), RTX 3050 | 42 | 0.408 (train-val 90/10) |
| T-002 | 2026-04-28 | YOLOv11 | YOLOv11-kaggle-v15: 50 epochs, imgsz=640, rect disabled by DDP, batch=16/GPU, Kaggle T4x2 | 42 | 0.377 (train-val 90/10) |
| T-003 | 2026-04-28 | Co-DETR | Co-DETR-R50-kaggle-v15: 12 epochs, torchrun T4x2, AMP, best val bbox mAP 0.326 | 42 | 0.326 (train-val 90/10) |
| T-004 | 2026-05-01 | Co-DETR | Co-DETR-R50-tensordock: 12 epochs, 768×432, batch=5/GPU, 2×RTX4090, official pretrain, shared split | 42 | 0.417 (shared-train-val-v1) |

---

### T-001 — YOLO26l local baseline (50 epochs)

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLO26l |
| Env | ultralytics (envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Duration | ~5.9 h |
| Seed | 42 |
| Inputs | yolo26l.pt (COCO pretrained) |
| Artifacts | artifacts/training-runs/T-001/, artifacts/checkpoints/T-001-yolo26l-local/best.pt, outputs/yolo26l-local-v1/run/last.pt |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | YOLO26l |
| Pretrained weights | yolo26l.pt (COCO) |
| Input resolution | imgsz=1152, rect=True (short-side ~640 for 16:9) |
| Config file | scripts/train_yolo.py |
| Modifications | — |

#### Dataset Config
| Field | Value |
|---|---|
| Train split | 90% of 6,463 training images (5,816 images) |
| Val split | 10% of training images (647 images) |
| Classes | all 32 |
| PU strategy | none |
| Augmentation | ultralytics defaults |

#### Hyperparameters
| Field | Value |
|---|---|
| Epochs | 50 |
| Batch size | auto (≈4 on 4GB VRAM) |
| Optimizer | ultralytics default (SGD) |
| AMP | True |

#### Results
| Metric | Value |
|---|---|
| val mAP@[.5:.95] | **0.408** |
| val mAP@.50 | 0.549 |
| Best epoch | 47 |
| Final train box loss | 1.001 |

#### Notes
Strong result vs YOLOv11 Kaggle baseline (0.377 mAP50-95). WandB run logged to `little_farms/fathomnet2026` as `yolov26-baseline` (moved from wrong project post-run — name was manually corrected in WandB UI).

---

### T-002 — YOLOv11 Kaggle baseline (50 epochs)

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLOv11 |
| Env | ultralytics (envs/ultralytics/) |
| Platform | Kaggle T4 x2 |
| Duration | ~2.6 h |
| Seed | 42 |
| Inputs | yolo11l.pt (COCO pretrained) |
| Artifacts | artifacts/training-runs/T-002/, artifacts/checkpoints/T-002-yolov11-kaggle/best.pt, artifacts/submissions/I-001-yolov11-kaggle-nonsahi/submission.csv |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | YOLOv11l |
| Pretrained weights | yolo11l.pt (COCO) |
| Input resolution | imgsz=640, rect requested but disabled by multi-GPU DDP |
| Config file | kaggle/yolov11-baseline/kernel_train.py |
| Modifications | — |

#### Dataset Config
| Field | Value |
|---|---|
| Train split | 90% of 6,463 training images |
| Val split | 10% of training images |
| Classes | all 32 |
| PU strategy | none |
| Augmentation | ultralytics defaults |

#### Hyperparameters
| Field | Value |
|---|---|
| Epochs | 50 |
| Batch size | 16 per GPU (32 effective on T4x2) |
| Optimizer | AdamW |
| AMP | True |

#### Results
| Metric | Value |
|---|---|
| val mAP@[.5:.95] | **0.377** |
| val mAP@.50 | 0.501 |
| Best epoch | 49 |
| Final train box loss | 0.910 |

#### Notes
Completed on Kaggle T4 x2. submission.csv produced (68,794 predictions; 68,795 CSV lines including header). Archived to `artifacts/`. Official non-SAHI submission `52122957` scored **0.0348** public LB. WandB run logged under `little_farms/fathomnet2026` as `fathomnet-yolov11-baseline`.

---

### T-003 — Co-DETR R50 Kaggle baseline (12 epochs)

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | Co-DETR |
| Env | mmdet (envs/mmdet/) |
| Platform | Kaggle T4 x2 |
| Duration | ~4 h 46 m |
| Seed | 42 |
| Inputs | co_dino_5scale_r50_8xb2_1x_coco.py base config + COCO-pretrained weights |
| Artifacts | artifacts/training-runs/T-003/, artifacts/checkpoints/T-003-codetr-kaggle/best_coco_bbox_mAP_epoch_12.pth, artifacts/checkpoints/T-003-codetr-kaggle/config_used.py, artifacts/checkpoints/T-003-codetr-kaggle/idx_to_cat_id.json |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | Co-DETR R50 (5-scale) |
| Pretrained weights | `co_dino_5scale_r50_8xb2_1x_coco.py` base config / COCO initialization |
| Input resolution | train: RandomChoiceResize [(384,216), (448,252), (512,288), (576,324), (640,360)] + final Resize(480,270); test: Resize(480,270), keep_ratio=True |
| Config file | kaggle/codetr-baseline/kernel_train.py |
| Modifications | RandomCrop(0.7), PhotoMetricDistortion, AMP, 2-GPU torchrun, post-inference checkpoint zip |

#### Dataset Config
| Field | Value |
|---|---|
| Train split | 90% of 6,463 training images |
| Val split | 10% of training images |
| Classes | all 32 |
| PU strategy | none |
| Augmentation | RandomCrop, RandomChoiceResize, PhotoMetricDistortion, RandomFlip |

#### Hyperparameters
| Field | Value |
|---|---|
| Epochs | 12 |
| Batch size | 8 per GPU (16 effective on T4x2) |
| Optimizer | AdamW |
| LR initial | 2e-4 |
| Weight decay | 1e-4 |
| LR scheduler | MultiStepLR milestone=[11], gamma=0.1 |
| AMP | True |

#### Results
| Metric | Value |
|---|---|
| val bbox mAP@[.5:.95] | **0.326** |
| val bbox mAP@.50 | 0.480 |
| Best epoch | 12 |
| Last logged train loss | 39.67 |

#### Notes
Downloaded from completed Kaggle kernel outputs. Kaggle non-SAHI inference from the best epoch checkpoint produced `submission.csv` with 360,718 predictions and scored **0.0318** public LB. WandB run archived inside the downloaded `vis_data/wandb/` tree.

---

### T-004 — Co-DETR R50 TensorDock corrected baseline (12 epochs)

| Field | Value |
|---|---|
| Date | 2026-05-01 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | TensorDock (2× RTX 4090, fathom1) |
| Duration | ~2h39m (13:30–16:09 UTC+2) |
| Seed | 42 |
| Inputs | Co-DINO R50 official pretrain (https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth) |
| Repo commit | 097354f58722855be2d775f1b7111072d70864e3 |
| Config source | envs/mmdet-src/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py |
| Resolved config | artifacts/training-runs/T-004/resolved-config.py |
| Stdout log | artifacts/training-runs/T-004/stdout.log |
| Metrics artifact | artifacts/training-runs/T-004/metrics.jsonl |
| Selected checkpoint | artifacts/checkpoints/T-004-codetr-tensordock/best.pth (epoch 7) |
| Artifacts | artifacts/training-runs/T-004/, artifacts/checkpoints/T-004-codetr-tensordock/ |
| WandB | https://wandb.ai/little_farms/fathomnet2026/runs/dmr1u4w4 |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | Co-DETR R50 (5-scale) |
| Pretrained weights | co_dino_5scale_r50_1x_coco-7481f903.pth (OpenMMLab official COCO pretrain) |
| Input resolution | 768×432 |
| Config file | envs/mmdet-src/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py |
| Modifications | Resolution 768×432, batch=5/GPU, AMP, 2-GPU torchrun (DDP), shared-split manifest |

#### Dataset Config
| Field | Value |
|---|---|
| Train split | shared-train-val-v1 train: 5,816 images / 20,015 annotations |
| Val split | shared-train-val-v1 val: 647 images / 2,210 annotations |
| Split manifest | artifacts/split-manifests/shared-train-val-v1.json |
| Classes | all 32 |
| PU strategy | none |
| Augmentation | mmdet defaults (RandomFlip, Normalize, Pad) |

#### Hyperparameters
| Field | Value |
|---|---|
| Epochs | 12 |
| Batch size | 5 per GPU (10 effective on 2× RTX 4090) |
| Optimizer | AdamW (mmdet Co-DETR defaults) |
| LR initial | 1e-4 |
| AMP | True |

#### Results
| Metric | Value |
|---|---|
| Val mAP@[.5:.95] by epoch | 0.218, 0.312, 0.326, 0.358, 0.401, 0.391, **0.417**, 0.395, 0.413, 0.402, 0.360, 0.392 |
| Best val mAP@[.5:.95] | **0.417** (epoch 7) |
| Best val mAP@.50 | **0.562** (epoch 7) |
| Best epoch | 7 |
| Final epoch val mAP@[.5:.95] | 0.392 |

#### Notes
Corrected Co-DETR baseline addressing issue #84: T-003 used a wrong 90/10 random split (not the shared manifest) and COCO weights via the Kaggle config path rather than the official OpenMMLab pretrain. This run uses the shared-train-val-v1 split manifest and the official pretrain — making results directly comparable to T-001 YOLO26l. T-004 (0.417) beats T-001 (0.408 mAP) by +0.009 mAP. WandB: https://wandb.ai/little_farms/fathomnet2026/runs/dmr1u4w4

---

## Training Entry Template

<!--
### T-001 — <short description>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Track | YOLOv11 \| Co-DETR |
| Env | ultralytics \| mmdet |
| Platform | TensorDock (GPU type) \| Kaggle (GPU type) \| local RTX 3050 |
| Duration | Xh Ym |
| Seed | 42 |
| Inputs | — (fresh start) \| T-NNN checkpoint, I-NNN pseudo-labels, etc. |
| Repo commit | <40-char git SHA used to launch the run> |
| Config source | scripts/train_yolo.py \| kaggle/codetr-baseline/kernel_train.py |
| Resolved config | artifacts/training-runs/T-001/resolved-config.json |
| Stdout log | artifacts/training-runs/T-001/stdout.log |
| Metrics artifact | artifacts/training-runs/T-001/results.csv \| metrics.jsonl |
| Selected checkpoint | artifacts/checkpoints/T-001-.../best.pt |
| Artifacts | artifacts/training-runs/T-001/, artifacts/checkpoints/T-001-.../ |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | YOLOv11l \| Co-DETR-5scale-SwinL |
| Pretrained weights | yolo11l.pt (COCO) \| co_dino_5scale_swin_l_16e_o365tococo.pth |
| Input resolution | 1280×1280 |
| Config file | configs/yolov11_baseline.yaml |
| Modifications | — |

#### Dataset Config
| Field | Value |
|---|---|
| Train split | all 6,463 training images |
| Val split | micro-val (define before first run) |
| Classes | all 32 |
| Class balancing / sampler | repeat-factor (t=0.001) \| none |
| PU strategy | none \| bg-weight sweep (α=0.3) \| Soft Teacher |
| Pseudo-labels from | — \| I-NNN (conf ≥ 0.60, N boxes) |
| Augmentation | mosaic=1.0, fliplr=0.5, copy-paste=0.3, ... |

#### Hyperparameters
| Field | Value |
|---|---|
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | AdamW |
| LR initial | 1e-3 |
| LR scheduler | cosine, warm-up 3 epochs |
| Weight decay | 5e-4 |
| Early stopping | patience=20 |
| num_workers | 8 |
| Loss config | box=7.5, cls=0.5, dfl=1.5 (YOLO defaults) |

#### Evaluation Config (training-time val only)
| Field | Value |
|---|---|
| Evaluator | pycocotools mAP@[.5:.95] |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.7 |

#### Results
| Metric | Value |
|---|---|
| μval mAP@[.5:.95] | — |
| μval mAP@.50 | — |
| Best epoch | — |
| Final train loss | — |

#### Notes
Free-text observations.
-->

---

## Training Conventions

- `T-NNN` IDs are zero-padded, unique, and never reused.
- Record output paths explicitly so the training run can be reconstructed from the entry alone.
- Every new tracked run must have a durable bundle under `artifacts/training-runs/T-NNN/`.
- `Repo commit`, `Config source`, `Resolved config`, `Stdout log`, `Metrics artifact`, and `Selected checkpoint` are mandatory for new entries.
- If a tracked experiment changes the model config, isolate that config change in its own commit before the run starts.
- `Seed`, `Env`, and `Inputs` are mandatory.
- Log entry must be committed to git when the training run completes.
- Weights are not committed by default. Commit via Git LFS only after validation confirms a checkpoint is worth keeping.
