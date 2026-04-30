# FathomNet-CLEF 2026 — Training Log

Metric: **mAP@[0.50:0.95]** (COCO standard, 10 IoU thresholds). Higher is better.
Micro-val: TBD — 200-image hand-audited stratified split (to be defined before first training run).

> **Format note:** This format will be adapted as the challenge progresses. If a new training
> dependency pattern requires changes, update the format and note it here with a date.
> Backfill of historical entries is not required — note the effective date of each format change.
> Last updated: 2026-04-28

---

## Training Index

| ID | Date | Track | Description | Seed | μval mAP |
|----|------|-------|-------------|------|----------|
| T-001 | 2026-04-28 | YOLO26l | YOLO26l-local-v1: 50 epochs, rect 1152px, batch≈4 (auto), RTX 3050 | 42 | 0.408 (train-val 90/10) |
| T-002 | 2026-04-28 | YOLOv11 | YOLOv11-kaggle-v15: 50 epochs, imgsz=640, rect disabled by DDP, batch=16/GPU, Kaggle T4x2 | 42 | 0.377 (train-val 90/10) |
| T-003 | 2026-04-28 | Co-DETR | Co-DETR-R50-kaggle-v15: 12 epochs, torchrun T4x2, AMP, best val bbox mAP 0.326 | 42 | 0.326 (train-val 90/10) |

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
| Artifacts | outputs/yolo26l-local-v1/run/best.pt, outputs/yolo26l-local-v1/run/last.pt, outputs/yolo26l-local-v1/run/results.csv |

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
| Artifacts | artifacts/checkpoints/T-002-yolov11-kaggle/best.pt, artifacts/submissions/I-001-yolov11-kaggle-nonsahi/submission.csv |

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
| Artifacts | artifacts/checkpoints/T-003-codetr-kaggle/best_coco_bbox_mAP_epoch_12.pth, artifacts/checkpoints/T-003-codetr-kaggle/epoch_12.pth, artifacts/checkpoints/T-003-codetr-kaggle/config_used.py, artifacts/checkpoints/T-003-codetr-kaggle/idx_to_cat_id.json |

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
| Artifacts | outputs/T-001/best.pt, outputs/T-001/last.pt, outputs/T-001/config.yaml |

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
- `Seed`, `Env`, and `Inputs` are mandatory.
- Log entry must be committed to git when the training run completes.
- Weights are not committed by default. Commit via Git LFS only after validation confirms a checkpoint is worth keeping.
