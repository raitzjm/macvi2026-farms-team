# FathomNet-CLEF 2026 — Experiment Log

Metric: **mAP@[0.50:0.95]** (COCO standard, 10 IoU thresholds). Higher is better.  
Micro-val: TBD — 200-image hand-audited stratified split (to be defined before first training run).

> **Format note:** This format will be adapted as the challenge progresses. If a new experiment
> type or dependency pattern requires changes, update the format and note it here with a date.
> Backfill of historical entries is not required — note the effective date of each format change.
> Last updated: 2026-04-27

---

## Training Index

| ID | Date | Track | Description | Seed | μval mAP |
|----|------|-------|-------------|------|----------|
| T-001 | 2026-04-28 | YOLO26l | YOLO26l-local-v1: 50 epochs, rect 1152px, batch≈4 (auto), RTX 3050 | 42 | 0.408 (train-val 90/10) |
| T-002 | 2026-04-28 | Co-DETR | Co-DETR R50 v14: 12 epochs, 1333×800 rect, Kaggle T4, no inference (bug) | 42 | 0.324 (train-val 90/10) |

## Inference Index

| ID | Date | Track | Inputs | Description | μval mAP | LB mAP | Sub ID |
|----|------|-------|--------|-------------|----------|--------|--------|
| — | — | — | — | — | — | — | — |

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
Strong result vs YOLO11l Kaggle baseline (0.372 mAP50-95). WandB run logged to `little_farms/fathomnet2026` as `yolov26-baseline` (moved from wrong project post-run — name was manually corrected in WandB UI).

---

### T-002 — Co-DETR R50 v14 (12 epochs, Kaggle T4)

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | Co-DETR |
| Env | mmdet (envs/mmdet/) |
| Platform | Kaggle T4 |
| Duration | ~3.8 h (training), inference aborted (bug) |
| Seed | 42 |
| Inputs | co_dino_5scale_r50_lsj_8xb2_1x_coco.pth (COCO pretrained) |
| Artifacts | none recoverable (kernel v14 output download blocked while v15 running) |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | Co-DETR R50 |
| Pretrained weights | co_dino_5scale_r50_lsj_8xb2_1x_coco.pth |
| Input resolution | 1333×800 (mmdet rect) |
| Config file | kaggle/codetr-baseline/kernel_train.py (inline config) |
| Modifications | AMP enabled; custom dataset (32 classes) |

#### Dataset Config
| Field | Value |
|---|---|
| Train split | 90% of 6,463 images |
| Val split | 10% of 6,463 images |
| Classes | all 32 |
| PU strategy | none |

#### Hyperparameters
| Field | Value |
|---|---|
| Epochs | 12 |
| Batch size | 2 |
| Optimizer | AdamW |
| LR initial | 1e-4 |

#### Results
| Metric | Value |
|---|---|
| val mAP@[.5:.95] | **0.324** |
| val mAP@.50 | — |
| Best epoch | 12 |
| Final train loss | — |

#### Notes
Training completed 12 epochs. Inference crashed: `mmdetection-src` was deleted (cleanup) before `init_detector()` — sub-module files needed at model construction time even after `sys.modules` cache. Fixed in v15 (cleanup moved to after submission.csv is written). Checkpoint zip not recovered — v15 was running when download attempted (403 Forbidden from Kaggle API).

---

## Training Entry Template

<!--
### T-001 — <short description>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Track | YOLO11l \| Co-DETR |
| Env | ultralytics \| mmdet |
| Platform | TensorDock (GPU type) \| Kaggle (GPU type) \| local RTX 3050 |
| Duration | Xh Ym |
| Seed | 42 |
| Inputs | — (fresh start) \| T-NNN checkpoint, I-NNN pseudo-labels, etc. |
| Artifacts | outputs/T-001/best.pt, outputs/T-001/last.pt, outputs/T-001/config.yaml |

#### Model & Architecture
| Field | Value |
|---|---|
| Architecture | YOLO11l \| Co-DETR-5scale-SwinL |
| Pretrained weights | yolo11l.pt (COCO) \| co_dino_5scale_swin_l_16e_o365tococo.pth |
| Input resolution | 1280×1280 |
| Config file | configs/yolo11l_baseline.yaml |
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

## Inference Entry Template

<!--
### I-001 — <short description>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Track | YOLO11l \| Co-DETR \| Ensemble |
| Env | ultralytics \| mmdet |
| Platform | local RTX 3050 \| TensorDock \| Kaggle |
| Duration | Xh Ym |
| Evaluated split | micro-val \| test set |
| Inputs | T-001 checkpoint, T-002 checkpoint, configs/per_class_thresholds.yaml |
| Artifacts | outputs/I-001/predictions_raw.json, outputs/I-001/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-001 | outputs/T-001/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 1280×1280 \| multi-scale [800, 1333] |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.7 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | yes \| no |
| Tile size | 640×640 |
| Overlap ratio | 0.20 |
| Full-image pass | yes \| no |
| Postprocess | NMM \| NMS \| WBF |
| Postprocess thresh | 0.50 |

#### TTA
| Field | Value |
|---|---|
| Enabled | yes \| no |
| Transforms | fliplr, scale [0.83, 1.0, 1.17] |

#### Ensemble
| Field | Value |
|---|---|
| Enabled | yes \| no |
| Method | WBF |
| Sources | T-001+SAHI predictions, T-002+SAHI predictions |
| WBF IoU thresh | 0.55 |
| WBF skip thresh | 0.001 |
| Fold coverage | all folds \| fold-0, fold-2 |

#### Per-Class Thresholds
| Field | Value |
|---|---|
| Strategy | uniform (conf=0.25) \| F1-optimal per class on micro-val |
| Tuned on | micro-val split (no leakage) |
| Threshold file | outputs/I-001/per_class_thresholds.json |

#### Results
| Metric | Value |
|---|---|
| μval mAP@[.5:.95] | — (blank if evaluated on test set) |
| μval mAP@.50 | — (blank if evaluated on test set) |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes \| no |
| Submission ID | — |
| Date | YYYY-MM-DD |
| Public LB mAP | — |
| LB rank | — |
| Notes | — |

#### Notes
Free-text observations.
-->

---

## Dependency / Lineage Conventions

- **Inputs field**: list all upstream dependencies explicitly — checkpoint paths, pseudo-label sources, threshold files. This is the primary dependency record.
- **Lineage column** (Inference Index only): optional one-line compact chain for multi-hop flows, e.g.:
  `T-001 → I-001 (pseudo-labels) → T-002 → I-002 + I-003 → I-004 (WBF)`
- **IDs**: `T-NNN` for training, `I-NNN` for inference, zero-padded to 3 digits. Never reused.
- **Artifacts**: record output paths explicitly so any experiment is reproducible from its entry alone.
- **Reproducibility**: `Seed` + `Env` + `Inputs` are mandatory for every training entry.
- **Commit rule**: log entry must be committed to git when the experiment completes.
- **Weights**: not committed by default. Commit via Git LFS only after LB validation confirms a checkpoint is worth keeping.
