# FathomNet-CLEF 2026 — Experiment Log

Metric: **mAP@[0.50:0.95]** (COCO standard, 10 IoU thresholds). Higher is better.  
Micro-val: TBD — 200-image hand-audited stratified split (to be defined before first training run).

> **Format note:** This format will be adapted as the challenge progresses. If a new experiment
> type or dependency pattern requires changes, update the format and note it here with a date.
> Last updated: 2026-04-27

---

## Training Index

| ID | Date | Model | Description | Seed | μval mAP | Status |
|----|------|-------|-------------|------|----------|--------|
| — | — | — | — | — | — | — |

## Inference Index

| ID | Date | Inputs | Description | μval mAP | LB mAP | Sub ID | Status |
|----|------|--------|-------------|----------|--------|--------|--------|
| — | — | — | — | — | — | — | — |

---

## Training Entry Template

<!--
### T-001 — <short description>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Status | planned \| running \| completed \| failed \| abandoned |
| Platform | TensorDock (GPU type) \| Kaggle (GPU type) \| local RTX 3050 |
| Duration | Xh Ym |
| Code version | git sha (short) |
| Data version | dataset manifest SHA or note |
| Seed | 42 |
| Inputs | — (fresh start) \| T-NNN checkpoint, I-NNN pseudo-labels, etc. |
| Artifacts | outputs/T-001/best.pt, outputs/T-001/last.pt, outputs/T-001/config.yaml |
| Failure reason | (if abandoned/failed — required) |

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

#### Evaluation Config
| Field | Value |
|---|---|
| Evaluator | pycocotools mAP@[.5:.95] |
| Conf threshold (val) | 0.001 |
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
| Status | planned \| running \| completed \| failed \| abandoned |
| Platform | local RTX 3050 \| TensorDock \| Kaggle |
| Duration | Xh Ym |
| Code version | git sha (short) |
| Evaluated split | micro-val \| full train (sanity) \| test set |
| Inputs | T-001 checkpoint, T-002 checkpoint, configs/per_class_thresholds.yaml |
| Artifacts | outputs/I-001/predictions_raw.json, outputs/I-001/submission.csv |
| Lineage | T-001 → I-001 (baseline) |
| Failure reason | (if abandoned/failed — required) |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-001 | outputs/T-001/best.pt |

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
| μval mAP@[.5:.95] | — |
| μval mAP@.50 | — |

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

- **Inputs field**: list all upstream dependencies explicitly — checkpoint paths, pseudo-label sources, threshold files. This is the primary dependency record. No retroactive child links needed.
- **Lineage field**: one-line compact chain for complex flows, e.g.:
  `T-001 → I-001 (pseudo-labels) → T-002 → I-002 + I-003 → I-004 (WBF ensemble)`
- **IDs**: `T-NNN` for training, `I-NNN` for inference, zero-padded to 3 digits. Never reused, even if abandoned.
- **Status values**: `planned | running | completed | failed | abandoned`
  - `failed` = crashed or produced invalid results
  - `abandoned` = intentionally stopped; `Failure reason` field required
- **Artifacts**: record output paths explicitly so any experiment is reproducible from its entry alone.
- **Reproducibility**: `Code version` (git sha) + `Data version` + `Seed` are mandatory for completed training runs.
