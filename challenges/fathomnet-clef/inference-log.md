# FathomNet-CLEF 2026 — Inference Log

Metric: **mAP@[0.50:0.95]** (COCO standard, 10 IoU thresholds). Higher is better.
Micro-val: TBD — 200-image hand-audited stratified split (to be defined before first inference evaluation).

> **Format note:** This format will be adapted as the challenge progresses. If a new inference,
> submission, or artifact-tracking pattern requires changes, update the format and note it here with a date.
> Backfill of historical entries is not required — note the effective date of each format change.
> Last updated: 2026-04-30

> **2026-04-30 artifact policy:** every new tracked inference run must have a tracked bundle at
> `artifacts/inference-runs/I-NNN/` with `metadata.json`, the resolved inference config snapshot,
> and a raw `stdout.log` whenever the launcher can capture it. Submission CSVs under
> `artifacts/submissions/` are local payloads by default under the current ignore rules, so tracked
> bundles and log entries must record enough metadata to identify the submitted file even when the
> CSV itself is not versioned.

---

## Inference Index

| ID | Date | Track | Inputs | Description | μval mAP | LB mAP | Sub ID |
|----|------|-------|--------|-------------|----------|--------|--------|
| I-001 | 2026-04-28 | YOLOv11 | T-002 | Kaggle kernel non-SAHI inference on test set | — | 0.0348 | 52122957 |
| I-002 | 2026-04-28 | YOLO26l | T-001 | Local non-SAHI inference on test set, corrected resubmission | — | 0.0340 | 52142348 |
| I-003 | 2026-04-28 | YOLO26l | T-001 | Local SAHI inference on test set | — | 0.0204 | 52141745 |
| I-004 | 2026-04-28 | YOLOv11 | T-002 | Local non-SAHI inference on test set; differs from Kaggle CSV | — | 0.0377 | 52141814 |
| I-005 | 2026-04-28 | YOLOv11 | T-002 | Local SAHI inference on test set | — | 0.0166 | 52142207 |
| I-006 | 2026-04-28 | Co-DETR | T-003 | Kaggle kernel non-SAHI inference on test set | — | 0.0318 | 52148404 |
| I-007 | 2026-04-28 | Co-DETR | T-003 | Local non-SAHI inference on test set; differs from Kaggle CSV | — | 0.0333 | 52149263 |
| I-008 | 2026-04-28 | Co-DETR | T-003 | Local SAHI inference on test set | — | 0.0053 | 52149270 |
| I-009 | 2026-05-01 | Co-DETR | T-004 | Local non-SAHI inference on test set; corrected baseline (official pretrain + shared split) | — | 0.0556 | see notes |
| I-010 | 2026-05-01 | Co-DETR | T-004 | Local non-SAHI, epoch_11.pth — T-005 PU checkpoint test | — | 0.0485 | see notes |
| I-011 | 2026-05-01 | Co-DETR | T-004 | Local non-SAHI, epoch_12.pth — T-005 PU checkpoint test | — | 0.0470 | see notes |
| I-012 | 2026-05-02 | Co-DETR | T-006 | Local non-SAHI, best_coco_bbox_mAP_epoch_5.pth — primary T-006 submission | — | 0.0535 | see notes |
| I-013 | 2026-05-02 | Co-DETR | T-006 | Local non-SAHI, epoch_11.pth — T-006 late-epoch hedge | — | 0.0508 | see notes |
| I-014 | 2026-05-03 | YOLO | T-007 best.pt | T-007 yolo26s best.pt (epoch~46) local non-SAHI | — | 0.0346 | see notes |
| I-015 | 2026-05-03 | YOLO | T-007 last.pt | T-007 yolo26s last.pt (epoch 50) local non-SAHI | — | 0.0329 | see notes |
| I-016 | 2026-05-03 | YOLO | T-007 best.pt | T-007 yolo26s best.pt SAHI (slice=640, conf=0.0001) | — | 0.0127 | see notes |
| I-017 | 2026-05-03 | Ensemble | T-004ep7 + T-007best | WBF ensemble Co-DETR(w=2) + YOLO(w=1), iou=0.55 | — | 0.0537 | see notes |
| I-019 | 2026-05-04 | Co-DETR | T-009 ep7 best | T-009 Co-DETR R50+PL ep7 best local non-SAHI | — | 0.0572 | see notes |
| I-020 | 2026-05-04 | Co-DETR | T-009 ep12 last | T-009 Co-DETR R50+PL ep12 last local non-SAHI | — | 0.0576 | see notes |
| I-021 | 2026-05-04 | Co-DETR | T-009 ep12 | TTA-hflip on T-009 ep12, WBF iou=0.6 | — | 0.0521 | see notes |
| I-022 | 2026-05-04 | Ensemble | T-009 ep7+ep12 | WBF(I-019, I-020) equal weights, iou=0.55 | — | 0.0523 | see notes |
| I-025 | 2026-05-05 | YOLO | T-012 ep39 best | T-012 YOLO26l best.pt imgsz=1280 conf=0.001 iou=0.6 local RTX3050 | — | 0.0459 | see notes |
| I-026 | 2026-05-05 | Ensemble | T-009ep12 + T-012best | WBF T-009ep12(w=2) + T-012best(w=1), iou_thr=0.55 | — | 0.0518 | see notes |
| I-027 | 2026-05-05 | Ensemble | T-009ep12 + T-012best | WBF T-009ep12(w=1) + T-012best(w=1), iou_thr=0.55 | — | 0.0546 | see notes |
| I-028 | 2026-05-05 | YOLO | T-011 best.pt | T-011 YOLO26s best.pt imgsz=1280 conf=0.001 iou=0.6 local RTX3050 | — | 0.0449 | see notes |
| I-029 | 2026-05-05 | YOLO | T-011 epoch30.pt | T-011 YOLO26s epoch30.pt imgsz=1280 conf=0.001 iou=0.6 local RTX3050 | — | 0.0467 | see notes |
| I-030 | 2026-05-05 | YOLO | T-011 best.pt | T-011 YOLO26s best.pt imgsz=1024 conf=0.001 iou=0.6 local RTX3050 | — | 0.0497 | see notes |
| I-031 | 2026-05-05 | Co-DETR | T-010 ep12 last | T-010 Co-DETR R50+2ndPL ep12 last local non-SAHI | — | 0.0503 | 52368031 |
| I-032 | 2026-05-05 | Co-DETR | T-010 ep10 best | T-010 Co-DETR R50+2ndPL ep10 (best val mAP) local non-SAHI | — | 0.0404 | 52368142 |
| I-033 | 2026-05-05 | Co-DETR | T-010 ep8 | T-010 Co-DETR R50+2ndPL ep8 local non-SAHI | — | 0.0384 | 52368235 |
| I-049 | 2026-05-07 | Ensemble | T-014 fold plateau set | 5-fold YOLO26s teacher ensemble, epoch35/40/45 two-stage WBF | — | 0.0582 | see notes |
| I-050 | 2026-05-08 | YOLO | T-019 epoch45 | T-019 Kaggle final retrain epoch45 single checkpoint, local non-SAHI | — | 0.0438 | 52441654 |
| I-051 | 2026-05-08 | YOLO | T-019 best.pt | T-019 Kaggle final retrain best.pt single checkpoint, local non-SAHI | — | 0.0450 | 52441691 |
| I-052 | 2026-05-08 | YOLO | T-019 last.pt | T-019 Kaggle final retrain last.pt single checkpoint, local non-SAHI | — | 0.0461 | 52441732 |
| I-053 | 2026-05-08 | YOLO | T-019 epoch40 | T-019 Kaggle final retrain epoch40 single checkpoint, local non-SAHI | — | 0.0444 | 52441772 |
| I-058 | 2026-05-08 | Ensemble | T-014 fold epoch45 set | 5-fold YOLO teacher epoch45-only equal-weight WBF | — | 0.0545 | 52441209 |
| I-059 | 2026-05-08 | Ensemble | T-014 fold epoch40+45 set | 2-stage WBF using epoch40+45 within each fold | — | 0.0571 | 52441208 |
| I-060 | 2026-05-08 | Ensemble | T-014 fold epoch40 set | 5-fold YOLO teacher epoch40-only equal-weight WBF | — | 0.0537 | 52441323 |
| I-061 | 2026-05-08 | Ensemble | T-014 fold epoch35 set | 5-fold YOLO teacher epoch35-only equal-weight WBF | — | 0.0589 | 52441327 |
| I-062 | 2026-05-08 | Ensemble | T-014 fold epoch35+40 set | 2-stage WBF using epoch35+40 within each fold | — | 0.0577 | 52441368 |
| I-063 | 2026-05-08 | Ensemble | T-009 ep12 + I-061 | Equal-weight WBF Co-DETR T-009 ep12 + teacher epoch35 5-fold | — | 0.0562 | 52441824 |

---

### I-001 — YOLOv11 Kaggle non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLOv11 |
| Env | ultralytics |
| Platform | Kaggle T4 x2 |
| Evaluated split | test set |
| Inputs | T-002 checkpoint + Kaggle kernel inference path |
| Artifacts | artifacts/inference-runs/I-001/, artifacts/submissions/I-001-yolov11-kaggle-nonsahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-002 | artifacts/checkpoints/T-002-yolov11-kaggle/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=640 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52122957 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0348 |
| Notes | Generated by Kaggle kernel and manually submitted from downloaded outputs |

#### Notes
Baseline official submission from the successful Kaggle YOLOv11 kernel run.

---

### I-002 — YOLO26l local non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLO26l |
| Env | ultralytics |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-001 checkpoint |
| Artifacts | artifacts/inference-runs/I-002/, artifacts/submissions/I-002-yolo26l-local-nonsahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-001 | artifacts/checkpoints/T-001-yolo26l-local/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1152 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52142348 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0340 |
| Notes | Corrected resubmission after filtering zero-area boxes |

#### Notes
Initial submission `52140950` errored with `bbox_height must be positive` due to 5 zero-height boxes. Exporter was fixed to drop non-positive-area boxes, the CSV was regenerated (82,561 predictions; 82,562 CSV lines including header), and the corrected resubmission scored 0.0340.

---

### I-003 — YOLO26l local SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLO26l |
| Env | ultralytics + SAHI |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-001 checkpoint |
| Artifacts | artifacts/inference-runs/I-003/, artifacts/submissions/I-003-yolo26l-local-sahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-001 | artifacts/checkpoints/T-001-yolo26l-local/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1152 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | yes |
| Tile size | 640x640 |
| Overlap ratio | 0.20 |
| Postprocess thresh | 0.50 |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52141745 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0204 |
| Notes | 237,201 predictions; substantially worse than non-SAHI |

#### Notes
SAHI materially increased box count but underperformed the direct non-SAHI pass on the public leaderboard.

---

### I-004 — YOLOv11 local non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLOv11 |
| Env | ultralytics |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-002 checkpoint + archived idx_to_cat_id map |
| Artifacts | artifacts/inference-runs/I-004/, artifacts/submissions/I-004-yolov11-local-nonsahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-002 | artifacts/checkpoints/T-002-yolov11-kaggle/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=640 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52141814 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0377 |
| Notes | Local PNG-based inference differed from archived Kaggle CSV |

#### Notes
Local non-SAHI output was not identical to the Kaggle-generated CSV: 61,110 predictions (61,111 CSV lines including header) locally versus 68,794 predictions in the archived Kaggle CSV. Because it differed, it was submitted separately and scored better than the Kaggle non-SAHI baseline.

---

### I-005 — YOLOv11 local SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | YOLOv11 |
| Env | ultralytics + SAHI |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-002 checkpoint + archived idx_to_cat_id map |
| Artifacts | artifacts/inference-runs/I-005/, artifacts/submissions/I-005-yolov11-local-sahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-002 | artifacts/checkpoints/T-002-yolov11-kaggle/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=640 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | yes |
| Tile size | 640x640 |
| Overlap ratio | 0.20 |
| Postprocess thresh | 0.50 |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52142207 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0166 |
| Notes | 223,671 predictions; much worse than non-SAHI |

#### Notes
SAHI substantially increased prediction count but hurt public leaderboard performance relative to both local and Kaggle non-SAHI YOLOv11 runs.

---

### I-006 — Co-DETR Kaggle non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | Co-DETR |
| Env | mmdet |
| Platform | Kaggle T4 x2 |
| Evaluated split | test set |
| Inputs | T-003 checkpoint + Kaggle kernel inference path |
| Artifacts | artifacts/inference-runs/I-006/, artifacts/submissions/I-006-codetr-kaggle-nonsahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-003 | artifacts/checkpoints/T-003-codetr-kaggle/best_coco_bbox_mAP_epoch_12.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | config_used.py test pipeline Resize(1333,800), keep_ratio=True |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52148404 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0318 |
| Notes | Downloaded Kaggle `submission.csv`; 360,718 predictions (360,719 CSV lines incl. header) |

#### Notes
The completed Kaggle kernel wrote `submission.csv` and the best epoch-12 checkpoint. This serves as the reference inference output for the local Co-DETR reproduction passes.

---

### I-007 — Co-DETR local non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | Co-DETR |
| Env | mmdet |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-003 checkpoint + downloaded `config_used.py` + downloaded `idx_to_cat_id.json` |
| Artifacts | artifacts/inference-runs/I-007/, artifacts/submissions/I-007-codetr-local-nonsahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-003 | artifacts/checkpoints/T-003-codetr-kaggle/best_coco_bbox_mAP_epoch_12.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | config_used.py test pipeline Resize(1333,800), keep_ratio=True |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52149263 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0333 |
| Notes | Local CSV differed from Kaggle output; 360,627 predictions (360,628 CSV lines incl. header) |

#### Notes
Local non-SAHI inference was not identical to the Kaggle kernel output and slightly outperformed it on the public leaderboard.

---

### I-008 — Co-DETR local SAHI inference

| Field | Value |
|---|---|
| Date | 2026-04-28 |
| Track | Co-DETR |
| Env | mmdet + SAHI |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-003 checkpoint + downloaded `config_used.py` + downloaded `idx_to_cat_id.json` |
| Artifacts | artifacts/inference-runs/I-008/, artifacts/submissions/I-008-codetr-local-sahi/submission.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-003 | artifacts/checkpoints/T-003-codetr-kaggle/best_coco_bbox_mAP_epoch_12.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | config_used.py via SAHI MMDet adapter |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | SAHI postprocess match threshold 0.50 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | yes |
| Tile size | 640x640 |
| Overlap ratio | 0.20 |
| Postprocess thresh | 0.50 |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | 52149270 |
| Date | 2026-04-28 |
| Public LB mAP | 0.0053 |
| Notes | 608,595 predictions (608,596 CSV lines incl. header); severe LB collapse |

#### Notes
The tiled Co-DETR pass dramatically increased prediction count and performed much worse than both Kaggle and local non-SAHI inference.

---

### I-009 — Co-DETR T-004 local non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-05-01 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | TensorDock (fathom1, 1× RTX 4090) |
| Evaluated split | test set |
| Inputs | T-004 checkpoint, best epoch 7 |
| Repo commit | 0c3fe90042e4af0c912c3297677e6a5f7019dfb4 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-009/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-009/stdout.log |
| Artifacts | artifacts/inference-runs/I-009/, artifacts/submissions/I-009-from-T-004-codetr-local-nonsahi/submission-I-009-from-T-004.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-004 | artifacts/checkpoints/T-004-codetr-tensordock/best.pth (epoch 7, val mAP=0.417) |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |
| (slice_size) | 640 (not active) |
| (overlap) | 0.20 (not active) |
| (postprocess_match_threshold) | 0.50 (not active) |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-01 18:24 UTC) |
| Date | 2026-05-01 |
| Public LB mAP | **0.0556** |
| Notes | +0.0179 vs I-004 anchor (0.0377); new best LB score; submitted via Kaggle CLI from local container |

#### Notes
First submission from T-004 corrected Co-DETR baseline (official pretrain + shared split manifest). Public LB score 0.0556 is +0.0179 above the previous best I-004 (0.0377 mAP), confirming that the corrected training setup translates to substantially better test-set performance. Val mAP improvement (+0.009 vs T-001 YOLO26l) closely mirrors LB improvement, supporting trust in the shared-split as a reliable proxy metric.

---

### I-010 — Co-DETR T-004 epoch11 local non-SAHI inference (T-005 PU checkpoint test)

| Field | Value |
|---|---|
| Date | 2026-05-01 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | TensorDock (fathom1, 1× RTX 4090) |
| Evaluated split | test set |
| Inputs | T-004 checkpoint, epoch 11 |
| Repo commit | 0c3fe90042e4af0c912c3297677e6a5f7019dfb4 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-010/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-010/stdout.log |
| Artifacts | artifacts/inference-runs/I-010/, artifacts/submissions/I-010-from-T-004ep11-codetr-local-nonsahi/submission-I-010-from-T-004ep11.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-004 | outputs/codetr-r50-bounded-fairness-T-004/mmdet_run/epoch_11.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-01 18:43 UTC) |
| Date | 2026-05-01 |
| Public LB mAP | **0.0485** |
| Notes | T-005 PU checkpoint test; epoch 11 scores below epoch 7 (0.0556), supporting PU selection of best_val epoch |

#### Notes
Part of T-005 PU checkpoint selection experiment. Epoch 11 on LB (0.0485) is -0.0071 vs epoch 7 (I-009, 0.0556), confirming that later training epochs do not improve test-set performance. Val mAP at epoch 11 was higher than epoch 7, which is consistent with the PU hypothesis: labeled positives in the train split are a biased subset, and overfitting to them degrades detection on the test set.

---

### I-011 — Co-DETR T-004 epoch12 local non-SAHI inference (T-005 PU checkpoint test)

| Field | Value |
|---|---|
| Date | 2026-05-01 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | TensorDock (fathom1, 1× RTX 4090) |
| Evaluated split | test set |
| Inputs | T-004 checkpoint, epoch 12 |
| Repo commit | 0c3fe90042e4af0c912c3297677e6a5f7019dfb4 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-011/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-011/stdout.log |
| Artifacts | artifacts/inference-runs/I-011/, artifacts/submissions/I-011-from-T-004ep12-codetr-local-nonsahi/submission-I-011-from-T-004ep12.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-004 | outputs/codetr-r50-bounded-fairness-T-004/mmdet_run/epoch_12.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-01 18:43 UTC) |
| Date | 2026-05-01 |
| Public LB mAP | **0.0470** |
| Notes | T-005 PU checkpoint test; epoch 12 scores below epoch 11 (0.0485) and well below epoch 7 (0.0556) |

#### Notes
Part of T-005 PU checkpoint selection experiment. Epoch 12 on LB (0.0470) is -0.0086 vs epoch 7 (I-009, 0.0556) and -0.0015 vs epoch 11 (I-010, 0.0485). The monotonic decline across epochs 7→11→12 strongly confirms the PU hypothesis: epoch 7 is the better checkpoint for test-set generalization despite epochs 11/12 having higher val mAP on the shared split.

---

### I-012 — Co-DETR T-006 epoch5 local non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-05-02 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | TensorDock (fathom1, 1× RTX 4090) |
| Evaluated split | test set |
| Inputs | T-006 checkpoint, best epoch 5 |
| Repo commit | a8c91d55e4fe7559c2f76bba029ef99b88fffd99 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-012/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-012/stdout.log |
| Artifacts | artifacts/inference-runs/I-012/, artifacts/submissions/I-012-from-T-006ep5-codetr-local-nonsahi/submission-I-012-from-T-006ep5.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-006 | artifacts/checkpoints/T-006-codetr-tensordock/best.pth (epoch 5, val mAP=0.427) |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-02 01:36 UTC) |
| Date | 2026-05-02 |
| Public LB mAP | **0.0535** |
| Notes | 401,611 predictions (401,612 CSV lines incl. header); primary T-006 submission; below I-009 best baseline (0.0556) by 0.0021 |

#### Notes
Primary submission for T-006 after the lead data scientist review favored the early best checkpoint. Epoch 5 beat the later T-006 hedge (I-013, 0.0508) by 0.0027 on the public leaderboard, which is consistent with the T-004/T-005 PU pattern: later training epochs do not reliably improve held-out test performance even when they remain competitive on validation.

---

### I-013 — Co-DETR T-006 epoch11 local non-SAHI inference

| Field | Value |
|---|---|
| Date | 2026-05-02 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | TensorDock (fathom1, 1× RTX 4090) |
| Evaluated split | test set |
| Inputs | T-006 checkpoint, epoch 11 |
| Repo commit | a8c91d55e4fe7559c2f76bba029ef99b88fffd99 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-013/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-013/stdout.log |
| Artifacts | artifacts/inference-runs/I-013/, artifacts/submissions/I-013-from-T-006ep11-codetr-local-nonsahi/submission-I-013-from-T-006ep11.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-006 | outputs/codetr-r50-extended-T-006/mmdet_run/epoch_11.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-02 01:39 UTC) |
| Date | 2026-05-02 |
| Public LB mAP | **0.0508** |
| Notes | 385,807 predictions (385,808 CSV lines incl. header); later-epoch hedge scored below epoch 5 (0.0535) by 0.0027 |

#### Notes
The only late T-006 checkpoint worth testing after epoch 5 was epoch 11 because its val mAP (0.426) nearly matched the best checkpoint and its val mAP@.50 was slightly higher. Kaggle rejected that hedge: epoch 11 underperformed both I-012 (epoch 5) and I-009 (T-004 epoch 7), strengthening the working PU hypothesis that later epochs overfit the labeled-positive subset rather than improving test-set generalization.

---

### I-014 — T-007 yolo26s best.pt local non-SAHI

| Field | Value |
|---|---|
| Date | 2026-05-03 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-007 best.pt (Ultralytics fitness best, epoch ~46) |
| Repo commit | e3f28322 |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-014/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-014/stdout.log |
| Artifacts | artifacts/inference-runs/I-014/, artifacts/submissions/I-014-from-T-007-yolo-local-nonsahi/submission-I-014-from-T-007.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-007 | artifacts/checkpoints/T-007-yolo26s-local/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1152 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |
| Batch | 8 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-03 05:36 UTC) |
| Date | 2026-05-03 |
| Public LB mAP | **0.0346** |
| Notes | 31,604 predictions; best.pt beats last.pt (I-015: 0.0329) by +0.0017, confirming the PU late-epoch val_cls drift does hurt LB, though the effect is milder than in Co-DETR |

#### Notes
T-007 yolo26s baseline (0.0346) vs T-001 yolo26l (0.0340): comparable despite smaller backbone, because T-007 uses the correct shared split and PU-aware augmentation. Still well below the Co-DETR anchor T-004/I-009 (0.0556), indicating the Co-DETR architecture advantage on this dataset is substantial.

---

### I-015 — T-007 yolo26s last.pt local non-SAHI

| Field | Value |
|---|---|
| Date | 2026-05-03 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-007 last.pt (final epoch 50) |
| Repo commit | e3f28322 |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-015/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-015/stdout.log |
| Artifacts | artifacts/inference-runs/I-015/, artifacts/submissions/I-015-from-T-007-yolo-local-nonsahi/submission-I-015-from-T-007.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-007 | artifacts/checkpoints/T-007-yolo26s-local/last.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1152 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |
| Batch | 8 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-03 05:36 UTC) |
| Date | 2026-05-03 |
| Public LB mAP | **0.0329** |
| Notes | 29,669 predictions; last.pt (epoch 50) underperforms best.pt (epoch 46) by -0.0017, confirming PU late-epoch drift in YOLO (mild but present) |

---

### I-016 — T-007 yolo26s best.pt local SAHI

| Field | Value |
|---|---|
| Date | 2026-05-03 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-007 best.pt (epoch ~46) with SAHI tiling |
| Repo commit | e3f28322 |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-016/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-016/stdout.log |
| Artifacts | artifacts/inference-runs/I-016/, artifacts/submissions/I-016-from-T-007-yolo-local-sahi/submission-I-016-from-T-007.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-007 | artifacts/checkpoints/T-007-yolo26s-local/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1152 |
| Conf threshold | 0.0001 (lowered for SAHI) |
| IoU threshold (NMS) | 0.6 |
| Batch | 8 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | yes |
| Slice size | 640 |
| Overlap | 0.2 |
| Postprocess match threshold | 0.5 (default) |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-03 05:36 UTC) |
| Date | 2026-05-03 |
| Public LB mAP | **0.0127** |
| Notes | 448,762 predictions; SAHI dramatically over-detected — ultra-low conf (0.0001) combined with slice tiling flooded the output with low-quality boxes, crushing precision. Compare I-003/I-005/I-008 — SAHI has consistently underperformed non-SAHI across all model families on this dataset |

#### Notes
SAHI on FathomNet appears consistently harmful (I-003=0.0204, I-005=0.0166, I-008=0.0053, I-016=0.0127) — the tiling and low-confidence threshold produce far too many false positives in PU data. SAHI should be deprioritized for future submissions unless a substantially different NMS/merge strategy is applied.

---

### I-017 — WBF ensemble T-004ep7 (Co-DETR) + T-007best (YOLO)

| Field | Value |
|---|---|
| Run ID | I-017 |
| Date | 2026-05-03 |
| Track | Ensemble |
| Method | Weighted Box Fusion (wbf_ensemble.py) |
| Repo commit | fa4238ce |
| Config source | scripts/wbf_ensemble.py |
| Resolved config | artifacts/inference-runs/I-017/metadata.json |
| Artifacts | artifacts/inference-runs/I-017/, artifacts/submissions/I-017-ensemble-wbf-t004ep7-t007best/submission-I-017-ensemble-wbf-2-1.csv |

#### Inputs
| Source | Run | Checkpoint | Weight |
|---|---|---|---|
| Co-DETR (T-004) | I-009 | epoch_7.pth | 2.0 |
| YOLO (T-007) | I-014 | best.pt (~epoch 46) | 1.0 |

#### Ensemble Config
| Field | Value |
|---|---|
| Method | WBF (ensemble_boxes.weighted_boxes_fusion) |
| WBF IoU threshold | 0.55 |
| WBF skip threshold | 0.001 |
| Weights | [2.0, 1.0] (Co-DETR:YOLO) |
| Scope | per (image_id, category_id) pair |
| Test JSON | /home/farms/data/datasets/fathomnet-2026/test_dataset.json |

#### Results
| Metric | Value |
|---|---|
| μval mAP@[.5:.95] | — |
| Public LB mAP | **0.0537** |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-03 |
| Public LB mAP | 0.0537 |
| Predictions in CSV | 334,831 |
| Notes | Slightly below I-009 alone (0.0556); YOLO diversity insufficient to lift Co-DETR |

#### Notes
WBF at weights [2,1] Co-DETR:YOLO produced 0.0537, below T-004/I-009 standalone (0.0556) by −0.0019. The YOLO component (I-014=0.0346) has lower absolute precision than Co-DETR and adding it at weight 1 introduced enough noise to drag the ensemble down. A higher Co-DETR weight [3,1] might recover the gap — the Lead noted this as a low-effort hedge worth testing before the deadline if a submission slot is available. The ensemble does confirm that T-004/I-009 remains the leaderboard best.

---

### I-018 — T-008 yolo26s best.pt local non-SAHI

| Field | Value |
|---|---|
| Date | 2026-05-03 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | T-008 best.pt (Ultralytics fitness best, epoch 44) |
| Repo commit | 137740a87881485f574c90c16efde5cd5505db5d |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-018/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-018/stdout.log |
| Artifacts | artifacts/inference-runs/I-018/, artifacts/submissions/I-018-from-T-008/submission-I-018-from-T-008.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-008 | artifacts/checkpoints/T-008-yolo26s-local/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1152 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |
| Batch | 8 |
| Duration | ~2 min |
| Predictions | 37,110 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-03 10:55 UTC) |
| Date | 2026-05-03 |
| Public LB mAP | **0.0421** |
| Notes | T-008 best.pt (ep44, workers=2) vs T-007 best.pt (ep46, workers=8): +0.0075 LB gain despite near-identical val mAP50-95 (0.366 vs 0.365). Likely run-to-run variance amplified by test PU distribution; workers=2 vs 8 may affect augmentation ordering despite seed=42 |

#### Notes
T-008 (LB=0.0421) outperforms T-007 (I-014, LB=0.0346) by +0.0075 despite essentially identical val mAP50-95 (0.366 vs 0.365). The only training differences are workers=2 (T-008) vs workers=8 (T-007) and save_period=5. Different worker counts change data loading concurrency and can produce different augmentation sequences even with the same seed, but the +0.0075 LB gap is large enough to suggest genuine run-to-run variance rather than just the workers change. T-008 is now the best YOLO checkpoint and a better input for ensembling.

---

### I-019 — T-009 Co-DETR R50 + pseudo-labels ep7 best local non-SAHI

> **Note on run ID:** This run was originally labeled I-016 by the inference script (ID conflict with the already-committed T-007 SAHI run I-016). The correct ID is I-019. The Kaggle submission filename retains the `I-016` prefix.

| Field | Value |
|---|---|
| Date | 2026-05-04 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | local (RTX 3050 4GB) |
| Evaluated split | test set |
| Inputs | T-009 checkpoint, best ep7 (best_coco_bbox_mAP_epoch_7.pth, val mAP=0.407) |
| Repo commit | bc586eae31d5e93235dfed28b1e0e96ab794c353 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-019/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-019/stdout.log |
| Artifacts | artifacts/inference-runs/I-019/, artifacts/submissions/I-019-from-T-009-codetr-best-ep7/submission-I-016-from-T-009.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-009 | artifacts/checkpoints/T-009/best_coco_bbox_mAP_epoch_7.pth (val mAP=0.407) |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-04 14:45 UTC, filename submission-I-016-from-T-009.csv) |
| Date | 2026-05-04 |
| Public LB mAP | **0.0572** |
| Notes | +0.0016 vs T-004/I-009 baseline (0.0556); pseudo-labels improve over T-004 baseline; ep7 is the val-best checkpoint |

#### Notes
T-009 ep7 (val mAP=0.407, pseudo-labels enabled) LB 0.0572 vs T-004 ep7 (val mAP=0.417, no PL) LB 0.0556: +0.0016 gain. Pseudo-labels contributed a small but positive signal. Notably, the last epoch (I-020) scored 0.0576, higher than the val-best epoch — contra the PU pattern observed in T-004/T-006. This suggests pseudo-label training may partially recover the late-epoch degradation seen in pure PU training.

---

### I-020 — T-009 Co-DETR R50 + pseudo-labels ep12 last local non-SAHI

> **Note on run ID:** This run was originally labeled I-017 by the inference script (ID conflict with the already-committed WBF ensemble run I-017). The correct ID is I-020. The Kaggle submission filename retains the `I-017` prefix.

| Field | Value |
|---|---|
| Date | 2026-05-04 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | local (RTX 3050 4GB) |
| Evaluated split | test set |
| Inputs | T-009 checkpoint, ep12 last (epoch_12.pth, val mAP=0.404) |
| Repo commit | bc586eae31d5e93235dfed28b1e0e96ab794c353 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-020/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-020/stdout.log |
| Artifacts | artifacts/inference-runs/I-020/, artifacts/submissions/I-020-from-T-009-codetr-last-ep12/submission-I-017-from-T-009.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-009 | artifacts/checkpoints/T-009/epoch_12.pth (val mAP=0.404) |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI (submitted 2026-05-04 14:45 UTC, filename submission-I-017-from-T-009.csv) |
| Date | 2026-05-04 |
| Public LB mAP | **0.0576** |
| Notes | +0.0020 vs T-004/I-009 baseline (0.0556); **new best LB score**; last epoch beats val-best epoch (I-019: 0.0572) by +0.0004 |

#### Notes
T-009 ep12 (val mAP=0.404) LB 0.0576 is the new sprint best, exceeding T-004/I-009 (0.0556) by +0.0020. Remarkably, ep12 last (val mAP=0.404) outscores ep7 val-best (val mAP=0.407) on the public leaderboard, reversing the PU early-stopping pattern seen in T-004 and T-006. This may indicate that pseudo-label training mitigates late-epoch PU label-noise overfitting, allowing later checkpoints to generalize better than they did without PLs.

---

### I-021 — T-009 Co-DETR R50 + pseudo-labels ep12 TTA-hflip

| Field | Value |
|---|---|
| Date | 2026-05-04 |
| Track | Co-DETR |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | local (RTX 3050 4GB) |
| Evaluated split | test set |
| Inputs | T-009 checkpoint, ep12 last (epoch_12.pth, val mAP=0.404) |
| Repo commit | 0c7a7318 |
| Config source | scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-021/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-021/stdout.log |
| Artifacts | artifacts/inference-runs/I-021/, artifacts/submissions/I-021-from-T-009-codetr-tta-ep12/ |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-009 | artifacts/checkpoints/T-009/epoch_12.pth (val mAP=0.404) |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 (matches training resolution via resolved config) |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | detector defaults |

#### TTA Config
| Field | Value |
|---|---|
| Enabled | yes |
| Transforms | horizontal flip |
| Merge method | WBF, weights=[1,1], iou_thr=0.6, skip_box_thr=0.001 |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-04 |
| Public LB mAP | 0.0521 |
| Notes | TTA-hflip hurt vs no-TTA (0.0521 vs 0.0576 for I-020). Suggests Co-DETR predictions are already well-calibrated and the hflip view introduces noise rather than signal at this dataset scale. |

---

### I-022 — WBF ensemble of T-009 ep7 + ep12

| Field | Value |
|---|---|
| Date | 2026-05-04 |
| Track | Ensemble |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | local (CPU) |
| Evaluated split | test set |
| Inputs | I-019 CSV (T-009 ep7, LB=0.0572) + I-020 CSV (T-009 ep12, LB=0.0576) |
| Repo commit | 0c7a7318 |
| Config source | scripts/wbf_ensemble.py |
| Artifacts | artifacts/inference-runs/I-022/, artifacts/submissions/I-022-wbf-t009ep7-t009ep12/ |

#### Ensemble Config
| Field | Value |
|---|---|
| Method | WBF |
| Sources | I-019 (T-009 ep7 best), I-020 (T-009 ep12 last) |
| WBF IoU thresh | 0.55 |
| WBF skip thresh | 0.001 |
| Weights | [1.0, 1.0] |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-04 |
| Public LB mAP | **0.0523** |
| Notes | **Regressed vs both sources** (I-019=0.0572, I-020=0.0576). Same-architecture WBF between similar checkpoints halves scores on non-overlapping boxes, hurting recall. Cross-architecture ensembling (e.g. Co-DETR + YOLO) is required for WBF to be beneficial. |

---

### I-025 — T-012 YOLO26l best.pt local inference

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4 GB |
| Evaluated split | test set |
| Inputs | T-012 ep39 best.pt |
| Repo commit | bb974cbdcb1a17d580e14dd1d375885410d58cbd |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-025/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-025/stdout.log |
| Artifacts | artifacts/inference-runs/I-025/, artifacts/submissions/submission-I-025-from-T-012.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-012 ep39 best | artifacts/checkpoints/T-012-yolo26l-kaggle/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| imgsz | 1280 |
| conf | 0.001 |
| iou | 0.6 |
| batch | 2 (reduced from 8 to avoid OOM on 3.68 GiB RTX 3050) |
| augment | false |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-05 |
| Public LB mAP | **0.0459** |
| Notes | First local YOLO26l inference from Kaggle-trained T-012. imgsz=1280 matches training resolution; batch=2 required due to RTX 3050 VRAM. YOLO26l (0.0459) significantly underperforms Co-DETR T-009 (0.0576). Cross-architecture WBF (I-026/I-027) is the next experiment. |

---

### I-026 — WBF ensemble T-009ep12(w=2) + T-012best(w=1)

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | Ensemble |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | local (CPU) |
| Evaluated split | test set |
| Inputs | I-020 CSV (T-009 ep12, LB=0.0576) + I-025 CSV (T-012 ep39, LB=0.0459) |
| Repo commit | 8e90326d |
| Config source | scripts/wbf_ensemble.py |
| Artifacts | artifacts/submissions/submission-I-026-wbf-t009ep12-t012best.csv |

#### Ensemble Config
| Field | Value |
|---|---|
| Method | WBF |
| Sources | I-020 (T-009 ep12 Co-DETR), I-025 (T-012 YOLO26l best) |
| WBF IoU thresh | 0.55 |
| WBF skip thresh | 0.001 |
| Weights | [2.0, 1.0] — Co-DETR weighted higher per LB ratio |
| Total predictions | 329,478 |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-05 |
| Public LB mAP | **0.0518** |
| Notes | Cross-architecture WBF (Co-DETR + YOLO) did not beat T-009 solo (0.0576). Up-weighting Co-DETR too aggressively (w=2) diluted YOLO's complementary detections. Compare with I-027 equal weights (0.0546). |

---

### I-027 — WBF ensemble T-009ep12(w=1) + T-012best(w=1) equal weights

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | Ensemble |
| Env | mmdet (challenges/fathomnet-clef/envs/mmdet/) |
| Platform | local (CPU) |
| Evaluated split | test set |
| Inputs | I-020 CSV (T-009 ep12, LB=0.0576) + I-025 CSV (T-012 ep39, LB=0.0459) |
| Repo commit | 8e90326d |
| Config source | scripts/wbf_ensemble.py |
| Artifacts | artifacts/submissions/submission-I-027-wbf-t009ep12-t012best-equal.csv |

#### Ensemble Config
| Field | Value |
|---|---|
| Method | WBF |
| Sources | I-020 (T-009 ep12 Co-DETR), I-025 (T-012 YOLO26l best) |
| WBF IoU thresh | 0.55 |
| WBF skip thresh | 0.001 |
| Weights | [1.0, 1.0] — equal weights ablation vs I-026 |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-05 |
| Public LB mAP | **0.0546** |
| Notes | Equal weights (0.0546) beat 2:1 weighting I-026 (0.0518), but both below T-009 solo (0.0576). The YOLO26l at LB=0.0459 is too weak to improve WBF — it adds low-confidence detections that dilute precision. Need stronger YOLO or better weight tuning. |

---

### I-028 — T-011 yolo26s best.pt local inference

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Duration | ~1m50s |
| Evaluated split | test set |
| Inputs | T-011 best.pt (Ultralytics fitness best, epoch 39) |
| Repo commit | 1c7db99ebe71219c6b4896ff11457956a1c2b116 |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-028/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-028/stdout.log |
| Artifacts | artifacts/inference-runs/I-028/, artifacts/submissions/submission-I-028-from-T-011.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-011 | artifacts/checkpoints/T-011-yolo26s-kaggle/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1280 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |
| Batch | 2 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI |
| Date | 2026-05-05 |
| Public LB mAP | **0.0449** |
| Notes | Fair apples-to-apples comparator against T-012 I-025, which used the same inference settings and scored 0.0459. T-011 best.pt lands only 0.0010 behind T-012 best.pt on LB despite the larger validation mAP gap. |

#### Notes
Produced 46,451 predictions (46,452 CSV lines including header). This establishes the direct S-vs-L comparison under matched inference settings: YOLO26s is slightly worse than YOLO26l, but the LB gap is small.

---

### I-029 — T-011 yolo26s epoch30.pt local inference

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Duration | ~1m50s |
| Evaluated split | test set |
| Inputs | T-011 epoch30.pt (nearest saved checkpoint after val_cls minimum at epoch 28) |
| Repo commit | 1c7db99ebe71219c6b4896ff11457956a1c2b116 |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-029/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-029/stdout.log |
| Artifacts | artifacts/inference-runs/I-029/, artifacts/submissions/submission-I-029-from-T-011.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-011 | artifacts/checkpoints/T-011-yolo26s-kaggle/epoch30.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1280 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |
| Batch | 2 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI |
| Date | 2026-05-05 |
| Public LB mAP | **0.0467** |
| Notes | Beats both T-011 best.pt at 1280 (I-028=0.0449) and T-012 best.pt at 1280 (I-025=0.0459). The val-cls-adjacent checkpoint generalized better than the fitness-best checkpoint on LB. |

#### Notes
Produced 65,251 predictions (65,252 CSV lines including header). This is the strongest evidence from the T-011 comparison set: for the smaller YOLO26s model, the checkpoint near the classifier-loss minimum transfers better to the leaderboard than the validation-fitness best checkpoint.

---

### I-030 — T-011 yolo26s best.pt local inference at train resolution

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | YOLO |
| Env | ultralytics (challenges/fathomnet-clef/envs/ultralytics/) |
| Platform | local RTX 3050 4GB |
| Duration | ~1m36s |
| Evaluated split | test set |
| Inputs | T-011 best.pt (Ultralytics fitness best, epoch 39) |
| Repo commit | 1c7db99ebe71219c6b4896ff11457956a1c2b116 |
| Config source | scripts/infer_yolo.py |
| Resolved config | artifacts/inference-runs/I-030/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-030/stdout.log |
| Artifacts | artifacts/inference-runs/I-030/, artifacts/submissions/submission-I-030-from-T-011.csv |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-011 | artifacts/checkpoints/T-011-yolo26s-kaggle/best.pt |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | imgsz=1024 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |
| Batch | 2 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | see Kaggle UI |
| Date | 2026-05-05 |
| Public LB mAP | **0.0497** |
| Notes | Best score of the T-011 comparison set. For YOLO26s, test-time inference at the training resolution outperformed the larger 1280 setting by +0.0048 over I-028. |

#### Notes
Produced 46,009 predictions (46,010 CSV lines including header). The smaller YOLO26s model appears to prefer train-resolution inference over the 1280 upscaling used for the T-012 comparison pass. Combined with I-029, this suggests YOLO26s is viable for lightweight side-track experiments and should not be judged only by the 1280 best.pt comparator.

---

### I-031 — T-010 Co-DETR R50+2ndPL epoch_12 last local non-SAHI

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | Co-DETR |
| Env | mmdet |
| Platform | local RTX 3050 4GB |
| Duration | ~8m |
| Evaluated split | test set |
| Inputs | T-010 epoch_12.pth |
| Config source | scripts/infer_codetr.py |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-010 | artifacts/checkpoints/T-010/epoch_12.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-05 |
| Kaggle submission ID | 52368031 |
| Public LB mAP | **0.0503** |
| Predictions | 300,674 |

#### Notes
Second PL iteration (T-010). Uses pseudo-labels from T-009 ep12 teacher (1,370 annotations). LB dropped vs T-009 ep12 baseline (0.0576 → 0.0503, -0.0073). Audit rerun on 2026-05-06 reproduced the previously-submitted local CSV byte-for-byte. Recorded local CSV fingerprint: `sha256=85693701ec15a473e742cec7d21c7ff8f851c236e344c3802265bc168c6d681b`, `300,675` lines including header. The submission CSV itself remains local-only and is ignored by git.

---

### I-032 — T-010 Co-DETR R50+2ndPL epoch_10 (best val mAP) local non-SAHI

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | Co-DETR |
| Env | mmdet |
| Platform | local RTX 3050 4GB |
| Duration | ~8m |
| Evaluated split | test set |
| Inputs | T-010 epoch_10.pth (best val mAP=0.353) |
| Config source | scripts/infer_codetr.py |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-010 | artifacts/checkpoints/T-010/epoch_10.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-05 |
| Kaggle submission ID | 52368142 |
| Public LB mAP | **0.0404** |
| Predictions | 278,952 |

#### Notes
Best val mAP checkpoint for T-010. It underperformed both ep12 and the T-009 baseline on the public LB. Audit rerun on 2026-05-06 reproduced the previously-submitted local CSV byte-for-byte. Recorded local CSV fingerprint: `sha256=4c0af357c70366afd91ce89db5c4ff587c1e6e7e67f0754f17b21e513082fd3f`, `278,953` lines including header. The submission CSV itself remains local-only and is ignored by git.

---

### I-033 — T-010 Co-DETR R50+2ndPL epoch_8 local non-SAHI

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Track | Co-DETR |
| Env | mmdet |
| Platform | local RTX 3050 4GB |
| Duration | ~8m |
| Evaluated split | test set |
| Inputs | T-010 epoch_8.pth |
| Config source | scripts/infer_codetr.py |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| primary | T-010 | artifacts/checkpoints/T-010/epoch_8.pth |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | 768×432 |
| Conf threshold | 0.001 |
| IoU threshold (NMS) | 0.6 |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Date | 2026-05-05 |
| Kaggle submission ID | 52368235 |
| Public LB mAP | **0.0384** |
| Predictions | 298,870 |

#### Notes
Early checkpoint (ep8) as hedge per codex recommendation. It finished below both ep12 and ep10 on the public LB. Audit rerun on 2026-05-06 reproduced the previously-submitted local CSV byte-for-byte. Recorded local CSV fingerprint: `sha256=7b1f5915657cd494de0974cb05137f702ea59fdd796b31bbeef81b9343175e29`, `298,871` lines including header. The submission CSV itself remains local-only and is ignored by git.

---

### I-049 — T-014 pre-retrain 5-fold YOLO26s teacher ensemble

| Field | Value |
|---|---|
| Date | 2026-05-07 |
| Track | Ensemble |
| Env | ultralytics |
| Platform | local RTX 3050 4GB |
| Duration | ~2h total raw inference + local WBF post-processing |
| Evaluated split | test set |
| Inputs | `I-034..I-048` raw fold/checkpoint CSVs from `T-014..T-018` |
| Repo commit | `845096ed1880ad4e043a2edaf547cfb3de0f3440` |
| Config source | `scripts/wbf_ensemble.py` |
| Resolved config | `artifacts/inference-runs/I-049/resolved-config.json` |
| Stdout log | — (manual assembly; no single launcher stdout) |
| Artifacts | `artifacts/inference-runs/I-049/`, `artifacts/submissions/submission-I-049-from-T-014-yolo5fold-ep35-40-45-wbf2stage.csv` |

#### Checkpoints Used
| Role | Source exp | Path |
|---|---|---|
| fold0 teachers | T-014 | `artifacts/submissions/T-014-test-teacher-ensemble/fold0-epoch35-I-034-from-T-014.csv`, `fold0-epoch40-I-035-from-T-014.csv`, `fold0-epoch45-I-036-from-T-014.csv` |
| fold1 teachers | T-015 | `artifacts/submissions/T-014-test-teacher-ensemble/fold1-epoch35-I-037-from-T-015.csv`, `fold1-epoch40-I-038-from-T-015.csv`, `fold1-epoch45-I-039-from-T-015.csv` |
| fold2 teachers | T-016 | `artifacts/submissions/T-014-test-teacher-ensemble/fold2-epoch35-I-040-from-T-016.csv`, `fold2-epoch40-I-041-from-T-016.csv`, `fold2-epoch45-I-042-from-T-016.csv` |
| fold3 teachers | T-017 | `artifacts/submissions/T-014-test-teacher-ensemble/fold3-epoch35-I-043-from-T-017.csv`, `fold3-epoch40-I-044-from-T-017.csv`, `fold3-epoch45-I-045-from-T-017.csv` |
| fold4 teachers | T-018 | `artifacts/submissions/T-014-test-teacher-ensemble/fold4-epoch35-I-046-from-T-018.csv`, `fold4-epoch40-I-047-from-T-018.csv`, `fold4-epoch45-I-048-from-T-018.csv` |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | `imgsz=1024` |
| Conf threshold | `0.001` |
| IoU threshold (NMS) | `0.6` |

#### SAHI Config
| Field | Value |
|---|---|
| Enabled | no |

#### Ensemble
| Field | Value |
|---|---|
| Enabled | yes |
| Method | two-stage WBF |
| Sources | `I-034..I-048` raw plateau CSVs |
| WBF IoU thresh | `0.55` |
| WBF skip thresh | `0.001` |
| Fold coverage | all 5 folds |

#### Submission
| Field | Value |
|---|---|
| Submitted | yes |
| Submission ID | not exposed by current Kaggle CLI output |
| Date | 2026-05-07 |
| Public LB mAP | **0.0582** |
| Notes | filename `submission-I-049-from-T-014-yolo5fold-ep35-40-45-wbf2stage.csv`; Kaggle description `T-014 pre-retrain 5-fold YOLO26s teacher ensemble (epoch35/40/45, 2-stage WBF)` |

#### Notes
This is the pre-retrain diagnostic submission required by the locked `T-014` plan. Stage 1 runs WBF within each fold across `epoch35/40/45` with equal weights (`iou_thr=0.55`, `skip_box_thr=0.001`), producing fold-level fused CSVs with row counts `42465`, `40141`, `41903`, `43116`, and `39419`. Stage 2 runs equal-weight WBF across those five fold-level CSVs to produce the final submission CSV.

The final CSV has `74,110` prediction rows and SHA-256 `43b4ff776d276d20eb3dbc749b4ad1b33a31e8a08010d33f189f32a7ffff4d3a`. On the public leaderboard it scored `0.0582`, which makes it the strongest tracked public result in the repo so far and justifies the mandatory final `T-019` retrain rather than stopping at the teacher-ensemble stage.

---

### I-050..I-053 — T-019 final retrain single-checkpoint sweep

| Field | Value |
|---|---|
| Date | 2026-05-08 |
| Track | YOLO |
| Env | ultralytics |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Inputs | `T-019` Kaggle final retrain checkpoints from `artifacts/checkpoints/T-019-kaggle-final/` |
| Repo commit | pending endgame commit |
| Config source | `scripts/infer_yolo.py` |
| Artifacts | `artifacts/inference-runs/I-050/` .. `I-053/`, `artifacts/submissions/T-019-final/` |

#### Checkpoints Used
| Run ID | Checkpoint |
|---|---|
| I-050 | `artifacts/checkpoints/T-019-kaggle-final/checkpoints/epoch45.pt` |
| I-051 | `artifacts/checkpoints/T-019-kaggle-final/best.pt` |
| I-052 | `artifacts/checkpoints/T-019-kaggle-final/last.pt` |
| I-053 | `artifacts/checkpoints/T-019-kaggle-final/checkpoints/epoch40.pt` |

#### Inference Config
| Field | Value |
|---|---|
| Input resolution | `imgsz=1024` |
| Conf threshold | `0.001` |
| IoU threshold (NMS) | `0.6` |
| SAHI | no |

#### Submission Results
| Run ID | CSV | Rows | SHA-256 | Kaggle Sub ID | Public LB |
|---|---|---:|---|---:|---:|
| I-050 | `epoch45-I-050-from-T-019.csv` | `36,439` | `c2fc7d025306f2a91c72241ec06938cf411ecc2752159db2e3ce6ab976d37668` | `52441654` | `0.0438` |
| I-051 | `best-I-051-from-T-019.csv` | `36,631` | `68dd9eea66cc37171db7bd86fad27dcfa85498850c2c2b1b510b383af126b95b` | `52441691` | `0.0450` |
| I-052 | `last-I-052-from-T-019.csv` | `37,101` | `c63b954e2d5f7fafece6852ed0c0bb0f16171a00e3ef99dfb6af80b9ebf50e1e` | `52441732` | `0.0461` |
| I-053 | `epoch40-I-053-from-T-019.csv` | `40,758` | `51d455a5682dd2fda1f70c402976a780c39633f7eba1c2d85e8a318334ee83ab` | `52441772` | `0.0444` |

#### Notes
The Kaggle final retrain completed all 50 epochs and exported `best.pt`, `last.pt`, and periodic checkpoints through `epoch45.pt`, but the resulting detector family underperformed badly on the public leaderboard. `best.pt` corresponds to epoch `49` in the exported `results.csv`, and `last.pt` is epoch `50`; both still trailed the best teacher-only submission family by a wide margin.

---

### I-058..I-063 — T-014 endgame ensemble sweep

| Field | Value |
|---|---|
| Date | 2026-05-08 |
| Track | Ensemble |
| Env | ultralytics |
| Platform | local RTX 3050 4GB |
| Evaluated split | test set |
| Repo commit | pending endgame commit |
| Config source | `scripts/wbf_ensemble.py` |
| Artifacts | `artifacts/inference-runs/I-058/` .. `I-063/`, `artifacts/submissions/T-019-final/` |

#### Ensemble Config
| Field | Value |
|---|---|
| Method | WBF |
| WBF IoU thresh | `0.55` |
| WBF skip thresh | `0.001` |

#### Submission Results
| Run ID | Sources | Rows | SHA-256 | Kaggle Sub ID | Public LB |
|---|---|---:|---|---:|---:|
| I-058 | fold epoch45-only 5-fold WBF | `73,014` | `472d6308fd4f42457d5f77197b10e8b97b6bf72cfca549a4fce04922234b98b1` | `52441209` | `0.0545` |
| I-059 | fold `(epoch40, epoch45)` within-fold WBF, then 5-fold WBF | `70,575` | `a07638f757c501131054560d910f9413636737019133c516bb63d6c0d84e0399` | `52441208` | `0.0571` |
| I-060 | fold epoch40-only 5-fold WBF | `84,709` | `0719b10665215d68858dbea93ebdb551f2279f71ec49d80ae8e791b6f16dbc40` | `52441323` | `0.0537` |
| I-061 | fold epoch35-only 5-fold WBF | `85,089` | `be0cd0788be81c6e2891ebead89d136ddbd00d2678add92d18fc74abc46095ff` | `52441327` | `0.0589` |
| I-062 | fold `(epoch35, epoch40)` within-fold WBF, then 5-fold WBF | `78,515` | `99d9e6bdfd609eabee859d783f6e490c6b0054aa21844d6b25cc0be84698af52` | `52441368` | `0.0577` |
| I-063 | equal-weight WBF of `I-020` (`T-009` ep12 Co-DETR) + `I-061` | `336,600` | `ec526e36c045650e24151ec59ec32f70c39cc427896d8d3efa406a42b9673476` | `52441824` | `0.0562` |

#### Notes
`I-061` became the strongest public result in the whole repo at `0.0589`, edging out the original `I-049` teacher ensemble (`0.0582`). After the `T-019` final retrain singles all failed, the last submission slot was spent on `I-063`, an equal-weight cross-family WBF hedge between the strongest historical Co-DETR parent (`T-009` ep12, `I-020`, public LB `0.0576`) and the strongest YOLO teacher parent (`I-061`). `I-063` finished at `0.0562`, so the final experiment outcome remained the teacher-only `I-061` operating point rather than the final retrain or the cross-family hedge.

---

## Inference Entry Template

<!--
### I-001 — <short description>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Track | YOLOv11 \| Co-DETR \| Ensemble |
| Env | ultralytics \| mmdet |
| Platform | local RTX 3050 \| TensorDock \| Kaggle |
| Duration | Xh Ym |
| Evaluated split | micro-val \| test set |
| Inputs | T-001 checkpoint, T-002 checkpoint, configs/per_class_thresholds.yaml |
| Repo commit | <40-char git SHA used to launch the run> |
| Config source | scripts/infer_yolo.py \| scripts/infer_codetr.py |
| Resolved config | artifacts/inference-runs/I-001/resolved-config.json |
| Stdout log | artifacts/inference-runs/I-001/stdout.log |
| Artifacts | artifacts/inference-runs/I-001/, artifacts/submissions/I-001-.../submission.csv |

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

## Inference Conventions

- `I-NNN` IDs are zero-padded, unique, and never reused.
- Record output paths explicitly so the inference run or submission can be reconstructed from the entry alone.
- Every new tracked run must have a durable bundle under `artifacts/inference-runs/I-NNN/`.
- `Repo commit`, `Config source`, `Resolved config`, and the submission CSV path are mandatory for new entries.
- The `Inputs` field is the primary lineage record: list all checkpoints, pseudo-label sources, and threshold files.
- Log entry must be committed to git when the inference/submission completes.
- Archive curated checkpoints and submission CSVs under `artifacts/`.
