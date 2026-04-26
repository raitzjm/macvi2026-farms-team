# FathomNet 2026 Competition Reference

## 1. Competition Overview and Task

**Title:** FathomNetCLEF2026: Positive-Unlabeled Object Detection in Marine Images
**Context:** Part of LifeCLEF 2026 (CLEF conference) and FGVC13 Workshop (CVPR 2026).
**Task:** Object Detection under a **positive-unlabeled (PU) learning regime**.
**Challenge:** Training labels are incomplete — experts may have labeled only specific species they recognize, leaving other organisms in the same image unlabeled. Models must learn to detect all relevant marine organisms despite missing labels in the training set. Participants are explicitly encouraged to use PU learning, semi-supervised, and self-supervised techniques.

The 32 categories were derived from the FathomVerse community game, which crowdsourced annotation of FathomNet imagery using consolidated taxonomic and morphological groupings.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Training images | 6,463 (incomplete/PU labels) |
| Training annotations | 22,225 (~3.44 per image average) |
| Training image resolution | 1,920–3,840 × 1,080–2,160 px |
| Test images | 1,425 (fully annotated, for evaluation) |
| Test image resolution | 640–2,048 × 366–1,080 px (smaller than training) |
| Annotation format | COCO Object Detection JSON |
| Image format | PNG / JPG (downloaded via `coco_url` in JSON files) |
| Total size | < 50 GB |
| Categories | 32 consolidated taxonomic/morphological groupings |
| Source | FathomNet Database |

> **Note on test resolution:** Test images are significantly smaller than training images. Models should be robust to multi-scale inference.

### 2.1 Category Table

Category IDs are **non-contiguous** (gaps at 12, 14, 16, 18, 21, 23, 31, 34, 39). Use the exact IDs below in all submissions. Verified against local JSON at `/home/farms/data/datasets/fathomnet-2026/train_dataset.json`.

| ID | Name | Train Annotations | ID | Name | Train Annotations |
|---|---|---|---|---|---|
| 1 | amphipod | 58 | 24 | physonect siphonophore | 101 |
| 2 | anemone | 1,124 | 25 | pyrosome | 36 |
| 3 | barnacle | 31 | 26 | sea cucumber | 378 |
| 4 | benthic worm | 60 | 27 | sea fan | 3,026 |
| 5 | bivalve | 56 | 28 | sea pen | 50 |
| 6 | black coral | 324 | 29 | sea slug | 7 |
| 7 | bony fish | 3,069 | 30 | sea snail | 52 |
| 8 | brittle star | 3,013 | 32 | sea squirt | 17 |
| 9 | calycophoran siphonophore | 46 | 33 | sea star | 624 |
| 10 | chiton | 36 | 35 | shrimp | 525 |
| 11 | crab | 395 | 36 | soft coral | 292 |
| 13 | feather star | 687 | 37 | sponge | 1,213 |
| 15 | hydroid | 41 | 38 | squat lobster | 189 |
| 17 | isopod | 28 | 40 | stony coral | 741 |
| 19 | jelly | 114 | 41 | urchin | 5,723 |
| 20 | larvacean | 47 | | | |
| 22 | octopus | 122 | | | |

**Class imbalance note:** Urchin (5,723), bony fish (3,069), sea fan (3,026), and brittle star (3,013) together account for ~60% of all annotations. Sea slug (7), sea squirt (17), and isopod (28) are severely underrepresented.

### 2.2 Dataset Files (inside container)

```
/home/farms/data/datasets/fathomnet-2026/
├── train_dataset.json       # COCO-format training annotations (PU labels)
├── test_dataset.json        # COCO-format test image list (no annotations)
├── train_set/               # Training images
├── test_set/                # Test images
├── download.py              # Download script
├── requirements.txt         # Python deps for download script
└── README.md
```

### 2.3 JSON Structure

Both JSON files follow the COCO format:

```json
{
  "info": {...},
  "licenses": [...],
  "categories": [{"id": 1, "name": "amphipod", ...}, ...],
  "images": [{"id": ..., "file_name": ..., "width": ..., "height": ..., "coco_url": ...}, ...],
  "annotations": [{"id": ..., "image_id": ..., "category_id": ..., "bbox": [x, y, w, h], "area": ..., "iscrowd": 0}, ...]
}
```

The `test_dataset.json` contains `images` only — no `annotations` field (those are held out for evaluation).

### 2.4 Licensing

Dataset images carry a mix of licenses from contributing institutions:
- **CC0** (public domain)
- **CC-BY** (attribution required)
- **CC-BY-NC-ND** (non-commercial, no derivatives)

Check the per-image metadata in the JSON annotations for the applicable license.

### 2.5 Download

Dataset is distributed via Kaggle. After accepting competition rules:

```bash
# From the dataset directory
python download.py [-o OUTPUT_DIR] [--max-workers MAX_WORKERS] dataset_path
```

### 2.6 Known Issues

- **1,730 training images** return nginx redirect errors when downloaded via `coco_url`. These images are known to be inaccessible and should be excluded from training pipelines. Tracked in the competition GitHub repo.

---

## 3. Evaluation Metric

- **Metric:** mean Average Precision (mAP)
- **IoU range:** 0.50 to 0.95 in 0.05 increments (standard COCO mAP@[0.5:0.95])
- **Computation:** Standard COCO mAP on the fully-labeled test set
- **Correct prediction:** Requires matching both `category_id` and spatial overlap (IoU) with a ground truth bounding box

Reference evaluation notebook: <https://www.kaggle.com/code/lauravchrobak/map50-95>

---

## 4. Submission Format

CSV file with the following columns:

| Column | Description |
|---|---|
| `annotation_id` | Unique integer ID for each predicted box |
| `image_id` | ID of the image (from COCO JSON) |
| `category_id` | Predicted category — use exact non-contiguous IDs from Section 2.1 |
| `bbox_x` | Bounding box left edge (pixels) |
| `bbox_y` | Bounding box top edge (pixels) |
| `bbox_width` | Bounding box width (pixels) |
| `bbox_height` | Bounding box height (pixels) |
| `score` | Confidence score (float, 0–1) |

Example:

```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_width,bbox_height,score
0,12345,5,100.5,200.0,50.2,80.0,0.95
1,12345,19,300.1,150.4,20.0,30.5,0.88
```

---

## 5. Rules and Constraints

| Rule | Detail |
|---|---|
| Team size | Maximum **10 members** |
| Daily submission limit | **10 submissions per day** |
| Final submissions | Up to **2 submissions** may be selected for final private leaderboard judging |
| External image data | **Prohibited** — no image datasets outside the provided Kaggle data and the official FathomNet Database |
| FathomNet Database | **Allowed** as an additional image source |
| Other marine image datasets | **Not allowed** |
| Pre-trained models | **Allowed** (e.g., ImageNet, iNaturalist checkpoints) — must be disclosed |
| Disclosure requirement | Teams must specify all additional data sources and pre-trained models used |
| Scientific requirement | A "working note paper" must be submitted to LifeCLEF to appear in the official ranking |
| Prize | Kudos (no cash prize) |

---

## 6. Timeline

| Event | Date |
|---|---|
| Registration Opens | November 17, 2025 |
| Competition Start | February 7, 2026 |
| Registration Closes | April 23, 2026 |
| **Competition Deadline** | **May 7, 2026** |
| Working Note Paper Deadline | May 28, 2026 |
| Notification of Acceptance | June 30, 2026 |
| Camera-Ready Paper Deadline | July 6, 2026 |
| CLEF 2026 Conference | September 21–24, 2026 |

All deadlines are in **CET (Central European Time)**.

---

## 7. Resources

| Resource | URL |
|---|---|
| Kaggle competition page | <https://www.kaggle.com/competitions/fathomnet-2026> |
| ImageCLEF task page | <https://www.imageclef.org/FathomNetCLEF2026> |
| GitHub repo (starter code) | <https://github.com/fathomnet/fgvc-comp-2026> |
| Evaluation notebook | <https://www.kaggle.com/code/lauravchrobak/map50-95> |
| FathomNet Database | <https://fathomnet.org> |
| LifeCLEF 2026 | <https://www.imageclef.org/LifeCLEF2026> |

---

## 8. Citation and Acknowledgments

If using results from this competition in a publication, cite the LifeCLEF 2026 working notes proceedings.

Acknowledgments: FathomNet, MBARI (Monterey Bay Aquarium Research Institute), Schmidt Ocean Institute, NOAA, and the broader FathomNet contributor community for providing the underlying image data.
