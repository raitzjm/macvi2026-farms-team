# Analysis 00 — Prerequisites Log

**Analyst:** Claude Sonnet 4.6 Agent
**Date started:** 2026-03-11
**Notebook:** notebooks/00_prerequisites.ipynb

---

## Environment & Package Setup

Conda env used: **macvi** (torch 2.1.2+cu121, cv2 4.10, PIL — the appropriate analysis env)
Packages installed ephemerally into macvi (not yet in Dockerfile):
- `imagehash==4.3.2` — pHash deduplication (64-bit perceptual hash)
- `scikit-learn==1.8.0` — normalization, t-SNE fallback
- `faiss-cpu` (latest) — embedding similarity search
- `openTSNE` (latest) — fast CPU t-SNE (replaces cuml which is unavailable in macvi)
- `weasyprint==68.1` — HTML→PDF generation
- `transformers==4.36.2` — RADIO model loading; pinned to 4.36.x because torch 2.1.2
  is incompatible with transformers ≥4.57 (pytree API change: `register_pytree_node` removed)

System packages installed via `sudo apt-get` (were missing from container):
- `libpango-1.0-0`, `libpangoft2-1.0-0`, `libpangocairo-1.0-0`, `libcairo2`, `libgdk-pixbuf2.0-0`
  — required by weasyprint for PDF rendering (need to be added to Dockerfile)

**Decision (env choice):** The `lars-analysis` conda env referenced in the issue no longer
exists. The `macvi` env was chosen over `macvi2` because it already has cv2 and PIL, and
doesn't risk breaking the mmseg 1.x training environment.

**Note for Dockerfile:** add all packages listed above + apt packages to make permanent.

---

## §0a — MULTIAQUA Class Verification

### Task
Confirm class IDs 3-7 in the MULTIAQUA_night dataset (arXiv:2512.17450).
Fallback: visual inspection of 10 images per class ID.

### CONFIRMED MAPPING (critical correction to issue's original assumption)
Full inspection of all 3098 files reveals classes 0-4 only (no 5-7).
The issue's claim "0=obstacle, 1=water, 2=sky" is **WRONG** for MULTIAQUA.

**Correct mapping verified via pixel distribution + visual overlay (figures/multiaqua_allclass_verify.png):**
| Value | Semantic | % of image | Location | → LaRS |
|-------|----------|-----------|----------|--------|
| 0 | dynamic obstacle (boats, buoys) | 0.6% | 100% bottom | 0 (obstacle) |
| 1 | static obstacle (shore, mountains) | 9.8% | 50/50 | 0 (obstacle) |
| 2 | recording boat (auxiliary) | 0.1% | 92% bottom | 255 (ignore) |
| 3 | water | 43.8% | 100% bottom | 1 (water) |
| 4 | sky | 45.7% | 97% top | 2 (sky) |

Validated against MULTIAQUA paper (arXiv:2512.17450): exactly 4 classes (sky, water,
static obstacle, dynamic obstacle) + recording boat auxiliary. Integer mapping verified
empirically since paper doesn't state IDs.

Script updated: `MAQUA_MAPPING = {0: 0, 1: 0, 2: 255, 3: 1, 4: 2}`

### Status
- [x] Classes 0-4 visually inspected — QA figures saved
- [x] Final mapping confirmed and script updated

### Decisions / Issues
- Only 3098 annotation files for 3298 images (200 images lack annotations — excluded)
- Issue's original mapping was wrong — this was the primary goal of §0a

---

## §0b — WaterScenes Class 0 QA

### Task
Sample 20 images where class 0 pixels exist, overlay mask, decide mapping.
Decision options: map to sky (2), map to ignore (255), or use free_space binary only.

### Dataset note
WaterScenes semantic annotations at: `water_scenes/semantic/SegmentationClass/{id}.png`
Observed classes: 0, 6, 8 (in spot check). Issue lists 9 classes (0-8):
- Class 8 → water (LaRS 1) is clear
- Classes 1-7 → obstacles (LaRS 0) per issue
- Class 0 → QA needed (sky? or ambiguous?)

### Decision: Class 0 → sky (2)
Visual QA figure (figures/waterscenes_class0_qa.png) clearly shows class 0 pixels
(yellow overlay) covering the sky/above-waterline region in all 20 sampled images.
The images are Chinese canal/river scenes. Class 0 consistently maps to the background
sky area → mapped to LaRS sky (2).

### Status
- [x] QA grid saved to figures/waterscenes_class0_qa.png
- [x] Decision: class 0 → LaRS sky (2)

---

## §0c — Within-Dataset Temporal Deduplication

### Task
pHash (64-bit) per image, cluster at Hamming ≤ 10, keep one per cluster.
Output: shared_data/dedup_stats.csv

### Dataset image counts (pre-dedup)
| Dataset | Images | Notes |
|---------|--------|-------|
| MULTIAQUA_night | 3,298 | 3,098 have annotations; 200 missing annotation excluded |
| OrcaTech (1280×640 train) | 121 | X-prefix (Wabash River) |
| OrcaTech (640×320 train) | 345 | H/N-prefix (different scenes) |
| OrcaTech test 1280×640 | 61 | |
| OrcaTech test 640×320 | 173 | |
| Tampere-WaterSeg | 600 | dock(200)+channel(200)+open(200) |
| WaterScenes | 54,120 | main bottleneck for dedup |
| ROSEBUD HD | 549 | |

### Status
- [ ] MULTIAQUA deduplicated
- [ ] OrcaTech deduplicated
- [ ] Tampere deduplicated
- [ ] WaterScenes deduplicated
- [ ] ROSEBUD deduplicated
- [ ] dedup_stats.csv written

### Decisions / Issues
_TBD_

---

## §0d — Sky/Obstacle Heuristic for Binary Datasets

### Task
For USVInland, Tampere-WaterSeg, (not ROSEBUD — has fluvial RGB masks):
per-column y_water profile to split non-water pixels into sky (2), obstacle (0), ignore (255).

### Decision: ROSEBUD skip heuristic — use rgb_fluvial_masks directly
ROSEBUD has 7-class RGB fluvial masks with confirmed color coding from README.txt:
- [255,255,255] = water → LaRS 1
- [255,0,255]   = sky → LaRS 2
- [0,0,255]     = shore/bank → LaRS 0 (obstacle)
- [0,255,0]     = bridges → LaRS 0
- [255,0,0]     = boats/cars/people → LaRS 0
- [255,255,0]   = trees/foliage → LaRS 0
- [0,255,255]   = debris → LaRS 0
- [0,0,0]       = unlabeled/background → LaRS 255 (ignore)

QA: sample 10 images per dataset, overlay, save figures/heuristic_qa_{dataset}.png.
Fallback: if >15% mislabeled → assign ALL non-water as ignore (255).

### Status
- [ ] USVInland heuristic applied + QA
- [ ] Tampere heuristic applied + QA
- [ ] ROSEBUD: direct RGB mapping applied (no heuristic needed)

### Decisions / Issues
- ROSEBUD rgb_fluvial_masks color table extracted from ROSEBUD_ReadMe.rtf directly —
  no Gemini/web lookup needed. Color confirmed: [0,0,0]=black=unlabeled (not in docs
  but observed in masks, treated as ignore 255).

---

## §0e — Label Harmonization

### Task
Produce shared_data/harmonized_{dataset}/masks/ (remapped PNGs) + images.txt index.
Also produce shared_data/class_balance.csv.

### Full mapping table (confirmed)
| Dataset | source class | → LaRS class |
|---------|-------------|--------------|
| MULTIAQUA | 0 | obstacle (0) |
| MULTIAQUA | 1 | water (1) |
| MULTIAQUA | 2 | sky (2) |
| MULTIAQUA | 3-4 | TBD after §0a visual QA |
| OrcaTech | 0 | non-water → heuristic §0d |
| OrcaTech | 1 | water (1) |
| Tampere | 0 | non-water → heuristic §0d |
| Tampere | 255 | water (1) |
| WaterScenes | 0 | sky (2) or ignore — see §0b |
| WaterScenes | 1-7 | obstacle (0) |
| WaterScenes | 8 | water (1) |
| ROSEBUD | [255,255,255] | water (1) |
| ROSEBUD | [255,0,255] | sky (2) |
| ROSEBUD | [0,0,255]/[0,255,0]/[255,0,0]/[255,255,0]/[0,255,255] | obstacle (0) |
| ROSEBUD | [0,0,0] | ignore (255) |

### Status
- [ ] MULTIAQUA harmonized
- [ ] OrcaTech harmonized
- [ ] Tampere harmonized
- [ ] WaterScenes harmonized
- [ ] ROSEBUD harmonized
- [ ] class_balance.csv written

### Decisions / Issues
_TBD_
