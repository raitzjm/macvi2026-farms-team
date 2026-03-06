# LaRS Embedding Distribution Analysis

Analysis of the LaRS dataset's train/test embedding distribution using NVIDIA RADIO-H,
covering both whole-image and spatial (patch-level) out-of-distribution (OOD) detection.
This work informs model selection and augmentation strategy for the embedded semantic
segmentation challenge.

## Purpose

The benchmark design guarantees that all 440 test sequences are absent from the training
set. This analysis quantifies *how different* those test images are from training — not
just a binary "seen/unseen" flag, but a continuous similarity score grounded in a
powerful vision foundation model.

**Key question answered:** Can a model trained on the LaRS training set generalise to
the test set, or is there a distribution shift severe enough to require special handling?

## Dataset Processed

| Split | Images |
|-------|--------|
| Train | 2,605  |
| Val   | 198    |
| Test  | 1,203  |
| **Total** | **4,006** |

Embeddings were computed with **NVIDIA C-RADIOv3-H** (653 M parameters) on an
RTX A6000. RADIO-H was chosen because it produces rich spatial token grids in a
single forward pass — no manual patch extraction required.

## Directory Structure

```
lars-analysis/
├── README.md                    # this file
├── shared_data/                 # LFS-tracked shared artefacts (used by both analyses)
│   ├── embeddings.db            # SQLite, 4006 rows, BLOB embeddings (Git-LFS)
│   └── tsne_results.npy         # precomputed t-SNE coordinates (Git-LFS)
├── reports/
│   ├── whole_image_ood_report.pdf
│   └── spatial_ood_report.pdf
├── whole_image_ood/             # Analysis 1 — whole-image embeddings
│   ├── whole_image_ood.ipynb    # interactive Plotly notebook
│   ├── build_notebook.py        # programmatic notebook builder
│   ├── generate_report.py       # standalone PDF generator
│   ├── GOTCHAS.md               # failures, OOM recovery, methodology corrections
│   ├── setup.md                 # full environment / package install log
│   └── logs/
│       └── analysis.log         # nbconvert output from whole-image run
└── spatial_ood/                 # Analysis 2 — patch-level spatial OOD
    ├── spatial_ood.ipynb        # interactive notebook with heatmaps
    ├── generate_spatial_report.py
    ├── figures/                 # exported PNG figures
    ├── heatmaps/                # per-image similarity heatmaps (.npy)
    └── logs/
        └── spatial_ood.log      # real-time cell-level log from spatial run
```

---

## Analysis 1 — Whole-Image OOD (`whole_image_ood/`)

Each of the 4,006 images is embedded as a **single whole-image vector** using RADIO-H's
CLS token (or mean-pooled spatial tokens). All embeddings are stored in `embeddings.db`.

### Methods

| Section | Method | Purpose |
|---------|--------|---------|
| §4 | t-SNE (cuML GPU-accelerated) | 2-D visualisation of the full embedding cloud |
| §6 | Per-sequence t-SNE overlay | Colour t-SNE by sequence ID to distinguish global shift from isolated unseen envs |
| §7 | Cosine similarity distribution | Quantitative OOD score: for every test/val image, find max cosine similarity to any training image |

PCA, UMAP, k-means, full k-NN, and summary statistics were **explicitly skipped**
as either redundant or answering a different question.

### Key Findings

- **Median test max-cosine-sim to training: 0.911** — visual content is well-covered
  despite no test sequences appearing in training.
- t-SNE shows train and test clouds **substantially overlapping** — no catastrophic
  domain shift.
- The 15 most OOD test images are listed in the report with thumbnails. Even these
  score above 0.70, suggesting atypical but not alien scenes.
- No special OOD mitigation (domain adaptation, synthetic augmentation) is required
  for Phase 1 baselines; the standard LaRS training split is sufficient.

### Notebooks & Reports

- **`whole_image_ood.ipynb`** — all interactive Plotly figures, OOD table, sequence
  breakdown. Run with `lars-analysis` conda kernel.
- **`generate_report.py`** — standalone script; reproduces `reports/whole_image_ood_report.pdf`
  without re-running the notebook.

---

## Analysis 2 — Spatial OOD (`spatial_ood/`)

Extends the whole-image analysis to **patch level**: each image is represented as a
grid of RADIO-H spatial tokens, and per-patch similarity to the nearest training patch
is computed. This produces a **spatial heatmap** — a spatial map of which pixels are
OOD relative to training.

### Pipeline

1. **Phase 1 — Index construction**: RADIO-H spatial tokens are extracted for all
   training images. A 10%-subsampled set (~1.4 M vectors, D=1280) is stored in a
   memory-mapped `.dat` file and indexed with FAISS `IndexFlatIP` for persistence.

2. **Phase 2 — Heatmap scoring**: For each test/val image, its spatial tokens are
   compared against the 1.4 M training vectors via a **PyTorch GPU matmul** (not
   FAISS CPU search, which benchmarked at ~2.2 min/image — ~51 hours total).
   Using float16 on the RTX A6000 (~3.6 GB VRAM) this runs at ~13–17 ms/image.
   One high-resolution image (`inhouse_seq_110_00149`, 15,960 tokens) was recovered
   via a 512-row chunked fallback after a CUDA OOM — all 1,401 heatmaps are present.

3. **Figures**: Heatmaps for selected sequences are saved to `figures/`. The
   `aggregate_heatmap.png` shows the mean OOD score averaged over all test images.

### Key Findings

- No consistent spatial region (e.g. horizon, shoreline) is systematically OOD.
- The most OOD patches tend to appear in unusual lighting conditions or rare vessel
  types — consistent with the whole-image cosine similarity results.
- `top20_ood_grid.png` shows the 20 test images with the lowest median patch similarity
  to training; these are candidates for targeted augmentation in future experiments.

### Notebooks & Reports

- **`spatial_ood.ipynb`** — interactive notebook; includes FAISS index build, GPU
  heatmap scoring, and figure export.
- **`generate_spatial_report.py`** — reproduces `reports/spatial_ood_report.pdf`.

---

## Environment

A dedicated conda environment (`lars-analysis`) is required. Full install log in
`setup.md`. Key packages:

| Package | Role |
|---------|------|
| `torch` + cu121 | RADIO-H inference, GPU matmul heatmap scoring |
| `transformers`, `timm`, `einops`, `open-clip-torch` | RADIO-H model loading |
| `cuml` 24.12 | GPU-accelerated t-SNE |
| `faiss-cpu` | FAISS index construction (Phase 1 only) |
| `plotly`, `kaleido` | Interactive figures + static export |
| `weasyprint` | HTML → PDF report generation |

The environment is **not committed** to git. Rebuild it from `whole_image_ood/setup.md` if lost.

Git-LFS tracked files (large artifacts not re-runnable quickly):

- `shared_data/embeddings.db` — 81 MB SQLite, 4006 rows
- `shared_data/tsne_results.npy` — precomputed t-SNE coordinates

Spatial OOD large artifacts (`train_patches.dat`, `faiss_index.bin`,
`train_patch_meta.npy`) are **gitignored** — regenerate from `spatial_ood.ipynb`.

---

## Reproducing Results

```bash
# 1. Activate the conda environment
conda activate lars-analysis

# 2. Whole-image OOD — run notebook (embeddings already in embeddings.db)
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=lars-analysis \
  whole_image_ood/whole_image_ood.ipynb

# 3. Whole-image PDF report (does not require re-running the notebook)
python whole_image_ood/generate_report.py

# 4. Spatial OOD — run notebook (rebuilds FAISS index + heatmaps from scratch)
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=lars-analysis \
  spatial_ood/spatial_ood.ipynb

# 5. Spatial PDF report
python spatial_ood/generate_spatial_report.py
```

---

## Known Issues / Lessons Learned

See `whole_image_ood/GOTCHAS.md` for the full log. Summary:

1. **Notebook SyntaxError** from f-strings in `nbformat` cell source — fixed by
   switching `build_notebook.py` to string concatenation.
2. **Missing RADIO-H deps** (`timm`, `einops`, `open-clip-torch`) discovered at
   import time — not listed in model card; added to `setup.md`.
3. **CPU FAISS too slow** for Phase 2 (~51 hours) — replaced with PyTorch GPU matmul.
4. **CUDA OOM** for one extra-large image — recovered with 512-row chunked fallback.
5. **Report prose drift** — methodology description in the PDF was wrong after the
   PyTorch GPU switch; caught and corrected by an independent review agent.
