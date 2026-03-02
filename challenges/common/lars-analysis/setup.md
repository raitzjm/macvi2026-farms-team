# LaRS Embedding Analysis — Environment Setup

## Overview

This document records every package installed and command run to set up the
`lars-analysis` conda environment used by `analysis.ipynb`.

## Conda Environment Creation

Using Miniforge at `/home/farms/miniforge3`:

```bash
/home/farms/miniforge3/bin/conda create -n lars-analysis python=3.11 -y
```

- Python: 3.11
- Location: `/home/farms/miniforge3/envs/lars-analysis`

## RAPIDS cuML (GPU t-SNE)

Attempted install of cuML 24.12 for GPU-accelerated t-SNE:

```bash
conda run -n lars-analysis conda install -c rapidsai -c conda-forge -c nvidia cuml=24.12 cuda-version=12.1 -y
```

**Status**: ✅ Succeeded. cuML 24.12 installed successfully and GPU t-SNE was used
(KL divergence 0.239, completed in seconds). The notebook falls back to
`sklearn.manifold.TSNE` automatically if cuML is unavailable.

## PyTorch (CUDA 12.1)

```bash
conda run -n lars-analysis pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- torch: latest stable with cu121 support
- torchvision: matching version

## HuggingFace / RADIO Model

```bash
conda run -n lars-analysis pip install transformers accelerate
conda run -n lars-analysis pip install timm einops open-clip-torch
```

- transformers: for `AutoModel.from_pretrained("nvidia/C-RADIOv3-H")`
- accelerate: for multi-GPU / mixed-precision support
- timm: required by RADIO-H at import time (discovered during execution)
- einops: required by RADIO-H at import time (discovered during execution)
- open-clip-torch: required by RADIO-H at import time (discovered during execution)

## Data / Numerical Stack

```bash
conda run -n lars-analysis pip install pillow numpy pandas tqdm
```

- pillow: image loading
- numpy: array operations, embedding storage
- pandas: DataFrame for analysis
- tqdm: progress bars

## Visualisation

```bash
conda run -n lars-analysis pip install plotly scikit-learn
conda run -n lars-analysis pip install kaleido
```

- plotly: interactive charts
- scikit-learn: PCA + fallback t-SNE
- kaleido: static image export from Plotly (used in PDF report)

## Jupyter / Notebook Runtime

```bash
conda run -n lars-analysis pip install jupyter nbconvert nbformat ipykernel ipywidgets
```

- jupyter: notebook server
- nbconvert: HTML and PDF export
- nbformat: notebook format utilities
- ipykernel: kernel for Jupyter
- ipywidgets: interactive widgets (progress bars in notebooks)

## PDF Export

```bash
conda run -n lars-analysis pip install weasyprint
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0
```

- weasyprint: HTML→PDF conversion
- libpango / libpangoft2: system libraries required by weasyprint for text
  rendering (missing by default in the container; must be installed via apt)

## Report Generator

```bash
conda run -n lars-analysis pip install matplotlib
```

- matplotlib: used by `generate_report.py` for static figures in the PDF report
  (installed separately from the notebook stack since the notebook uses Plotly)

## Kernel Registration

```bash
conda run -n lars-analysis python -m ipykernel install --user --name lars-analysis --display-name "lars-analysis"
```

Registers the `lars-analysis` kernel so `jupyter nbconvert --execute` can
find it by `--ExecutePreprocessor.kernel_name=lars-analysis`.

## FAISS (Nearest-Neighbour Search for Spatial OOD)

```bash
conda run -n lars-analysis pip install faiss-cpu
```

- **faiss-cpu** was used (version 1.13.2). `faiss-gpu` is not available via pip and the
  conda-forge `faiss-gpu=1.9.0` package corrupted the environment's numpy install on
  this machine (Python 3.11 / CUDA 12.1 / RTX A6000). Fallback to CPU FAISS was chosen.
- FAISS is used only for **index build** (Section 2 of `spatial_ood.ipynb`): building
  `IndexFlatIP` from 1-in-10 subsampled training patches (~1.4 M vectors) and saving
  to `faiss_index.bin`.
- **Phase 2 heatmap search uses PyTorch GPU matmul, not FAISS CPU search**. The 1.4 M
  subsampled vectors are re-read from the memmap, moved to GPU as float16 (3.6 GB
  VRAM), and per-image similarity is computed as `(query_tokens @ train.T).max(dim=1)`.
  Benchmark: ~13-17 ms/image on RTX A6000 (A6000 has 48 GB VRAM); 1,396 images
  completed in ~17 minutes total including RADIO-H inference.
- CPU FAISS flat search was measured at ~2.2 min/image (unusable; ~51 hours total).
- `faiss.get_num_gpus()` returns 0 in this environment; all FAISS operations are CPU-side.

## Spatial OOD Extension

The packages above (torch, faiss-cpu, matplotlib, etc.) are reused by the spatial OOD
extension notebook at `spatial_ood/spatial_ood.ipynb`. No additional packages beyond
what is already installed are required.

## Notes

- The conda environment itself is NOT committed to git.
- Embeddings are stored in `embeddings.db` (Git-LFS tracked).
- t-SNE results are stored in `tsne_results.npy` (Git-LFS tracked).
- All heavy computation is protected by existence checks so cells only run once.
- Spatial OOD large artifacts (`train_patches.dat`, `faiss_index.bin`,
  `train_patch_meta.npy`) are NOT committed (listed in `.gitignore`).
  Per-image heatmap `.npy` files and figure PNGs are committed.
