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

**Status**: Attempted. If the install succeeds, cuML t-SNE is used automatically.
If it fails or causes OOM errors at runtime, the notebook falls back to
`sklearn.manifold.TSNE`. The notebook cell indicates which backend was used.

## PyTorch (CUDA 12.1)

```bash
conda run -n lars-analysis pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- torch: latest stable with cu121 support
- torchvision: matching version

## HuggingFace / RADIO Model

```bash
conda run -n lars-analysis pip install transformers accelerate
```

- transformers: for `AutoModel.from_pretrained("nvidia/C-RADIOv3-H")`
- accelerate: for multi-GPU / mixed-precision support

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
```

- weasyprint: HTML→PDF conversion (fallback if LaTeX/webpdf not available)

## Kernel Registration

```bash
conda run -n lars-analysis python -m ipykernel install --user --name lars-analysis --display-name "lars-analysis"
```

Registers the `lars-analysis` kernel so `jupyter nbconvert --execute` can
find it by `--ExecutePreprocessor.kernel_name=lars-analysis`.

## Notes

- The conda environment itself is NOT committed to git.
- Embeddings are stored in `embeddings.db` (Git-LFS tracked).
- t-SNE results are stored in `tsne_results.npy` (Git-LFS tracked).
- All heavy computation is protected by existence checks so cells only run once.
