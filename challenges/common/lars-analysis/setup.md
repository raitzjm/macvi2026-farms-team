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

## Notes

- The conda environment itself is NOT committed to git.
- Embeddings are stored in `embeddings.db` (Git-LFS tracked).
- t-SNE results are stored in `tsne_results.npy` (Git-LFS tracked).
- All heavy computation is protected by existence checks so cells only run once.
