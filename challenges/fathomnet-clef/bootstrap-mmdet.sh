#!/usr/bin/env bash
# =============================================================================
# bootstrap-mmdet.sh — create the `mmdet` conda env
#
# Creates: <challenge-dir>/envs/mmdet/  (gitignored, persistent via repo mount)
# Stack:   Python 3.11, conda-forge CUDA toolchain (cuda-nvcc, cudatoolkit,
#          cudnn), PyTorch 2.1.2+cu121, mmcv (CUDA ops), mmdet, mmengine,
#          pycocotools, wandb, ensemble-boxes, pandas, numpy
#
# Run once. Because the challenge directory is part of the mounted repo volume,
# the env survives container recreations — no need to re-run on restart.
# NOTE: conda records absolute paths (shebangs, RECORD). If the repo is moved
# to a different host path, delete envs/mmdet/ and re-run this script.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA=/home/farms/miniforge3/bin/conda
ENV_DIR="${SCRIPT_DIR}/envs/mmdet"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

# --- Preflight checks --------------------------------------------------------

if [ ! -x "${CONDA}" ]; then
    echo "ERROR: conda not found at ${CONDA}" >&2
    echo "       Is Miniforge installed at /home/farms/miniforge3/?" >&2
    exit 1
fi

if [ -d "${ENV_DIR}" ]; then
    echo "ERROR: env already exists at ${ENV_DIR}" >&2
    echo "       To reinstall, delete it: rm -rf '${ENV_DIR}'" >&2
    exit 1
fi

# --- Create env --------------------------------------------------------------

echo ">>> Creating conda env at ${ENV_DIR} (Python 3.11)..."
mkdir -p "$(dirname "${ENV_DIR}")"
"${CONDA}" create -p "${ENV_DIR}" python=3.11 -y

# --- PyTorch 2.1.2 + CUDA 12.1 -----------------------------------------------

echo ">>> Installing PyTorch 2.1.2+cu121..."
"${PIP}" install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- OpenMMLab stack ---------------------------------------------------------

echo ">>> Installing setuptools<81 (for pkg_resources), numpy<2, and mmcv..."
"${PIP}" install "setuptools<81" "numpy<2"
MMCV_WITH_OPS=1 FORCE_CUDA=1 "${PIP}" install mmcv==2.1.0 mmengine mmdet -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# --- Supporting packages -----------------------------------------------------

echo ">>> Installing pycocotools, wandb, ensemble-boxes, pandas..."
"${PIP}" install pycocotools wandb ensemble-boxes pandas

# --- Co-DETR source (mmdet projects/ are not in the pip wheel) ---------------

CODETR_SRC="${SCRIPT_DIR}/envs/mmdet-src"
if [ ! -d "${CODETR_SRC}" ]; then
    echo ">>> Cloning mmdetection v3.3.0 for projects/CO-DETR..."
    git clone --depth 1 --branch v3.3.0 \
        https://github.com/open-mmlab/mmdetection.git "${CODETR_SRC}"
fi

# --- Smoke check -------------------------------------------------------------

echo ">>> Running import smoke check..."
"${PYTHON}" -c "import torch, mmcv, mmdet, mmengine; from mmcv.ops import nms; print(torch.__version__, mmcv.__version__, mmdet.__version__, mmengine.__version__)"

echo ""
echo "=== bootstrap-mmdet.sh complete ==="
echo "    Env: ${ENV_DIR}"
echo "    To activate: conda activate '${ENV_DIR}'"
echo "    Or use:      conda run -p '${ENV_DIR}' python ..."
