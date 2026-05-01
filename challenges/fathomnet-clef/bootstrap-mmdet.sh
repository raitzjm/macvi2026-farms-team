#!/usr/bin/env bash
# =============================================================================
# bootstrap-mmdet.sh — create the `mmdet` conda env
#
# Creates: <challenge-dir>/envs/mmdet/  (gitignored, persistent via repo mount)
# Stack:   Python 3.11, conda-forge CUDA toolchain (cuda-nvcc, cudatoolkit,
#          cudnn), PyTorch 2.1.2+cu121, mmcv (CUDA ops), mmdet, mmengine,
#          pycocotools, wandb, ensemble-boxes, pandas, numpy, sahi
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
REQ_FILE="${SCRIPT_DIR}/requirements-mmdet.txt"
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

if [ ! -f "${REQ_FILE}" ]; then
    echo "ERROR: requirements file not found at ${REQ_FILE}" >&2
    exit 1
fi

# --- Create env --------------------------------------------------------------

echo ">>> Creating conda env at ${ENV_DIR} (Python 3.11)..."
mkdir -p "$(dirname "${ENV_DIR}")"
"${CONDA}" create -p "${ENV_DIR}" python=3.11 -y

# --- PyTorch 2.1.2 + CUDA 12.1 -----------------------------------------------

echo ">>> Installing pinned MMDetection stack from ${REQ_FILE}..."
MMCV_WITH_OPS=1 FORCE_CUDA=1 "${PIP}" install \
    --requirement "${REQ_FILE}" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

echo ">>> Reasserting numpy<2 after dependency resolution..."
"${PIP}" install "numpy<2"

echo ">>> Pinning setuptools<81 so MMEngine can still import pkg_resources..."
"${PIP}" install "setuptools<81"

echo ">>> Installing SAHI without dependency resolution to preserve pinned torch..."
"${PIP}" install sahi==0.11.36 --no-deps

# --- Co-DETR source (mmdet projects/ are not in the pip wheel) ---------------

CODETR_SRC="${SCRIPT_DIR}/envs/mmdet-src"
if [ ! -d "${CODETR_SRC}" ]; then
    echo ">>> Cloning mmdetection v3.3.0 for projects/CO-DETR..."
    git clone --depth 1 --branch v3.3.0 \
        https://github.com/open-mmlab/mmdetection.git "${CODETR_SRC}"
fi

# --- Smoke check -------------------------------------------------------------

echo ">>> Running import smoke check..."
PYTHONPATH="${CODETR_SRC}/projects/CO-DETR:${PYTHONPATH:-}" \
    "${PYTHON}" -c "import torch, mmcv, mmdet, mmengine, sahi, codetr, numpy; from mmcv.ops import nms; assert torch.__version__ == '2.1.2+cu121', torch.__version__; assert numpy.__version__ == '1.26.4', numpy.__version__; print(torch.__version__, mmcv.__version__, mmdet.__version__, mmengine.__version__, sahi.__version__, numpy.__version__)"

echo ""
echo "=== bootstrap-mmdet.sh complete ==="
echo "    Env: ${ENV_DIR}"
echo "    To activate: conda activate '${ENV_DIR}'"
echo "    Or use:      conda run -p '${ENV_DIR}' python ..."
