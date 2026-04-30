#!/usr/bin/env bash
# =============================================================================
# bootstrap-ultralytics.sh — create the `ultralytics` conda env
#
# Creates: <challenge-dir>/envs/ultralytics/  (gitignored, persistent via repo mount)
# Stack:   Python 3.11, PyTorch 2.1.2+cu121, ultralytics (YOLO11), SAHI,
#          pycocotools, ensemble-boxes, wandb, opencv-python-headless,
#          pandas, numpy, tqdm, pyyaml
#
# Run once. Because the challenge directory is part of the mounted repo volume,
# the env survives container recreations — no need to re-run on restart.
# NOTE: conda records absolute paths (shebangs, RECORD). If the repo is moved
# to a different host path, delete envs/ultralytics/ and re-run this script.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA=/home/farms/miniforge3/bin/conda
ENV_DIR="${SCRIPT_DIR}/envs/ultralytics"
REQ_FILE="${SCRIPT_DIR}/requirements-ultralytics.txt"
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

# --- PyTorch 2.1.2 + CUDA 12.1 (prebuilt wheel) ------------------------------

echo ">>> Installing pinned ultralytics stack from ${REQ_FILE}..."
"${PIP}" install --requirement "${REQ_FILE}" --extra-index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing SAHI without dependency resolution to preserve pinned torch..."
"${PIP}" install sahi==0.11.36 --no-deps

# --- Smoke check -------------------------------------------------------------

echo ">>> Running import smoke check..."
"${PYTHON}" -c "import torch, ultralytics, sahi, cv2, pandas, numpy, yaml, tqdm, wandb, pycocotools, ensemble_boxes; assert torch.__version__ == '2.1.2+cu121', torch.__version__; assert numpy.__version__ == '1.26.4', numpy.__version__; print(torch.__version__, ultralytics.__version__, sahi.__version__, numpy.__version__)"

echo ""
echo "=== bootstrap-ultralytics.sh complete ==="
echo "    Env: ${ENV_DIR}"
echo "    To activate: conda activate '${ENV_DIR}'"
echo "    Or use:      conda run -p '${ENV_DIR}' python ..."
