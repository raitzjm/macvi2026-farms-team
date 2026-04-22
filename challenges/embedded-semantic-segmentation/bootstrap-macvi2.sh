#!/usr/bin/env bash
# =============================================================================
# bootstrap-macvi2.sh — recreate the `macvi2` conda env (Phase 2+)
#
# Creates: /home/farms/miniforge3/envs/macvi2/
# Stack:   Python 3.11, PyTorch 2.1.2+cu121, mmcv 2.1.0,
#          mmsegmentation 1.2.2 (stock), wandb 0.25.0, onnx 1.20.1,
#          onnxruntime 1.24.2, timm
#
# IMPORTANT — envs are EPHEMERAL per container.
# If the container is removed and recreated (docker run / podman run on a fresh
# container), run this script again to restore the env.
#
# Original pinned recipe: git show macvi-embedded-semantic-segmentation-2026:container/Dockerfile
# =============================================================================

set -euo pipefail

CONDA=/home/farms/miniforge3/bin/conda
PIP=/home/farms/miniforge3/envs/macvi2/bin/pip
PYTHON=/home/farms/miniforge3/envs/macvi2/bin/python
ENV_DIR=/home/farms/miniforge3/envs/macvi2
WANDB_VALIDATION_PY="${ENV_DIR}/lib/python3.11/site-packages/wandb/sdk/lib/wbauth/validation.py"

# --- Preflight checks --------------------------------------------------------

if [ ! -x "${CONDA}" ]; then
    echo "ERROR: conda not found at ${CONDA}" >&2
    echo "       Is Miniforge installed at /home/farms/miniforge3/?" >&2
    exit 1
fi

if [ -d "${ENV_DIR}" ]; then
    echo "ERROR: env 'macvi2' already exists at ${ENV_DIR}" >&2
    echo "       To reinstall, run: /home/farms/miniforge3/bin/conda env remove -n macvi2" >&2
    exit 1
fi

# --- Create env --------------------------------------------------------------

echo ">>> Creating conda env 'macvi2' (Python 3.11)..."
"${CONDA}" create -n macvi2 python=3.11 -y

# --- PyTorch 2.1.2 + CUDA 12.1 -----------------------------------------------

echo ">>> Installing PyTorch 2.1.2+cu121..."
"${PIP}" install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- mmcv 2.1.0 --------------------------------------------------------------

echo ">>> Installing mmcv 2.1.0..."
"${PIP}" install \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# --- mmsegmentation 1.2.2 ----------------------------------------------------

echo ">>> Installing mmsegmentation 1.2.2..."
"${PIP}" install mmsegmentation==1.2.2

# --- Compatibility pins + missing mmseg 1.2.2 deps ---------------------------
# setuptools<70: torch 2.1.2 cpp_extension.py imports pkg_resources, removed in setuptools 71+.
# numpy<2: mmcv 2.1.0 compiled against NumPy 1.x ABI.
# opencv-python<4.11: 4.11+ requires numpy>=2 (conflicts with above pin).
# ftfy, regex: required by mmseg.utils.tokenizer (BEiT/CLIP code path).

echo ">>> Installing compatibility pins (setuptools<70, numpy<2, opencv-python<4.11, ftfy, regex)..."
"${PIP}" install \
    "setuptools<70" "numpy<2" "opencv-python<4.11" ftfy regex

# --- WandB + ONNX tools -------------------------------------------------------

echo ">>> Installing wandb 0.25.0, onnx 1.20.1, onnxruntime 1.24.2..."
"${PIP}" install \
    wandb==0.25.0 onnx==1.20.1 onnxruntime==1.24.2

# --- timm --------------------------------------------------------------------

echo ">>> Installing timm..."
"${PIP}" install timm

# --- wandb validation.py patch -----------------------------------------------
# wandb 0.25.0 rejects keys starting with "wandb_" (new key format) because
# the 40-char minimum check fires after stripping the "wandb_" prefix.
# Insert an early-return for the wandb_ prefix before the length check.

echo ">>> Applying wandb validation.py patch..."
"${PYTHON}" -c '
import pathlib
p = pathlib.Path("'"${WANDB_VALIDATION_PY}"'")
src = p.read_text()
needle = "    # On-prem API keys have a variable-length prefix"
ins = "    # New wandb_ key format (36+ chars) -- accept as-is.\n    if key.startswith(\"wandb_\"):\n        return None\n\n"
assert needle in src, "wandb validation.py patch anchor not found"
p.write_text(src.replace(needle, ins + needle))
print("validation.py patched OK")
'

# --- Smoke check -------------------------------------------------------------

echo ">>> Running import smoke check..."
"${PYTHON}" -c "import torch; import mmseg; import mmcv; print(torch.__version__, mmseg.__version__, mmcv.__version__)"

echo ""
echo "=== bootstrap-macvi2.sh complete ==="
echo "    Env: ${ENV_DIR}"
echo "    Patches applied: wandb validation.py"
