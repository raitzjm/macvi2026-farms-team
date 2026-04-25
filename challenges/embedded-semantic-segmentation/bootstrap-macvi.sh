#!/usr/bin/env bash
# =============================================================================
# bootstrap-macvi.sh — create the `macvi` conda env (Phase 1)
#
# Creates: <challenge-dir>/envs/macvi/  (gitignored, persistent via repo mount)
# Stack:   Python 3.11, PyTorch 2.1.2+cu121, mmcv-full 1.7.2,
#          mmsegmentation 0.30.0 (mmsegmentation-macvi fork), wandb 0.25.0
#
# Run once. Because the challenge directory is part of the mounted repo volume,
# the env survives container recreations — no need to re-run on restart.
# NOTE: conda records absolute paths (shebangs, RECORD). If the repo is moved
# to a different host path, delete envs/macvi/ and re-run this script.
#
# Original pinned recipe: git show macvi-embedded-semantic-segmentation-2026:container/Dockerfile
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA=/home/farms/miniforge3/bin/conda
ENV_DIR="${SCRIPT_DIR}/envs/macvi"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"
MMSEG_SRC="${SCRIPT_DIR}/mmsegmentation-macvi"
FUNCTIONS_PY="${ENV_DIR}/lib/python3.11/site-packages/mmcv/parallel/_functions.py"
DISTRIBUTED_PY="${ENV_DIR}/lib/python3.11/site-packages/mmcv/parallel/distributed.py"
WANDB_VALIDATION_PY="${ENV_DIR}/lib/python3.11/site-packages/wandb/sdk/lib/wbauth/validation.py"

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

# --- mmcv-full 1.7.2 ---------------------------------------------------------

echo ">>> Installing mmcv-full 1.7.2..."
"${PIP}" install \
    mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# --- Compatibility pins -------------------------------------------------------

echo ">>> Installing compatibility pins (setuptools<70, numpy<2, opencv-python<4.11)..."
"${PIP}" install "setuptools<70" "numpy<2" "opencv-python<4.11"

# --- mmcv scatter patch -------------------------------------------------------
# PyTorch 2.1.x changed _get_stream() to require torch.device instead of int.
# mmcv/parallel/_functions.py still passes a plain int, causing a runtime error
# on any multi-GPU scatter operation.

echo ">>> Applying mmcv scatter patch (_functions.py)..."
sed -i \
    's/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device("cuda", device) if isinstance(device, int) else device) for device in target_gpus]/' \
    "${FUNCTIONS_PY}"

# --- mmcv distributed.py DDP-eval patch ---------------------------------------
# PyTorch 2.1.2 removed _use_replicated_tensor_module from DistributedDataParallel
# in some builds. Guard with getattr so eval mode doesn't crash when the attribute
# is absent.
# Verification: grep for the exact pattern before patching, then confirm replacement.

echo ">>> Applying mmcv distributed.py DDP-eval patch..."

BEFORE_PATTERN='self._use_replicated_tensor_module'
AFTER_PATTERN='getattr(self, '"'"'_use_replicated_tensor_module'"'"', False)'

before_count=$(grep -c "${BEFORE_PATTERN}" "${DISTRIBUTED_PY}" || true)
if [ "${before_count}" -eq 0 ]; then
    echo "ERROR: distributed.py patch anchor '${BEFORE_PATTERN}' not found." >&2
    echo "       The patch has either already been applied or the file changed." >&2
    echo "       Inspect: ${DISTRIBUTED_PY}" >&2
    exit 1
fi

sed -i \
    "s/self\._use_replicated_tensor_module/getattr(self, '_use_replicated_tensor_module', False)/g" \
    "${DISTRIBUTED_PY}"

after_count=$(grep -c "getattr(self, '_use_replicated_tensor_module', False)" "${DISTRIBUTED_PY}" || true)
if [ "${after_count}" -eq 0 ]; then
    echo "ERROR: distributed.py patch: replacement not found after sed." >&2
    echo "       The sed substitution may have silently failed." >&2
    echo "       Inspect: ${DISTRIBUTED_PY}" >&2
    exit 1
fi

echo "    distributed.py patched OK (${before_count} occurrence(s) replaced)"

# --- cityscapesscripts + mmsegmentation-macvi ---------------------------------

echo ">>> Installing cityscapesscripts..."
"${PIP}" install cityscapesscripts

echo ">>> Installing mmsegmentation-macvi (non-editable)..."
# Remove any stale egg-info that may have been left by a local pip install in
# the workspace — a stale egg-info causes PEP 517 get_requires_for_build_wheel
# to fail when paths recorded inside don't match the source directory.
rm -rf "${MMSEG_SRC}"/*.egg-info
"${PIP}" install "${MMSEG_SRC}"

# --- wandb 0.25.0 ------------------------------------------------------------

echo ">>> Installing wandb 0.25.0..."
"${PIP}" install wandb==0.25.0

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
echo "=== bootstrap-macvi.sh complete ==="
echo "    Env: ${ENV_DIR}"
echo "    Patches: mmcv scatter, mmcv distributed.py DDP-eval, wandb validation.py"
echo "    To activate: conda activate '${ENV_DIR}'"
echo "    Or use:      conda run -p '${ENV_DIR}' python ..."
