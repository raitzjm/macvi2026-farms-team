#!/usr/bin/env bash
# run_phase3.sh — run a single Phase 3 model training run.
#
# Usage (from repo root, inside container, with macvi2 env active):
#   source challenges/embedded-semantic-segmentation/phase3/scripts/setup_wandb.sh
#   bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh <model>
#
# <model> is one of:
#   pidnets | stdc2 | pidchad_s_021 | segchad_022
#
# The script:
#   1. Rebuilds the LaRS symlink tree if not present
#   2. Sets PYTHONPATH and WANDB_DIR
#   3. Launches torchrun DDP training with mmseg 1.x tools/train.py (macvi2 env)
#
# Example:
#   bash challenges/embedded-semantic-segmentation/phase3/scripts/run_phase3.sh pidnets

set -uo pipefail

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    echo "ERROR: No model specified. Usage: run_phase3.sh <model>" >&2
    exit 1
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CHALLENGE="challenges/embedded-semantic-segmentation"
CONFIGS="$REPO_ROOT/$CHALLENGE/phase3/configs"
OUTPUTS="$REPO_ROOT/$CHALLENGE/outputs"
WORK_DIR="$OUTPUTS/phase3/$MODEL"
NUM_GPUS=2

MMSEG_TRAIN="/home/farms/miniforge3/envs/macvi2/lib/python3.11/site-packages/mmseg/.mim/tools/train.py"
CONDA_RUN="/home/farms/miniforge3/bin/conda"

# ── Config file resolution ─────────────────────────────────────────────────────
CONFIG_FILE="${MODEL}_768x384_lars_phase3.py"
if [ ! -f "$CONFIGS/$CONFIG_FILE" ]; then
    echo "ERROR: Config not found: $CONFIGS/$CONFIG_FILE" >&2
    exit 1
fi

# ── Env setup ──────────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Add conda env lib dir to LD_LIBRARY_PATH so DataLoader worker processes
# can find libicui18n.so.78 → libstdc++ (CXXABI_1.3.15) correctly.
# Without this, worker subprocesses spawned by num_workers>0 fail to import
# sqlite3 (used lazily by CopyPasteObstacles._load_candidates).
CONDA_LIB="/home/farms/miniforge3/envs/macvi2/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# WANDB_DIR must NOT be on PYTHONPATH (else wandb/ cache shadows the package).
export WANDB_DIR="$OUTPUTS/phase3"
mkdir -p "$WANDB_DIR"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set. Run: source $CHALLENGE/phase3/scripts/setup_wandb.sh" >&2
    exit 1
fi

# ── LaRS symlink tree ──────────────────────────────────────────────────────────
LARS_SRC="/home/farms/data/datasets/lars_dataset"
LARS_LINKED="/home/farms/lars_linked"

if [ ! -d "$LARS_LINKED/train/images" ]; then
    echo "Rebuilding LaRS symlink tree at $LARS_LINKED ..."
    for split in train val test; do
        mkdir -p "$LARS_LINKED/$split"
        [ ! -L "$LARS_LINKED/$split/images" ] && \
            ln -sf "$LARS_SRC/img_dir/$split/images" "$LARS_LINKED/$split/images"
        [ ! -L "$LARS_LINKED/$split/semantic_masks" ] && \
            ln -sf "$LARS_SRC/ann_dir/$split/semantic_masks" "$LARS_LINKED/$split/semantic_masks"
        if [ -f "$LARS_SRC/img_dir/$split/image_list.txt" ] && \
           [ ! -L "$LARS_LINKED/$split/image_list.txt" ]; then
            ln -sf "$LARS_SRC/img_dir/$split/image_list.txt" \
                   "$LARS_LINKED/$split/image_list.txt"
        fi
    done
    echo "LaRS symlink tree ready."
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"

echo "========================================================"
echo "Phase 3 — $MODEL — $(date -u +'%Y-%m-%d %H:%M UTC')"
echo "Config:   $CONFIGS/$CONFIG_FILE"
echo "Work dir: $WORK_DIR"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo 'N/A')"
echo "========================================================"

cd "$REPO_ROOT"

"$CONDA_RUN" run --no-capture-output -n macvi2 \
    env PYTHONPATH="$PYTHONPATH" \
        WANDB_API_KEY="$WANDB_API_KEY" \
        WANDB_DIR="$WANDB_DIR" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$MMSEG_TRAIN" --launcher pytorch \
        "$CONFIGS/$CONFIG_FILE" \
        --work-dir "$WORK_DIR" \
        "${@:2}"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Phase 3 — $MODEL DONE ($(date -u +'%H:%M UTC'))"
else
    echo "Phase 3 — $MODEL FAILED exit=$EXIT_CODE ($(date -u +'%H:%M UTC'))"
fi
exit $EXIT_CODE
