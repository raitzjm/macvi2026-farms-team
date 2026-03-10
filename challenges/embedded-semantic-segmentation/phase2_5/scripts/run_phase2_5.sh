#!/usr/bin/env bash
# run_phase2_5.sh — run a single Phase 2.5 model training run.
#
# Usage (from repo root, inside container, with macvi2 env active):
#   source challenges/embedded-semantic-segmentation/phase2_5/scripts/setup_wandb.sh
#   bash challenges/embedded-semantic-segmentation/phase2_5/scripts/run_phase2_5.sh <model>
#
# <model> is one of:
#   stdc2 | pidnets
#
# The script:
#   1. Rebuilds the LaRS symlink tree if not present
#   2. Sets PYTHONPATH and WANDB_DIR
#   3. Activates macvi2 conda env
#   4. Launches torchrun DDP training with mmseg 1.x tools/train.py
#
# Example:
#   bash challenges/embedded-semantic-segmentation/phase2_5/scripts/run_phase2_5.sh stdc2

set -uo pipefail

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    echo "ERROR: No model specified. Usage: run_phase2_5.sh <model>" >&2
    exit 1
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CHALLENGE="challenges/embedded-semantic-segmentation"
CONFIGS="$REPO_ROOT/$CHALLENGE/phase2_5/configs"
OUTPUTS="$REPO_ROOT/$CHALLENGE/outputs"
WORK_DIR="$OUTPUTS/phase2_5/$MODEL"
NUM_GPUS=2

# mmseg 1.x tools path (installed with pip via .mim)
MMSEG_TRAIN="/home/farms/miniforge3/envs/macvi2/lib/python3.11/site-packages/mmseg/.mim/tools/train.py"

# Conda activation
CONDA_RUN="/home/farms/miniforge3/bin/conda"

# ── Config file resolution ────────────────────────────────────────────────────
CONFIG_FILE="${MODEL}_768x384_lars_phase2_5.py"
if [ ! -f "$CONFIGS/$CONFIG_FILE" ]; then
    echo "ERROR: Config not found: $CONFIGS/$CONFIG_FILE" >&2
    exit 1
fi

# ── Env setup ─────────────────────────────────────────────────────────────────
# Expose repo root to Python so custom_imports resolve correctly.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# WANDB_DIR must NOT be on PYTHONPATH (else wandb/ cache dir shadows the package).
export WANDB_DIR="$OUTPUTS/phase2_5"
mkdir -p "$WANDB_DIR"

# Verify WANDB_API_KEY is set (call setup_wandb.sh first).
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set. Run: source $CHALLENGE/phase2_5/scripts/setup_wandb.sh" >&2
    exit 1
fi

# ── LaRS symlink tree ─────────────────────────────────────────────────────────
# Rebuild if not present (container sessions are ephemeral).
LARS_SRC="/home/farms/data/lars_dataset"
LARS_LINKED="/home/farms/lars_linked"

if [ ! -d "$LARS_LINKED/train/images" ]; then
    echo "Rebuilding LaRS symlink tree at $LARS_LINKED ..."
    for split in train val test; do
        mkdir -p "$LARS_LINKED/$split"
        # Images
        if [ ! -L "$LARS_LINKED/$split/images" ]; then
            ln -sf "$LARS_SRC/img_dir/$split/images" "$LARS_LINKED/$split/images"
        fi
        # Semantic masks
        if [ ! -L "$LARS_LINKED/$split/semantic_masks" ]; then
            ln -sf "$LARS_SRC/ann_dir/$split/semantic_masks" "$LARS_LINKED/$split/semantic_masks"
        fi
        # Image list (only present for train and val)
        if [ -f "$LARS_SRC/img_dir/$split/image_list.txt" ] && [ ! -L "$LARS_LINKED/$split/image_list.txt" ]; then
            ln -sf "$LARS_SRC/img_dir/$split/image_list.txt" "$LARS_LINKED/$split/image_list.txt"
        fi
    done
    echo "LaRS symlink tree ready."
fi

# ── Launch ────────────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"

echo "========================================================"
echo "Phase 2.5 — $MODEL — $(date -u +'%Y-%m-%d %H:%M UTC')"
echo "Config:   $CONFIGS/$CONFIG_FILE"
echo "Work dir: $WORK_DIR"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo 'N/A')"
echo "========================================================"

cd "$REPO_ROOT"

"$CONDA_RUN" run --no-capture-output -n macvi2 \
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$MMSEG_TRAIN" --launcher pytorch \
        "$CONFIGS/$CONFIG_FILE" \
        --work-dir "$WORK_DIR" \
        "${@:2}"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Phase 2.5 — $MODEL DONE ($(date -u +'%H:%M UTC'))"
else
    echo "Phase 2.5 — $MODEL FAILED exit=$EXIT_CODE ($(date -u +'%H:%M UTC'))"
fi
exit $EXIT_CODE
