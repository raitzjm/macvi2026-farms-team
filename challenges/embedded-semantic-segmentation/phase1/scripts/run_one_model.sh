#!/usr/bin/env bash
# run_one_model.sh — run a single Phase 1 model. Called by the monitoring loop.
# Usage: bash run_one_model.sh <exp_id> <model_name> <config_filename>
# Example: bash run_one_model.sh EXP-001 stdc2 stdc2_768x384_lars_phase1.py

set -uo pipefail

EXP_ID="$1"
MODEL="$2"
CONFIG_FILE="$3"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CHALLENGE="challenges/embedded-semantic-segmentation"
MMSEG="$REPO_ROOT/$CHALLENGE/mmsegmentation-macvi"
CONFIGS="$REPO_ROOT/$CHALLENGE/phase1/configs"
OUTPUTS="$REPO_ROOT/$CHALLENGE/outputs"
WORK_DIR="$OUTPUTS/phase1/$MODEL"
NUM_GPUS=2

# WandB: write cache to outputs dir so it never shadows the wandb package via PYTHONPATH
export WANDB_DIR="$OUTPUTS/phase1"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Load WandB key
WANDB_KEY_FILE="${HOME}/data/wandb_key.txt"
if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "ERROR: WandB key file not found at $WANDB_KEY_FILE" >&2
    exit 1
fi
export WANDB_API_KEY
WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_KEY_FILE")

# Verify symlink tree
if [ ! -d /home/farms/lars_linked/train/images ]; then
    echo "ERROR: /home/farms/lars_linked/ symtree not found. Rebuild it first." >&2
    exit 1
fi

mkdir -p "$WORK_DIR"
mkdir -p "$WANDB_DIR"

echo "========================================================"
echo "$EXP_ID — $MODEL — $(date -u +'%Y-%m-%d %H:%M UTC')"
echo "Config:   $CONFIGS/$CONFIG_FILE"
echo "Work dir: $WORK_DIR"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo 'N/A')"
echo "========================================================"

cd "$REPO_ROOT"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    "$MMSEG/tools/train.py" --launcher pytorch \
    "$CONFIGS/$CONFIG_FILE" --work-dir "$WORK_DIR"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "$EXP_ID — $MODEL DONE ($(date -u +'%H:%M UTC'))"
else
    echo "$EXP_ID — $MODEL FAILED exit=$EXIT_CODE ($(date -u +'%H:%M UTC'))"
fi
exit $EXIT_CODE
