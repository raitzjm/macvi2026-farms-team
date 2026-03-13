#!/usr/bin/env bash
# run_qat.sh — run MMRazor QAT fine-tuning for Phase 3.
#
# Usage:
#   source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
#   bash challenges/embedded-semantic-segmentation/phase3/scripts/run_qat.sh <model_type>
#
# <model_type> is one of:
#   segnext_t | segnext_t_uperhead | pidnets | pidnets_p2_5

set -uo pipefail

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    echo "ERROR: No model specified. Usage: run_qat.sh <model>" >&2
    exit 1
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CHALLENGE="challenges/embedded-semantic-segmentation"
CONFIGS="$REPO_ROOT/$CHALLENGE/phase3/configs"
OUTPUTS="$REPO_ROOT/$CHALLENGE/outputs"
WORK_DIR="$OUTPUTS/phase3/$MODEL"
NUM_GPUS=1 # Single GPU for local hardware constraint

# Custom train tool path
TRAIN_SCRIPT="$REPO_ROOT/$CHALLENGE/phase3/scripts/train_qat.py"

# Conda activation
CONDA_RUN="/home/farms/miniforge3/bin/conda"

# ── Config file resolution ────────────────────────────────────────────────────
CONFIG_FILE="${MODEL}_768x384_lars_qat.py"
if [ ! -f "$CONFIGS/$CONFIG_FILE" ]; then
    echo "ERROR: Config not found: $CONFIGS/$CONFIG_FILE" >&2
    exit 1
fi

# ── Env setup ─────────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$OUTPUTS/phase3"
mkdir -p "$WANDB_DIR"

# Verify WANDB_API_KEY is set.
# if [ -z "${WANDB_API_KEY:-}" ]; then
#     echo "ERROR: WANDB_API_KEY not set." >&2
#     exit 1
# fi

# ── Launch ────────────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"

echo "========================================================"
echo "Phase 3 QAT — $MODEL — $(date -u +'%Y-%m-%d %H:%M UTC')"
echo "Config:   $CONFIGS/$CONFIG_FILE"
echo "Work dir: $WORK_DIR"
echo "========================================================"

cd "$REPO_ROOT"

"$CONDA_RUN" run --no-capture-output -n macvi2 \
    python "$TRAIN_SCRIPT" \
        "$CONFIGS/$CONFIG_FILE" \
        --work-dir "$WORK_DIR" \
        "${@:2}"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Phase 3 QAT — $MODEL DONE"
else
    echo "Phase 3 QAT — $MODEL FAILED exit=$EXIT_CODE"
fi
exit $EXIT_CODE
