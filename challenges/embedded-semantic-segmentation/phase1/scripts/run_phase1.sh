#!/usr/bin/env bash
# run_phase1.sh — run all 4 Phase 1 baseline training runs sequentially.
#
# Usage (from repo root, inside container, with macvi env active):
#   source challenges/embedded-semantic-segmentation/phase1/scripts/setup_wandb.sh
#   bash challenges/embedded-semantic-segmentation/phase1/scripts/run_phase1.sh 2>&1 | tee phase1_training.log
#
# Runs: EXP-001 (stdc2) → EXP-002 (bisenetv2) → EXP-003 (segnext_t) → EXP-004 (segformer_mit-b2)

set -uo pipefail   # -e removed: individual model failures must not abort remaining runs

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CHALLENGE="challenges/embedded-semantic-segmentation"
MMSEG="$CHALLENGE/mmsegmentation-macvi"
CONFIGS="$CHALLENGE/phase1/configs"
OUTPUTS="$CHALLENGE/outputs"
NUM_GPUS=2

echo "========================================================"
echo "Phase 1 training — $(date -u +'%Y-%m-%d %H:%M UTC')"
echo "Repo root: $REPO_ROOT"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "NUM_GPUS: $NUM_GPUS"
echo "========================================================"

cd "$REPO_ROOT"

# Expose repo root to Python so custom_imports (challenges.*.pipelines) resolves
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep wandb local cache out of the repo root so it never shadows the wandb package
export WANDB_DIR="$REPO_ROOT/$OUTPUTS/phase1"

# Verify WANDB_API_KEY is set
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set. Run: source $CHALLENGE/phase1/scripts/setup_wandb.sh"
    exit 1
fi

# Verify symlink tree exists
if [ ! -d /home/farms/lars_linked/train/images ]; then
    echo "ERROR: /home/farms/lars_linked/ symtree not found. Rebuild it first."
    exit 1
fi

run_model() {
    local exp_id="$1"
    local model="$2"
    local config="$3"
    local work_dir="$OUTPUTS/phase1/$model"

    echo ""
    echo "--------------------------------------------------------"
    echo "$exp_id — $model — $(date -u +'%H:%M UTC')"
    echo "Config:   $config"
    echo "Work dir: $work_dir"
    echo "--------------------------------------------------------"
    mkdir -p "$work_dir"

    bash "$MMSEG/tools/dist_train.sh" "$CONFIGS/$config" "$NUM_GPUS" --work-dir "$work_dir"
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "$exp_id — $model DONE ($(date -u +'%H:%M UTC'))"
    else
        echo "$exp_id — $model FAILED with exit code $exit_code ($(date -u +'%H:%M UTC'))"
    fi
    return $exit_code
}

FAILED=()
run_model "EXP-001" "stdc2"            "stdc2_768x384_lars_phase1.py"            || FAILED+=("EXP-001 stdc2")
run_model "EXP-002" "bisenetv2"        "bisenetv2_768x384_lars_phase1.py"        || FAILED+=("EXP-002 bisenetv2")
run_model "EXP-003" "segnext_t"        "segnext_t_768x384_lars_phase1.py"        || FAILED+=("EXP-003 segnext_t")
run_model "EXP-004" "segformer_mit-b2" "segformer_mit-b2_768x384_lars_phase1.py" || FAILED+=("EXP-004 segformer_mit-b2")

echo ""
echo "========================================================"
echo "All Phase 1 runs complete — $(date -u +'%Y-%m-%d %H:%M UTC')"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED runs: ${FAILED[*]}"
else
    echo "All runs succeeded."
fi
echo "========================================================"
