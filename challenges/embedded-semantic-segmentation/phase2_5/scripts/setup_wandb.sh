#!/usr/bin/env bash
# setup_wandb.sh — load WANDB_API_KEY from key file into the environment.
#
# Source this script before running any training command:
#   source challenges/embedded-semantic-segmentation/phase2_5/scripts/setup_wandb.sh
#
# The key file must exist at ~/data/wandb_key.txt (mounted read-only dataset dir).
# The key is NEVER written to any config or committed to the repo.

WANDB_KEY_FILE="${HOME}/data/wandb_key.txt"

if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "ERROR: WandB key file not found at $WANDB_KEY_FILE" >&2
    return 1 2>/dev/null || exit 1
fi

export WANDB_API_KEY
WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_KEY_FILE")

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WandB key file is empty: $WANDB_KEY_FILE" >&2
    return 1 2>/dev/null || exit 1
fi

echo "WANDB_API_KEY loaded (${#WANDB_API_KEY} chars). WandB logging is active."
