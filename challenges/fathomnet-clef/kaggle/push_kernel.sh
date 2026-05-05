#!/usr/bin/env bash
# push_kernel.sh — push a Kaggle kernel with WandB key injected at push time.
#
# Usage:
#   ./push_kernel.sh <kernel-dir> --wandb-key $WANDB_API_KEY [--no-follow]
#
# Behavior:
#   1. Copy the kernel dir to a temp dir
#   2. sed-replace __WANDB_KEY_PLACEHOLDER__ with the provided key in *.py
#   3. kaggle kernels push from the temp dir
#   4. trap: always rm -rf the temp dir on exit (no key left on disk)
#
# Why: Kaggle kernels run as `python <code_file>` — no CLI args, no env vars.
# UserSecretsClient requires manual UI setup per machine/kernel (rejected).
# Tracked source contains only the placeholder; pushed Kaggle copy has the key
# (user rotates post-sprint per AGENTS.md guidance).
set -euo pipefail

KERNEL_DIR="${1:?Usage: $0 <kernel-dir> --wandb-key <key> [--no-follow]}"
shift

WANDB_KEY=""
FOLLOW=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb-key) WANDB_KEY="${2:?--wandb-key needs a value}"; shift 2 ;;
        --no-follow) FOLLOW=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$WANDB_KEY" ]] || { echo "missing --wandb-key" >&2; exit 2; }

KERNEL_DIR_ABS="$(cd "$KERNEL_DIR" && pwd)"
SLUG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$KERNEL_DIR_ABS/kernel-metadata.json")
TMP="$(mktemp -d "/tmp/kaggle-push-${SLUG//\//-}-XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

cp -r "$KERNEL_DIR_ABS"/. "$TMP/"

# Substitute placeholder; use | as sed delim to avoid escaping slashes in the key.
find "$TMP" -maxdepth 2 -type f -name '*.py' -print0 \
    | xargs -0 sed -i "s|__WANDB_KEY_PLACEHOLDER__|$WANDB_KEY|g"

# Sanity check: any remaining placeholder means we missed a file.
if grep -RIln '__WANDB_KEY_PLACEHOLDER__' "$TMP" 2>/dev/null | grep -q .; then
    echo "[push] ERROR: placeholder remained after substitution — check kernel .py files" >&2
    exit 3
fi

(cd "$TMP" && kaggle kernels push -p "$(pwd)")
echo "[push] Pushed $SLUG"

if [[ "$FOLLOW" -eq 1 ]]; then
    echo "[push] Following logs for $SLUG (Ctrl-C to stop following, kernel keeps running)"
    kaggle kernels logs "$SLUG" --follow
fi
