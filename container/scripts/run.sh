#!/usr/bin/env bash
# Run the macvi-farms container.
# Defaults to Docker; pass --podman to use Podman instead.
# Data directory is mounted read-only at /home/farms/data inside the container.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="docker"

# Defaults — all mounts are optional
CLAUDE_DIR=""
GITCONFIG=""
SSH_DIR=""
DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --podman)
            RUNTIME="podman"
            shift
            ;;
        --claude-dir)
            CLAUDE_DIR="$2"
            shift 2
            ;;
        --gitconfig)
            GITCONFIG="$2"
            shift 2
            ;;
        --ssh-dir)
            SSH_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --podman            Use Podman instead of Docker"
            echo "  --claude-dir DIR    Path to .claude directory (for Claude Code / agentic work)"
            echo "  --gitconfig FILE    Path to .gitconfig file (for correct git commit attribution)"
            echo "  --ssh-dir DIR       Path to .ssh directory (for git over SSH)"
            echo "  --data-dir DIR      Path to datasets directory (mounted read-only at /home/farms/data)"
            echo ""
            echo "All mounts are optional; omitted ones are simply not mounted."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

VOLUMES=(-v "${SCRIPT_DIR}/../../:/home/farms/macvi2026-farms-team")

if [[ -n "$CLAUDE_DIR" ]]; then
    VOLUMES+=(-v "${CLAUDE_DIR}:/home/farms/.claude")
fi

if [[ -n "$GITCONFIG" ]]; then
    VOLUMES+=(-v "${GITCONFIG}:/home/farms/.gitconfig:ro")
fi

if [[ -n "$SSH_DIR" ]]; then
    VOLUMES+=(-v "${SSH_DIR}:/home/farms/.ssh:ro")
fi

if [[ -n "$DATA_DIR" ]]; then
    VOLUMES+=(-v "${DATA_DIR}:/home/farms/data:ro")
fi

if [[ "$RUNTIME" == "podman" ]]; then
    GPU_ARGS=(--device=nvidia.com/gpu=all --security-opt=label=disable --userns=keep-id)
else
    GPU_ARGS=(--gpus all)
fi

echo "Running with: $RUNTIME"
"$RUNTIME" run -it --rm \
    "${GPU_ARGS[@]}" \
    --network=host \
    -e GH_TOKEN="${MACVI_GH_TOKEN}" \
    "${VOLUMES[@]}" \
    macvi-farms:latest
