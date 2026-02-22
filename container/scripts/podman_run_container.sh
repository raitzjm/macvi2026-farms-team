#!/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
CLAUDE_DIR="${HOME}/.claude"
GITCONFIG="${HOME}/.gitconfig"
SSH_DIR=""
DATA_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "  --claude-dir DIR    Path to .claude directory (default: ~/.claude)"
            echo "  --gitconfig FILE    Path to .gitconfig file (default: ~/.gitconfig)"
            echo "  --ssh-dir DIR       Path to .ssh directory (mounted read-only, not mounted if omitted)"
            echo "  --data-dir DIR      Path to datasets directory (mounted at /home/farms/data)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build volume arguments
VOLUMES=(
    -v "${SCRIPT_DIR}/../../:/home/farms/macvi2026-farms-team"
    -v "${CLAUDE_DIR}:/home/farms/.claude"
    -v "${GITCONFIG}:/home/farms/.gitconfig:ro"
)

if [[ -n "$SSH_DIR" ]]; then
    VOLUMES+=(-v "${SSH_DIR}:/home/farms/.ssh:ro")
fi

if [[ -n "$DATA_DIR" ]]; then
    VOLUMES+=(-v "${DATA_DIR}:/home/farms/macvi2026-farms-team/data:ro")
fi

podman run -it --rm --userns=keep-id \
    --device=nvidia.com/gpu=all \
    --security-opt=label=disable \
    --network=host \
    -e GH_TOKEN="${MACVI_GH_TOKEN}" \
    "${VOLUMES[@]}" \
    macvi-farms:latest