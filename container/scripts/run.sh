#!/usr/bin/env bash
# Run the macvi-farms container.
# Defaults to Docker; pass --podman to use Podman instead.
# Data directory is mounted read-only at /home/farms/data inside the container.
# Use --claude-dir / --gemini-dir / --codex-dir to persist AI assistant config across runs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="docker"

# Defaults — all mounts are optional
CLAUDE_DIR=""
CLAUDE_JSON=""
GEMINI_DIR=""
CODEX_DIR=""
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
        --claude-json)
            CLAUDE_JSON="$2"
            shift 2
            ;;
        --gemini-dir)
            GEMINI_DIR="$2"
            shift 2
            ;;
        --codex-dir)
            CODEX_DIR="$2"
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
            echo "  --claude-json FILE  Path to .claude.json file (mounted at /home/farms/.claude.json)"
            echo "  --gemini-dir DIR    Path to .gemini directory (for Gemini CLI / agentic work)"
            echo "  --codex-dir DIR     Path to .codex directory (for OpenAI Codex CLI / agentic work)"
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

if [[ -n "$CLAUDE_JSON" ]]; then
    VOLUMES+=(-v "${CLAUDE_JSON}:/home/farms/.claude.json")
fi

if [[ -n "$GEMINI_DIR" ]]; then
    VOLUMES+=(-v "${GEMINI_DIR}:/home/farms/.gemini")
fi

if [[ -n "$CODEX_DIR" ]]; then
    VOLUMES+=(-v "${CODEX_DIR}:/home/farms/.codex")
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

HOST_SHM_BYTES=$(df --block-size=1 /dev/shm | awk 'NR==2{print $2}')

echo "Running with: $RUNTIME (shm-size=${HOST_SHM_BYTES} bytes)"
"$RUNTIME" run -it --rm \
    "${GPU_ARGS[@]}" \
    --network=host \
    --shm-size="${HOST_SHM_BYTES}" \
    --ulimit memlock=-1 \
    -e GH_TOKEN="${MACVI_GH_TOKEN}" \
    "${VOLUMES[@]}" \
    macvi-farms:latest
