#!/usr/bin/env bash
# Run the macvi-farms container.
# Defaults to Docker; pass --podman to use Podman instead.
# Data directory is mounted read-only at /home/farms/data inside the container.
# Use --claude-dir / --gemini-dir / --opencode-dir to persist AI assistant config across runs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="docker"

# Defaults — all mounts are optional
CLAUDE_DIR=""
CLAUDE_JSON=""
GEMINI_DIR=""
OPENCODE_DIR=""
GITCONFIG=""
SSH_DIR=""
DATA_DIR=""

# Credential overrides — flags take precedence over host env vars
GH_TOKEN_OVERRIDE=""
KAGGLE_USERNAME_OVERRIDE=""
KAGGLE_KEY_OVERRIDE=""

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
        --opencode-dir)
            OPENCODE_DIR="$2"
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
        --gh-token)
            GH_TOKEN_OVERRIDE="$2"
            shift 2
            ;;
        --kaggle-username)
            KAGGLE_USERNAME_OVERRIDE="$2"
            shift 2
            ;;
        --kaggle-key)
            KAGGLE_KEY_OVERRIDE="$2"
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
            echo "  --opencode-dir DIR  Path to OpenCode config directory (~/.local/share/opencode)"
            echo "  --gitconfig FILE    Path to .gitconfig file (for correct git commit attribution)"
            echo "  --ssh-dir DIR       Path to .ssh directory (for git over SSH)"
            echo "  --data-dir DIR      Path to datasets directory (mounted read-only at /home/farms/data)"
            echo "  --gh-token TOKEN    GitHub token (overrides \$MACVI_GH_TOKEN from host env)"
            echo "  --kaggle-username U Kaggle username (overrides \$KAGGLE_USERNAME from host env)"
            echo "  --kaggle-key KEY    Kaggle API key (overrides \$KAGGLE_KEY from host env)"
            echo ""
            echo "All mounts and credentials are optional; omitted ones fall back to host env vars."
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

if [[ -n "$OPENCODE_DIR" ]]; then
    VOLUMES+=(-v "${OPENCODE_DIR}:/home/farms/.local/share/opencode")
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

# Resolve credentials: flag overrides host env var
_GH_TOKEN="${GH_TOKEN_OVERRIDE:-${MACVI_GH_TOKEN}}"
_KAGGLE_USERNAME="${KAGGLE_USERNAME_OVERRIDE:-${KAGGLE_USERNAME}}"
_KAGGLE_KEY="${KAGGLE_KEY_OVERRIDE:-${KAGGLE_KEY}}"

echo "Running with: $RUNTIME (shm-size=${HOST_SHM_BYTES} bytes)"
"$RUNTIME" run -it --rm \
    "${GPU_ARGS[@]}" \
    --network=host \
    --shm-size="${HOST_SHM_BYTES}" \
    --ulimit memlock=-1 \
    -e GH_TOKEN="${_GH_TOKEN}" \
    -e KAGGLE_USERNAME="${_KAGGLE_USERNAME}" \
    -e KAGGLE_KEY="${_KAGGLE_KEY}" \
    "${VOLUMES[@]}" \
    macvi-farms:latest
