#!/usr/bin/env bash
# Build the macvi-farms container image.
# Defaults to Docker; pass --podman to use Podman instead.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="docker"

while [[ $# -gt 0 ]]; do
    case $1 in
        --podman)
            RUNTIME="podman"
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--podman]"
            echo ""
            echo "Options:"
            echo "  --podman    Use Podman instead of Docker"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Building with: $RUNTIME"
"$RUNTIME" build -t macvi-farms:latest -f "${SCRIPT_DIR}/../../container/Dockerfile" "${SCRIPT_DIR}/../../"
