#!/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
podman build -t macvi-farms:latest -f  ${SCRIPT_DIR}/../../container/Dockerfile  
