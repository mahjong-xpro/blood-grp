#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

python -m blood.training.runner \
    --cfg configs/default.yaml \
    "$@"
