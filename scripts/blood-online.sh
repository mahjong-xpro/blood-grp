#!/bin/bash
# One-command orchestration for online training on a single node.
#
# Usage:
#   ./scripts/blood-online.sh up   [--clients N] [--gpu-start N] [--bg|--fg]
#   ./scripts/blood-online.sh down
#   ./scripts/blood-online.sh status
#
# Defaults:
# - clients: 7
# - gpu-start: 1 (leave GPU0 for trainer)
#
# Notes:
# - server + clients are started in background
# - trainer runs in foreground by default (Ctrl+C to stop), unless BLOOD_TRAIN_BG=1 or --bg is set

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLIENTS="${BLOOD_CLIENTS:-7}"
GPU_START="${BLOOD_CLIENT_GPU_START:-1}"
TRAIN_BG="${BLOOD_TRAIN_BG:-0}"

CMD="${1:-}"
shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    --clients) CLIENTS="${2:-}"; shift 2 ;;
    --gpu-start) GPU_START="${2:-}"; shift 2 ;;
    --bg) TRAIN_BG="1"; shift ;;
    --fg) TRAIN_BG="0"; shift ;;
    -h|--help)
      CMD="help"
      shift || true
      ;;
    *)
      echo "Unknown argument: $1" 1>&2
      exit 1
      ;;
  esac
done

case "$CMD" in
  up)
    "$SCRIPT_DIR/blood-server.sh" start || true
    "$SCRIPT_DIR/blood-client.sh" start-many --num "$CLIENTS" --gpu-start "$GPU_START" || true
    if [ "$TRAIN_BG" = "1" ]; then
      "$SCRIPT_DIR/blood-train.sh" start online
      echo "Trainer started in background. Use:"
      echo "  ./scripts/blood-train.sh logs"
    else
      exec "$SCRIPT_DIR/blood-train.sh" online
    fi
    ;;
  down)
    "$SCRIPT_DIR/blood-train.sh" stop || true
    "$SCRIPT_DIR/blood-client.sh" stop-all || true
    "$SCRIPT_DIR/blood-server.sh" stop || true
    ;;
  status)
    "$SCRIPT_DIR/blood-server.sh" status || true
    echo ""
    "$SCRIPT_DIR/blood-train.sh" status || true
    echo ""
    "$SCRIPT_DIR/blood-client.sh" status-all || true
    ;;
  help|"")
    cat <<EOF
Bloody Battle Mahjong Online Training Orchestrator

Usage:
  $0 up     [--clients N] [--gpu-start N] [--bg|--fg]
  $0 down
  $0 status

Environment variables:
  BLOOD_CLIENTS=7
  BLOOD_CLIENT_GPU_START=1
  BLOOD_TRAIN_BG=0   # set 1 to background trainer

Examples:
  $0 up --clients 7 --gpu-start 1
  $0 up --bg
  BLOOD_TRAIN_BG=1 $0 up
EOF
    ;;
  *)
    echo "Unknown command: $CMD (try --help)" 1>&2
    exit 1
    ;;
esac

