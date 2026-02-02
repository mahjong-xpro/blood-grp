#!/bin/bash
# Bloody Battle Mahjong Training Client Management Script
# Usage:
#   ./scripts/blood-client.sh start        [--id N] [--gpu N] [--log-dir DIR] [--base-cfg FILE]
#   ./scripts/blood-client.sh stop         [--id N]
#   ./scripts/blood-client.sh restart      [--id N] [--gpu N] [--log-dir DIR] [--base-cfg FILE]
#   ./scripts/blood-client.sh status       [--id N]
#   ./scripts/blood-client.sh logs         [--id N]
#   ./scripts/blood-client.sh start-many   --num N [--gpu-start N] [--id-start N] [--log-root DIR] [--base-cfg FILE]
#   ./scripts/blood-client.sh stop-all
#   ./scripts/blood-client.sh status-all
#
# Notes:
# - Each client MUST use a unique log_dir, otherwise they will delete each other's self-play dir.
# - This script auto-generates a per-client config in /tmp and a unique TRAIN_PLAY_PROFILE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"

PYTHON_CMD="${PYTHON_CMD:-python3}"
BASE_CFG_DEFAULT="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"

PID_DIR="${BLOOD_PID_DIR:-/tmp}"
LOG_DIR="${BLOOD_LOG_DIR:-/tmp}"
CFG_DIR="${BLOOD_CFG_DIR:-/tmp}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
  cat <<EOF
Bloody Battle Mahjong Training Client Management

Usage:
  $0 start      [--id N] [--gpu N] [--log-dir DIR] [--base-cfg FILE]
  $0 stop       [--id N]
  $0 restart    [--id N] [--gpu N] [--log-dir DIR] [--base-cfg FILE]
  $0 status     [--id N]
  $0 logs       [--id N]

  $0 start-many --num N [--gpu-start N] [--id-start N] [--log-root DIR] [--base-cfg FILE]
  $0 stop-all
  $0 status-all

Examples (typical single-node 8xGPU):
  # Start 7 clients on GPU1..GPU7 (leave GPU0 for trainer)
  $0 start-many --num 7 --gpu-start 1

  # Start one client on GPU3
  $0 start --id 3 --gpu 3

Environment variables:
  MORTAL_CFG        - Base config.toml path (default: mortal/config.toml)
  PYTHON_CMD        - Python command (default: python3)
  BLOOD_PID_DIR     - PID dir (default: /tmp)
  BLOOD_LOG_DIR     - Log dir (default: /tmp)
  BLOOD_CFG_DIR     - Generated config dir (default: /tmp)
EOF
}

die() {
  echo -e "${RED}Error: $*${NC}" 1>&2
  exit 1
}

ensure_python() {
  command -v "$PYTHON_CMD" >/dev/null 2>&1 || die "Python not found: $PYTHON_CMD"
}

pid_file() {
  local id="$1"
  echo "$PID_DIR/blood-client-$id.pid"
}

log_file() {
  local id="$1"
  echo "$LOG_DIR/blood-client-$id.log"
}

generated_cfg_file() {
  local id="$1"
  echo "$CFG_DIR/blood-client-$id.toml"
}

client_profile() {
  local id="$1"
  echo "client$id"
}

is_running() {
  local id="$1"
  local pf
  pf="$(pid_file "$id")"
  if [ -f "$pf" ]; then
    local pid
    pid="$(cat "$pf")"
    if ps -p "$pid" >/dev/null 2>&1; then
      return 0
    fi
    rm -f "$pf" || true
  fi
  return 1
}

infer_default_log_dir() {
  local base_cfg="$1"
  "$PYTHON_CMD" - <<'PY' "$base_cfg"
import sys
import toml
cfg_path = sys.argv[1]
cfg = toml.load(cfg_path)
tp = cfg.get("train_play", {})
default = tp.get("default", {})
log_dir = default.get("log_dir", "")
if not isinstance(log_dir, str):
    log_dir = str(log_dir)
print(log_dir)
PY
}

generate_client_cfg() {
  local id="$1"
  local base_cfg="$2"
  local profile="$3"
  local log_dir="$4"
  local out_cfg="$5"

  "$PYTHON_CMD" - <<'PY' "$base_cfg" "$profile" "$log_dir" "$out_cfg"
import sys
import copy
import toml

base_cfg, profile, log_dir, out_cfg = sys.argv[1:]
cfg = toml.load(base_cfg)
tp = cfg.setdefault("train_play", {})
default = tp.get("default")
if default is None:
    raise SystemExit("config missing [train_play.default]")

tp[profile] = copy.deepcopy(default)
tp[profile]["log_dir"] = log_dir

with open(out_cfg, "w", encoding="utf-8") as f:
    toml.dump(cfg, f)
print(out_cfg)
PY
}

start_one() {
  local id="$1"
  local gpu="$2"       # empty allowed
  local log_dir="$3"   # empty allowed
  local base_cfg="$4"

  ensure_python
  [ -d "$MORTAL_DIR" ] || die "mortal dir not found: $MORTAL_DIR"
  [ -f "$MORTAL_DIR/client.py" ] || die "client.py not found: $MORTAL_DIR/client.py"
  [ -f "$base_cfg" ] || die "Config file not found: $base_cfg"

  local pf lf cfgf profile
  pf="$(pid_file "$id")"
  lf="$(log_file "$id")"
  cfgf="$(generated_cfg_file "$id")"
  profile="$(client_profile "$id")"

  if is_running "$id"; then
    echo -e "${YELLOW}Client $id is already running (PID: $(cat "$pf"))${NC}"
    return 1
  fi

  if [ -z "$log_dir" ]; then
    local default_log_dir
    default_log_dir="$(infer_default_log_dir "$base_cfg")"
    if [ -n "$default_log_dir" ]; then
      log_dir="$default_log_dir/$profile"
    else
      log_dir="/tmp/mortal-$profile"
    fi
  fi

  mkdir -p "$(dirname "$log_dir")" || true
  mkdir -p "$PID_DIR" "$LOG_DIR" "$CFG_DIR" || true

  generate_client_cfg "$id" "$base_cfg" "$profile" "$log_dir" "$cfgf" >/dev/null

  echo -e "${GREEN}Starting training client $id${NC}"
  echo "  Base config : $base_cfg"
  echo "  Client cfg  : $cfgf"
  echo "  Profile     : $profile"
  echo "  Log dir     : $log_dir"
  if [ -n "$gpu" ]; then
    echo "  GPU         : $gpu (CUDA_VISIBLE_DEVICES)"
  else
    echo "  GPU         : (inherit)"
  fi
  echo "  Log file    : $lf"

  (
    cd "$MORTAL_DIR"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export TRAIN_PLAY_PROFILE="$profile"
    export MORTAL_CFG="$cfgf"
    if [ -n "$gpu" ]; then
      export CUDA_VISIBLE_DEVICES="$gpu"
    fi
    nohup "$PYTHON_CMD" client.py >> "$lf" 2>&1 &
    echo $! > "$pf"
  )

  sleep 1
  if is_running "$id"; then
    echo -e "${GREEN}✓ Client $id started successfully (PID: $(cat "$pf"))${NC}"
    return 0
  fi
  echo -e "${RED}✗ Client $id failed to start. Check: $lf${NC}"
  rm -f "$pf" || true
  return 1
}

stop_one() {
  local id="$1"
  local pf
  pf="$(pid_file "$id")"
  if ! is_running "$id"; then
    echo -e "${YELLOW}Client $id is not running${NC}"
    return 1
  fi

  local pid
  pid="$(cat "$pf")"
  echo -e "${YELLOW}Stopping client $id (PID: $pid)...${NC}"
  kill "$pid" 2>/dev/null || true

  for _ in {1..10}; do
    if ! ps -p "$pid" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ps -p "$pid" >/dev/null 2>&1; then
    echo -e "${YELLOW}Force killing client $id...${NC}"
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$pf" || true
  echo -e "${GREEN}✓ Client $id stopped${NC}"
}

status_one() {
  local id="$1"
  local pf lf
  pf="$(pid_file "$id")"
  lf="$(log_file "$id")"

  if is_running "$id"; then
    local pid
    pid="$(cat "$pf")"
    echo -e "${GREEN}✓ Client $id is running${NC}"
    echo "  PID: $pid"
    echo "  Log: $lf"
    echo ""
    ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || true
    return 0
  fi
  echo -e "${RED}✗ Client $id is not running${NC}"
  return 1
}

logs_one() {
  local id="$1"
  local lf
  lf="$(log_file "$id")"
  if [ -f "$lf" ]; then
    echo "Showing client $id logs (Ctrl+C to exit):"
    echo "---"
    tail -f "$lf"
    return 0
  fi
  echo -e "${YELLOW}Log file not found: $lf${NC}"
  return 1
}

start_many() {
  local num="$1"
  local gpu_start="$2"
  local id_start="$3"
  local log_root="$4"
  local base_cfg="$5"

  [[ "$num" =~ ^[0-9]+$ ]] || die "--num must be an integer"
  [[ "$gpu_start" =~ ^[0-9]+$ ]] || die "--gpu-start must be an integer"
  [[ "$id_start" =~ ^[0-9]+$ ]] || die "--id-start must be an integer"

  for ((i=0; i<num; i++)); do
    local id gpu log_dir
    id=$((id_start + i))
    gpu=$((gpu_start + i))
    if [ -n "$log_root" ]; then
      log_dir="$log_root/$(client_profile "$id")"
    else
      log_dir=""
    fi
    start_one "$id" "$gpu" "$log_dir" "$base_cfg" || true
  done
}

stop_all() {
  shopt -s nullglob
  local files=("$PID_DIR"/blood-client-*.pid)
  if [ ${#files[@]} -eq 0 ]; then
    echo -e "${YELLOW}No client PID files found in $PID_DIR${NC}"
    return 0
  fi
  for pf in "${files[@]}"; do
    local id
    id="$(basename "$pf" | sed -E 's/^blood-client-([0-9]+)\.pid$/\1/')"
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      stop_one "$id" || true
    fi
  done
}

status_all() {
  shopt -s nullglob
  local files=("$PID_DIR"/blood-client-*.pid)
  if [ ${#files[@]} -eq 0 ]; then
    echo -e "${YELLOW}No client PID files found in $PID_DIR${NC}"
    return 0
  fi
  local any_running=0
  for pf in "${files[@]}"; do
    local id
    id="$(basename "$pf" | sed -E 's/^blood-client-([0-9]+)\.pid$/\1/')"
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      if is_running "$id"; then
        any_running=1
        echo -e "${GREEN}✓ Client $id is running (PID: $(cat "$(pid_file "$id")"))${NC}"
      else
        echo -e "${RED}✗ Client $id is not running${NC}"
      fi
    fi
  done
  [ "$any_running" -eq 1 ] && return 0
  return 1
}

# -------------------------
# CLI parsing
# -------------------------

CMD="${1:-}"
shift || true

ID="0"
GPU=""
LOG_DIR_ARG=""
LOG_ROOT=""
BASE_CFG="$BASE_CFG_DEFAULT"
NUM=""
GPU_START="1"
ID_START="1"

while [ $# -gt 0 ]; do
  case "$1" in
    --id) ID="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --log-dir) LOG_DIR_ARG="${2:-}"; shift 2 ;;
    --log-root) LOG_ROOT="${2:-}"; shift 2 ;;
    --base-cfg) BASE_CFG="${2:-}"; shift 2 ;;
    --num) NUM="${2:-}"; shift 2 ;;
    --gpu-start) GPU_START="${2:-}"; shift 2 ;;
    --id-start) ID_START="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (try --help)" ;;
  esac
done

case "$CMD" in
  start)
    [[ "$ID" =~ ^[0-9]+$ ]] || die "--id must be an integer"
    start_one "$ID" "$GPU" "$LOG_DIR_ARG" "$BASE_CFG"
    ;;
  stop)
    [[ "$ID" =~ ^[0-9]+$ ]] || die "--id must be an integer"
    stop_one "$ID"
    ;;
  restart)
    [[ "$ID" =~ ^[0-9]+$ ]] || die "--id must be an integer"
    stop_one "$ID" || true
    start_one "$ID" "$GPU" "$LOG_DIR_ARG" "$BASE_CFG"
    ;;
  status)
    [[ "$ID" =~ ^[0-9]+$ ]] || die "--id must be an integer"
    status_one "$ID"
    ;;
  logs)
    [[ "$ID" =~ ^[0-9]+$ ]] || die "--id must be an integer"
    logs_one "$ID"
    ;;
  start-many)
    [ -n "$NUM" ] || die "start-many requires --num N"
    start_many "$NUM" "$GPU_START" "$ID_START" "$LOG_ROOT" "$BASE_CFG"
    ;;
  stop-all)
    stop_all
    ;;
  status-all)
    status_all
    ;;
  ""|-h|--help)
    usage
    exit 0
    ;;
  *)
    die "Unknown command: $CMD (try --help)"
    ;;
esac

