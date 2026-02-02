#!/bin/bash
# Bloody Battle Mahjong Training Script
# Usage:
#   ./blood-train.sh offline|online
#   ./blood-train.sh start offline|online
#   ./blood-train.sh stop|restart|status|logs
# Notes:
# - Training is a foreground process by default.
# - Use "start" to run in background (PID/log in /tmp by default).

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
CONFIG_FILE="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

PID_DIR="${BLOOD_PID_DIR:-/tmp}"
LOG_DIR="${BLOOD_LOG_DIR:-/tmp}"
PID_FILE="$PID_DIR/blood-trainer.pid"
LOG_FILE="$LOG_DIR/blood-trainer.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo ""
    echo "Please create a config.toml file:"
    echo "  cd mortal"
    echo "  cp config.example.toml config.toml"
    echo "  # Edit config.toml with your settings"
    exit 1
fi

# Check if python exists
if ! command -v "$PYTHON_CMD" > /dev/null 2>&1; then
    echo -e "${RED}Error: Python not found: $PYTHON_CMD${NC}"
    echo "Set PYTHON_CMD to a valid python executable (e.g. PYTHON_CMD=python3.12)"
    exit 1
fi

# Check if train.py exists
if [ ! -f "$MORTAL_DIR/train.py" ]; then
    echo -e "${RED}Error: train.py not found: $MORTAL_DIR/train.py${NC}"
    exit 1
fi

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
        rm -f "$PID_FILE" || true
    fi
    return 1
}

stop_trainer() {
    if ! is_running; then
        echo -e "${YELLOW}Trainer is not running${NC}"
        return 1
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    echo -e "${YELLOW}Stopping trainer (PID: $pid)...${NC}"
    kill "$pid" 2>/dev/null || true
    for _ in {1..15}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}Force killing trainer...${NC}"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" || true
    echo -e "${GREEN}✓ Trainer stopped${NC}"
}

status_trainer() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo -e "${GREEN}✓ Trainer is running${NC}"
        echo "  PID: $pid"
        echo "  Log: $LOG_FILE"
        echo ""
        ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || true
        return 0
    fi
    echo -e "${RED}✗ Trainer is not running${NC}"
    return 1
}

logs_trainer() {
    if [ -f "$LOG_FILE" ]; then
        echo "Showing trainer logs (Ctrl+C to exit):"
        echo "---"
        tail -f "$LOG_FILE"
        return 0
    fi
    echo -e "${YELLOW}Log file not found: $LOG_FILE${NC}"
    return 1
}

cfg_online_value() {
    "$PYTHON_CMD" - <<'PY' "$CONFIG_FILE" 2>/dev/null || true
import sys
import toml
cfg = toml.load(sys.argv[1])
v = cfg.get("control", {}).get("online", None)
print("" if v is None else str(v).lower())
PY
}

ensure_mode_matches_config() {
    local mode="$1"
    local v
    v="$(cfg_online_value)"
    if [ -z "$v" ]; then
        return 0
    fi
    if [ "$mode" = "online" ] && [ "$v" != "true" ]; then
        echo -e "${YELLOW}Warning: you selected online mode, but config says [control].online = $v${NC}"
        echo -e "${YELLOW}         Set it to true in $CONFIG_FILE (or set BLOOD_IGNORE_MODE_MISMATCH=1 to ignore).${NC}"
        [ "${BLOOD_IGNORE_MODE_MISMATCH:-0}" = "1" ] || exit 1
    fi
    if [ "$mode" = "offline" ] && [ "$v" = "true" ]; then
        echo -e "${YELLOW}Warning: you selected offline mode, but config says [control].online = true${NC}"
        echo -e "${YELLOW}         Set it to false in $CONFIG_FILE (or set BLOOD_IGNORE_MODE_MISMATCH=1 to ignore).${NC}"
        [ "${BLOOD_IGNORE_MODE_MISMATCH:-0}" = "1" ] || exit 1
    fi
}

run_trainer_foreground() {
    local mode="$1"
    ensure_mode_matches_config "$mode"

    if [ "$mode" = "online" ]; then
        # Check if server is running; if not, start it.
        if [ -f "${BLOOD_PID_DIR:-/tmp}/blood-server.pid" ]; then
            local spid
            spid="$(cat "${BLOOD_PID_DIR:-/tmp}/blood-server.pid")"
            if ps -p "$spid" > /dev/null 2>&1; then
                echo -e "${YELLOW}Training server is already running (PID: $spid)${NC}"
            else
                echo -e "${YELLOW}Starting training server...${NC}"
                "$SCRIPT_DIR/blood-server.sh" start
                sleep 2
            fi
        else
            echo -e "${YELLOW}Starting training server...${NC}"
            "$SCRIPT_DIR/blood-server.sh" start
            sleep 2
        fi

        echo -e "${BLUE}Clients:${NC}"
        echo "  ./scripts/blood-client.sh start-many --num 7 --gpu-start 1"
        echo ""
        if [ -n "${BLOOD_CLIENTS:-}" ]; then
            GPU_START="${BLOOD_CLIENT_GPU_START:-1}"
            echo -e "${YELLOW}Auto-starting clients: BLOOD_CLIENTS=${BLOOD_CLIENTS} (gpu-start=${GPU_START})${NC}"
            "$SCRIPT_DIR/blood-client.sh" start-many --num "$BLOOD_CLIENTS" --gpu-start "$GPU_START" || true
            echo ""
        fi
    fi

    echo -e "${GREEN}Starting trainer (foreground)...${NC}"
    echo "  Config: $CONFIG_FILE"
    echo "  Mode  : $mode"
    echo ""
    cd "$MORTAL_DIR" || exit 1
    exec "$PYTHON_CMD" train.py
}

start_trainer_background() {
    local mode="$1"
    ensure_mode_matches_config "$mode"

    if is_running; then
        echo -e "${YELLOW}Trainer is already running (PID: $(cat "$PID_FILE"))${NC}"
        return 1
    fi

    mkdir -p "$PID_DIR" "$LOG_DIR" || true

    echo -e "${GREEN}Starting trainer (background)...${NC}"
    echo "  Config : $CONFIG_FILE"
    echo "  Mode   : $mode"
    echo "  PID    : $PID_FILE"
    echo "  Log    : $LOG_FILE"
    echo ""

    # In online mode, ensure server is up (same behavior as foreground).
    if [ "$mode" = "online" ]; then
        "$SCRIPT_DIR/blood-server.sh" start || true
    fi

    (
        cd "$MORTAL_DIR"
        export PYTHONUNBUFFERED=1
        nohup "$PYTHON_CMD" train.py >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
    )

    sleep 1
    if is_running; then
        echo -e "${GREEN}✓ Trainer started successfully (PID: $(cat "$PID_FILE"))${NC}"
        return 0
    fi
    echo -e "${RED}✗ Trainer failed to start. Check: $LOG_FILE${NC}"
    rm -f "$PID_FILE" || true
    return 1
}

# Determine training mode
CMD="${1:-offline}"
MODE="${2:-}"

case "$CMD" in
    offline|online)
        run_trainer_foreground "$CMD"
        ;;
    start)
        [ -n "$MODE" ] || { echo -e "${RED}Error: start requires offline|online${NC}"; exit 1; }
        start_trainer_background "$MODE"
        ;;
    stop)
        stop_trainer
        ;;
    restart)
        [ -n "$MODE" ] || { echo -e "${RED}Error: restart requires offline|online${NC}"; exit 1; }
        stop_trainer || true
        start_trainer_background "$MODE"
        ;;
    status)
        status_trainer
        ;;
    logs)
        logs_trainer
        ;;
    *)
        echo "Bloody Battle Mahjong Training Script"
        echo ""
        echo "Usage:"
        echo "  $0 offline|online"
        echo "  $0 start offline|online"
        echo "  $0 stop|restart offline|online|status|logs"
        echo ""
        echo "Environment variables:"
        echo "  MORTAL_CFG   - Path to config.toml (default: mortal/config.toml)"
        echo "  PYTHON_CMD   - Python command (default: python3)"
        echo "  BLOOD_PID_DIR - PID dir (default: /tmp)"
        echo "  BLOOD_LOG_DIR - Log dir (default: /tmp)"
        echo "  BLOOD_CLIENTS - (online only) auto-start N clients via blood-client.sh"
        echo "  BLOOD_CLIENT_GPU_START - (online only) first GPU index for auto-started clients (default: 1)"
        echo "  BLOOD_IGNORE_MODE_MISMATCH=1 - ignore config [control].online mismatch"
        echo ""
        echo "Examples:"
        echo "  $0 offline"
        echo "  $0 online"
        echo "  $0 start online   # run trainer in background"
        echo "  $0 logs           # follow background log"
        exit 1
        ;;
esac
