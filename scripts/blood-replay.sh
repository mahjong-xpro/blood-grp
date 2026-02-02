#!/bin/bash
# Bloody Battle Mahjong Log Replay Web Service
# Usage: ./blood-replay.sh [port]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_VIEWER_DIR="$PROJECT_DIR/log-viewer"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check if Flask is installed
if ! "$PYTHON_CMD" -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Flask not found. Installing dependencies...${NC}"
    cd "$LOG_VIEWER_DIR" || exit 1
    "$PYTHON_CMD" -m pip install -r requirements.txt
fi

# Port (default 5000)
PORT="${1:-5000}"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use.${NC}"
    echo -e "${YELLOW}Please specify a different port or stop the existing service.${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}Starting Mahjong Log Replay Web Service${NC}"
echo "  Directory: $LOG_VIEWER_DIR"
echo "  Port: $PORT"
echo "  URL: http://localhost:$PORT"
echo ""

cd "$LOG_VIEWER_DIR" || exit 1
exec "$PYTHON_CMD" app.py --host 0.0.0.0 --port "$PORT"
