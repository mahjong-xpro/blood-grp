#!/bin/bash
# Bloody Battle Mahjong TensorBoard Launch Script
# Usage: ./blood-tensorboard.sh [tensorboard_dir] [port]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
CONFIG_FILE="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
TENSORBOARD_CMD="${TENSORBOARD_CMD:-tensorboard}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if tensorboard is available
if ! command -v "$TENSORBOARD_CMD" &> /dev/null; then
    echo -e "${RED}Error: TensorBoard not found.${NC}"
    echo ""
    echo "Please install TensorBoard:"
    echo "  pip install tensorboard"
    exit 1
fi

# Check python (used for config parsing in other scripts; keep consistent)
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo -e "${RED}Error: Python not found: $PYTHON_CMD${NC}"
    exit 1
fi

# Try to read tensorboard_dir from config file
TENSORBOARD_DIR=""
if [ -f "$CONFIG_FILE" ]; then
    # Try to extract tensorboard_dir from config.toml using grep/sed
    TENSORBOARD_DIR=$(grep -E "^tensorboard_dir\s*=" "$CONFIG_FILE" 2>/dev/null | sed -E "s/^tensorboard_dir\s*=\s*['\"](.*)['\"]\s*$/\1/" | head -1 || echo "")
    # If not found, try without quotes
    if [ -z "$TENSORBOARD_DIR" ]; then
        TENSORBOARD_DIR=$(grep -E "^tensorboard_dir\s*=" "$CONFIG_FILE" 2>/dev/null | sed -E "s/^tensorboard_dir\s*=\s*(.*)\s*$/\1/" | head -1 || echo "")
    fi
fi

# Override with command line argument if provided
if [ -n "$1" ]; then
    TENSORBOARD_DIR="$1"
fi

# Default tensorboard directory if not specified
if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR="$MORTAL_DIR/logs/tensorboard"
    echo -e "${YELLOW}Warning: tensorboard_dir not found in config or specified.${NC}"
    echo -e "${YELLOW}Using default: $TENSORBOARD_DIR${NC}"
    echo ""
fi

# Check if directory exists
if [ ! -d "$TENSORBOARD_DIR" ]; then
    echo -e "${YELLOW}Warning: TensorBoard directory does not exist: $TENSORBOARD_DIR${NC}"
    echo -e "${YELLOW}Creating directory...${NC}"
    mkdir -p "$TENSORBOARD_DIR"
fi

# Port (default 6006)
PORT="${2:-6006}"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use.${NC}"
    echo -e "${YELLOW}Please specify a different port or stop the existing TensorBoard instance.${NC}"
    echo ""
    echo "To stop existing TensorBoard:"
    echo "  lsof -ti:$PORT | xargs kill"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}Starting TensorBoard${NC}"
echo "  Directory: $TENSORBOARD_DIR"
echo "  Port: $PORT"
echo "  URL: http://localhost:$PORT"
echo ""

cd "$PROJECT_DIR" || exit 1

# Start TensorBoard
exec "$TENSORBOARD_CMD" --logdir="$TENSORBOARD_DIR" --port="$PORT" --host=0.0.0.0
