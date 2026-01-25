#!/bin/bash
# Bloody Battle Mahjong Training Script
# Usage: ./blood-train.sh [offline|online]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
CONFIG_FILE="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

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

# Check if train.py exists
if [ ! -f "$MORTAL_DIR/train.py" ]; then
    echo -e "${RED}Error: train.py not found: $MORTAL_DIR/train.py${NC}"
    exit 1
fi

# Determine training mode
MODE="${1:-offline}"

case "$MODE" in
    offline)
        echo -e "${GREEN}Starting Offline Training${NC}"
        echo "  Config: $CONFIG_FILE"
        echo "  Mode: Offline (using existing dataset)"
        echo ""
        
        cd "$MORTAL_DIR" || exit 1
        exec "$PYTHON_CMD" train.py
        ;;
    
    online)
        echo -e "${GREEN}Starting Online Training${NC}"
        echo "  Config: $CONFIG_FILE"
        echo "  Mode: Online (distributed training)"
        echo ""
        
        # Check if server is running
        if [ -f "/tmp/blood-server.pid" ]; then
            PID=$(cat "/tmp/blood-server.pid")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${YELLOW}Training server is already running (PID: $PID)${NC}"
            else
                echo -e "${YELLOW}Starting training server...${NC}"
                "$SCRIPT_DIR/blood-server.sh" start
                sleep 3
            fi
        else
            echo -e "${YELLOW}Starting training server...${NC}"
            "$SCRIPT_DIR/blood-server.sh" start
            sleep 3
        fi
        
        echo -e "${BLUE}Note: You need to start training clients separately:${NC}"
        echo "  cd mortal"
        echo "  python client.py"
        echo ""
        echo -e "${GREEN}Starting training main process...${NC}"
        
        cd "$MORTAL_DIR" || exit 1
        exec "$PYTHON_CMD" train.py
        ;;
    
    *)
        echo "Bloody Battle Mahjong Training Script"
        echo ""
        echo "Usage: $0 [offline|online]"
        echo ""
        echo "Modes:"
        echo "  offline  - Offline training using existing dataset (default)"
        echo "  online   - Online training with distributed clients"
        echo ""
        echo "Examples:"
        echo "  $0              # Start offline training"
        echo "  $0 offline      # Start offline training"
        echo "  $0 online       # Start online training"
        echo ""
        echo "Environment variables:"
        echo "  MORTAL_CFG   - Path to config.toml (default: mortal/config.toml)"
        echo "  PYTHON_CMD   - Python command (default: python3)"
        echo ""
        echo "For online training, you also need to:"
        echo "  1. Start training server: ./scripts/blood-server.sh start"
        echo "  2. Start training clients: cd mortal && python client.py"
        echo "  3. Start training main: $0 online"
        exit 1
        ;;
esac
