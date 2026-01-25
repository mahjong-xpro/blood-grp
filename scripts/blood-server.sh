#!/bin/bash
# Bloody Battle Mahjong Training Server Management Script
# Usage: ./blood-server.sh {start|stop|restart|status|logs}

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
CONFIG_FILE="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
PID_FILE="/tmp/blood-server.pid"
LOG_FILE="/tmp/blood-server.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Start the server
start() {
    if is_running; then
        echo -e "${YELLOW}Server is already running (PID: $(cat $PID_FILE))${NC}"
        return 1
    fi

    echo -e "${GREEN}Starting Bloody Battle Mahjong Training Server...${NC}"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        echo "Please create a config.toml file or set MORTAL_CFG environment variable"
        return 1
    fi

    # Check if server.py exists
    if [ ! -f "$MORTAL_DIR/server.py" ]; then
        echo -e "${RED}Error: server.py not found: $MORTAL_DIR/server.py${NC}"
        return 1
    fi

    # Change to mortal directory
    cd "$MORTAL_DIR" || exit 1

    # Start server in background
    echo "Starting server..."
    echo "  Config: $CONFIG_FILE"
    echo "  Python: $PYTHON_CMD"
    echo "  Log: $LOG_FILE"
    
    nohup "$PYTHON_CMD" server.py >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Wait a moment to check if it started successfully
    sleep 2
    if is_running; then
        echo -e "${GREEN}✓ Server started successfully (PID: $PID)${NC}"
        echo "  Log file: $LOG_FILE"
        echo "  Use './scripts/blood-server.sh logs' to view logs"
        return 0
    else
        echo -e "${RED}✗ Server failed to start. Check log file: $LOG_FILE${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Stop the server
stop() {
    if ! is_running; then
        echo -e "${YELLOW}Server is not running${NC}"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}Stopping server (PID: $PID)...${NC}"
    
    # Try graceful shutdown first
    kill "$PID" 2>/dev/null || true
    
    # Wait for process to stop
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Force killing server...${NC}"
        kill -9 "$PID" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✓ Server stopped${NC}"
}

# Restart the server
restart() {
    echo "Restarting server..."
    stop
    sleep 2
    start
}

# Show server status
status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}✓ Server is running${NC}"
        echo "  PID: $PID"
        
        # Show process info
        echo ""
        echo "Process information:"
        ps -p "$PID" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || echo "  (Process info unavailable)"
        
        # Show port info from config
        if [ -f "$CONFIG_FILE" ]; then
            PORT=$(grep -E '^\s*port\s*=' "$CONFIG_FILE" | head -1 | sed 's/.*=\s*\([0-9]*\).*/\1/' || echo "N/A")
            HOST=$(grep -E '^\s*host\s*=' "$CONFIG_FILE" | head -1 | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "127.0.0.1")
            echo ""
            echo "Configuration:"
            echo "  Host: $HOST"
            echo "  Port: $PORT"
        fi
        
        echo ""
        echo "Log file: $LOG_FILE"
    else
        echo -e "${RED}✗ Server is not running${NC}"
        return 1
    fi
}

# Show logs
logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "Showing server logs (Ctrl+C to exit):"
        echo "---"
        tail -f "$LOG_FILE"
    else
        echo -e "${YELLOW}Log file not found: $LOG_FILE${NC}"
        echo "Server may not have been started yet."
        return 1
    fi
}

# Main command handler
case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Bloody Battle Mahjong Training Server Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the training server"
        echo "  stop    - Stop the training server"
        echo "  restart - Restart the training server"
        echo "  status  - Show server status and information"
        echo "  logs    - Show and follow server logs"
        echo ""
        echo "Environment variables:"
        echo "  MORTAL_CFG   - Path to config.toml (default: mortal/config.toml)"
        echo "  PYTHON_CMD   - Python command (default: python3)"
        echo ""
        echo "Examples:"
        echo "  $0 start              # Start server with default config"
        echo "  MORTAL_CFG=/path/to/config.toml $0 start  # Start with custom config"
        exit 1
        ;;
esac
