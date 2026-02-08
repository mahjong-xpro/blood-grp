#!/bin/bash
# Resolve project root (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Navigate to blood-arena directory
cd "$PROJECT_ROOT/blood-arena"

# Add project root to PYTHONPATH so we can import libblood/mortal
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT"

# Start Uvicorn
echo "Starting Blood Arena Game Server..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8800 --reload
