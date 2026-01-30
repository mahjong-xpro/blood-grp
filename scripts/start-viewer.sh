#!/bin/bash
set -e

# Default to port 8080 if not specified
PORT=${1:-8080}

# Activate virtual environment if available
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:.

echo "Starting Mahjong Log Viewer on port $PORT..."
echo "Monitoring directory: /data/mortal/drain (Data currently being trained on)"
echo "Note: Files here are transient. They are cached in memory by the viewer."

python3 log-viewer/app.py --port $PORT --log-dir /data/mortal/drain
