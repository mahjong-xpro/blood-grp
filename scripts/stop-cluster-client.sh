#!/bin/bash

echo "Stopping all Mortal Clients..."
# Kill all processes matching "mortal/client.py"
pkill -f "mortal/client.py" || echo "No clients found running."

echo "Cleaning up temporary config files..."
rm -f mortal/config_client_*.toml

echo "Done. All clients stopped and configs cleaned."
