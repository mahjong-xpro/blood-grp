#!/bin/bash
set -e
# Defaults to config.toml

echo "Starting Mortal Server (Model/Data Broker)..."
python mortal/server.py
