#!/bin/bash
set -e
# Defaults to config.toml
export MORTAL_CFG=mortal/config.toml

echo "Starting Mortal Server (Model/Data Broker)..."
python mortal/server.py
