#!/bin/bash
set -e
# Defaults to config.toml
export MORTAL_CFG=mortal/config.toml

echo "Starting Mortal Client..."
python mortal/client.py
