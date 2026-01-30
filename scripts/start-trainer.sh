#!/bin/bash
set -e
# Defaults to config.toml
export MORTAL_CFG=mortal/config.toml

echo "Starting Mortal Trainer..."
python mortal/train.py
