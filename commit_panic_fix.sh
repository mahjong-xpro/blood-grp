#!/bin/bash
set -e

# Add the modified files for the panic fix
git add libblood/src/consts.rs
git add libblood/src/dataset/invisible.rs

# Commit with a clear message explaining the fix
git commit -m "fix(dataset): resolve shape mismatch panic in invisible.rs caused by opponent_waits feature (134 != 118)"

# Push changes
git push origin main
