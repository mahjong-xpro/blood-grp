#!/bin/bash
set -e

# Add modified core files
git add mortal/train.py
git add mortal/dataloader.py
git add mortal/config.toml
git add libblood/src/state/update.rs
git add LOCAL_TRAINING_ANALYSIS.md

# Add task tracking files (only if in repo, otherwise skip)
# git add task.md 
# git add action_plan.md

# Commit with a descriptive message
git commit -m "feat(phase6): Prepare for Phase 6 'Pure Greed' training

- configured config.toml for Phase 6 (disable rewards, epsilon 0.05)
- implemented Opponent Wait Prediction auxiliary task
- updated dataloader.py to extract opponent_waits
- updated train.py to support opponent_waits in training loop
- fixed Chankan bug in libblood/src/state/update.rs
- updated analysis report with Step 741k metrics"

# Push to remote
git push origin main
