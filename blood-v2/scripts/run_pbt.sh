#!/usr/bin/env bash
# Population-Based Training launcher for Blood Mahjong.
#
# Starts N parallel training instances coordinated by PBTController.
# Each instance runs the elite training stage with perturbed hyperparameters.
#
# Usage:
#   ./scripts/run_pbt.sh [population_size] [base_config]
#
# Example:
#   ./scripts/run_pbt.sh 4 configs/elite.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

POPULATION_SIZE="${1:-4}"
BASE_CONFIG="${2:-configs/elite.yaml}"
PBT_DIR="pbt_runs"
EVAL_EVERY="${PBT_EVAL_EVERY:-1000000}"

echo "=== Blood Mahjong PBT Training ==="
echo "Population size: $POPULATION_SIZE"
echo "Base config: $BASE_CONFIG"
echo "PBT directory: $PBT_DIR"
echo "Eval every: $EVAL_EVERY steps"
echo ""

# Ensure PYTHONPATH is set
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"

# Create PBT work directory
mkdir -p "$PBT_DIR"

# Initialize population
python -c "
from blood.training.pbt import PBTController
import yaml

with open('$BASE_CONFIG') as f:
    base_cfg = yaml.safe_load(f)

# Extract tunable hyperparameters from base config
base_hp = {k: v for k, v in base_cfg.items() if isinstance(v, (int, float))}

controller = PBTController(
    population_size=$POPULATION_SIZE,
    eval_every=$EVAL_EVERY,
    work_dir='$PBT_DIR',
)
members = controller.initialize_population(base_hp)
for m in members:
    print(f'  Member {m.member_id}: lr={m.hyperparams.get(\"learning_rate\", \"N/A\"):.6f}, '
          f'entropy={m.hyperparams.get(\"exploration_loss_coeff\", \"N/A\"):.4f}')
print(f'Initialized {len(members)} members')
"

echo ""
echo "Population initialized. To start training each member:"
echo ""
for i in $(seq 0 $((POPULATION_SIZE - 1))); do
    echo "  # Member $i:"
    echo "  python -m blood.training.runner --config $BASE_CONFIG \\"
    echo "    --experiment blood_pbt_member_$i \\"
    echo "    --train_dir $PBT_DIR/member_$i"
    echo ""
done

echo "Monitor with: tensorboard --logdir $PBT_DIR"
echo ""
echo "After each eval_every steps, run the PBT step:"
echo "  python -c \"from blood.training.pbt import PBTController; c = PBTController(work_dir='$PBT_DIR'); c.load_state(); actions = c.step(); print(actions)\""
