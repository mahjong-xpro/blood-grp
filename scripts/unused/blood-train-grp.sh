#!/bin/bash
# GRP 训练脚本
# Usage: ./blood-train-grp.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
CONFIG_FILE="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo ""
    echo "Please create a config.toml file:"
    echo "  cd mortal"
    echo "  cp config.example.toml config.toml"
    exit 1
fi

# Check if train_grp.py exists
if [ ! -f "$MORTAL_DIR/train_grp.py" ]; then
    echo -e "${RED}Error: train_grp.py not found: $MORTAL_DIR/train_grp.py${NC}"
    exit 1
fi

echo -e "${GREEN}Starting GRP Training${NC}"
echo "  Config: $CONFIG_FILE"
echo "  Mode: GRP (Global Ranking Prediction)"
echo ""

# Check GRP data
GRP_TRAIN_GLOBS=$(python3 -c "import toml; c=toml.load(open('$CONFIG_FILE')); print(' '.join(c['grp']['dataset']['train_globs']))" 2>/dev/null || echo "")
if [ -z "$GRP_TRAIN_GLOBS" ] || [[ "$GRP_TRAIN_GLOBS" == *"/path/to"* ]]; then
    echo -e "${YELLOW}警告: GRP 训练数据路径未配置或指向示例路径${NC}"
    echo "  请检查配置文件中的 [grp.dataset].train_globs"
    echo ""
    echo "  对于自博弈训练，可以指向自对局数据："
    echo "    train_globs = ['/data/mortal/train_play/**/*.json.gz']"
    echo ""
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd "$MORTAL_DIR" || exit 1
exec "$PYTHON_CMD" train_grp.py
