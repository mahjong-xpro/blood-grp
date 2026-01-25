#!/bin/bash
# Bloody Battle Mahjong Mortal Player Script
#
# 这个脚本用于启动一个 Mortal AI 玩家实例，该玩家通过 mjai 协议与其他玩家对局。
#
# mjai 协议说明：
# - 通过标准输入（stdin）接收 JSON 格式的游戏事件
# - 通过标准输出（stdout）输出 JSON 格式的动作响应
# - 类似于国际象棋的 UCI 协议
#
# 使用场景：
# 1. 本地测试：启动 4 个玩家实例进行对局
# 2. 在线平台：配置平台使用此脚本作为 AI 玩家
# 3. 训练数据收集：在训练模式下记录对局日志
#
# Usage: ./blood-mortal.sh <player_id>
#   player_id: 0, 1, 2, or 3 (血战到底需要 4 个玩家)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Player ID required${NC}"
    echo ""
    echo "Usage: $0 <player_id>"
    echo ""
    echo "Arguments:"
    echo "  player_id  - 玩家 ID，必须是 0、1、2 或 3"
    echo ""
    echo "说明："
    echo "  这个脚本启动一个 Mortal AI 玩家，通过 mjai 协议进行对局。"
    echo "  玩家通过标准输入/输出（stdin/stdout）与游戏服务器通信。"
    echo ""
    echo "示例："
    echo "  $0 0  # 启动玩家 0"
    echo ""
    echo "详细说明请查看：scripts/MORTAL_EXPLANATION.md"
    exit 1
fi

PLAYER_ID=$1

# Validate player ID
if ! [[ "$PLAYER_ID" =~ ^[0-3]$ ]]; then
    echo -e "${RED}Error: Invalid player ID${NC}"
    echo "玩家 ID 必须是 0、1、2 或 3（血战到底需要 4 个玩家）"
    exit 1
fi

# Check if mortal.py exists
if [ ! -f "$MORTAL_DIR/mortal.py" ]; then
    echo -e "${RED}Error: mortal.py not found: $MORTAL_DIR/mortal.py${NC}"
    exit 1
fi

# Change to mortal directory
cd "$MORTAL_DIR" || exit 1

echo -e "${GREEN}Starting Bloody Battle Mahjong Mortal Player${NC}"
echo "  Player ID: $PLAYER_ID"
echo "  Protocol: mjai (stdin/stdout)"
echo ""
echo "等待游戏事件输入（stdin）..."
echo "---"

# Run mortal player
# mortal.py 会：
# 1. 加载 AI 模型（Brain + DQN）
# 2. 创建 MortalEngine（决策引擎）
# 3. 创建 Bot（mjai 协议接口）
# 4. 进入事件循环，从 stdin 读取事件，向 stdout 输出响应
exec "$PYTHON_CMD" mortal.py "$PLAYER_ID"
