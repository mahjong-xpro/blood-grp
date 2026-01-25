#!/bin/bash
# 配置文件设置脚本
# 用于在服务器上快速设置配置文件

set -e

# 配置参数
DATA_DIR="${DATA_DIR:-/data/mortal}"
SOURCE_DIR="${SOURCE_DIR:-/data/blood}"
MORTAL_DIR="$SOURCE_DIR/mortal"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Bloody Battle Mahjong 配置文件设置${NC}"
echo "=================================="
echo "数据目录: $DATA_DIR"
echo "源码目录: $SOURCE_DIR"
echo ""

# 检查源码目录
if [ ! -d "$MORTAL_DIR" ]; then
    echo -e "${RED}错误: 源码目录不存在: $MORTAL_DIR${NC}"
    exit 1
fi

# 检查示例配置文件
if [ ! -f "$MORTAL_DIR/config.example.toml" ]; then
    echo -e "${RED}错误: 示例配置文件不存在: $MORTAL_DIR/config.example.toml${NC}"
    exit 1
fi

# 创建数据目录结构
echo -e "${YELLOW}创建数据目录结构...${NC}"
mkdir -p "$DATA_DIR"/{train_play,test_play,1v3,logs,buffer,drain,dataset/{train,val}}

# 复制并更新配置文件
echo -e "${YELLOW}创建配置文件...${NC}"
CONFIG_FILE="$MORTAL_DIR/config.toml"

if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}配置文件已存在，创建备份...${NC}"
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
fi

# 使用 sed 替换路径（如果配置文件已存在）
if [ -f "$CONFIG_FILE" ]; then
    # 更新路径
    sed -i.bak "s|/path/to|$DATA_DIR|g" "$CONFIG_FILE"
    sed -i.bak "s|mortal\.pth|$DATA_DIR/mortal.pth|g" "$CONFIG_FILE"
    sed -i.bak "s|best\.pth|$DATA_DIR/best.pth|g" "$CONFIG_FILE"
    sed -i.bak "s|baseline\.pth|$DATA_DIR/baseline.pth|g" "$CONFIG_FILE"
    sed -i.bak "s|train_play|$DATA_DIR/train_play|g" "$CONFIG_FILE"
    sed -i.bak "s|test_play|$DATA_DIR/test_play|g" "$CONFIG_FILE"
    sed -i.bak "s|1v3|$DATA_DIR/1v3|g" "$CONFIG_FILE"
    sed -i.bak "s|logs|$DATA_DIR/logs|g" "$CONFIG_FILE"
    sed -i.bak "s|buffer|$DATA_DIR/buffer|g" "$CONFIG_FILE"
    sed -i.bak "s|drain|$DATA_DIR/drain|g" "$CONFIG_FILE"
    rm -f "$CONFIG_FILE.bak"
else
    # 从示例文件创建
    cp "$MORTAL_DIR/config.example.toml" "$CONFIG_FILE"
    
    # 更新路径
    sed -i.bak "s|/path/to|$DATA_DIR|g" "$CONFIG_FILE"
    sed -i.bak "s|mortal\.pth|$DATA_DIR/mortal.pth|g" "$CONFIG_FILE"
    sed -i.bak "s|best\.pth|$DATA_DIR/best.pth|g" "$CONFIG_FILE"
    sed -i.bak "s|baseline\.pth|$DATA_DIR/baseline.pth|g" "$CONFIG_FILE"
    sed -i.bak "s|train_play|$DATA_DIR/train_play|g" "$CONFIG_FILE"
    sed -i.bak "s|test_play|$DATA_DIR/test_play|g" "$CONFIG_FILE"
    sed -i.bak "s|1v3|$DATA_DIR/1v3|g" "$CONFIG_FILE"
    sed -i.bak "s|logs|$DATA_DIR/logs|g" "$CONFIG_FILE"
    sed -i.bak "s|buffer|$DATA_DIR/buffer|g" "$CONFIG_FILE"
    sed -i.bak "s|drain|$DATA_DIR/drain|g" "$CONFIG_FILE"
    rm -f "$CONFIG_FILE.bak"
fi

echo -e "${GREEN}✓ 配置文件已创建: $CONFIG_FILE${NC}"
echo ""
echo -e "${YELLOW}下一步：${NC}"
echo "  1. 检查配置文件: cat $CONFIG_FILE"
echo "  2. 根据需要调整参数（特别是 device、batch_size 等）"
echo "  3. 开始训练: cd $MORTAL_DIR && python train.py"
echo "  或使用脚本: ./scripts/blood-train.sh offline"
