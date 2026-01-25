#!/bin/bash
# 配置文件检查脚本
# 检查配置文件是否完整和正确

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MORTAL_DIR="$PROJECT_DIR/mortal"
CONFIG_FILE="${MORTAL_CFG:-$MORTAL_DIR/config.toml}"

echo -e "${BLUE}配置文件检查${NC}"
echo "=================================="
echo "配置文件: $CONFIG_FILE"
echo ""

# 检查文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ 配置文件不存在: $CONFIG_FILE${NC}"
    echo ""
    echo "请先创建配置文件："
    echo "  cd mortal"
    echo "  cp config.example.toml config.toml"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件存在${NC}"

# 检查 TOML 语法
echo ""
echo -e "${BLUE}检查 TOML 语法...${NC}"
if python3 -c "import tomli; tomli.load(open('$CONFIG_FILE', 'rb'))" 2>/dev/null; then
    echo -e "${GREEN}✓ TOML 语法正确${NC}"
elif python3 -c "import toml; toml.load(open('$CONFIG_FILE'))" 2>/dev/null; then
    echo -e "${GREEN}✓ TOML 语法正确${NC}"
else
    echo -e "${RED}✗ TOML 语法错误${NC}"
    exit 1
fi

# 检查关键配置项
echo ""
echo -e "${BLUE}检查关键配置项...${NC}"

check_config() {
    local key=$1
    local desc=$2
    if grep -q "$key" "$CONFIG_FILE"; then
        echo -e "${GREEN}✓ $desc${NC}"
        return 0
    else
        echo -e "${RED}✗ 缺少: $desc${NC}"
        return 1
    fi
}

check_path() {
    local key=$1
    local desc=$2
    local line=$(grep -E "^\s*$key\s*=" "$CONFIG_FILE" | head -1)
    if [ -z "$line" ]; then
        echo -e "${RED}✗ 缺少: $desc${NC}"
        return 1
    fi
    # 提取值（处理引号和注释）
    local value=$(echo "$line" | sed 's/#.*//' | sed 's/.*=\s*"\([^"]*\)".*/\1/' | sed 's/.*=\s*\([^"]*\).*/\1/' | xargs)
    if [[ "$value" == /data/mortal/* ]] || [[ "$value" == /data/blood/* ]]; then
        echo -e "${GREEN}✓ $desc: $value${NC}"
        return 0
    elif [[ "$value" == /path/to/* ]]; then
        echo -e "${YELLOW}⚠ $desc: $value (需要更新为实际路径)${NC}"
        return 1
    else
        echo -e "${YELLOW}⚠ $desc: $value (请确认路径正确)${NC}"
        return 0
    fi
}

ERRORS=0

# 检查必需配置
check_config "state_file" "模型文件路径" || ERRORS=$((ERRORS+1))
check_config "device" "计算设备" || ERRORS=$((ERRORS+1))
check_config "batch_size" "批次大小" || ERRORS=$((ERRORS+1))
check_config "train_play" "自对局配置" || ERRORS=$((ERRORS+1))
check_config "dataset" "数据集配置" || ERRORS=$((ERRORS+1))
check_config "baseline" "Baseline 配置" || ERRORS=$((ERRORS+1))

# 检查路径
echo ""
echo -e "${BLUE}检查路径配置...${NC}"
check_path "state_file" "模型文件路径"
check_path "log_dir" "自对局数据目录"
check_path "globs" "数据集路径"

# 检查设备配置
echo ""
echo -e "${BLUE}检查设备配置...${NC}"
DEVICE_LINE=$(grep -E "^\s*device\s*=" "$CONFIG_FILE" | head -1)
DEVICE=$(echo "$DEVICE_LINE" | sed 's/#.*//' | sed "s/.*=\s*'\([^']*\)'.*/\1/" | sed 's/.*=\s*"\([^"]*\)".*/\1/' | xargs)
if [[ "$DEVICE" == cuda* ]]; then
    echo -e "${GREEN}✓ 使用 GPU: $DEVICE${NC}"
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU 信息:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /' || echo -e "${YELLOW}    ⚠ 无法获取 GPU 信息${NC}"
    else
        echo -e "${YELLOW}  ⚠ nvidia-smi 不可用，请确认 GPU 配置正确${NC}"
    fi
elif [[ "$DEVICE" == cpu ]]; then
    echo -e "${YELLOW}⚠ 使用 CPU（训练速度较慢）${NC}"
elif [ -z "$DEVICE" ]; then
    echo -e "${RED}✗ 无法解析设备配置${NC}"
    ERRORS=$((ERRORS+1))
else
    echo -e "${YELLOW}⚠ 设备配置: $DEVICE (请确认正确)${NC}"
fi

# 检查目录权限
echo ""
echo -e "${BLUE}检查目录权限...${NC}"
TRAIN_PLAY_LINE=$(grep -E "^\s*log_dir\s*=" "$CONFIG_FILE" | grep -A 5 "train_play" | head -1)
TRAIN_PLAY_DIR=$(echo "$TRAIN_PLAY_LINE" | sed 's/#.*//' | sed 's/.*=\s*"\([^"]*\)".*/\1/' | xargs)
if [ -z "$TRAIN_PLAY_DIR" ]; then
    # 尝试从 [train_play.default] 部分获取
    TRAIN_PLAY_LINE=$(grep -A 2 "\[train_play.default\]" "$CONFIG_FILE" | grep "log_dir" | head -1)
    TRAIN_PLAY_DIR=$(echo "$TRAIN_PLAY_LINE" | sed 's/#.*//' | sed 's/.*=\s*"\([^"]*\)".*/\1/' | xargs)
fi
if [ -n "$TRAIN_PLAY_DIR" ]; then
    if [ -d "$TRAIN_PLAY_DIR" ]; then
        if [ -w "$TRAIN_PLAY_DIR" ]; then
            echo -e "${GREEN}✓ 自对局目录可写: $TRAIN_PLAY_DIR${NC}"
        else
            echo -e "${RED}✗ 自对局目录不可写: $TRAIN_PLAY_DIR${NC}"
            ERRORS=$((ERRORS+1))
        fi
    else
        echo -e "${YELLOW}⚠ 自对局目录不存在（训练时会自动创建）: $TRAIN_PLAY_DIR${NC}"
    fi
else
    echo -e "${YELLOW}⚠ 无法找到自对局目录配置${NC}"
fi

# 检查 Python 模块
echo ""
echo -e "${BLUE}检查 Python 模块...${NC}"
cd "$MORTAL_DIR" || exit 1
if python3 -c "import libblood" 2>/dev/null; then
    echo -e "${GREEN}✓ libblood 模块可用${NC}"
else
    echo -e "${YELLOW}⚠ libblood 模块不可用（可能未编译）${NC}"
    echo "  提示: 在服务器上需要先编译 libblood 模块"
    echo "  运行: cd /data/blood && cargo build --release -p libblood"
    # 不将其视为错误，因为可能在不同环境检查
fi

# 总结
echo ""
echo "=================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 配置文件检查通过${NC}"
    echo ""
    echo "可以开始训练："
    echo "  ./scripts/blood-train.sh offline"
    exit 0
else
    echo -e "${RED}✗ 发现 $ERRORS 个问题，请修复后重试${NC}"
    exit 1
fi
