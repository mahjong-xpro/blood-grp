#!/bin/bash
# SSH 隧道：通过本地端口访问服务器上的端口（本地端口转发）
# 用法: ./ssh-tunnel.sh [选项] [user@服务器] [服务器端口] [本机端口]
#
# 示例:
#   ./ssh-tunnel.sh user@my-server 6007 6007
#   ./ssh-tunnel.sh -p 2222 user@my-server 6007 6007
#   ./ssh-tunnel.sh -P 6007 -L 8080 -p 2222 user@my-server

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BACKGROUND=false
SERVER=""
SSH_PORT=""
SSH_PORT_OPT=""
REMOTE_PORT=""
REMOTE_PORT_OPT=""
LOCAL_PORT=""
LOCAL_PORT_OPT=""
EXTRA_SSH_OPTS=()

usage() {
    echo "用法: $0 [-f] [-N] [-p SSH端口] [-P 服务器端口] [-L 本机端口] [user@服务器] [服务器端口] [本机端口]"
    echo ""
    echo "SSH 隧道：通过本地端口访问服务器上的端口。"
    echo "适用于服务器端口无法从外网直接访问的场景（如 TensorBoard、Jupyter 等）。"
    echo ""
    echo "选项:"
    echo "  -f  后台运行 (ssh -f)"
    echo "  -N  仅做端口转发，不执行远程命令 (ssh -N)"
    echo "  -p  SSH 连接端口（默认 22，可用环境变量 SSH_PORT）"
    echo "  -P  服务器上的服务端口（可用环境变量 SSH_REMOTE_PORT）"
    echo "  -L  本机监听端口（可用环境变量 SSH_LOCAL_PORT）"
    echo ""
    echo "参数:"
    echo "  user@服务器  SSH 登录地址（可用环境变量 SSH_HOST）"
    echo "  服务器端口   要转发的远程端口，如 6007"
    echo "  本机端口     本地监听端口，不写则与服务器端口相同"
    echo ""
    echo "环境变量: SSH_HOST, SSH_PORT, SSH_REMOTE_PORT, SSH_LOCAL_PORT"
    echo ""
    echo "示例:"
    echo "  $0 user@my-server 6007                  # 服务器 6007 → 本机 6007"
    echo "  $0 user@my-server 6007 8080              # 服务器 6007 → 本机 8080"
    echo "  $0 -p 2222 user@my-server 6007           # SSH 走 2222 端口"
    echo "  $0 -f -N user@my-server 6007             # 后台运行，仅转发"
    echo "  $0 -P 6007 -L 8080 user@my-server        # 用选项指定端口"
    echo "  SSH_HOST=user@server SSH_REMOTE_PORT=6007 $0"
}

while getopts "fNL:P:p:h" opt; do
    case "$opt" in
        f) BACKGROUND=true ;;
        N) EXTRA_SSH_OPTS+=(-N) ;;
        L) LOCAL_PORT_OPT="$OPTARG" ;;
        P) REMOTE_PORT_OPT="$OPTARG" ;;
        p) SSH_PORT_OPT="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# 从位置参数或环境变量获取值
SERVER="${1:-${SSH_HOST:-}}"
ARG2="${2:-${SSH_REMOTE_PORT:-}}"
ARG3="${3:-${SSH_LOCAL_PORT:-}}"

REMOTE_PORT="${REMOTE_PORT_OPT:-$ARG2}"
REMOTE_PORT="${REMOTE_PORT:-${SSH_REMOTE_PORT:-}}"
LOCAL_PORT="${LOCAL_PORT_OPT:-$ARG3}"
LOCAL_PORT="${LOCAL_PORT:-${SSH_LOCAL_PORT:-}}"
SSH_PORT="${SSH_PORT_OPT:-${SSH_PORT:-}}"

# 未指定本机端口时与服务器端口一致
if [ -z "$LOCAL_PORT" ]; then
    LOCAL_PORT="$REMOTE_PORT"
fi

if [ -z "$SERVER" ] || [ -z "$REMOTE_PORT" ]; then
    echo -e "${RED}错误: 缺少必要参数。${NC}"
    echo ""
    usage
    exit 1
fi

# 检查本机端口是否被占用
if command -v lsof &>/dev/null; then
    if lsof -Pi ":$LOCAL_PORT" -sTCP:LISTEN -t &>/dev/null; then
        echo -e "${YELLOW}警告: 本机端口 $LOCAL_PORT 已被占用。${NC}"
        echo "  可改用其它本机端口，或先结束占用进程："
        echo "  lsof -Pi :$LOCAL_PORT -sTCP:LISTEN"
        exit 1
    fi
fi

# 构建 SSH 参数
FORWARD="-L ${LOCAL_PORT}:localhost:${REMOTE_PORT}"
KEEPALIVE="-o ServerAliveInterval=60 -o ServerAliveCountMax=3"

SSH_OPTS=($KEEPALIVE "$FORWARD" ${EXTRA_SSH_OPTS[@]+"${EXTRA_SSH_OPTS[@]}"})
[ -n "$SSH_PORT" ] && SSH_OPTS=(-p "$SSH_PORT" "${SSH_OPTS[@]}")
[ "$BACKGROUND" = true ] && SSH_OPTS+=(-f)

echo -e "${GREEN}SSH 隧道${NC}"
echo "  服务器:     ${SERVER}${SSH_PORT:+ (SSH 端口 ${SSH_PORT})}"
echo "  远程端口:   ${REMOTE_PORT}"
echo "  本机端口:   ${LOCAL_PORT}"
echo "  访问地址:   http://localhost:${LOCAL_PORT}"
echo ""
[ "$BACKGROUND" = true ] && echo "后台运行模式 (ssh -f)" && echo ""

exec ssh "${SSH_OPTS[@]}" "$SERVER"
