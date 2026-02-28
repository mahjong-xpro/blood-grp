#!/bin/bash
# 训练指标快速检查脚本
# 用法: ./scripts/check_metrics.sh [tensorboard_url]

set -euo pipefail

TB_URL="${1:-http://localhost:6006}"
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}=== Blood-V2 训练指标快照 ===${NC}"
echo -e "TensorBoard: ${TB_URL}"
echo ""

# 检查TensorBoard是否可访问
if ! curl -s "${TB_URL}" > /dev/null 2>&1; then
    echo -e "${RED}错误: 无法连接到 TensorBoard (${TB_URL})${NC}"
    echo "请确保 TensorBoard 正在运行:"
    echo "  ./scripts/manage.sh monitor"
    exit 1
fi

get_metric() {
    local tag="$1"
    local format="${2:-%.6f}"
    curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=${tag}&run=." 2>/dev/null | \
    python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data:
        value = data[-1][2]
        print('${format}' % value)
    else:
        print('无数据')
except:
    print('错误')
" 2>/dev/null || echo "N/A"
}

get_metric_trend() {
    local tag="$1"
    curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=${tag}&run=." 2>/dev/null | \
    python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if len(data) >= 2:
        recent = data[-10:] if len(data) >= 10 else data
        old = data[-20:-10] if len(data) >= 20 else data[:len(data)//2]
        recent_avg = sum(x[2] for x in recent) / len(recent)
        old_avg = sum(x[2] for x in old) / len(old) if old else recent_avg
        change = ((recent_avg - old_avg) / old_avg * 100) if old_avg != 0 else 0
        if change > 5:
            print('↑ +%.1f%%' % change)
        elif change < -5:
            print('↓ %.1f%%' % change)
        else:
            print('→ %.1f%%' % change)
    else:
        print('--')
except:
    print('--')
" 2>/dev/null || echo "--"
}

# 1. 优势标准差
echo -e "${CYAN}1. 优势标准差 (Advantages Std)${NC}"
adv_std=$(get_metric "train%2Fadvantages_std" "%.4f")
adv_trend=$(get_metric_trend "train%2Fadvantages_std")
if [[ "$adv_std" != "N/A" && "$adv_std" != "无数据" ]]; then
    adv_val=$(echo "$adv_std" | awk '{print $1}')
    if (( $(echo "$adv_val > 3.0" | bc -l) )); then
        echo -e "   ${GREEN}✓ ${adv_std}${NC} ${adv_trend} (健康)"
    elif (( $(echo "$adv_val > 1.0" | bc -l) )); then
        echo -e "   ${YELLOW}⚠ ${adv_std}${NC} ${adv_trend} (改善中)"
    else
        echo -e "   ${RED}✗ ${adv_std}${NC} ${adv_trend} (过低)"
    fi
else
    echo -e "   ${RED}${adv_std}${NC}"
fi
echo "   目标: > 3.0 (当前修复前: 0.30)"
echo ""

# 2. 学习率
echo -e "${CYAN}2. 学习率 (Learning Rate)${NC}"
lr=$(get_metric "train%2Flr" "%.6f")
lr_trend=$(get_metric_trend "train%2Flr")
if [[ "$lr" != "N/A" && "$lr" != "无数据" ]]; then
    lr_val=$(echo "$lr" | awk '{print $1}')
    if (( $(echo "$lr_val > 0.00008" | bc -l) )); then
        echo -e "   ${GREEN}✓ ${lr}${NC} ${lr_trend} (已解锁)"
    else
        echo -e "   ${RED}✗ ${lr}${NC} ${lr_trend} (锁定)"
    fi
else
    echo -e "   ${RED}${lr}${NC}"
fi
echo "   目标: 1e-4 到 3e-4 动态调整 (修复前: 锁定在 5e-5)"
echo ""

# 3. 价值损失
echo -e "${CYAN}3. 价值损失 (Value Loss)${NC}"
val_loss=$(get_metric "train%2Fvalue_loss" "%.4f")
val_trend=$(get_metric_trend "train%2Fvalue_loss")
if [[ "$val_loss" != "N/A" && "$val_loss" != "无数据" ]]; then
    if [[ "$val_trend" == *"↓"* ]]; then
        echo -e "   ${GREEN}✓ ${val_loss}${NC} ${val_trend} (下降中)"
    elif [[ "$val_trend" == *"→"* ]]; then
        echo -e "   ${YELLOW}⚠ ${val_loss}${NC} ${val_trend} (稳定)"
    else
        echo -e "   ${RED}✗ ${val_loss}${NC} ${val_trend} (上升)"
    fi
else
    echo -e "   ${RED}${val_loss}${NC}"
fi
echo "   目标: 下降趋势 (修复前: 上升 +20%)"
echo ""

# 4. PPO裁剪比例
echo -e "${CYAN}4. PPO裁剪比例 (PPO Clip Ratio)${NC}"
ppo_clip=$(get_metric "train%2Fppo_clip_ratio" "%.4f")
if [[ "$ppo_clip" != "N/A" && "$ppo_clip" != "无数据" ]]; then
    ppo_pct=$(echo "$ppo_clip * 100" | bc -l | xargs printf "%.2f")
    ppo_val=$(echo "$ppo_clip" | awk '{print $1}')
    if (( $(echo "$ppo_val > 0.10" | bc -l) )); then
        echo -e "   ${GREEN}✓ ${ppo_pct}%${NC} (健康)"
    elif (( $(echo "$ppo_val > 0.05" | bc -l) )); then
        echo -e "   ${YELLOW}⚠ ${ppo_pct}%${NC} (偏低)"
    else
        echo -e "   ${RED}✗ ${ppo_pct}%${NC} (过低)"
    fi
else
    echo -e "   ${RED}${ppo_clip}${NC}"
fi
echo "   目标: 10-20% (修复前: 2.2%)"
echo ""

# 5. 平均回报
echo -e "${CYAN}5. 平均回报 (Mean Return)${NC}"
mean_ret=$(get_metric "train%2Fmean_return" "%.2f")
ret_trend=$(get_metric_trend "train%2Fmean_return")
if [[ "$mean_ret" != "N/A" && "$mean_ret" != "无数据" ]]; then
    if [[ "$ret_trend" == *"↑"* ]]; then
        echo -e "   ${GREEN}✓ ${mean_ret}${NC} ${ret_trend} (增长中)"
    elif [[ "$ret_trend" == *"→"* ]]; then
        echo -e "   ${YELLOW}⚠ ${mean_ret}${NC} ${ret_trend} (停滞)"
    else
        echo -e "   ${RED}✗ ${mean_ret}${NC} ${ret_trend} (下降)"
    fi
else
    echo -e "   ${RED}${mean_ret}${NC}"
fi
echo "   目标: 持续增长 >5%/100K steps (修复前: 仅 3.5%/671K steps)"
echo ""

# 6. KL散度
echo -e "${CYAN}6. KL散度 (KL Divergence)${NC}"
kl=$(get_metric "train%2Fkl_divergence" "%.6f")
if [[ "$kl" != "N/A" && "$kl" != "无数据" ]]; then
    kl_val=$(echo "$kl" | awk '{print $1}')
    if (( $(echo "$kl_val > 0.001 && $kl_val < 0.003" | bc -l) )); then
        echo -e "   ${GREEN}✓ ${kl}${NC} (健康范围)"
    elif (( $(echo "$kl_val < 0.0005" | bc -l) )); then
        echo -e "   ${RED}✗ ${kl}${NC} (过低)"
    else
        echo -e "   ${YELLOW}⚠ ${kl}${NC} (偏高)"
    fi
else
    echo -e "   ${RED}${kl}${NC}"
fi
echo "   目标: 0.001-0.002"
echo ""

# 总结
echo -e "${CYAN}=== 诊断总结 ===${NC}"
echo ""

# 计算健康指标数量
healthy=0
total=6

if [[ "$adv_std" != "N/A" && "$adv_std" != "无数据" ]]; then
    adv_val=$(echo "$adv_std" | awk '{print $1}')
    (( $(echo "$adv_val > 1.0" | bc -l) )) && ((healthy++))
fi

if [[ "$lr" != "N/A" && "$lr" != "无数据" ]]; then
    lr_val=$(echo "$lr" | awk '{print $1}')
    (( $(echo "$lr_val > 0.00008" | bc -l) )) && ((healthy++))
fi

if [[ "$val_trend" == *"↓"* ]]; then
    ((healthy++))
fi

if [[ "$ppo_clip" != "N/A" && "$ppo_clip" != "无数据" ]]; then
    ppo_val=$(echo "$ppo_clip" | awk '{print $1}')
    (( $(echo "$ppo_val > 0.05" | bc -l) )) && ((healthy++))
fi

if [[ "$ret_trend" == *"↑"* ]]; then
    ((healthy++))
fi

if [[ "$kl" != "N/A" && "$kl" != "无数据" ]]; then
    kl_val=$(echo "$kl" | awk '{print $1}')
    (( $(echo "$kl_val > 0.0005" | bc -l) )) && ((healthy++))
fi

if (( healthy >= 5 )); then
    echo -e "${GREEN}✓ 训练状态良好 (${healthy}/${total} 指标健康)${NC}"
elif (( healthy >= 3 )); then
    echo -e "${YELLOW}⚠ 训练状态一般 (${healthy}/${total} 指标健康)${NC}"
    echo "   建议: 继续观察，可能需要进一步调整"
else
    echo -e "${RED}✗ 训练状态不佳 (${healthy}/${total} 指标健康)${NC}"
    echo "   建议: 检查配置，考虑重启训练"
fi

echo ""
echo "详细分析: blood-v2/METRICS_UPDATE_GUIDE.md"
echo "配置修复: blood-v2/CONFIG_CHANGES_SUMMARY.md"