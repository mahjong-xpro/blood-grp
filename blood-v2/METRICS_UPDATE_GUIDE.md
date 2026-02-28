# 训练指标更新指南

## 当前状态（2026-02-28）

### 已完成的修复
1. ✅ **定缺系统重构** - 移除均匀先验，实现渐进式学习
2. ✅ **配置紧急修复** - competitive_distill.yaml 和 elite.yaml
3. ✅ **学习率解锁** - lr_adaptive_min: 5e-5 → 1e-4
4. ✅ **KL阈值提升** - 0.0005 → 0.002（允许更大策略更新）
5. ✅ **优势裁剪移除** - adv_clip: 2.0 → 0.0（释放学习信号）
6. ✅ **PPO裁剪放宽** - 0.15 → 0.2（允许更大更新）
7. ✅ **价值损失降权** - 1.0 → 0.5（平衡Actor-Critic）
8. ✅ **奖励塑形增强** - 2-3倍提升

### 训练状态
- **当前状态**: 未运行
- **需要操作**: 重启训练以应用新配置

## 重启训练命令

```bash
cd /Users/twosson/Mahjong/blood/blood-v2

# 如果是 competitive_distill 阶段
./scripts/manage.sh train distill --resume --device gpu

# 如果是 elite 阶段
./scripts/manage.sh train elite --resume --device gpu

# 启动 TensorBoard 监控
./scripts/manage.sh monitor --port 6006
```

## 关键指标监控

### 1. 优势标准差 (train/advantages_std)
**目标**: 从 0.30 提升到 2.0-10.0

**检查命令**:
```bash
curl -s "http://localhost:6006/data/plugin/scalars/scalars?tag=train%2Fadvantages_std&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('最新:', data[-1][2] if data else '无数据')"
```

**预期改善**:
- 前100K steps: 0.5-1.0（初期恢复）
- 100K-500K steps: 1.0-3.0（稳定增长）
- 500K+ steps: 3.0-10.0（健康学习）

### 2. 学习率 (train/lr)
**目标**: 不再锁定在 5e-5

**检查命令**:
```bash
curl -s "http://localhost:6006/data/plugin/scalars/scalars?tag=train%2Flr&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('最新:', data[-1][2] if data else '无数据')"
```

**预期改善**:
- 应该在 1e-4 到 3e-4 之间动态调整
- 不应该长时间停留在最低值

### 3. 价值损失 (train/value_loss)
**目标**: 停止上升，开始下降

**检查命令**:
```bash
curl -s "http://localhost:6006/data/plugin/scalars/scalars?tag=train%2Fvalue_loss&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); recent = data[-100:] if len(data) > 100 else data; print('最近100步均值:', sum(x[2] for x in recent)/len(recent) if recent else '无数据')"
```

**预期改善**:
- 前50K steps: 可能继续上升（调整期）
- 50K-200K steps: 开始稳定
- 200K+ steps: 逐步下降

### 4. PPO裁剪比例 (train/ppo_clip_ratio)
**目标**: 从 2.2% 提升到 10-20%

**检查命令**:
```bash
curl -s "http://localhost:6006/data/plugin/scalars/scalars?tag=train%2Fppo_clip_ratio&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('最新:', data[-1][2]*100, '%' if data else '无数据')"
```

**预期改善**:
- 应该立即提升到 10-15%
- 表示策略更新更积极

### 5. 平均回报 (train/mean_return)
**目标**: 加速增长

**检查命令**:
```bash
curl -s "http://localhost:6006/data/plugin/scalars/scalars?tag=train%2Fmean_return&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('最新:', data[-1][2] if data else '无数据')"
```

**预期改善**:
- 每100K steps应该有明显提升（>5%）
- 不应该长时间停滞

### 6. KL散度 (train/kl_divergence)
**目标**: 0.001-0.002（健康更新范围）

**检查命令**:
```bash
curl -s "http://localhost:6006/data/plugin/scalars/scalars?tag=train%2Fkl_divergence&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('最新:', data[-1][2] if data else '无数据')"
```

**预期改善**:
- 不应该长时间低于 0.0005
- 应该在 0.001-0.002 之间波动

## 一键检查所有指标

```bash
# 保存为 check_all_metrics.sh
#!/bin/bash
TB_URL="http://localhost:6006"

echo "=== 训练指标快照 ==="
echo ""

echo "1. 优势标准差:"
curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=train%2Fadvantages_std&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('  ', data[-1][2] if data else '无数据')"

echo "2. 学习率:"
curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=train%2Flr&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('  ', data[-1][2] if data else '无数据')"

echo "3. 价值损失:"
curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=train%2Fvalue_loss&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('  ', data[-1][2] if data else '无数据')"

echo "4. PPO裁剪比例:"
curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=train%2Fppo_clip_ratio&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('  ', data[-1][2]*100, '%' if data else '无数据')"

echo "5. 平均回报:"
curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=train%2Fmean_return&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('  ', data[-1][2] if data else '无数据')"

echo "6. KL散度:"
curl -s "${TB_URL}/data/plugin/scalars/scalars?tag=train%2Fkl_divergence&run=." | \
python3 -c "import sys, json; data = json.load(sys.stdin); print('  ', data[-1][2] if data else '无数据')"

echo ""
echo "=== 检查完成 ==="
```

## 监控时间表

### 立即检查（重启后1小时内）
- ✅ PPO裁剪比例是否提升
- ✅ 学习率是否解锁
- ✅ 优势标准差是否开始增长

### 短期检查（24小时内）
- ✅ 优势标准差是否达到 1.0+
- ✅ 价值损失是否稳定
- ✅ 回报增长是否加速

### 中期检查（3-7天）
- ✅ 优势标准差是否达到 3.0+
- ✅ 价值损失是否下降
- ✅ 定缺决策是否改善

### 长期检查（2-4周）
- ✅ 优势标准差是否达到 5.0+
- ✅ 整体性能是否显著提升
- ✅ 是否需要进一步优化

## 问题诊断

### 如果优势标准差仍然很低（<0.5）
**可能原因**:
1. 奖励信号仍然太弱
2. 环境多样性不足
3. 需要更激进的探索

**解决方案**:
1. 进一步提升奖励塑形系数
2. 增加熵正则化系数
3. 降低价值损失权重

### 如果学习率仍然锁定
**可能原因**:
1. KL散度阈值仍然太低
2. 策略更新太保守

**解决方案**:
1. 进一步提升 lr_schedule_kl_threshold
2. 增大 ppo_clip_ratio

### 如果价值损失继续上升
**可能原因**:
1. Critic学习速度跟不上Actor
2. 价值损失权重仍然太高

**解决方案**:
1. 进一步降低 value_loss_coeff
2. 增加Critic网络容量
3. 使用更大的batch size

## 成功标准

训练被认为成功修复，当满足以下条件：

1. ✅ **优势标准差** > 3.0（持续100K steps）
2. ✅ **学习率** 在 1e-4 到 3e-4 之间动态调整
3. ✅ **价值损失** 呈下降趋势
4. ✅ **PPO裁剪比例** > 10%
5. ✅ **回报增长** 每100K steps > 5%
6. ✅ **定缺决策** 明显改善（通过对局观察）

## 下一步行动

1. **立即**: 重启训练，应用新配置
2. **1小时后**: 检查立即指标（PPO裁剪、学习率）
3. **24小时后**: 检查短期指标（优势标准差、价值损失）
4. **3天后**: 评估中期效果，决定是否需要进一步调整
5. **1周后**: 全面评估，考虑是否实施Rust侧增强

---

**文档创建时间**: 2026-02-28  
**配置版本**: competitive_distill.yaml v2, elite.yaml v2  
**状态**: 等待训练重启