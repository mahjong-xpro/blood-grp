# 配置文件修改总结

**修改时间**: 2026-02-28  
**原因**: 训练停滞，紧急修复  
**影响阶段**: Competitive Distill + Elite

---

## Competitive Distill 配置修改

### 文件: `configs/competitive_distill.yaml`

#### 1. 学习率配置 (关键修复)

```yaml
# 修改前
lr_schedule_kl_threshold: 0.0005
lr_adaptive_min: 5e-5

# 修改后
lr_schedule_kl_threshold: 0.002    # 4倍放宽
lr_adaptive_min: 1e-4              # 2倍提升
```

**原因**:
- 学习率95%时间锁定在5e-5
- KL均值0.001-0.002，阈值0.0005过严
- 导致学习几乎停滞

**预期效果**:
- 学习率解锁，稳定在1e-4
- 学习速度提升3-5倍

#### 2. PPO配置

```yaml
# 修改前
ppo_clip_ratio: 0.15

# 修改后
ppo_clip_ratio: 0.2
```

**原因**:
- 当前裁剪比例仅2.2%
- PPO机制几乎不起作用

**预期效果**:
- 裁剪比例提升到5-10%
- 策略更新更有效

#### 3. 价值损失权重

```yaml
# 修改前
value_loss_coeff: 1.0

# 修改后
value_loss_coeff: 0.5
```

**原因**:
- Critic学习失败，价值损失上升
- 降低权重减少错误影响

#### 4. 优势裁剪

```yaml
# 修改前
adv_clip: 2.0

# 修改后
adv_clip: 0.0  # 禁用
```

**原因**:
- 优势标准差已极低(0.30)
- 裁剪进一步削弱学习信号

**预期效果**:
- 恢复完整学习信号
- 优势标准差上升到0.5-1.0

#### 5. 奖励塑形增强

```yaml
# 修改前
reward_tsumo_bonus: 0.1
reward_deal_in_penalty: 0.05
reward_shanten_progress: 0.003
reward_rank_bonus: 0.15

# 修改后
reward_tsumo_bonus: 0.2        # 2倍
reward_deal_in_penalty: 0.1    # 2倍
reward_shanten_progress: 0.01  # 3.3倍
reward_rank_bonus: 0.3         # 2倍
```

**原因**:
- 增加奖励方差
- 提升学习信号强度

**预期效果**:
- 优势标准差上升
- 学习信号增强

#### 6. 联赛配置

```yaml
# 修改前
league_add_every: 50000
league_self_play_prob: 0.2

# 修改后
league_add_every: 25000        # 2倍频率
league_self_play_prob: 0.1     # 减半
```

**原因**:
- 增加对手多样性
- 减少自博弈比例

#### 7. Arena评估

```yaml
# 修改前
blood_arena_eval_every: 500000

# 修改后
blood_arena_eval_every: 100000  # 5倍频率
```

**原因**:
- 更频繁监控进度
- 及时发现问题

---

## Elite 配置优化

### 文件: `configs/elite.yaml`

#### 1. 预防性学习率配置

```yaml
# 修改前
lr_schedule_kl_threshold: 0.002
lr_adaptive_min: 5e-5

# 修改后
lr_schedule_kl_threshold: 0.003  # 预防性放宽
lr_adaptive_min: 1e-4            # 预防性提升
```

**原因**:
- 吸取Phase2b教训
- 预防学习率锁死

#### 2. PPO配置

```yaml
# 修改前
ppo_clip_ratio: 0.15

# 修改后
ppo_clip_ratio: 0.2
```

#### 3. 优势裁剪策略

```yaml
# 修改前
adv_clip: 5.0
blood_schedule_adv_clip: "linear,5.0,3.0,10000000,160000000"

# 修改后
adv_clip: 0.0  # 初期禁用
blood_schedule_adv_clip: "linear,0.0,2.0,50000000,150000000"
```

**原因**:
- 初期禁用裁剪，恢复学习信号
- 待优势标准差恢复后逐步启用

#### 4. 奖励塑形

```yaml
# 修改前
reward_tsumo_bonus: 0.1
reward_deal_in_penalty: 0.05
reward_shanten_progress: 0.003
reward_rank_bonus: 0.2

# 修改后
reward_tsumo_bonus: 0.2
reward_deal_in_penalty: 0.1
reward_shanten_progress: 0.01
reward_rank_bonus: 0.3
```

#### 5. 联赛配置

```yaml
# 修改前
league_add_every: 100000
league_self_play_prob: 0.2

# 修改后
league_add_every: 50000
league_self_play_prob: 0.1
```

#### 6. Arena评估

```yaml
# 修改前
blood_arena_eval_every: 2000000

# 修改后
blood_arena_eval_every: 500000
```

---

## 修改影响分析

### 短期影响 (24小时)

| 指标 | 修改前 | 预期修改后 | 改善幅度 |
|------|--------|------------|----------|
| 学习率 | 5e-5 (锁定) | 1e-4 (解锁) | 2倍 |
| 优势标准差 | 0.30 | 0.5-1.0 | 1.7-3.3倍 |
| 回报改善速度 | 0.0066/150K | 0.02+/150K | 3倍+ |
| 裁剪比例 | 2.2% | 5-10% | 2-4倍 |

### 中期影响 (1周)

- 价值损失下降: 1.08 → 0.8-
- 实际打牌效果改善
- Elo提升: 1500 → 1550+

### 长期影响 (2-4周)

- 完成Competitive Distill阶段
- 顺利过渡到Elite阶段
- 目标Elo: 1600+

---

## 回滚方案

如果修改导致训练不稳定，可以回滚到保守方案：

### 保守配置

```yaml
# competitive_distill.yaml
lr_adaptive_min: 7.5e-5           # 而非1e-4
lr_schedule_kl_threshold: 0.001   # 而非0.002
adv_clip: 1.0                     # 而非0.0
ppo_clip_ratio: 0.175             # 而非0.2
value_loss_coeff: 0.75            # 而非0.5

# 奖励塑形适度增强
reward_tsumo_bonus: 0.15          # 而非0.2
reward_deal_in_penalty: 0.075     # 而非0.1
reward_shanten_progress: 0.006    # 而非0.01
reward_rank_bonus: 0.225          # 而非0.3
```

---

## 监控计划

### 每小时检查

```bash
# 快速检查关键指标
curl -s 'http://localhost:6006/data/plugin/scalars/scalars?run=blood_v2_competitive_distill%2F.summary%2F0&tag=train%2Flr' | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'LR: {data[-1][2]:.6f}')"
```

### 每天检查

```bash
python3 /tmp/fetch_metrics.py
```

### 警告阈值

- KL散度 >0.01: 降低学习率
- 梯度范数 >10: 降低学习率或max_grad_norm
- 价值损失持续上升: 进一步降低value_loss_coeff
- 回报下降: 检查是否过拟合

---

## 总结

### 核心修改

1. ✅ 学习率解锁 (5e-5 → 1e-4)
2. ✅ KL阈值放宽 (0.0005 → 0.002)
3. ✅ 优势裁剪禁用 (2.0 → 0.0)
4. ✅ 奖励塑形增强 (2-3倍)
5. ✅ 对手多样性提升

### 预期结果

- 24小时内看到明显改善
- 1周内恢复正常学习
- 2-4周内完成当前阶段

### 风险控制

- 密切监控前24小时
- 准备好回滚方案
- 设置警告阈值

**行动**: 配置已修改，重启训练，开始监控！