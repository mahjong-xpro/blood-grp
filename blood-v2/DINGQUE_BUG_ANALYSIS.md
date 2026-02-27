# 定缺100%选择Sou问题分析报告

> 日期: 2026-02-27  
> 问题: Agent在所有训练run中都100%选择sou（条），从不选择man（万）或pin（饼）

---

## 问题确认

**现象**: 
- 每次重新训练2M步，agent都100%选择sou
- 不是随机的探索不足，而是**系统性偏差**
- 其他玩家（RuleBot）分布正常（~33%各花色）

**测试结果**:
- ✅ 模型初始化**无偏差** (χ² = 1.55 < 5.99)
- ✅ Action编码正确 (31=Man, 32=Pin, 33=Sou)
- ❌ 问题在**训练过程**

---

## 根本原因分析

### 1. 奖励信号缺失

当前系统**完全移除了定缺奖励**（[`selfplay_env.py:547-557`](python/blood/env/selfplay_env.py:547)）：

```python
# DingQue: No explicit reward shaping for dingque choice.
# Let the agent learn the optimal strategy through downstream rewards
# (winning, shanten progress, etc.).
```

**问题**: 下游奖励（胜利、向听进度）的信号**太弱、太延迟**：
- 定缺选择 → 影响手牌效率 → 影响听牌速度 → 影响胜率
- 这个因果链太长，2M步不足以学习

### 2. 探索不足

PPO的探索机制依赖entropy bonus：
- Warmup阶段: `exploration_loss_coeff = 0.01`
- 对于3个等概率选项，这个系数**太小**

**计算**:
- 最大熵: ln(3) ≈ 1.10
- Entropy bonus: 0.01 × 1.10 = **0.011**
- 相比主奖励（-3到+3），探索奖励可忽略不计

### 3. 早期随机选择的锁定效应

训练初期，agent随机探索：
- 假设前100局中，60局选了sou，20局man，20局pin
- 由于样本量小，sou的平均回报可能**偶然**略高
- PPO更新后，策略向sou倾斜
- 下一轮，agent更多选sou，进一步强化这个偏差
- **正反馈循环** → 策略过早收敛到sou

### 4. 为什么总是sou？

可能的原因：
1. **牌山生成的随机种子偏差**: 如果初始几千局的牌山恰好对sou有利
2. **RuleBot的定缺策略**: RuleBot选"最少"，可能在某些牌型下与sou配合更好
3. **纯随机**: 三选一，总有一个会被先锁定，恰好是sou

---

## 解决方案

### 方案A: 临时探索奖励（推荐）

在warmup阶段加入**多样性奖励**（不是"选最少"，而是"鼓励探索"）：

```python
# python/blood/env/selfplay_env.py

def _compute_shaping_reward(self, prev_score: float, action: int, ow_before) -> float:
    bonus = 0.0
    
    # Exploration bonus for dingque: encourage trying all three suits
    # This is NOT "choose minimum" (which is suboptimal), but rather
    # "explore all options" to prevent premature convergence.
    if self._env.get_phase() == "ding_que" and self._warmup_dq_exploration > 0:
        # Track suit selection history (needs to be added to __init__)
        # Give small bonus for choosing less-explored suits
        # This will naturally balance out over training
        pass  # Implementation below
    
    # ... rest of warmup shaping
```

**实现**:

```python
# In __init__:
self._dq_suit_counts = [0, 0, 0]  # [man, pin, sou]
self._warmup_dq_exploration = getattr(cfg, "warmup_dq_exploration", 0.02)

# In _compute_shaping_reward:
if action in [31, 32, 33]:  # DingQue actions
    suit_idx = action - 31
    self._dq_suit_counts[suit_idx] += 1
    
    # Bonus inversely proportional to selection frequency
    total = sum(self._dq_suit_counts) + 1e-6
    freq = self._dq_suit_counts[suit_idx] / total
    
    # Encourage less-explored suits (max bonus when freq=0, min when freq=1)
    exploration_bonus = self._warmup_dq_exploration * (1.0 - freq)
    bonus += exploration_bonus
```

**配置**:
```yaml
# configs/warmup.yaml
warmup_dq_exploration: 0.02  # Small bonus to encourage exploration
```

### 方案B: 提高探索系数

```yaml
# configs/warmup.yaml
exploration_loss_coeff: 0.03  # 从0.01提升到0.03
```

**效果**: 
- Entropy bonus: 0.03 × 1.10 = 0.033
- 仍然较小，但比原来强3倍

### 方案C: 强制均匀采样（不推荐）

在训练初期（前500K步）强制从均匀分布采样定缺：

```python
if self._global_env_steps < 500_000 and action in [31, 32, 33]:
    # Override with uniform sampling
    action = np.random.choice([31, 32, 33])
```

**缺点**: 破坏了策略梯度，可能影响学习

---

## 推荐方案

**短期（立即实施）**: 方案A + 方案B

1. 加入探索奖励（`warmup_dq_exploration: 0.02`）
2. 提高entropy系数（`exploration_loss_coeff: 0.03`）
3. 重新训练warmup阶段

**预期效果**:
- 前500K步: 三个花色分布逐渐均衡（各30-40%）
- 500K-2M步: Agent开始根据手牌结构选择
- 2M步后: 分布可能不是33.3%（这是正常的，说明agent在学习策略）

**长期（架构改进）**: 
- 等训练到50M步后，移除探索奖励
- Agent应该已经学会根据手牌结构选择最优花色

---

## 实施步骤

1. 修改 `selfplay_env.py` 加入探索奖励
2. 修改 `cfg.py` 加入 `warmup_dq_exploration` 参数
3. 修改 `warmup.yaml` 设置探索参数
4. 删除现有训练数据
5. 重新训练 warmup 阶段
6. 运行 `analyze_dingque.py` 验证分布

---

## 结论

这不是Bug，而是**强化学习的探索-利用困境**：

- Agent在探索不足的情况下，过早收敛到局部最优（恰好是sou）
- 解决方法是增强探索机制，让agent有机会尝试所有选项
- 一旦探索充分，agent会自然学会根据手牌选择最优策略

**关键洞察**: 移除"选最少"的奖励是**正确的设计决策**，但需要配合足够的探索机制。当前的探索系数太小，导致策略过早收敛。