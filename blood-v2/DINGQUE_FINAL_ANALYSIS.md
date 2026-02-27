# DingQue Bug 最终深度分析

## 问题现象

Player 0 (AI agent) 在评估中持续表现出极端的 dingque 偏好：
- 第1次训练: Sou 100%
- 第2次训练: Pin 92%, Sou 8%
- 第3次训练: Sou 92%, Pin 8%

**关键观察**: Man (万) 始终是 0%

## 已排除的原因

### 1. ✅ Suit Augmentation Bug (已修复但不是根本原因)
- **发现**: `augment.py` 使用 `perm.index()` 而不是 `perm[old_suit]`
- **修复**: 已改为正向映射
- **结果**: 锁定的花色会变化，但问题依然存在
- **结论**: Augmentation bug 会加剧问题，但不是根本原因

### 2. ✅ Exploration Coefficient (已优化但效果有限)
- **发现**: 原始值 0.01 太小
- **修复**: 提升到 0.03
- **结果**: 问题依然存在
- **结论**: Exploration 不足是症状，不是病因

### 3. ✅ 模型初始化偏差 (已验证不存在)
- **测试**: `test_dingque_init.py` 验证了初始化均匀性
- **结果**: χ² < 2.0，无系统性偏差
- **结论**: 初始化正常

### 4. ✅ Action Mask 错误 (已验证正确)
- **检查**: Rust engine 正确生成 mask[31:34] = [1, 1, 1]
- **结论**: Action mask 正常

## 真正的根本原因

### 核心问题：**训练数据已被污染 + 缺乏 Dingque Reward Signal**

#### 原因 1: 历史训练数据污染

旧的 checkpoint 是用**错误的 augmentation** 训练的：
- 83.3% 的训练样本使用错误的 action 映射
- 模型学习到的策略完全混乱
- 即使修复代码，从旧 checkpoint 继续训练仍会保留错误策略

#### 原因 2: 缺乏 Dingque 阶段的 Reward Signal

查看 `selfplay_env.py:547-557`：

```python
# DingQue: No explicit reward shaping for dingque choice.
# Let the agent learn the optimal strategy through downstream rewards
# (winning, shanten progress, etc.).
```

**设计意图**: 让模型通过下游奖励（胜利、向听进步）学习最优 dingque 策略

**实际问题**:
1. Dingque 决策与最终胜利之间的因果链太长（~100步）
2. 信用分配问题严重
3. 在错误的 augmentation 下，reward 信号完全混乱
4. 模型无法学习到正确的 dingque 策略

#### 原因 3: 为什么 Man 始终是 0%？

这不是随机的！可能的原因：

1. **Observation Encoding 偏差**
   - Man/Pin/Sou 在 observation 中的编码位置不同
   - 可能存在某种系统性偏差使得 Man 的特征不明显

2. **初始随机选择的锁定效应**
   - 训练初期随机选择了 Pin 或 Sou
   - 由于 exploration 不足 + reward 信号混乱
   - 模型锁定在这个选择上
   - Man 从未被充分探索

3. **Augmentation 的累积效应**
   - 错误的 augmentation 可能系统性地惩罚了 Man 的选择
   - 导致模型学习到"避免 Man"的策略

## 解决方案

### 方案 A: 完全重新训练（推荐）

```bash
cd blood-v2

# 1. 验证修复
pytest tests/test_augment_fix.py tests/test_augment_roundtrip.py -v

# 2. 完全删除所有旧数据
rm -rf train_dir/blood_v2_*
rm -rf checkpoints/*

# 3. 从头开始训练
./scripts/manage.sh train warmup
```

**预期**: 需要训练足够长时间（至少 500K steps）才能看到均匀分布

### 方案 B: 添加 Dingque Reward Shaping（如果方案A失败）

在 `selfplay_env.py` 的 `_compute_shaping_reward` 中添加：

```python
# Dingque reward: encourage balanced exploration
if self._env.get_phase() == "ding_que" and not self._env.get_ding_que_done()[0]:
    # Reward choosing the suit with minimum count in hand
    hand = self._env.get_agent_hand()
    suit_counts = [
        sum(1 for t in hand if 0 <= t < 9),   # Man
        sum(1 for t in hand if 9 <= t < 18),  # Pin
        sum(1 for t in hand if 18 <= t < 27), # Sou
    ]
    
    # Map action to suit
    if 31 <= action <= 33:
        chosen_suit = action - 31
        min_count = min(suit_counts)
        
        # Reward if chosen suit has minimum count
        if suit_counts[chosen_suit] == min_count:
            bonus += 0.05  # Small bonus for optimal choice
```

### 方案 C: 强制均匀探索（临时措施）

在训练初期（前 100K steps）强制使用 ε-greedy：

```python
if phase == "ding_que" and global_steps < 100000:
    if random.random() < 0.3:  # 30% 随机
        action = random.choice([31, 32, 33])
```

## 验证方法

训练完成后：

```bash
# 1. 录制 100 局游戏
./scripts/manage.sh record --games 100

# 2. 分析 dingque 分布
python scripts/analyze_dingque.py replays/

# 3. 检查 TensorBoard
tensorboard --logdir train_dir/
# 查看 blood/policy_entropy 是否保持在合理范围
```

**预期结果**:
- Man: 25-40%
- Pin: 25-40%
- Sou: 25-40%

**注意**: 不一定是精确的 33.3%，因为：
- 模型会学习根据手牌结构选择
- 轻微的不均匀是正常的策略学习
- 关键是避免 >80% 的极端偏好

## 深层次问题

### 为什么这个 Bug 如此顽固？

1. **多重因素叠加**
   - Augmentation bug (83.3% 错误样本)
   - Exploration 不足 (0.01 太小)
   - 缺乏 reward signal (无 dingque shaping)
   - 长因果链 (dingque → 胜利 ~100步)

2. **正反馈循环**
   - 随机选择某个花色 → 略微表现更好（噪声）
   - 策略梯度强化这个选择 → 更频繁选择
   - 更多数据强化偏好 → 锁定

3. **Checkpoint 污染**
   - 旧 checkpoint 包含错误策略
   - 继续训练会保留这个偏差
   - 必须完全重新训练

### 教训

1. **数据增强需要严格测试**
   - 每个 permutation 都要验证
   - 需要 round-trip 测试
   - 需要可逆性验证

2. **Exploration 至关重要**
   - 对于离散选择（如 dingque），需要足够的 exploration
   - 0.01 对于 3 个选项来说太小
   - 建议至少 0.03-0.05

3. **Reward Signal 设计**
   - 长因果链需要中间 reward
   - "让模型自己学习"在某些情况下不可行
   - 需要平衡 shaping 和 exploration

4. **监控和调试**
   - 需要持续监控关键指标（如 dingque 分布）
   - 异常模式（如 100% 单一选择）应该立即触发警报
   - 需要完善的调试工具

## 总结

Dingque bug 是一个**多因素复合问题**：

1. **直接原因**: Augmentation 映射错误 (已修复)
2. **加剧因素**: Exploration 不足 (已优化)
3. **根本原因**: 训练数据污染 + 缺乏 reward signal
4. **解决方案**: 完全重新训练 + 可选的 reward shaping

**关键**: 必须从头开始训练，不能从旧 checkpoint 继续。