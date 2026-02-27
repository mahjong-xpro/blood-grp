# DingQue 偏差完整分析

## 当前状态
- Man: 43% ✓
- Pin: 6% ⚠️（严重偏低）
- Sou: 51% ⚠️

## 已修复的问题

### 1. 定缺检测逻辑错误 ✅
**位置**: [`factory.py:267`](blood-v2/python/blood/model/factory.py:267), [`losses.py:87`](blood-v2/python/blood/training/losses.py:87)

**修复前**:
```python
is_dingque = dq_mask.any(dim=-1)  # 错误：只要有任何定缺动作合法
```

**修复后**:
```python
is_dingque = dq_mask.all(dim=-1) & (~other_mask.any(dim=-1))  # 正确：31/32/33全合法且其他全非法
```

### 2. 先验策略优化 ✅
**位置**: [`factory.py:276-283`](blood-v2/python/blood/model/factory.py:276-283)

**策略**: 将3个定缺动作的logits设为它们的均值，确保softmax后完全均匀

## 剩余问题分析

### 问题1: 观测编码中的花色统计特征
**位置**: [`student.rs:100-112`](blood-v2/crates/engine/src/obs/student.rs:100-112)

**代码**:
```rust
if board.phase == Phase::DingQue {
    for suit in Suit::all() {
        let count = (suit.start()..suit.end())
            .filter(|&t| p.hand[t] > 0)
            .map(|t| p.hand[t] as u32)
            .sum::<u32>();
        fill_ch!(ch + suit as usize, count as f32 / 13.0);
    }
}
```

**问题**:
1. 这个特征告诉模型"Man有X张，Pin有Y张，Sou有Z张"
2. 如果初始手牌分配有系统性偏差（如Pin平均比其他花色少），模型会学习到"Pin少→选Pin"
3. 但我们的先验强制均匀分布，导致冲突

**验证方法**:
```python
# 统计1000局初始手牌的花色分布
import numpy as np
from blood_mahjong_env import BloodMahjongEnv

suit_counts = {'man': [], 'pin': [], 'sou': []}
for _ in range(1000):
    env = BloodMahjongEnv()
    obs = env.reset()
    # 从观测中提取花色统计（channels 5-7）
    man_count = obs['obs'][5*27:(5+1)*27].sum()
    pin_count = obs['obs'][6*27:(6+1)*27].sum()
    sou_count = obs['obs'][7*27:(7+1)*27].sum()
    suit_counts['man'].append(man_count)
    suit_counts['pin'].append(pin_count)
    suit_counts['sou'].append(sou_count)

print(f"Man: {np.mean(suit_counts['man']):.2f} ± {np.std(suit_counts['man']):.2f}")
print(f"Pin: {np.mean(suit_counts['pin']):.2f} ± {np.std(suit_counts['pin']):.2f}")
print(f"Sou: {np.mean(suit_counts['sou']):.2f} ± {np.std(suit_counts['sou']):.2f}")
```

### 问题2: RuleBot对手的定缺策略
**位置**: [`opponent.rs:15-27`](blood-v2/crates/pybind/src/opponent.rs:15-27)

**代码**:
```rust
pub fn choose_ding_que(&self, board: &BoardState, player_id: usize) -> Action {
    let p = &board.players[player_id];
    let mut best_suit = Suit::Man;
    let mut min_count = u8::MAX;
    for suit in Suit::all() {
        let count = suit_tile_count(&p.hand, suit);
        if count < min_count {
            min_count = count;
            best_suit = suit;
        }
    }
    Action::DingQue(best_suit)
}
```

**问题**:
- RuleBot选择最少的花色（正确策略）
- 如果初始手牌分配有偏差，RuleBot会系统性地偏向某个花色
- 这会影响训练数据的分布

### 问题3: 样本量不足
- 当前只有100个样本
- 统计波动可能导致6%的Pin比例
- 需要更多样本（1000+）才能确认是否有系统性偏差

## 解决方案

### 方案A: 移除定缺阶段的花色统计特征（推荐）
**优点**:
- 消除观测编码与先验的冲突
- 强制模型依赖先验而非学习偏差
- 简单直接

**缺点**:
- 需要重新训练模型（观测维度变化）
- 模型无法基于手牌统计做智能选择

**实现**:
```rust
// student.rs line 100-112
if board.phase == Phase::DingQue {
    // 移除花色统计，只保留phase标记
    // 模型将完全依赖先验做出均匀选择
}
```

### 方案B: 增强先验强度（当前方案）
**优点**:
- 不需要重新训练
- 保留模型学习智能策略的能力

**缺点**:
- 如果观测编码有强偏差，先验可能不够强
- 需要更多训练步数才能收敛

**当前实现**: 已完成，使用均值先验

### 方案C: 验证并修复初始手牌分配
**步骤**:
1. 运行上述验证脚本，检查初始手牌分布
2. 如果发现偏差，检查 `generate_deck` 函数
3. 确保洗牌算法无偏

## 推荐行动

### 立即执行
1. **运行验证脚本**：确认初始手牌分布是否均匀
2. **增加样本量**：训练至少10K步，收集1000+定缺样本
3. **监控分布收敛**：观察Pin比例是否逐渐接近33%

### 如果Pin持续<20%
1. **检查初始手牌分配**：可能是随机数生成器或洗牌算法有问题
2. **考虑方案A**：移除定缺阶段的花色统计特征
3. **增强先验**：将均值先验改为固定值（如全部设为0）

### 长期优化
1. **动态先验**：根据训练进度逐渐减弱先验强度
2. **对抗训练**：使用多样化的对手策略避免偏差
3. **数据增强**：人工平衡定缺样本的花色分布

## 预期时间线
- **10K步**：Pin应该>15%
- **50K步**：Pin应该>25%
- **200K步**：Pin应该稳定在30-35%

如果50K步后Pin仍<20%，说明有系统性问题需要深入修复。