# DingQue Bug - 根本原因确认

## 🎯 找到了！

### Bug位置

**文件**: `crates/engine/src/obs/student.rs`  
**行数**: 94-100  
**Section**: Section 3: DING QUE (17 ch)

### 问题代码

```rust
// === Section 3: DING QUE (17 ch) ===
if let Some(suit) = p.ding_que {
    for t in suit.start()..suit.end() {
        w!(ch + suit as usize, t, 1.0);  // ← BUG!
    }
}
ch += 3;
```

### 问题分析

**在DingQue决策阶段**：
- `p.ding_que` 是 `None`（还没选择）
- Section 3的前3个通道**全是0**
- 模型**看不到手牌的花色分布信息**！

**在DingQue之后**：
- `p.ding_que` 是 `Some(suit)`
- Section 3标记了已选择的花色
- 但这时已经不需要做DingQue决策了

### 为什么会导致极端偏差

1. **缺少关键信息**：模型在做DingQue决策时，Section 3（本应提供花色统计）是空的
2. **依赖其他信号**：模型只能从Section 1（手牌one-hot）中隐式学习花色分布
3. **训练不稳定**：由于信息不足，模型容易学到spurious correlations
4. **Augmentation放大问题**：
   - 83.3%的样本使用了错误的action映射（已修复）
   - 但即使修复后，observation encoding仍然缺少关键信息
   - 导致模型无法学习到正确的花色选择策略

### 正确的实现应该是

**在DingQue阶段**，Section 3应该编码：
```rust
// 通道0: Man花色的牌数量（归一化）
// 通道1: Pin花色的牌数量（归一化）
// 通道2: Sou花色的牌数量（归一化）
```

**在DingQue之后**，Section 3应该编码：
```rust
// 标记已选择的花色（当前实现）
```

## 修复方案

### 方案A: 修改Observation Encoding（推荐）

在DingQue阶段，Section 3应该提供花色统计：

```rust
// === Section 3: DING QUE (17 ch) ===
if let Some(suit) = p.ding_que {
    // DingQue已完成：标记选择的花色
    for t in suit.start()..suit.end() {
        w!(ch + suit as usize, t, 1.0);
    }
} else if board.phase == Phase::DingQue {
    // DingQue阶段：提供花色统计信息
    for suit in Suit::all() {
        let count = (suit.start()..suit.end())
            .filter(|&t| p.hand[t] > 0)
            .map(|t| p.hand[t] as u32)
            .sum::<u32>();
        fill_ch!(ch + suit as usize, count as f32 / 13.0);
    }
}
ch += 3;
```

**注意**：这会改变observation的语义，需要**重新训练模型**！

### 方案B: 添加Reward Shaping（临时方案）

如果不想改observation，可以添加explicit reward：

```python
# 在 selfplay_env.py 的 _compute_shaping_reward 中
if action in [31, 32, 33]:  # DingQue actions
    suit = action - 31
    hand = self._env.get_agent_hand()
    suit_counts = [
        sum(1 for t in hand if 0 <= t < 9),   # Man
        sum(1 for t in hand if 9 <= t < 18),  # Pin  
        sum(1 for t in hand if 18 <= t < 27), # Sou
    ]
    # 奖励选择最少的花色
    min_count = min(suit_counts)
    if suit_counts[suit] == min_count:
        bonus += 0.1
```

### 方案C: 强制Exploration（最简单）

```yaml
# 在 warmup.yaml 中
exploration_loss_coeff: 0.15  # 从0.03提升到0.15
```

## 为什么之前没发现

1. **Section 3看起来是对的**：代码逻辑没有明显错误
2. **其他Section有手牌信息**：Section 1有完整的手牌one-hot
3. **但信息不够显式**：模型需要从27维one-hot中自己学习花色统计
4. **Augmentation bug掩盖了问题**：之前的bug太明显，掩盖了这个更深层的问题

## 下一步

1. **立即修复observation encoding**（方案A）
2. **重新训练模型**（必须，因为observation语义改变）
3. **验证修复效果**

这个bug解释了为什么：
- ✅ 权重是均匀的（模型本身没问题）
- ✅ Mask是正确的（环境没问题）
- ✅ Augmentation已修复（数据流没问题）
- ❌ 但行为仍然极端（**observation缺少关键信息**）