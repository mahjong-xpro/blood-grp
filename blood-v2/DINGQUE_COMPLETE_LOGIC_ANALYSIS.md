# 定缺完整逻辑分析：从业务到代码

## 问题：均匀先验会阻碍AI学习真正的定缺策略吗？

## 第一部分：定缺的业务逻辑

### 什么是定缺？
血战麻将的第一步：每个玩家选择一个花色（万/筒/索），之后必须打光该花色的所有牌。

### 最优定缺策略是什么？

#### 基础策略：选择最少的花色
```
手牌: 万4张, 筒3张, 索6张
最优: 选筒（最少）
原因: 最快打光，最早完成定缺
```

#### 进阶策略：考虑手牌结构
```
情况1: 万4张但都是孤张 [1m, 3m, 7m, 9m]
       筒3张但是顺子 [4p, 5p, 6p]
       → 应该选万（虽然多，但都是废牌）

情况2: 万5张 [1m, 1m, 2m, 3m, 4m] (有面子)
       筒3张 [1p, 5p, 9p] (孤张)
       → 应该选筒（虽然少，但万有价值）
```

#### 高级策略：考虑番数潜力
```
情况: 万7张 [1m-7m] (一条龙潜力)
      筒2张 [1p, 9p]
      → 可能选筒，保留一条龙番数
```

### 真正的定缺思想
**核心**: 选择对和牌最不利的花色，最大化剩余手牌的和牌潜力和番数。

## 第二部分：代码实现的完整流程

### 1. 观测编码（信息输入）
**位置**: [`student.rs:94-113`](blood-v2/crates/engine/src/obs/student.rs:94-113)

```rust
// Section 3: DING QUE (17 ch)
if board.phase == Phase::DingQue {
    // 通道0-2: 提供花色统计信息
    for suit in Suit::all() {
        let count = (suit.start()..suit.end())
            .filter(|&t| p.hand[t] > 0)
            .map(|t| p.hand[t] as u32)
            .sum::<u32>();
        fill_ch!(ch + suit as usize, count as f32 / 13.0);
    }
}
```

**提供的信息**:
- 万有多少张（归一化到0-1）
- 筒有多少张（归一化到0-1）
- 索有多少张（归一化到0-1）

**问题**: 只提供数量，不提供结构（是否有面子、孤张等）

### 2. 模型推理（策略学习）
**位置**: [`factory.py:247-292`](blood-v2/python/blood/model/factory.py:247-292)

```python
# 模型前向传播
action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_features)

# 应用动作掩码
illegal = ~mask.bool()
mask_value = torch.finfo(action_distribution_params.dtype).min
action_distribution_params = action_distribution_params.masked_fill(illegal, mask_value)

# 定缺均匀先验
dq_mask = mask[:, 31:34]
other_mask = mask[:, :31]
is_dingque = dq_mask.all(dim=-1) & (~other_mask.any(dim=-1))

if is_dingque.any():
    dq_logits = action_distribution_params[:, 31:34]
    dq_mean = dq_logits.mean(dim=-1, keepdim=True)
    uniform_logits = dq_mean.expand_as(dq_logits)
    action_distribution_params[:, 31:34] = torch.where(
        is_dingque.unsqueeze(-1),
        uniform_logits,
        action_distribution_params[:, 31:34]
    )
```

**关键问题**: 均匀先验**完全覆盖**了模型的输出！

### 3. 动作采样（决策执行）
```python
# Softmax + 采样
probs = F.softmax(action_distribution_params, dim=-1)
action = torch.multinomial(probs, 1)
```

**结果**: 定缺阶段永远是均匀采样（33.3% / 33.3% / 33.3%）

### 4. 奖励反馈（学习信号）
**位置**: [`selfplay_env.py:543-557`](blood-v2/python/blood/env/selfplay_env.py:543-557)

```python
def _compute_shaping_reward(self, prev_score: float, action: int, ow_before) -> float:
    bonus = 0.0
    
    # DingQue reward shaping: DISABLED
    # 定缺奖励塑形已禁用
    pass
    
    # ... 其他奖励 ...
    return bonus
```

**问题**: 定缺决策**没有即时奖励**，只能通过游戏结束时的最终得分学习。

## 第三部分：问题诊断

### 问题1: 均匀先验完全覆盖模型输出 ❌

**当前代码**:
```python
uniform_logits = dq_mean.expand_as(dq_logits)
action_distribution_params[:, 31:34] = torch.where(
    is_dingque.unsqueeze(-1),
    uniform_logits,  # 完全替换为均值
    action_distribution_params[:, 31:34]
)
```

**效果**: 
- 模型输出: [2.0, 1.0, 3.0] (模型认为索子最好)
- 先验后: [2.0, 2.0, 2.0] (强制均匀)
- Softmax: [0.333, 0.333, 0.333]

**结果**: **模型永远学不到定缺策略**，因为它的输出被完全忽略了！

### 问题2: 观测编码信息不足 ⚠️

**当前观测**: 只有花色数量
```
万: 0.31 (4张)
筒: 0.23 (3张)
索: 0.46 (6张)
```

**缺失信息**:
- 手牌结构（是否有面子、搭子）
- 孤张数量
- 番数潜力
- 向听数影响

**结果**: 即使模型能学习，信息也不够做出最优决策。

### 问题3: 奖励信号稀疏 ⚠️

**当前**: 定缺决策没有即时奖励
**问题**: 
- 定缺 → 打牌 → ... → 游戏结束 → 得分
- 信用分配困难：很难将最终得分归因到定缺决策

## 第四部分：正确的设计应该是什么？

### 方案A: 渐进式先验（推荐）⭐⭐⭐

```python
# 训练初期：强先验（防止偏差）
# 训练后期：弱先验（允许学习）

prior_strength = max(0.0, 1.0 - global_steps / warmup_steps)
# 0-50K步: prior_strength = 1.0 → 0.5 (强制均匀)
# 50K-200K步: prior_strength = 0.5 → 0.0 (逐渐放开)

mixed = dq_logits * (1.0 - prior_strength) + dq_mean * prior_strength
```

**优势**:
- 初期防止偏差自强化
- 后期允许模型学习真正策略
- 平滑过渡，训练稳定

### 方案B: 增强观测编码 ⭐⭐

```rust
// 不仅提供数量，还提供结构信息
if board.phase == Phase::DingQue {
    for suit in Suit::all() {
        // 1. 总数量
        let count = suit_tile_count(&p.hand, suit);
        fill_ch!(ch, count as f32 / 13.0);
        ch += 1;
        
        // 2. 孤张数量
        let isolated = count_isolated_tiles(&p.hand, suit);
        fill_ch!(ch, isolated as f32 / 9.0);
        ch += 1;
        
        // 3. 面子/搭子数量
        let groups = count_groups(&p.hand, suit);
        fill_ch!(ch, groups as f32 / 4.0);
        ch += 1;
    }
}
```

**优势**:
- 提供足够信息让模型学习进阶策略
- 不需要修改Python代码
- 需要重新训练模型（观测维度变化）

### 方案C: 添加定缺奖励塑形 ⭐

```python
def _compute_shaping_reward(self, prev_score: float, action: int, ow_before) -> float:
    bonus = 0.0
    
    if 31 <= action <= 33:
        # 计算定缺质量
        suit_counts = get_suit_counts()
        isolated_counts = get_isolated_counts()
        
        chosen_suit = action - 31
        # 奖励选择孤张多的花色
        bonus += 0.01 * isolated_counts[chosen_suit]
        # 惩罚选择有面子的花色
        bonus -= 0.02 * group_counts[chosen_suit]
    
    return bonus
```

**优势**:
- 提供即时学习信号
- 引导模型学习正确策略
- 不需要重新训练

## 第五部分：当前设计的问题总结

### ❌ 致命问题：模型无法学习

**原因**: 均匀先验**完全覆盖**模型输出
```python
# 这行代码让模型的定缺输出完全无效
uniform_logits = dq_mean.expand_as(dq_logits)
```

**后果**:
1. 模型永远输出均匀分布
2. 梯度无法反向传播到定缺决策
3. 模型无法从经验中学习定缺策略

### ⚠️ 次要问题：信息和奖励不足

1. **观测编码**: 只有数量，缺少结构信息
2. **奖励信号**: 没有即时反馈，信用分配困难

## 第六部分：推荐的修复方案

### 立即修复：渐进式先验

```python
# factory.py 添加全局步数追踪
self._dingque_prior_warmup_steps = getattr(cfg, "dingque_prior_warmup_steps", 100_000)

# forward_tail 中修改先验逻辑
if is_dingque.any():
    # 计算先验强度（随训练步数衰减）
    global_steps = getattr(self, "_global_training_steps", 0)
    prior_strength = max(0.0, 1.0 - global_steps / self._dingque_prior_warmup_steps)
    
    dq_logits = action_distribution_params[:, 31:34]
    dq_mean = dq_logits.mean(dim=-1, keepdim=True)
    
    # 混合策略：初期强制均匀，后期允许学习
    mixed = dq_logits * (1.0 - prior_strength) + dq_mean * prior_strength
    
    action_distribution_params[:, 31:34] = torch.where(
        is_dingque.unsqueeze(-1),
        mixed,
        action_distribution_params[:, 31:34]
    )
```

### 长期优化：增强观测 + 奖励塑形

1. 修改 `student.rs` 添加手牌结构信息
2. 启用定缺奖励塑形（基于孤张数量）
3. 重新训练模型

## 结论

### 当前设计的问题
❌ **均匀先验完全阻止了AI学习定缺策略**
- 模型输出被完全覆盖
- 梯度无法传播
- 永远是33.3%均匀分布

### AI能学会真正的定缺思想吗？
**当前设计**: ❌ **不能**
- 均匀先验 = 硬编码策略
- 模型没有学习机会

**修复后**: ✅ **可以**
- 渐进式先验：初期防偏差，后期允许学习
- 增强观测：提供足够信息
- 奖励塑形：提供学习信号

### 推荐行动
1. **立即**: 实现渐进式先验（100K步衰减）
2. **中期**: 添加定缺奖励塑形
3. **长期**: 增强观测编码（需要重新训练）

当前的"均匀先验"是为了**修复偏差bug**的临时方案，但它**牺牲了学习能力**。需要改为**渐进式先验**才能让AI真正学会定缺策略。