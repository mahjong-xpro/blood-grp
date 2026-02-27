# 定缺系统完全重构设计

## 问题诊断

### 当前系统的致命缺陷
1. **均匀先验完全覆盖模型输出** - 导致梯度无法传播，模型永远无法学习
2. **观测编码信息不足** - 只有花色数量统计，缺少手牌结构信息
3. **奖励信号稀疏** - 定缺决策没有即时反馈，只能依赖最终得分
4. **Oracle初始化偏差** - Man 40.9%, Pin 31.1%, Sou 28.0%（来自RuleBot）

### 根本原因
- 均匀先验是为了修复偏差bug的**临时方案**，但牺牲了学习能力
- 模型无法从经验中学习最优定缺策略
- 缺少引导信号帮助模型理解"什么是好的定缺"

---

## 新系统设计

### 核心理念
**渐进式学习**：训练初期强先验防止偏差，后期弱先验允许自主学习

### 1. 渐进式先验机制

#### 设计原理
```python
# 先验强度随训练步数线性衰减
prior_strength = max(0.0, 1.0 - global_steps / warmup_steps)

# 混合模型输出和均匀先验
logits_mixed = (1 - prior_strength) * logits_model + prior_strength * logits_uniform
```

#### 三阶段训练
1. **Phase 1 (0-50K steps)**: `prior_strength = 1.0 → 0.5`
   - 强先验主导，确保均匀分布
   - 模型开始接收梯度信号
   
2. **Phase 2 (50K-100K steps)**: `prior_strength = 0.5 → 0.1`
   - 先验逐渐减弱，模型逐渐主导
   - 允许模型探索不同策略
   
3. **Phase 3 (100K+ steps)**: `prior_strength = 0.1 → 0.0`
   - 模型完全自主决策
   - 先验仅作为微弱正则化

#### 实现位置
- **文件**: `blood-v2/python/blood/model/factory.py`
- **方法**: `PolicyModel.forward()` 中的定缺动作处理
- **配置**: `dingque_prior_warmup_steps` (default: 100000)

---

### 2. 增强观测编码

#### 当前观测 (student.rs Section 3)
```rust
// 通道 18-20: 定缺花色 one-hot
// 通道 21: 定缺完成标志
```

#### 新增观测通道
```rust
// Section 3 扩展 (新增 9 个通道)
// 通道 22-24: 各花色孤张数量 / 13.0
//   - 孤张 = 无法组成面子/搭子的牌
//   - 定缺时应优先打光孤张多的花色
//
// 通道 25-27: 各花色面子数量 / 4.0
//   - 面子 = 刻子/顺子
//   - 面子多的花色应保留
//
// 通道 28-30: 各花色搭子数量 / 6.0
//   - 搭子 = 两面/嵌张/边张
//   - 搭子多的花色有潜力
```

#### 实现位置
- **文件**: `blood-v2/crates/engine/src/obs/student.rs`
- **方法**: `encode_section_3()` 扩展
- **依赖**: 需要实现孤张/面子/搭子检测算法

---

### 3. 定缺奖励塑形

#### 设计原理
基于手牌结构的即时奖励，引导模型学习"好的定缺"

#### 奖励公式
```python
# 定缺后立即给予奖励
dingque_reward = base_reward * structure_quality

# 结构质量评分 (0-1)
structure_quality = (
    0.5 * isolated_tile_ratio +  # 孤张比例
    0.3 * (1 - meld_ratio) +      # 面子稀缺度
    0.2 * (1 - taatsu_ratio)      # 搭子稀缺度
)

# 基础奖励
base_reward = 0.01  # 小奖励，不干扰主要得分信号
```

#### 实现位置
- **文件**: `blood-v2/python/blood/env/selfplay_env.py`
- **方法**: `_compute_shaping_reward()` 中启用定缺部分
- **配置**: `dingque_reward_shaping` (default: True)

---

### 4. Oracle处理策略

#### 问题
Oracle (RuleBot) 的定缺分布本身有偏差 (Man 40.9%, Pin 31.1%, Sou 28.0%)

#### 解决方案A: 均匀Oracle
```python
# 定缺阶段强制Oracle输出均匀分布
if is_dingque_phase:
    oracle_logits[31:34] = 0.0  # 均匀logits
```

#### 解决方案B: 降低权重
```python
# 定缺阶段降低Oracle蒸馏权重
oracle_weight = base_weight * (0.1 if is_dingque_phase else 1.0)
```

#### 推荐方案
**方案A（均匀Oracle）** - 更简单直接，避免引入偏差

#### 实现位置
- **文件**: `blood-v2/python/blood/training/losses.py`
- **方法**: `_oracle_ce_loss()` 中添加定缺检测
- **配置**: `oracle_dingque_uniform` (default: True)

---

## 实现计划

### Step 1: 渐进式先验 (factory.py)
```python
class PolicyModel(nn.Module):
    def __init__(self, cfg):
        # ...
        self.dingque_prior_warmup_steps = getattr(cfg, 'dingque_prior_warmup_steps', 100000)
        self.global_steps = 0
    
    def forward(self, obs, mask):
        # ... 正常前向传播 ...
        
        # 定缺阶段检测
        dingque_mask = (mask[:, 31:34].sum(dim=1) > 0.5)
        
        if dingque_mask.any():
            # 计算先验强度
            prior_strength = max(0.0, 1.0 - self.global_steps / self.dingque_prior_warmup_steps)
            
            # 均匀先验 logits
            uniform_logits = torch.zeros_like(logits[:, 31:34])
            
            # 混合模型输出和先验
            logits[dingque_mask, 31:34] = (
                (1 - prior_strength) * logits[dingque_mask, 31:34] +
                prior_strength * uniform_logits
            )
        
        return logits, value
```

### Step 2: 增强观测编码 (student.rs)
```rust
// 在 encode_section_3() 中添加
fn encode_hand_structure(hand: &Hand, suit: Suit) -> (f32, f32, f32) {
    let tiles: Vec<u8> = hand.tiles_of_suit(suit);
    
    // 孤张检测：无法组成面子/搭子的牌
    let isolated = count_isolated_tiles(&tiles);
    
    // 面子检测：刻子/顺子
    let melds = count_melds(&tiles);
    
    // 搭子检测：两面/嵌张/边张
    let taatsu = count_taatsu(&tiles);
    
    (isolated as f32 / 13.0, melds as f32 / 4.0, taatsu as f32 / 6.0)
}
```

### Step 3: 定缺奖励塑形 (selfplay_env.py)
```python
def _compute_dingque_reward(self, chosen_suit: int) -> float:
    """计算定缺即时奖励"""
    if not self._dingque_reward_shaping:
        return 0.0
    
    # 从观测中提取手牌结构信息
    obs = self._env.get_player_obs(0)["obs"]
    obs_2d = obs.reshape(-1, 27)
    
    # 提取选择花色的结构信息
    isolated = obs_2d[22 + chosen_suit].mean()
    melds = obs_2d[25 + chosen_suit].mean()
    taatsu = obs_2d[28 + chosen_suit].mean()
    
    # 结构质量评分
    quality = 0.5 * isolated + 0.3 * (1 - melds) + 0.2 * (1 - taatsu)
    
    return 0.01 * quality
```

### Step 4: Oracle均匀化 (losses.py)
```python
def _oracle_ce_loss(self, oracle_logits, actions, advantages, mask):
    # 定缺阶段检测
    dingque_actions = (actions >= 31) & (actions <= 33)
    
    if self.oracle_dingque_uniform and dingque_actions.any():
        # 定缺动作强制均匀Oracle
        oracle_logits[dingque_actions, 31:34] = 0.0
    
    # ... 正常CE计算 ...
```

---

## 配置参数

### 新增配置项 (configs/default.yaml)
```yaml
# 定缺系统配置
dingque_prior_warmup_steps: 100000  # 先验衰减步数
dingque_reward_shaping: true        # 是否启用奖励塑形
oracle_dingque_uniform: true        # Oracle是否强制均匀
```

---

## 验证指标

### 训练过程监控
1. **定缺分布**: 每10K步记录 Man/Pin/Sou 比例
2. **先验强度**: 记录当前 `prior_strength` 值
3. **定缺奖励**: 记录平均定缺奖励值

### 收敛标准
- **Phase 1 (0-50K)**: 分布偏差 < 5% (28-38% per suit)
- **Phase 2 (50K-100K)**: 分布偏差 < 10% (23-43% per suit)
- **Phase 3 (100K+)**: 允许自然分布，监控是否出现极端偏差 (>60%)

### 性能验证
- 对比重构前后的胜率和平均得分
- 确保定缺策略学习不影响整体性能

---

## 风险与缓解

### 风险1: 渐进式先验可能仍然偏差
**缓解**: 
- 监控训练过程，如果Phase 1结束时仍有偏差，延长warmup
- 可调整衰减曲线（线性 → 指数衰减）

### 风险2: 奖励塑形可能与最优策略冲突
**缓解**:
- 使用小奖励值 (0.01)，不干扰主要信号
- 可通过配置完全禁用

### 风险3: 观测编码增加计算开销
**缓解**:
- 孤张/面子/搭子检测可复用现有shanten算法
- 仅在定缺阶段计算，开销可控

---

## 实施顺序

1. **Step 1**: 实现渐进式先验 (factory.py) - **最关键**
2. **Step 2**: Oracle均匀化 (losses.py) - **防止偏差传播**
3. **Step 3**: 增强观测编码 (student.rs) - **提供更好信息**
4. **Step 4**: 定缺奖励塑形 (selfplay_env.py) - **加速学习**

前两步是**必需**的，后两步是**增强**的。

---

## 总结

这个重构设计彻底解决了当前系统的根本问题：
- ✅ 允许模型学习（渐进式先验）
- ✅ 提供更好信息（增强观测）
- ✅ 加速学习过程（奖励塑形）
- ✅ 防止偏差传播（Oracle均匀化）

**不是简单修补，而是完全重新设计。**