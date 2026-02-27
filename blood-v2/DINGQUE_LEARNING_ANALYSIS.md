# 定缺系统学习能力分析

## 问题1: 配置文件是否正确？

### ✅ 所有配置文件已更新

已在以下配置文件中添加定缺系统参数：

1. **default.yaml** ✅
2. **warmup.yaml** ✅
3. **competitive.yaml** ✅
4. **elite.yaml** ✅

### 配置参数说明

```yaml
# DingQue system — progressive prior + Oracle uniformization
dingque_prior_enabled: true           # 启用渐进式先验
dingque_prior_warmup_steps: 100000    # 先验从1.0衰减到0.0的步数
oracle_dingque_uniform: true          # Oracle强制输出均匀分布
dingque_reward_shaping: false         # 奖励塑形（暂未实现）
```

### 配置一致性验证

所有训练阶段使用**相同的定缺参数**：
- Warmup (2M steps) → 前100K步渐进式先验
- Competitive (1M steps) → 继承warmup权重，继续学习
- Elite (200M steps) → 完全自主决策

---

## 问题2: 模型能自己学到定缺策略吗？

### 答案：✅ 能！新设计允许模型学习

### 关键机制对比

#### 旧系统（无法学习）
```
定缺阶段:
  模型输出 logits_model → 被完全覆盖 → logits_final = [0, 0, 0] (均匀)
  
梯度反向传播:
  ∂Loss/∂logits_final = [...] → ∂Loss/∂logits_model = 0 (梯度消失)
  
结果: 模型参数不更新，无法学习
```

#### 新系统（可以学习）
```
定缺阶段:
  模型输出 logits_model → 混合先验 → logits_final = (1-α)*logits_model + α*[0,0,0]
  
梯度反向传播:
  ∂Loss/∂logits_final = [...] → ∂Loss/∂logits_model = (1-α) * ∂Loss/∂logits_final
  
结果: 梯度按比例传播，模型可以学习
```

### 学习过程三阶段

#### Phase 1: 强先验主导 (0-50K steps)
```
prior_strength: 1.0 → 0.5
梯度传播: 0% → 50%

行为:
- 输出接近均匀分布 (33.3% ± 5%)
- 模型开始接收梯度信号
- 学习"定缺是什么"

示例:
step=0:    logits_final = 0.0*logits_model + 1.0*[0,0,0] = [0,0,0]
step=25K:  logits_final = 0.5*logits_model + 0.5*[0,0,0] = 混合
step=50K:  logits_final = 0.5*logits_model + 0.5*[0,0,0] = 混合
```

#### Phase 2: 渐进式过渡 (50K-100K steps)
```
prior_strength: 0.5 → 0.1
梯度传播: 50% → 90%

行为:
- 模型逐渐主导决策
- 探索不同定缺策略
- 学习"什么是好的定缺"

示例:
step=75K:  logits_final = 0.75*logits_model + 0.25*[0,0,0]
step=100K: logits_final = 0.9*logits_model + 0.1*[0,0,0]
```

#### Phase 3: 完全自主 (100K+ steps)
```
prior_strength: 0.1 → 0.0
梯度传播: 90% → 100%

行为:
- 模型完全自主决策
- 可能学到非均匀策略
- 基于手牌结构优化

示例:
step=150K: logits_final = 1.0*logits_model + 0.0*[0,0,0] = logits_model
```

---

## 学习信号来源

### 1. 主要信号：最终得分
```python
# 定缺决策影响最终得分
好的定缺 → 更快听牌 → 更高和牌概率 → 更高得分
坏的定缺 → 慢听牌 → 低和牌概率 → 低得分

# PPO通过优势函数学习
advantage = returns - value_baseline
policy_loss = -log_prob(action) * advantage

# 好的定缺动作获得正优势，坏的获得负优势
```

### 2. 辅助信号：向听数进退
```python
# reward_shanten_progress = 0.003
# 定缺后向听数改善 → 即时正奖励
好的定缺 → 向听数-1 → +0.003 reward
坏的定缺 → 向听数+0 → 0 reward
```

### 3. 观测信号：手牌结构
```python
# Section 3 观测编码提供花色统计
obs[18:21] = 各花色牌数 / 13.0
obs[21] = 定缺完成标志

# 模型可以学习:
# "如果某花色只有3张 → 选择该花色"
# "如果某花色有8张 → 不选择该花色"
```

---

## 学习能力验证

### 理论保证

1. **梯度传播** ✅
   - 渐进式先验保证 `∂Loss/∂logits_model ≠ 0`
   - 梯度强度随训练增加: 0% → 100%

2. **探索空间** ✅
   - Phase 1: 强先验确保均匀探索
   - Phase 2-3: 模型可以探索非均匀策略

3. **奖励信号** ✅
   - 主信号: 最终得分（延迟但准确）
   - 辅助信号: 向听进退（即时但粗糙）

### 实验预测

#### 预期学习曲线
```
定缺分布:
0-50K:   Man 33% / Pin 33% / Sou 34%  (均匀)
50-100K: Man 30% / Pin 35% / Sou 35%  (开始偏离)
100K+:   Man 28% / Pin 36% / Sou 36%  (自然分布)

原因: 
- 统计上Pin/Sou略优（中张多，容易成型）
- 模型可能学到这个模式
```

#### 监控指标
```python
# 1. 定缺分布偏差
max_deviation = max(abs(ratio - 0.333) for ratio in [man, pin, sou])

# 2. 定缺后向听数改善
avg_shanten_delta = mean(shanten_after - shanten_before)

# 3. 定缺决策熵
entropy = -sum(p * log(p) for p in [man_prob, pin_prob, sou_prob])
```

---

## 与旧系统对比

### 旧系统问题
```
问题: 定缺分布严重偏差 (Man 0% / Pin 0% / Sou 100%)
原因: 
1. 初始化偏差 → 模型倾向选Sou
2. 均匀先验完全覆盖 → 无法学习纠正
3. 偏差永久固化

解决方案: 均匀先验
副作用: 模型无法学习任何定缺策略
```

### 新系统优势
```
优势1: 防止初期偏差
- 强先验确保训练初期均匀分布
- 避免陷入局部最优

优势2: 允许后期学习
- 先验逐渐减弱，模型逐渐主导
- 可以学习基于手牌结构的策略

优势3: 平滑过渡
- 渐进式衰减避免突变
- 训练稳定性好
```

---

## 潜在学习策略

### 模型可能学到的策略

#### 策略1: 孤张最多原则
```python
# 选择孤张（无法组成面子/搭子）最多的花色
isolated_count = {
    'Man': count_isolated(hand, suit='Man'),
    'Pin': count_isolated(hand, suit='Pin'),
    'Sou': count_isolated(hand, suit='Sou'),
}
best_suit = max(isolated_count, key=isolated_count.get)
```

#### 策略2: 面子最少原则
```python
# 选择已成型面子最少的花色
meld_count = {
    'Man': count_melds(hand, suit='Man'),
    'Pin': count_melds(hand, suit='Pin'),
    'Sou': count_melds(hand, suit='Sou'),
}
best_suit = min(meld_count, key=meld_count.get)
```

#### 策略3: 综合评分
```python
# 综合考虑多个因素
score = {
    suit: (
        0.5 * isolated_ratio +
        0.3 * (1 - meld_ratio) +
        0.2 * (1 - taatsu_ratio)
    )
    for suit in ['Man', 'Pin', 'Sou']
}
best_suit = max(score, key=score.get)
```

### 学习难度分析

#### 容易学习 ✅
- **数量原则**: "选择牌数最少的花色"
  - 信号明确，观测直接
  - 预计50K步内学会

#### 中等难度 ⚙️
- **孤张原则**: "选择孤张最多的花色"
  - 需要理解牌型结构
  - 预计100K-200K步学会

#### 困难学习 ⚠️
- **番数潜力**: "选择番数潜力高的花色"
  - 需要长期规划
  - 可能需要200K+步或额外信号

---

## 风险与缓解

### 风险1: 学习速度慢
**症状**: 100K步后仍然接近均匀分布

**原因**:
- 定缺信号稀疏（每局只有1次）
- 奖励延迟（定缺 → 和牌间隔长）

**缓解**:
- 延长warmup步数到150K或200K
- 实现定缺奖励塑形（即时反馈）
- 增强观测编码（提供更多信息）

### 风险2: 学到次优策略
**症状**: 分布偏离均匀但胜率下降

**原因**:
- 模型过拟合某种模式
- 探索不足陷入局部最优

**缓解**:
- 监控胜率和Elo变化
- 增加探索系数（entropy coeff）
- 使用更多样化的对手（league）

### 风险3: 训练不稳定
**症状**: 定缺分布剧烈波动

**原因**:
- 先验衰减过快
- 梯度噪声大

**缓解**:
- 调整衰减曲线（线性 → 指数）
- 增加batch size
- 降低学习率

---

## 总结

### 核心结论

✅ **模型能够学习定缺策略**

理由:
1. 渐进式先验允许梯度传播
2. 观测编码提供手牌结构信息
3. 奖励信号（得分+向听）提供学习方向

### 学习路径

```
Phase 1 (0-50K):
  先验主导 → 均匀分布 → 防止偏差

Phase 2 (50K-100K):
  渐进过渡 → 开始探索 → 学习基础策略

Phase 3 (100K+):
  完全自主 → 优化策略 → 可能超越均匀
```

### 预期结果

**保守估计**: 模型学会"选择牌数最少的花色"
- 简单有效的启发式
- 50K-100K步内学会

**乐观估计**: 模型学会"综合评估手牌结构"
- 考虑孤张、面子、搭子
- 100K-200K步内学会

**最优情况**: 模型学会"番数潜力评估"
- 长期规划，最大化期望得分
- 需要200K+步或额外增强

### 与旧系统对比

| 维度 | 旧系统 | 新系统 |
|------|--------|--------|
| 梯度传播 | ❌ 完全阻断 | ✅ 渐进式传播 |
| 学习能力 | ❌ 无法学习 | ✅ 可以学习 |
| 初期偏差 | ✅ 强制均匀 | ✅ 渐进式均匀 |
| 后期优化 | ❌ 永远均匀 | ✅ 可以优化 |
| 训练稳定性 | ✅ 非常稳定 | ⚙️ 需要监控 |

**新系统在保持初期稳定性的同时，允许模型学习和优化定缺策略。**