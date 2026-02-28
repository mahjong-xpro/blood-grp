# 训练指标深度分析指南

## TensorBoard访问

```bash
# 启动TensorBoard
tensorboard --logdir=train_dir/ --port=6006

# 访问
http://localhost:6006/
```

---

## 核心指标分析

### 1. 定缺系统指标 🎯

#### 1.1 定缺分布 (DingQue Distribution)
**指标名**: `blood/dingque_man_ratio`, `blood/dingque_pin_ratio`, `blood/dingque_sou_ratio`

**健康标准**:
```
Phase 1 (0-50K steps):
  Man: 28-38% (目标: 33.3% ± 5%)
  Pin: 28-38%
  Sou: 28-38%
  
Phase 2 (50K-100K steps):
  Man: 23-43% (目标: 33.3% ± 10%)
  Pin: 23-43%
  Sou: 23-43%
  
Phase 3 (100K+ steps):
  允许自然分布，监控极端偏差 (>60%)
```

**异常模式**:
- ❌ **极端偏差**: 某花色 >60% 或 <10%
  - 原因: 先验衰减过快，模型过拟合
  - 解决: 延长warmup步数，增加探索

- ❌ **剧烈波动**: 分布在短期内大幅变化 (>20%)
  - 原因: 训练不稳定，学习率过高
  - 解决: 降低学习率，增加batch size

- ✅ **渐进式偏离**: 从均匀逐渐偏向某花色
  - 正常: 模型学习到统计规律
  - 监控: 确保不超过合理范围

#### 1.2 先验强度 (Prior Strength)
**指标名**: `blood/dingque_prior_strength`

**预期曲线**:
```
Steps    | Prior Strength | 行为
---------|----------------|------------------
0        | 1.0            | 完全均匀先验
25K      | 0.75           | 先验主导
50K      | 0.5            | 混合决策
75K      | 0.25           | 模型主导
100K     | 0.0            | 完全自主
```

**检查点**:
- Step 50K: 应该 ≈ 0.5
- Step 100K: 应该 = 0.0
- 如果不符合，检查步数更新逻辑

---

### 2. 策略质量指标 📊

#### 2.1 策略熵 (Policy Entropy)
**指标名**: `train/entropy`

**健康范围**:
```
Warmup:      0.5 - 1.5  (高探索)
Competitive: 0.3 - 0.8  (中等探索)
Elite:       0.2 - 0.5  (低探索，收敛)
```

**异常模式**:
- ❌ **熵过低** (<0.1): 策略过于确定，可能过拟合
  - 解决: 提高exploration_loss_coeff
  - 解决: 启用entropy floor

- ❌ **熵过高** (>2.0): 策略过于随机，学习不足
  - 解决: 降低exploration_loss_coeff
  - 解决: 检查奖励信号

- ✅ **渐进式下降**: 熵随训练逐渐降低
  - 正常: 策略从探索到利用

#### 2.2 KL散度 (KL Divergence)
**指标名**: `train/kl_divergence`

**健康范围**:
```
Warmup:      0.001 - 0.01
Competitive: 0.0005 - 0.005
Elite:       0.0002 - 0.002
```

**异常模式**:
- ❌ **KL过大** (>0.02): 策略更新过激
  - 原因: 学习率过高，PPO clip过大
  - 解决: 降低学习率，减小ppo_clip_ratio

- ❌ **KL过小** (<0.0001): 策略几乎不更新
  - 原因: 学习率过低，梯度消失
  - 解决: 提高学习率，检查梯度范数

#### 2.3 学习率 (Learning Rate)
**指标名**: `train/learning_rate`

**自适应调度**:
```
KL < threshold × 0.5  → LR × 1.5 (最高lr_adaptive_max)
KL > threshold × 2.0  → LR × 0.5 (最低lr_adaptive_min)
```

**监控**:
- 检查LR是否频繁触碰上下限
- 如果长期锁定在下限，说明学习停滞
- 如果长期锁定在上限，说明KL过低

---

### 3. 价值函数指标 💰

#### 3.1 价值损失 (Value Loss)
**指标名**: `train/value_loss`

**健康范围**:
```
Warmup:      0.5 - 2.0  (初期较高)
Competitive: 0.1 - 0.5  (逐渐降低)
Elite:       0.05 - 0.2 (收敛)
```

**异常模式**:
- ❌ **损失不降**: 价值函数学习失败
  - 原因: 奖励信号噪声大，学习率不当
  - 解决: 检查奖励设计，调整value_loss_coeff

- ❌ **损失爆炸**: 突然大幅上升
  - 原因: 梯度爆炸，数值不稳定
  - 解决: 降低学习率，启用梯度裁剪

#### 3.2 优势标准差 (Advantage Std)
**指标名**: `blood/raw_adv_std`

**健康范围**:
```
Warmup:      5.0 - 20.0
Competitive: 2.0 - 10.0
Elite:       1.0 - 5.0
```

**解读**:
- 高标准差: 奖励方差大，环境随机性高
- 低标准差: 奖励稳定，策略收敛
- 过低 (<0.5): 可能所有样本奖励相同，检查环境

---

### 4. 辅助任务指标 🎓

#### 4.1 向听数预测 (Shanten Prediction)
**指标名**: `blood/aux_loss`, `blood/shanten_accuracy`

**健康标准**:
```
Loss:     < 0.5 (CE loss)
Accuracy: > 60% (3个对手平均)
```

**异常模式**:
- ❌ **准确率低** (<40%): 模型无法推断对手进度
  - 原因: 观测信息不足，模型容量不够
  - 解决: 检查观测编码，增加aux_shanten_weight

#### 4.2 听牌预测 (Waiting Tiles)
**指标名**: `blood/ow_loss`, `blood/ow_precision`, `blood/ow_recall`

**健康标准**:
```
Loss:      < 0.3 (Focal Loss)
Precision: > 50%
Recall:    > 40%
```

**解读**:
- Precision高: 预测的听牌准确
- Recall高: 能识别大部分听牌
- 两者平衡: F1 score > 0.45

---

### 5. Oracle蒸馏指标 🎯

#### 5.1 Oracle CE损失
**指标名**: `blood/oracle_ce`

**健康范围**:
```
Warmup:      1.5 - 3.0 (Oracle disabled)
Competitive: 0.5 - 1.5
Elite:       0.3 - 0.8
```

**定缺阶段特殊处理**:
- 定缺阶段Oracle强制均匀 → CE ≈ 1.1 (log(3))
- 非定缺阶段应该更低

#### 5.2 Oracle蒸馏损失
**指标名**: `blood/distill_loss`

**健康范围**:
```
Competitive: 0.3 - 1.0
Elite:       0.2 - 0.6
```

**异常模式**:
- ❌ **损失不降**: 学生无法学习Oracle
  - 原因: 温度设置不当，权重过低
  - 解决: 调整oracle_distill_temperature

---

### 6. 奖励信号指标 🏆

#### 6.1 平均回报 (Mean Return)
**指标名**: `train/returns_mean`

**健康趋势**:
```
Warmup:      -0.5 → 0.0  (从负到零)
Competitive:  0.0 → 0.5  (从零到正)
Elite:        0.5 → 1.0  (持续提升)
```

**解读**:
- 负回报: 输多赢少
- 零回报: 平衡
- 正回报: 赢多输少

#### 6.2 胜率 (Win Rate)
**指标名**: `blood/win_rate`

**健康标准**:
```
vs RuleBot:
  Warmup:      10-20%
  Competitive: 25-35%
  Elite:       40-50%+
```

#### 6.3 番数分布 (Fan Distribution)
**指标名**: `blood/fan_1_ratio`, `blood/fan_2_ratio`, ..., `blood/fan_6_ratio`

**健康分布**:
```
1番: 30-40%  (最常见)
2番: 20-30%
3番: 15-25%
4番: 10-15%
5番: 5-10%
6番: 2-5%   (封顶)
```

**异常模式**:
- ❌ **低番过多** (1番>60%): 策略过于保守
  - 解决: 增加shanten_fan_bonus_scale

- ❌ **高番过少** (4-6番<10%): 未学会追求高番
  - 解决: 检查番数奖励设计

---

## 训练阶段诊断

### Warmup阶段 (0-2M steps)

**关键指标**:
1. ✅ 定缺分布均匀 (33% ± 5%)
2. ✅ 胜率逐渐提升 (0% → 20%)
3. ✅ 策略熵高 (0.8-1.5)
4. ✅ 向听数预测准确率 >50%

**常见问题**:
- 定缺偏差: 检查先验强度曲线
- 胜率不涨: 检查奖励设计
- 训练不稳定: 降低学习率

### Competitive阶段 (2M-3M steps)

**关键指标**:
1. ✅ 定缺分布开始偏离 (允许±10%)
2. ✅ 胜率持续提升 (20% → 35%)
3. ✅ Oracle CE降低 (<1.0)
4. ✅ 策略熵降低 (0.5-0.8)

**常见问题**:
- 定缺极端偏差: 延长warmup
- 自博弈崩溃: 检查league采样
- Oracle不收敛: 调整蒸馏权重

### Elite阶段 (3M-200M steps)

**关键指标**:
1. ✅ 定缺策略稳定 (自然分布)
2. ✅ 胜率高 (>40%)
3. ✅ 策略熵低但不过低 (0.3-0.5)
4. ✅ 高番比例提升 (4-6番>15%)

**常见问题**:
- 策略过拟合: 提高entropy floor
- 胜率停滞: 增加league多样性
- 训练不稳定: 调整adv_clip

---

## 异常诊断流程

### 问题1: 定缺分布严重偏差

**症状**: 某花色 >60% 或 <10%

**诊断步骤**:
1. 检查先验强度: `blood/dingque_prior_strength`
   - 应该在100K步时降到0
   - 如果不是，检查步数更新逻辑

2. 检查探索系数: `train/entropy`
   - 如果过低 (<0.2)，提高exploration_loss_coeff

3. 检查Oracle: `blood/oracle_ce`
   - 定缺阶段应该 ≈ 1.1
   - 如果不是，检查oracle_dingque_uniform

**解决方案**:
```yaml
# 延长warmup
dingque_prior_warmup_steps: 150000

# 提高探索
exploration_loss_coeff: 0.05

# 确保Oracle均匀化
oracle_dingque_uniform: true
```

### 问题2: 训练不稳定

**症状**: 指标剧烈波动，loss突然爆炸

**诊断步骤**:
1. 检查梯度范数: `train/grad_norm`
   - 如果频繁触碰max_grad_norm，说明梯度爆炸

2. 检查KL散度: `train/kl_divergence`
   - 如果 >0.02，说明策略更新过激

3. 检查优势裁剪: `blood/adv_clip_ratio`
   - 如果 >50%，说明优势方差过大

**解决方案**:
```yaml
# 降低学习率
learning_rate: 1e-4

# 减小PPO clip
ppo_clip_ratio: 0.1

# 启用优势裁剪
adv_clip: 2.0

# 增加batch size
batch_size: 2048
```

### 问题3: 学习停滞

**症状**: 所有指标不再改善

**诊断步骤**:
1. 检查学习率: `train/learning_rate`
   - 如果锁定在下限，说明KL过大

2. 检查策略熵: `train/entropy`
   - 如果过低，说明策略过于确定

3. 检查league多样性: `blood/league_pool_size`
   - 如果过小，对手不够多样

**解决方案**:
```yaml
# 提高学习率下限
lr_adaptive_min: 5e-5

# 提高entropy floor
blood_entropy_floor: 0.015

# 增加league容量
league_max_pool_size: 200
```

---

## 监控脚本

### 自动异常检测

```python
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

def check_training_health(logdir):
    """检查训练健康度"""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    issues = []
    
    # 检查定缺分布
    if 'blood/dingque_man_ratio' in ea.Tags()['scalars']:
        man_ratio = ea.Scalars('blood/dingque_man_ratio')[-1].value
        if man_ratio > 0.6 or man_ratio < 0.1:
            issues.append(f"定缺Man偏差严重: {man_ratio:.1%}")
    
    # 检查策略熵
    if 'train/entropy' in ea.Tags()['scalars']:
        entropy = ea.Scalars('train/entropy')[-1].value
        if entropy < 0.1:
            issues.append(f"策略熵过低: {entropy:.3f}")
    
    # 检查KL散度
    if 'train/kl_divergence' in ea.Tags()['scalars']:
        kl = ea.Scalars('train/kl_divergence')[-1].value
        if kl > 0.02:
            issues.append(f"KL散度过大: {kl:.4f}")
    
    return issues

# 使用
issues = check_training_health('train_dir/blood_v2_warmup')
if issues:
    print("⚠️ 训练异常:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ 训练健康")
```

---

## 总结

### 核心监控指标

**必看**:
1. `blood/dingque_*_ratio` - 定缺分布
2. `train/returns_mean` - 平均回报
3. `train/entropy` - 策略熵
4. `blood/win_rate` - 胜率

**重要**:
5. `train/kl_divergence` - KL散度
6. `train/value_loss` - 价值损失
7. `blood/oracle_ce` - Oracle损失
8. `train/learning_rate` - 学习率

**辅助**:
9. `blood/aux_loss` - 辅助任务
10. `blood/fan_*_ratio` - 番数分布

### 健康训练的特征

✅ 定缺分布渐进式偏离（不极端）
✅ 回报持续提升
✅ 策略熵渐进式下降
✅ KL散度稳定在合理范围
✅ 价值损失逐渐降低
✅ 胜率持续提升

### 异常训练的特征

❌ 定缺分布极端偏差 (>60%)
❌ 回报剧烈波动或不涨
❌ 策略熵过低 (<0.1) 或过高 (>2.0)
❌ KL散度过大 (>0.02)
❌ 价值损失不降或爆炸
❌ 胜率停滞或下降

**建议：定期检查TensorBoard，及时发现和解决问题。**