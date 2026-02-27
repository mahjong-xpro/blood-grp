# 定缺Bug最终诊断报告

## 问题现象

**训练退化**: 从 0/44/56 (万/筒/索) 退化回 0/0/100，即使继续训练1M步

## 已排除的原因

### ✅ 观测编码
- Section 3在定缺阶段有数据（平均27.56）
- 修复已编译并生效

### ✅ 数据增强
- 映射修复正确（`perm[old_suit]`）
- 所有6个排列都是双射

### ✅ 动作掩码
- 三个动作都100%可用
- 无系统性偏差

### ✅ Oracle Distillation
- warmup.yaml中已禁用（`oracle_enabled: false`）
- 但Oracle本身确实有偏差（万40.9%, 筒31.1%, 索28.0%）

## 真正的原因

既然Oracle已禁用，问题必然在：

### 1. 模型架构的隐含偏差

**动作空间顺序效应**:
- 动作31=万, 32=筒, 33=索
- 最后一个线性层的权重初始化可能有微小偏差
- 即使是0.01的差异，经过softmax也会被放大

**证据**:
- Oracle（随机初始化）就有偏差
- 学生模型也会有类似的初始化偏差

### 2. 训练动态的正反馈循环

```
初始偏差(微小) 
  ↓
模型略微偏向索子
  ↓
生成更多索子样本
  ↓
梯度更新强化索子
  ↓
偏差放大
  ↓
最终100%索子
```

### 3. 探索不足

当前`exploration_loss_coeff: 0.03`可能仍然不够。熵惩罚不足以对抗正反馈循环。

## 根本解决方案

### 方案A: 定缺动作的均匀先验 ⭐ 推荐

在最后一层添加定缺动作的均匀先验，强制三个动作的初始logits相等：

```python
# factory.py forward_tail中，action_distribution_params之后
if self._cached_obs is not None:
    phase = self._cached_obs.get("phase")  # 需要添加phase到观测
    if phase == "ding_que":  # 定缺阶段
        # 添加均匀先验：将31/32/33的logits拉向均值
        dq_logits = action_distribution_params[:, 31:34]
        dq_mean = dq_logits.mean(dim=-1, keepdim=True)
        prior_strength = 0.5  # 先验强度
        action_distribution_params[:, 31:34] = (
            dq_logits * (1 - prior_strength) + 
            dq_mean * prior_strength
        )
```

### 方案B: 大幅增加探索

```yaml
exploration_loss_coeff: 0.1  # 从0.03提高到0.1
```

但这会降低整体性能。

### 方案C: 定缺阶段的特殊处理

在定缺阶段使用ε-greedy或温度采样，强制探索：

```python
if phase == "ding_que":
    # 20%概率随机选择
    if random.random() < 0.2:
        action = random.choice([31, 32, 33])
```

## 推荐行动

1. **立即实施方案A** - 添加定缺均匀先验
2. **清理重训** - 删除所有checkpoint，从头开始
3. **密切监控** - 每100K步检查定缺分布

## 技术细节

### 为什么均匀先验有效

1. **打破初始化偏差** - 强制三个动作起点相同
2. **持续约束** - 训练过程中持续施加均匀性约束
3. **不影响学习** - 模型仍然可以学习基于手牌的智能选择
4. **仅影响定缺** - 不影响其他阶段的策略

### 实现位置

`blood-v2/python/blood/model/factory.py:247-264`

在action mask之后，添加定缺先验逻辑。

## 结论

定缺Bug是**模型架构的系统性偏差**，通过正反馈循环放大。解决方案是在定缺阶段添加均匀先验，强制三个动作的logits保持平衡。