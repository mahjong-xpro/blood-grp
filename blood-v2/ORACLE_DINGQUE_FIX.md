# Oracle DingQue Distillation Fix

## 问题描述

Oracle模型在定缺(DingQue)阶段存在初始化偏差：
- 万(Man): 40.9%
- 筒(Pin): 31.1%  
- 索(Sou): 28.0%

通过Oracle distillation，这个偏差会传播给学生模型，导致定缺分布不均匀。

## 解决方案：方案A - 定缺阶段禁用Oracle Distillation

### 修改文件
[`blood-v2/python/blood/training/losses.py`](blood-v2/python/blood/training/losses.py:77-101)

### 关键改动

```python
# 检测是否在定缺阶段
dq_mask = action_mask[:, 31:34] if action_mask is not None else None
is_dingque_phase = dq_mask is not None and dq_mask.all()

# 仅在非定缺阶段使用Oracle distillation
if not is_dingque_phase:
    # ... Oracle KL distillation代码
```

### 工作原理

1. **定缺阶段检测**: 检查动作31-33(定缺动作)是否全部合法
2. **条件性跳过**: 在定缺阶段跳过Oracle KL distillation
3. **保留其他损失**: Oracle CE loss和value distillation不受影响

### 优势

✅ **精准定位**: 仅在定缺阶段禁用，其他阶段正常使用Oracle  
✅ **保持性能**: 弃牌、副露等决策仍受益于Oracle指导  
✅ **最小侵入**: 无需修改Oracle初始化或模型架构  
✅ **向后兼容**: 不影响现有训练配置

## 训练配置建议

### Warmup阶段
```yaml
oracle_enabled: false  # 当前配置，保持不变
```
- 对手是RuleBot，Oracle价值有限
- 定缺均匀先验(0.3强度)已提供保护

### Competitive_distill/Elite阶段
```yaml
oracle_enabled: true  # 可以安全启用
```
- 自博弈阶段Oracle价值显现
- 定缺阶段自动跳过distillation
- 其他阶段享受Oracle指导

## 技术细节

### 定缺动作编号
- 31: 定万(Man)
- 32: 定筒(Pin)
- 33: 定索(Sou)

### 检测逻辑
```python
# action_mask shape: (batch_size, num_actions)
# 定缺阶段: action_mask[:, 31:34] 全为True
dq_mask = action_mask[:, 31:34]
is_dingque_phase = dq_mask.all()  # 所有样本的31-33都合法
```

### 影响范围
- ✅ **跳过**: Oracle KL distillation (定缺阶段)
- ✅ **保留**: Oracle CE loss (所有阶段)
- ✅ **保留**: Oracle value head loss (所有阶段)
- ✅ **保留**: Oracle value distillation (所有阶段)

## 验证方法

### 训练后检查
```bash
python scripts/test_live_dingque.py
```

预期结果：
- 早期训练: 可能有轻微偏差 (25/35/40)
- 中期训练: 逐渐收敛 (30/33/37)
- 最终状态: 均衡分布 33/33/33 (±5%)

### 日志监控
```python
# 检查summaries中是否有distill_loss
# 定缺阶段应该没有distill_loss记录
```

## 与其他修复的协同

此修复与之前的修复协同工作：

1. **观察编码修复** ([`student.rs:100-112`](blood-v2/crates/engine/src/obs/student.rs:100-112))
   - 提供定缺阶段的花色分布信息

2. **均匀先验** ([`factory.py:260-271`](blood-v2/python/blood/model/factory.py:260-271))
   - 防止模型初始化偏差放大

3. **Oracle定缺跳过** (本修复)
   - 防止Oracle偏差传播

三重保护确保定缺分布均匀。

## 总结

通过在定缺阶段禁用Oracle distillation，我们：
- 避免了Oracle初始化偏差的传播
- 保留了Oracle在其他阶段的价值
- 实现了最小化的代码改动
- 提供了清晰的训练策略

这是解决Oracle定缺偏差问题的最优方案。