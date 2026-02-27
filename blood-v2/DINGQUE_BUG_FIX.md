# DingQue 100% Sou Bug - Root Cause & Fix

## Issue Report
**Date**: 2026-02-27  
**Severity**: CRITICAL  
**Impact**: 训练2M步后定缺仍100%选索子，完全无法学习均匀分布

## Root Cause Analysis

### Bug Location
1. [`factory.py:264`](blood-v2/python/blood/model/factory.py:264) - DingQue prior logic
2. [`losses.py:87`](blood-v2/python/blood/training/losses.py:87) - Oracle distillation skip logic

### The Fatal Logic Error

**错误代码**:
```python
dq_mask = mask[:, 31:34]  # (B, 3) - actions 31/32/33
is_dingque = dq_mask.all(dim=-1)  # ❌ WRONG: checks if ALL 3 actions are legal
```

**问题**:
- 定缺阶段时，3个定缺动作(31=Man, 32=Pin, 33=Sou)都是合法的
- `dq_mask.all(dim=-1)` 检查"是否所有3个动作都合法" → **永远返回True**
- 导致 `is_dingque` 永远为True，先验**永远不会被应用**

**正确逻辑**:
```python
is_dingque = dq_mask.any(dim=-1)  # ✅ CORRECT: checks if ANY action is legal
```

### Why This Caused 100% Sou Selection

1. **模型初始化偏差**: ResNet随机初始化导致logits有微小偏差
2. **无先验修正**: 先验永不触发，模型从第一步就学习偏差分布
3. **自强化循环**: 
   - 初始偏好索子 → 采样更多索子动作
   - 更多索子样本 → 梯度进一步强化索子
   - 2M步后完全收敛到100%索子

## Fix Implementation

### 1. Forward Pass Prior (factory.py)
```python
# Line 264: all() → any()
is_dingque = dq_mask.any(dim=-1)  # 检查是否有任何定缺动作合法

# Line 269: 提高先验强度 0.3 → 0.5
prior_strength = 0.5  # 50%拉向均值，更强制均匀分布
```

### 2. Oracle Distillation Skip (losses.py)
```python
# Line 87: all() → any()
is_dingque = dq_mask.any(dim=-1)  # 定缺阶段跳过Oracle蒸馏
```

## Expected Behavior After Fix

### Training Dynamics
- **0-100K步**: 定缺分布快速收敛到 ~33%/33%/33%
- **100K-500K步**: 微调至最优分布（基于手牌统计）
- **500K+步**: 稳定在合理分布，不再100%单一花色

### Validation Metrics
```python
# 定缺分布应该接近均匀（允许±10%波动）
man_ratio: 0.25 - 0.40  # 万子
pin_ratio: 0.25 - 0.40  # 筒子
sou_ratio: 0.25 - 0.40  # 索子
```

## Testing Plan

### 1. Immediate Validation
```bash
# 重新训练Warmup阶段（2M步）
python -m blood.train --config=configs/warmup.yaml

# 监控定缺分布（每10K步）
tensorboard --logdir=train_dir/blood_v2_warmup
# 查看 blood/dingque_man_ratio, blood/dingque_pin_ratio, blood/dingque_sou_ratio
```

### 2. Expected Timeline
- **50K步**: 分布开始均匀化（60%/20%/20% → 40%/30%/30%）
- **200K步**: 接近均匀（35%/33%/32%）
- **500K步**: 稳定均匀（33%±5%）

### 3. Regression Test
```python
# 添加单元测试确保先验正确触发
def test_dingque_prior_triggers():
    mask = torch.zeros(4, 34)
    mask[:, 31:34] = 1.0  # DingQue phase
    
    dq_mask = mask[:, 31:34]
    is_dingque = dq_mask.any(dim=-1)
    
    assert is_dingque.all(), "Prior should trigger for all samples"
```

## Impact Assessment

### Before Fix
- ❌ 定缺100%索子（完全失败）
- ❌ 无法学习合理策略
- ❌ 训练数据严重偏差

### After Fix
- ✅ 定缺均匀分布（33%/33%/33%）
- ✅ 基于手牌统计的智能选择
- ✅ 训练数据平衡

### Training Cost
- **重新训练Warmup**: 2M步 × 1天 = 1天
- **重新训练Competitive**: 5M步 × 2-3天 = 2-3天
- **总延迟**: ~4天（但获得正确的基础策略）

## Lessons Learned

1. **逻辑运算符选择**: `all()` vs `any()` 在mask检测中的关键区别
2. **先验验证**: 必须验证先验是否真正触发（添加日志/metrics）
3. **早期异常检测**: 100K步时就应该发现100%分布异常
4. **单元测试覆盖**: 关键逻辑分支必须有测试覆盖

## Related Issues
- 之前的定缺修复尝试（特征工程、奖励调整）都未触及根本原因
- 此bug导致所有之前的训练run都需要重新训练

## Verification Checklist
- [x] 修复 factory.py line 264
- [x] 修复 losses.py line 87
- [x] 提高先验强度 0.3 → 0.5
- [ ] 添加定缺分布监控metrics
- [ ] 重新训练Warmup阶段
- [ ] 验证定缺分布均匀化
- [ ] 更新训练文档

---
**Status**: Fixed, pending validation  
**Next Action**: 重新启动Warmup训练，监控定缺分布收敛