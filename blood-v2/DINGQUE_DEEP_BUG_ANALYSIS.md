# DingQue深度Bug分析

## 已修复Bug回顾

1. ✅ 观察编码缺失 (student.rs)
2. ✅ 数据增强映射错误 (augment.py)
3. ✅ 均匀先验batch检测 (factory.py)
4. ✅ Oracle跳过batch检测 (losses.py)

## 新发现的潜在问题

### Bug #6: Oracle CE Loss在定缺阶段仍然生效 ⚠️

**位置**: [`losses.py:105-111`](blood-v2/python/blood/training/losses.py:105-111)

**问题描述**:
```python
# Oracle CE (advantage-weighted)
oracle_ce = self._oracle_ce_loss(
    oracle_logits, mb.actions, getattr(mb, "advantages", None), action_mask
)
oracle_ce_weight = getattr(ac, "oracle_ce_weight", 0.1)
extra_loss = extra_loss + oracle_ce_weight * oracle_ce
summaries["oracle_ce"] = oracle_ce.detach()
```

Oracle CE loss在定缺阶段**没有被跳过**，会使用Oracle的偏差logits训练。

**影响**:
- Oracle定缺偏差: 万40.9%, 筒31.1%, 索28.0%
- CE loss会惩罚与Oracle不一致的动作
- 即使跳过了KL distillation，CE loss仍会传播偏差

**严重程度**: 🔴 高 - 直接导致偏差传播

**建议修复**:
```python
# Oracle CE (advantage-weighted) - skip for DingQue samples
if has_non_dingque:
    oracle_ce = self._oracle_ce_loss(
        oracle_logits, mb.actions, getattr(mb, "advantages", None), action_mask
    )
    oracle_ce_weight = getattr(ac, "oracle_ce_weight", 0.1)
    extra_loss = extra_loss + oracle_ce_weight * oracle_ce
    summaries["oracle_ce"] = oracle_ce.detach()
```

### Bug #7: 反向动作映射的数学验证问题 🟡

**位置**: [`blood_env.py:212-238`](blood-v2/python/blood/env/blood_env.py:212-238)

**当前实现**:
```python
def _inverse_action(self, action):
    if self._current_perm is None:
        return action
    # Compute inverse permutation
    inv_perm = tuple(self._current_perm.index(i) for i in range(3))
    return augment_action(action, inv_perm)
```

**数学验证**:

设 `perm = (2, 0, 1)`，表示:
- 新位置0 ← 旧花色2 (索)
- 新位置1 ← 旧花色0 (万)
- 新位置2 ← 旧花色1 (筒)

**正向映射** (引擎→智能体):
- 旧动作31(万) → `perm[0]=2` → 新动作33(索) ✓
- 旧动作32(筒) → `perm[1]=0` → 新动作31(万) ✓
- 旧动作33(索) → `perm[2]=1` → 新动作32(筒) ✓

**反向映射** (智能体→引擎):
智能体输出33(索)，需要找到原始动作。

当前方法: `inv_perm = (perm.index(0), perm.index(1), perm.index(2))`
- `perm.index(0) = 1` (万在新位置1)
- `perm.index(1) = 2` (筒在新位置2)
- `perm.index(2) = 0` (索在新位置0)
- `inv_perm = (1, 2, 0)`

应用反向映射:
- 智能体动作33(索，新位置2) → `inv_perm[2]=0` → 引擎动作31(万) ✓

**验证结果**: ✅ 数学正确

但注释中的解释有误导性，建议改进注释。

### Bug #8: 定缺阶段的Advantage加权可能有问题 🟡

**位置**: [`losses.py:158-179`](blood-v2/python/blood/training/losses.py:158-179)

**问题分析**:

Oracle CE loss使用advantage加权:
```python
if advantages is not None:
    adv_std = max(advantages.detach().std().item(), 0.1)
    normed = (advantages.detach() / adv_std).clamp(-10.0, 10.0)
    adv_w = F.softmax(normed, dim=0)
    adv_w = adv_w * len(adv_w)
    return (ce_raw * adv_w).mean()
```

**定缺阶段的问题**:
1. 定缺决策的advantage可能很小（短期影响小）
2. Softmax加权可能过度放大某些样本
3. 如果batch混合了定缺和非定缺，定缺样本可能被过度加权

**严重程度**: 🟡 中 - 可能影响训练稳定性

**建议**: 监控定缺阶段的advantage分布

### Bug #9: 均匀先验的prior_strength可能需要动态调整 🟢

**位置**: [`factory.py:267`](blood-v2/python/blood/model/factory.py:267)

**当前实现**:
```python
prior_strength = 0.3  # 固定30%
```

**问题**:
- 训练初期: 模型输出随机，30%可能不够强
- 训练后期: 模型已学会均匀，30%可能过强，限制学习

**建议**: 考虑curriculum learning
```python
# 根据训练步数动态调整
# 初期强先验(0.5) → 后期弱先验(0.1)
warmup_steps = 500_000
current_step = getattr(ac, '_training_step', 0)
prior_strength = 0.5 * (1.0 - min(current_step / warmup_steps, 1.0)) + 0.1
```

**严重程度**: 🟢 低 - 优化建议

### Bug #10: 数据增强概率可能影响定缺学习 🟢

**位置**: [`blood_env.py:148`](blood-v2/python/blood/env/blood_env.py:148)

**当前设置**:
```python
self._augment_prob = 0.5  # 50%概率增强
```

**分析**:
- 50%的episode使用原始花色顺序
- 50%的episode使用增强花色顺序
- 如果模型在原始顺序上过拟合，增强可能不够

**建议**: 考虑提高增强概率
```python
self._augment_prob = 0.8  # 80%概率增强
```

或使用完全随机:
```python
# 总是随机选择一个排列（包括identity）
idx = int(self._rng.integers(0, 6))
self._current_perm = SUIT_PERMUTATIONS[idx]
```

**严重程度**: 🟢 低 - 优化建议

## Bug优先级总结

### 🔴 高优先级 - 立即修复
1. **Bug #6**: Oracle CE Loss在定缺阶段生效
   - 直接传播Oracle偏差
   - 需要立即修复

### 🟡 中优先级 - 建议修复
2. **Bug #7**: 改进反向映射注释
   - 代码正确但注释误导
   - 建议改进文档

3. **Bug #8**: 定缺阶段advantage加权
   - 可能影响训练稳定性
   - 建议监控

### 🟢 低优先级 - 优化建议
4. **Bug #9**: 动态调整prior_strength
   - 优化训练效率
   - 可选改进

5. **Bug #10**: 提高数据增强概率
   - 减少过拟合风险
   - 可选改进

## 推荐修复顺序

1. **立即**: 修复Bug #6 (Oracle CE Loss跳过)
2. **短期**: 改进Bug #7注释，监控Bug #8
3. **长期**: 考虑Bug #9和#10的优化

## 测试建议

修复Bug #6后，运行完整测试:
```bash
# 1. 重新编译Rust引擎
cd blood-v2/crates/pybind
maturin develop --release

# 2. 清理旧checkpoint
rm -rf train_dir/blood_v2_warmup_*

# 3. 开始训练
python -m blood.train --config=configs/warmup.yaml

# 4. 监控定缺分布
python scripts/test_live_dingque.py
```

预期结果:
- 早期: 25/35/40 (轻微偏差)
- 中期: 30/33/37 (逐渐收敛)
- 最终: 33/33/33 ±5% (均衡)