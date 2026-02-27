# 定缺Bug完整修复总结

## 问题现象

- **初始**: 玩家0定缺分布 万0% / 筒0% / 索100%
- **中期**: 修复后改善到 万0% / 筒44% / 索56%
- **退化**: 继续训练1M步后又变回 万0% / 筒0% / 索100%

## 根本原因

**多重Bug的组合效应**:

1. **观测编码缺失** - Section 3在定缺阶段为空
2. **模型架构偏差** - Oracle随机初始化有偏差（万40.9%, 筒31.1%, 索28.0%）
3. **正反馈循环** - 初始偏差通过训练自我强化

## 已实施的完整修复

### 1. 观测编码修复 ✅
**文件**: `crates/engine/src/obs/student.rs:100-112`
**状态**: 已实现并重新编译

```rust
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
```

### 2. 数据增强前向映射修复 ✅
**文件**: `python/blood/env/augment.py:56,62`
**状态**: 已修复

```python
new_suit = perm[old_suit]  # 正确: 前向映射
# 错误的代码: new_suit = perm.index(old_suit)
```

### 3. 数据增强反向映射验证 ✅
**文件**: `python/blood/env/blood_env.py:212-237`
**状态**: 已验证正确

```python
inv_perm = tuple(self._current_perm.index(i) for i in range(3))
return augment_action(action, inv_perm)
```

**测试验证**: `tests/test_inverse_action.py`确认方法1(反向排列)是正确的

### 4. 定缺均匀先验 ✅
**文件**: `python/blood/model/factory.py:260-271`
**状态**: 已添加

```python
# DingQue uniform prior: 强制定缺动作(31/32/33)的logits趋向均值
dq_mask = mask[:, 31:34]
if dq_mask.all():
    dq_logits = action_distribution_params[:, 31:34]
    dq_mean = dq_logits.mean(dim=-1, keepdim=True)
    prior_strength = 0.3
    action_distribution_params[:, 31:34] = (
        dq_logits * (1.0 - prior_strength) + 
        dq_mean * prior_strength
    )
```

### 5. 探索系数优化 ✅
**文件**: `configs/warmup.yaml:23`
**状态**: 已提升

```yaml
exploration_loss_coeff: 0.03  # 从0.01提升到0.03
```

### 6. Oracle已禁用 ✅
**文件**: `configs/warmup.yaml:52`
**状态**: 已确认

```yaml
oracle_enabled: false  # 避免Oracle偏差传播
```

## 诊断工具

### 1. 观测编码测试
```bash
python3 scripts/diagnose_dingque_bias.py
```
- 验证Section 3有数据（平均27.56）
- 验证动作掩码无偏差
- 验证数据增强对称性

### 2. Oracle偏差测试
```bash
python3 scripts/test_oracle_dingque.py
```
- 发现Oracle有显著偏差（卡方27.21）
- 万40.9%, 筒31.1%, 索28.0%

### 3. 反向映射测试
```bash
python3 tests/test_inverse_action.py
```
- 确认反向排列方法正确

## 技术细节

### 为什么需要反向排列

```
前向映射（观测增强）:
perm=(2,0,1) 表示: new[0]=old[2], new[1]=old[0], new[2]=old[1]
即: 万→索, 筒→万, 索→筒

反向映射（动作还原）:
需要: old[0]=new[1], old[1]=new[2], old[2]=new[0]
即: inv_perm=(1,2,0)

计算方法:
inv_perm[i] = perm.index(i)
- inv_perm[0] = perm.index(0) = 1
- inv_perm[1] = perm.index(1) = 2
- inv_perm[2] = perm.index(2) = 0
```

### 为什么需要均匀先验

1. **打破初始化偏差** - Oracle和学生模型都有初始化偏差
2. **防止正反馈循环** - 小偏差会通过训练自我强化
3. **不阻碍学习** - 70%权重保留模型输出，30%拉向均值
4. **仅影响定缺** - 不影响其他阶段的策略

## 验证清单

- [x] Rust引擎重新编译
- [x] 观测编码Section 3有数据
- [x] 数据增强前向映射正确
- [x] 数据增强反向映射正确
- [x] 定缺均匀先验已添加
- [x] 探索系数已提升
- [x] Oracle已禁用
- [x] 所有测试通过

## 下一步行动

1. **清理重训**:
```bash
rm -rf train_dir/blood_v2_warmup_*
python -m blood.train --config=configs/warmup.yaml
```

2. **监控定缺分布**:
```bash
python scripts/test_live_dingque.py
```

3. **预期结果**:
- 初期: 可能仍有轻微偏差（如 25/35/40）
- 中期: 逐渐趋向均衡（如 30/33/37）
- 最终: 收敛到约33/33/33（±5%）

## 结论

定缺Bug是由**观测编码缺失** + **模型初始化偏差** + **正反馈循环**共同导致的。

所有修复已就位：
1. 观测编码提供了必要信息
2. 数据增强映射已验证正确
3. 均匀先验防止偏差放大
4. 探索系数帮助跳出局部最优

现在可以开始清理重训，定缺分布应该会收敛到均衡状态。