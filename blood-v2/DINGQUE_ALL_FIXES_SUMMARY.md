# DingQue所有修复总结

## 修复清单

### ✅ 已完成的修复

#### 1. 观察编码修复
- **文件**: [`blood-v2/crates/engine/src/obs/student.rs:100-112`](blood-v2/crates/engine/src/obs/student.rs:100-112)
- **问题**: Section 3在定缺阶段为空
- **修复**: 添加花色统计编码
```rust
} else if board.phase == Phase::DingQue {
    for suit in Suit::all() {
        let count = (suit.start()..suit.end())
            .filter(|&t| p.hand[t] > 0)
            .map(|t| p.hand[t] as u32)
            .sum::<u32>();
        fill_ch!(ch + suit as usize, count as f32 / 13.0);
    }
}
```

#### 2. 数据增强映射修复
- **文件**: [`blood-v2/python/blood/env/augment.py:56,62`](blood-v2/python/blood/env/augment.py:56,62)
- **问题**: 使用反向查找 `perm.index(old_suit)` 而非正向映射
- **修复**: 改为 `perm[old_suit]`
```python
new_suit = perm[old_suit]  # 正向映射
```

#### 3. 均匀先验batch检测修复
- **文件**: [`blood-v2/python/blood/model/factory.py:260-275`](blood-v2/python/blood/model/factory.py:260-275)
- **问题**: `dq_mask.all()` 要求整个batch都在定缺阶段
- **修复**: 逐样本检测 + torch.where
```python
is_dingque = dq_mask.all(dim=-1)  # (B,)
if is_dingque.any():
    mixed = dq_logits * 0.7 + dq_mean * 0.3
    action_distribution_params[:, 31:34] = torch.where(
        is_dingque.unsqueeze(-1),
        mixed,
        dq_logits
    )
```

#### 4. Oracle KL distillation跳过修复
- **文件**: [`blood-v2/python/blood/training/losses.py:84-103`](blood-v2/python/blood/training/losses.py:84-103)
- **问题**: 混合batch中Oracle偏差传播
- **修复**: 逐样本检测 + 条件跳过
```python
is_dingque = dq_mask.all(dim=-1)
has_non_dingque = (~is_dingque).any()
if has_non_dingque:
    # ... KL distillation
```

#### 5. Oracle CE Loss跳过修复
- **文件**: [`blood-v2/python/blood/training/losses.py:105-112`](blood-v2/python/blood/training/losses.py:105-112)
- **问题**: Oracle CE loss在定缺阶段传播偏差
- **修复**: 添加条件跳过
```python
if has_non_dingque:
    oracle_ce = self._oracle_ce_loss(...)
    extra_loss = extra_loss + oracle_ce_weight * oracle_ce
```

#### 6. DingQue奖励塑形移除
- **文件**: [`blood-v2/python/blood/env/selfplay_env.py:547-569`](blood-v2/python/blood/env/selfplay_env.py:547-569)
- **问题**: 奖励"最少花色"与最优策略冲突
- **修复**: 完全禁用，依赖均匀先验和观察编码
```python
# DingQue reward shaping: DISABLED
# Uniform prior + observation encoding provide sufficient guidance
pass
```

#### 7. 反向映射验证
- **文件**: [`blood-v2/python/blood/env/blood_env.py:212-238`](blood-v2/python/blood/env/blood_env.py:212-238)
- **状态**: 数学验证正确，无需修改
- **方法**: `inv_perm = tuple(perm.index(i) for i in range(3))`

## 配置建议

### Warmup阶段 (当前)
```yaml
# configs/warmup.yaml
oracle_enabled: false          # ✅ 正确 - 禁用Oracle
suit_augment_prob: 0.5         # 建议提高到0.8
exploration_loss_coeff: 0.03   # ✅ 正确 - 已提高
```

### Competitive阶段 (后续)
```yaml
# configs/competitive.yaml, competitive_distill.yaml, elite.yaml
oracle_enabled: true           # ✅ 安全 - 定缺阶段自动跳过
suit_augment_prob: 0.5         # 建议提高到0.8
```

## 需要重新编译

修改了Rust代码，必须重新编译：
```bash
cd blood-v2/crates/pybind
maturin develop --release
```

## 训练流程

### 1. 清理旧数据
```bash
rm -rf train_dir/blood_v2_warmup_*
```

### 2. 开始训练
```bash
python -m blood.train --config=configs/warmup.yaml
```

### 3. 监控定缺分布
```bash
# 每100K步检查
python scripts/test_live_dingque.py
```

### 4. 预期结果
- **早期** (0-500K): 25/35/40 (轻微偏差可接受)
- **中期** (500K-1M): 30/33/37 (逐渐收敛)
- **最终** (1M+): 33/33/33 ±5% (均衡)

## 异常处理

### 如果分布仍然偏差

1. **检查Rust编译**
```bash
cd blood-v2/crates/pybind
maturin develop --release
python -c "from blood._engine import RustMahjongEnv; print('OK')"
```

2. **检查checkpoint清理**
```bash
ls -la train_dir/blood_v2_warmup_*
# 应该为空或不存在
```

3. **检查Oracle状态**
```bash
grep "oracle_enabled" configs/warmup.yaml
# 应该显示: oracle_enabled: false
```

4. **提高先验强度**
如果仍有偏差，可以临时提高prior_strength:
```python
# factory.py:267
prior_strength = 0.5  # 从0.3提高到0.5
```

## 优化建议

### 短期优化
1. **提高数据增强概率**
```yaml
suit_augment_prob: 0.8  # 从0.5提高到0.8
```

2. **监控advantage分布**
```python
# 在losses.py中添加日志
if is_dingque.any():
    log.info("DingQue samples in batch: %d/%d", 
             is_dingque.sum(), len(is_dingque))
```

### 长期优化
1. **动态prior_strength** (curriculum learning)
```python
# factory.py
warmup_steps = 500_000
current_step = getattr(ac, '_training_step', 0)
prior_strength = 0.5 * (1.0 - min(current_step / warmup_steps, 1.0)) + 0.1
```

2. **完全随机增强**
```python
# blood_env.py
idx = int(self._rng.integers(0, 6))  # 包括identity
self._current_perm = SUIT_PERMUTATIONS[idx]
```

## 技术细节

### 三重保护机制
1. **观察编码** - 提供花色分布信息给模型
2. **均匀先验** - 防止模型输出偏差放大
3. **Oracle跳过** - 防止Oracle偏差传播

### 定缺动作编号
- 31: 定万 (Man)
- 32: 定筒 (Pin)
- 33: 定索 (Sou)

### Oracle偏差来源
- 模型初始化随机性
- 测试显示: 万40.9%, 筒31.1%, 索28.0%
- 通过跳过定缺阶段的distillation避免传播

### Batch检测重要性
- 训练中batch常混合定缺和非定缺样本
- `dq_mask.all()` 要求全batch都在定缺阶段（过于严格）
- `dq_mask.all(dim=-1)` 逐样本检测（正确）

## 测试验证

### 单元测试
```bash
# 数据增强
python -m pytest tests/test_augment.py -v

# 观察编码
cd blood-v2/crates/engine
cargo test test_obs -- --nocapture

# 反向映射
python tests/test_inverse_action.py
```

### 集成测试
```bash
# 定缺分布
python scripts/test_dingque_logits.py

# Oracle偏差
python scripts/test_oracle_dingque.py

# 完整诊断
python scripts/diagnose_dingque_bias.py
```

## 相关文档

- [`DINGQUE_FINAL_COMPLETE_ANALYSIS.md`](DINGQUE_FINAL_COMPLETE_ANALYSIS.md) - 完整分析
- [`DINGQUE_DEEP_BUG_ANALYSIS.md`](DINGQUE_DEEP_BUG_ANALYSIS.md) - 深度Bug分析
- [`DINGQUE_FIX_VERIFICATION.md`](DINGQUE_FIX_VERIFICATION.md) - 修复验证
- [`ORACLE_DINGQUE_FIX.md`](ORACLE_DINGQUE_FIX.md) - Oracle修复说明

## 总结

所有关键bug已修复：
- ✅ 观察编码 (Rust)
- ✅ 数据增强映射 (Python)
- ✅ 均匀先验batch检测 (Python)
- ✅ Oracle KL跳过 (Python)
- ✅ Oracle CE跳过 (Python)
- ✅ DingQue奖励塑形移除 (Python)
- ✅ 反向映射验证 (Python)

系统已准备好重新训练，预期定缺分布将收敛到33/33/33 ±5%。