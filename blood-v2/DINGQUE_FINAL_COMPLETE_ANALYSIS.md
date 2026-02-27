# DingQue完整修复分析与总结

## 发现的所有Bug

### 🔴 关键Bug (已修复)

#### Bug #1: 观察编码缺失
- **位置**: [`student.rs:94-113`](blood-v2/crates/engine/src/obs/student.rs:94-113)
- **问题**: Section 3在定缺阶段为空，模型无花色分布信息
- **修复**: 添加花色统计编码 `count / 13.0`
- **影响**: 模型无法做出明智的定缺决策

#### Bug #2: 数据增强映射错误
- **位置**: [`augment.py:56,62`](blood-v2/python/blood/env/augment.py:56,62)
- **问题**: 使用 `perm.index(old_suit)` 反向查找而非 `perm[old_suit]` 正向映射
- **修复**: 改为正向映射
- **影响**: 数据增强失效，模型在特定花色顺序上过拟合

#### Bug #3: 均匀先验batch检测
- **位置**: [`factory.py:260-271`](blood-v2/python/blood/model/factory.py:260-271)
- **问题**: `dq_mask.all()` 要求整个batch都在定缺阶段
- **修复**: 改为逐样本检测 `dq_mask.all(dim=-1)`
- **影响**: 混合batch中先验不生效

#### Bug #4: Oracle KL distillation跳过
- **位置**: [`losses.py:84-103`](blood-v2/python/blood/training/losses.py:84-103)
- **问题**: `dq_mask.all()` 要求整个batch都在定缺阶段
- **修复**: 改为逐样本检测，仅对非定缺样本计算
- **影响**: 混合batch中Oracle偏差传播

#### Bug #6: Oracle CE Loss跳过
- **位置**: [`losses.py:105-111`](blood-v2/python/blood/training/losses.py:105-111)
- **问题**: Oracle CE loss在定缺阶段仍然生效
- **修复**: 添加 `if has_non_dingque:` 条件
- **影响**: 直接传播Oracle偏差 (万40.9%, 筒31.1%, 索28.0%)

### ✅ 已验证正确

#### Bug #7: 反向动作映射
- **位置**: [`blood_env.py:212-238`](blood-v2/python/blood/env/blood_env.py:212-238)
- **状态**: 数学验证正确
- **建议**: 改进注释说明

### 🟡 潜在问题

#### Bug #8: 定缺阶段Advantage加权
- **位置**: [`losses.py:158-179`](blood-v2/python/blood/training/losses.py:158-179)
- **问题**: Softmax加权可能在混合batch中过度放大定缺样本
- **建议**: 监控定缺阶段的advantage分布

#### Bug #11: DingQue奖励塑形可能引入偏差
- **位置**: [`selfplay_env.py:547-569`](blood-v2/python/blood/env/selfplay_env.py:547-569)
- **问题**: 奖励"选择最少花色"可能与最优策略冲突
- **分析**:
  ```python
  # 当前逻辑: 奖励选择牌数最少的花色
  if suit_counts[chosen_suit] == min_count:
      bonus += 0.05
  elif suit_counts[chosen_suit] == max_count:
      bonus -= 0.02
  ```
- **问题场景**:
  - 万: 5张 (包含多个刻子/顺子潜力)
  - 筒: 3张 (孤张)
  - 索: 5张
  - 当前逻辑奖励选筒，但最优可能是选万或索
- **严重程度**: 🟡 中 - 仅在warmup阶段生效
- **建议**: 
  1. 降低奖励强度 (0.05→0.02)
  2. 或完全移除，依赖均匀先验和观察编码
  3. 或改为奖励"选择孤张最多的花色"

### 🟢 优化建议

#### Bug #9: 动态调整prior_strength
- 建议curriculum learning: 0.5→0.1

#### Bug #10: 提高数据增强概率
- 建议: 50%→80% 或完全随机

## 修复优先级

### 立即修复 (已完成)
1. ✅ Bug #1: 观察编码
2. ✅ Bug #2: 数据增强映射
3. ✅ Bug #3: 均匀先验batch检测
4. ✅ Bug #4: Oracle KL跳过
5. ✅ Bug #6: Oracle CE跳过

### 短期改进
6. 🟡 Bug #11: 评估DingQue奖励塑形影响
7. 🟡 Bug #7: 改进反向映射注释
8. 🟡 Bug #8: 监控advantage分布

### 长期优化
9. 🟢 Bug #9: 动态prior_strength
10. 🟢 Bug #10: 提高增强概率

## 系统性问题分析

### 根本原因
1. **模型初始化偏差**: Oracle和学生模型都有初始化偏差
2. **正反馈循环**: 小偏差 → 更多训练样本 → 强化偏差 → 100%单一花色
3. **多重传播路径**: KL distillation + CE loss + 奖励塑形

### 三重保护机制
1. **观察编码**: 提供花色分布信息
2. **均匀先验**: 防止模型输出偏差放大
3. **Oracle跳过**: 防止Oracle偏差传播

### 训练建议

#### 清理重训
```bash
# 1. 重新编译Rust引擎
cd blood-v2/crates/pybind
maturin develop --release

# 2. 清理旧checkpoint
rm -rf train_dir/blood_v2_warmup_*

# 3. 开始训练
python -m blood.train --config=configs/warmup.yaml
```

#### 监控指标
```bash
# 每100K步检查定缺分布
python scripts/test_live_dingque.py
```

#### 预期结果
- **早期** (0-500K): 25/35/40 (轻微偏差可接受)
- **中期** (500K-1M): 30/33/37 (逐渐收敛)
- **最终** (1M+): 33/33/33 ±5% (均衡)

#### 异常处理
如果分布仍然偏差:
1. 检查Rust引擎是否重新编译
2. 检查是否清理了旧checkpoint
3. 检查Oracle是否正确禁用 (warmup阶段)
4. 考虑提高prior_strength (0.3→0.5)

## 配置建议

### Warmup阶段
```yaml
# configs/warmup.yaml
oracle_enabled: false  # 禁用Oracle
suit_augment_prob: 0.8  # 提高增强概率
exploration_loss_coeff: 0.03  # 增加探索
```

### Competitive阶段
```yaml
# configs/competitive_distill.yaml
oracle_enabled: true  # 启用Oracle (定缺阶段自动跳过)
suit_augment_prob: 0.8
```

## 测试验证

### 单元测试
```bash
# 测试数据增强
python -m pytest tests/test_augment.py

# 测试观察编码
cargo test --package engine test_obs

# 测试反向映射
python tests/test_inverse_action.py
```

### 集成测试
```bash
# 测试定缺分布
python scripts/test_dingque_logits.py

# 测试Oracle偏差
python scripts/test_oracle_dingque.py
```

## 总结

### 修复清单
1. ✅ 观察编码 (student.rs)
2. ✅ 数据增强映射 (augment.py)
3. ✅ 均匀先验 + batch检测 (factory.py)
4. ✅ Oracle KL跳过 + batch检测 (losses.py)
5. ✅ Oracle CE跳过 + batch检测 (losses.py)
6. ✅ 反向映射验证 (blood_env.py)

### 关键洞察
- **单一修复不够**: 需要多重保护机制
- **Batch检测重要**: 混合batch场景常见
- **Oracle是双刃剑**: 有益但需谨慎使用
- **奖励塑形需谨慎**: 可能引入意外偏差

### 下一步
1. 清理重训，验证修复效果
2. 监控定缺分布收敛
3. 评估DingQue奖励塑形影响
4. 考虑长期优化建议

所有关键bug已修复，系统已准备好重新训练。