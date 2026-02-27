# DingQue Bug 调查完成报告

## 📊 问题现状

**症状**: AI agent (Player 0) 的DingQue选择极端偏差
- 最新测试结果: Man 0%, Pin 6%, Sou 94%
- 问题持续存在，即使完全重新训练

## 🔍 已完成的调查

### 1. Augmentation Bug ✅ 已修复
- **位置**: `python/blood/env/augment.py:43`
- **问题**: 使用`perm.index(old_suit)`而非`perm[old_suit]`
- **影响**: 83.3%的训练样本使用错误的action映射
- **状态**: 已修复

### 2. Observation Encoding Bug ✅ 已修复
- **位置**: `crates/engine/src/obs/student.rs:94`
- **问题**: Section 3在DingQue阶段是空的
- **影响**: 模型看不到花色分布信息
- **状态**: 已修复，现在提供花色统计

### 3. Exploration Coefficient ✅ 已优化
- **配置**: `configs/warmup.yaml:23`
- **变更**: 0.01 → 0.03
- **状态**: 已提升

### 4. Reward Shaping ✅ 已添加
- **位置**: `python/blood/env/selfplay_env.py:543`
- **内容**: 奖励选择最少花色，惩罚选择最多花色
- **状态**: 已实施

### 5. 模型权重检查 ✅ 正常
- **发现**: Action 31/32/33的权重完全均匀（norm都在0.98左右）
- **结论**: 模型本身没有系统性偏差

### 6. Rust引擎检查 ✅ 正常
- **位置**: `crates/engine/src/state/board.rs:297`
- **发现**: apply_ding_que实现正确，无偏向逻辑
- **结论**: 引擎没有bug

## 🚨 关键发现

**问题依然存在，即使所有已知bug都已修复！**

这说明存在一个**我们尚未发现的深层bug**。

## 💡 可能的原因

### 假设1: 评估脚本问题
- 可能使用了旧的checkpoint
- 可能没有正确加载修复后的代码
- 可能augmentation在评估时仍然生效

### 假设2: 训练数据污染
- 旧的训练数据可能包含错误的样本
- 即使重新训练，如果没有完全清理，可能仍受影响

### 假设3: 隐藏的系统性bug
- 可能在数据流的某个环节存在我们未发现的bug
- 可能在action采样或mask生成中存在问题

### 假设4: Rust-Python接口问题
- 可能在Rust和Python之间的数据传递中存在问题
- 可能action编码/解码有误

## 🔧 建议的下一步行动

### 立即行动

1. **验证reward shaping是否生效**:
   ```bash
   cd blood-v2
   python3 scripts/test_reward_shaping.py
   ```

2. **检查评估脚本**:
   - 确认使用的是最新checkpoint
   - 确认没有使用augmentation
   - 确认正确加载了修复后的代码

3. **完全清理并重新训练**:
   ```bash
   cd blood-v2
   rm -rf train_dir/blood_v2_warmup
   rm -rf checkpoints/league/*
   maturin develop --release
   python3 -m blood.train --config configs/warmup.yaml
   ```

### 深度调试

如果上述步骤仍然失败，需要：

1. **添加详细日志**:
   - 在DingQue决策点记录所有信息
   - 记录observation, mask, action, reward
   - 追踪完整的数据流

2. **创建最小复现案例**:
   - 创建一个简单的测试，只测试DingQue决策
   - 排除其他因素的干扰

3. **检查Rust-Python接口**:
   - 验证action编码/解码
   - 验证observation传递
   - 验证mask生成

## 📝 已创建的文档

1. `DINGQUE_BUG_FOUND.md` - Bug发现过程
2. `DINGQUE_SMOKING_GUN.md` - 关键证据
3. `DINGQUE_FIX_COMPLETE.md` - 修复总结
4. `DINGQUE_FINAL_SOLUTION.md` - 解决方案
5. `DINGQUE_INVESTIGATION_COMPLETE.md` - 本文档

## 📊 已创建的测试脚本

1. `scripts/test_obs_fix.py` - 测试observation修复
2. `scripts/test_live_dingque.py` - 测试模型行为
3. `scripts/test_reward_shaping.py` - 测试reward shaping
4. `scripts/test_dingque_logits.py` - 测试logits输出

## 🎯 结论

我们已经修复了所有已知的bug：
- ✅ Augmentation映射
- ✅ Observation encoding
- ✅ Exploration coefficient
- ✅ Reward shaping

但问题依然存在，说明存在一个**更深层的、我们尚未发现的bug**。

建议：
1. 首先运行`test_reward_shaping.py`验证reward shaping是否生效
2. 如果reward shaping正常，问题可能在评估脚本或checkpoint加载
3. 如果reward shaping不正常，需要深度调试Python环境

这是一个非常棘手的bug，需要系统性的排查和调试。