# DingQue Bug - 最终调查报告

## 📊 问题现状

**症状**：AI agent的DingQue选择极端偏差
- 初始：Man 0%, Pin 0%, Sou 100%
- 完全重新训练2M步后：**依然** Man 0%, Pin 0%, Sou 100%

## ✅ 已验证正常的部分

1. **环境功能** ✅
   - 所有DingQue action (31/32/33) 都能正确执行
   - Action mask正确生成
   - Phase转换正常

2. **模型权重** ✅
   - Action 31/32/33的权重均匀（norm都在0.98左右）
   - 没有系统性偏差

3. **Rust引擎** ✅
   - apply_ding_que实现正确
   - 无偏向逻辑

4. **评估流程** ✅
   - Arena类正确处理agent_seat
   - 评估时augmentation已禁用

## 🔧 已实施的修复

1. **Augmentation映射错误** ✅
   - 文件：`python/blood/env/augment.py:43`
   - 修复：`perm.index(old_suit)` → `perm[old_suit]`

2. **Observation Encoding缺陷** ✅
   - 文件：`crates/engine/src/obs/student.rs:94`
   - 修复：在DingQue阶段提供花色统计信息

3. **Exploration不足** ✅
   - 文件：`configs/warmup.yaml:23`
   - 优化：0.01 → 0.03

4. **Reward Shaping缺失** ✅
   - 文件：`python/blood/env/selfplay_env.py:543`
   - 添加：奖励选择最少花色，惩罚选择最多花色

## 🚨 关键发现

**即使完全重新训练，问题依然存在！**

这说明：
1. 我们的修复可能没有真正生效，或
2. 存在一个更深层的、系统性的bug，在训练过程或模型采样中

## 🔍 可能的根本原因

### 假设1: Observation修复未生效
- Rust代码可能未正确重新编译
- 或者observation encoding的修复有误

### 假设2: Reward Shaping未生效
- SelfPlayEnv可能没有被正确使用
- 或者reward shaping的实现有误

### 假设3: 训练过程中的系统性偏差
- 可能在数据采集或经验回放中存在偏差
- 可能在action采样时存在bug

### 假设4: 模型架构问题
- 可能模型架构本身存在某种偏向
- 可能LSTM状态或其他机制导致偏差

### 假设5: Sample Factory框架问题
- 可能训练框架本身存在bug
- 可能action采样或分布计算有问题

## 📋 建议的深度调试步骤

### 1. 验证Observation修复
创建脚本直接检查Rust引擎输出的observation：
```python
# 检查Section 3在DingQue阶段是否有值
obs = env.reset()
obs_2d = obs['obs'].reshape(473, 27)
section3 = obs_2d[18:21]  # Section 3
print(f"Section 3 values: {section3[:, 0]}")  # 应该非零
```

### 2. 验证Reward Shaping
在训练过程中添加日志：
```python
# 在 selfplay_env.py 的 _compute_shaping_reward 中
if 31 <= action <= 33:
    print(f"DingQue action {action}, bonus: {bonus}")
```

### 3. 追踪训练过程
在训练时记录DingQue action分布：
```python
# 每1000步统计一次
dingque_actions = [31, 32, 33]
action_counts = {a: 0 for a in dingque_actions}
# ... 统计并记录到TensorBoard
```

### 4. 检查模型采样
验证模型在推理时的logits和概率：
```python
# 在DingQue阶段
logits = model(obs)
probs = softmax(logits[31:34])
print(f"DingQue probs: Man={probs[0]}, Pin={probs[1]}, Sou={probs[2]}")
```

### 5. 简化测试
创建一个最小的训练脚本，只训练DingQue决策：
- 使用固定的手牌
- 只训练DingQue阶段
- 验证模型是否能学习

## 🎯 下一步行动

### 立即行动
1. **验证observation修复是否生效**
   - 运行脚本检查Section 3的值
   - 确认Rust代码已正确重新编译

2. **验证reward shaping是否生效**
   - 在训练时添加日志
   - 确认bonus确实被计算和应用

3. **检查训练日志**
   - 查看TensorBoard
   - 确认DingQue action的分布

### 如果上述都正常
则问题可能在：
1. **Sample Factory框架**的action采样
2. **模型架构**的某种隐含偏向
3. **训练算法**（PPO）的某种问题

### 最后的手段
如果所有调试都失败：
1. **强制均匀采样**：在训练初期强制DingQue使用均匀随机
2. **简化问题**：创建一个只训练DingQue的最小示例
3. **寻求外部帮助**：这可能是一个非常罕见的bug

## 📝 总结

这是一个**极其罕见和棘手的bug**：
- 环境正常 ✅
- 权重正常 ✅
- 所有已知bug已修复 ✅
- **但问题依然存在** ❌

这说明存在一个**非常深层的、系统性的bug**，可能涉及：
- 训练框架（Sample Factory）
- 模型架构
- 训练算法（PPO）
- 或者某个我们完全没有想到的地方

**建议**：
1. 首先验证所有修复是否真的生效
2. 然后进行深度调试，追踪完整的数据流
3. 如果仍然失败，考虑简化问题或寻求外部帮助

这个bug已经超出了常规调试的范围，需要非常深入的系统性调查。