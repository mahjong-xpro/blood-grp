# DingQue Bug - 确凿证据

## 关键发现

### 测试结果

**权重检查**（checkpoint_000000672）:
```
Action 31 (Man): norm=0.986, mean=-0.001
Action 32 (Pin): norm=0.983, mean=-0.005
Action 33 (Sou): norm=0.983, mean=-0.001
```

**实际行为**（100局游戏）:
```
Man:  4%
Pin:  0%
Sou: 96%
```

### 结论

**权重均匀，但行为极端偏差** → 问题在**输入数据**或**采样过程**，不在模型本身！

## 可能的根本原因

### 1. Observation Encoding偏差（最可能）

Man的特征在observation中可能被系统性地编码为不利的值，导致即使权重均匀，输出logits也偏向Pin/Sou。

**需要检查**: `crates/engine/src/obs/student.rs` 中Section 3的dingque encoding

### 2. Augmentation在评估时的影响

虽然评估时`suit_augment_prob=0.0`，但可能存在其他地方的augmentation逻辑。

### 3. Action Sampling的Bug

在`factory.py:263`的`_maybe_sample_actions`中可能存在采样偏差。

### 4. Reward Signal的系统性偏差

训练过程中Man的选择可能系统性地获得更低的reward，导致模型学习到避免Man。

## 下一步诊断

### 方案A: 检查Observation Encoding

```bash
# 创建测试脚本检查observation中dingque相关的通道
# 查看Section 3 (通道18-20) 的值是否对Man有偏差
```

### 方案B: 直接测试Logits

创建脚本直接检查模型对相同observation的logits输出，看是否action 31的logit系统性地更低。

### 方案C: 添加Dingque Reward Shaping

如果无法找到确切bug，添加explicit reward shaping强制均匀探索：

```python
# 在selfplay_env.py的_compute_shaping_reward中添加
if self._env.get_phase() == "ding_que":
    # 奖励选择最少的花色
    hand = self._env.get_agent_hand()
    suit_counts = [
        sum(1 for t in hand if 0 <= t < 9),   # Man
        sum(1 for t in hand if 9 <= t < 18),  # Pin
        sum(1 for t in hand if 18 <= t < 27), # Sou
    ]
    if 31 <= action <= 33:
        chosen_suit = action - 31
        min_count = min(suit_counts)
        if suit_counts[chosen_suit] == min_count:
            bonus += 0.05
```

## 临时解决方案

如果无法快速找到bug，可以：

1. **强制ε-greedy探索**：训练初期30%随机选择dingque
2. **添加reward shaping**：奖励选择最少花色
3. **增加exploration coefficient**：从0.03提升到0.10

## 为什么这个Bug如此难找

1. **权重正常** - 看起来模型没问题
2. **Mask正确** - 环境生成的mask是对的
3. **Augmentation已修复** - 映射逻辑是对的
4. **但行为极端** - 说明问题在数据流的某个环节

这是一个**数据流bug**，不是模型bug。需要追踪从observation → logits → action的完整流程。