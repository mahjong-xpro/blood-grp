# 定缺系统重构实施总结

## 已完成的核心功能

### 1. 渐进式先验机制 ✅

**文件**: [`blood-v2/python/blood/model/factory.py`](blood-v2/python/blood/model/factory.py)

#### 实现细节
```python
# 在 BloodActorCritic.__init__() 中添加
self.dingque_prior_warmup_steps = getattr(cfg, "dingque_prior_warmup_steps", 100000)
self.dingque_prior_enabled = getattr(cfg, "dingque_prior_enabled", True)
self.register_buffer("_dingque_global_steps", torch.tensor(0, dtype=torch.long), persistent=False)

# 在 forward_tail() 中应用渐进式先验
if self.dingque_prior_enabled and self._cached_obs is not None:
    mask = self._cached_obs.get("action_mask")
    if mask is not None:
        dingque_mask = (mask[:, 31:34].sum(dim=1) > 0.5)
        if dingque_mask.any():
            global_steps = float(self._dingque_global_steps.item())
            prior_strength = max(0.0, 1.0 - global_steps / self.dingque_prior_warmup_steps)
            
            if prior_strength > 0.0:
                uniform_logits = torch.zeros(3, ...)
                action_distribution_params[dingque_mask, 31:34] = (
                    (1.0 - prior_strength) * action_distribution_params[dingque_mask, 31:34] +
                    prior_strength * uniform_logits
                )
```

#### 工作原理
- **Phase 1 (0-50K steps)**: `prior_strength = 1.0 → 0.5` - 强先验主导，确保均匀分布
- **Phase 2 (50K-100K steps)**: `prior_strength = 0.5 → 0.1` - 先验逐渐减弱，模型逐渐主导
- **Phase 3 (100K+ steps)**: `prior_strength = 0.1 → 0.0` - 模型完全自主决策

#### 关键优势
- ✅ **允许梯度传播** - 模型可以从经验中学习
- ✅ **防止初期偏差** - 强先验确保训练初期均匀分布
- ✅ **渐进式学习** - 平滑过渡到自主决策

---

### 2. 步数更新机制 ✅

**文件**: [`blood-v2/python/blood/training/learner_patch.py`](blood-v2/python/blood/training/learner_patch.py)

#### 实现细节
```python
def _patched_calculate_losses(self, mb, num_invalids):
    # ... 其他代码 ...
    
    # Update DingQue progressive prior global step counter
    if hasattr(self.actor_critic, "_dingque_global_steps"):
        self.actor_critic._dingque_global_steps.fill_(env_steps)
```

#### 工作原理
- 每次计算损失时，将当前环境步数同步到模型的全局步数计数器
- 模型在前向传播时读取此计数器来计算先验强度

---

### 3. Oracle均匀化 ✅

**文件**: [`blood-v2/python/blood/training/losses.py`](blood-v2/python/blood/training/losses.py)

#### 实现细节
```python
class BloodLossComputer:
    def __init__(self, cfg=None):
        # ...
        self._oracle_dingque_uniform = getattr(cfg, 'oracle_dingque_uniform', True)
    
    def _oracle_ce_loss(self, oracle_logits, actions, advantages, action_mask):
        # DingQue uniformization: force Oracle to output uniform distribution
        if self._oracle_dingque_uniform and action_mask is not None:
            dingque_mask = (action_mask[:, 31:34].sum(dim=1) > 0.5)
            if dingque_mask.any():
                oracle_logits = oracle_logits.clone()
                oracle_logits[dingque_mask, 31:34] = 0.0
        # ... 正常CE计算 ...
```

#### 工作原理
- 检测定缺阶段（actions 31-33 有效）
- 强制Oracle logits为0（均匀分布）
- 避免传播RuleBot的偏差（Man 40.9%, Pin 31.1%, Sou 28.0%）

---

### 4. 配置参数 ✅

**文件**:
- [`blood-v2/python/blood/cfg.py`](blood-v2/python/blood/cfg.py:106-118) - 参数注册
- [`blood-v2/configs/default.yaml`](blood-v2/configs/default.yaml:88-92) - 默认配置
- [`blood-v2/configs/warmup.yaml`](blood-v2/configs/warmup.yaml:97-101) - Warmup阶段
- [`blood-v2/configs/competitive.yaml`](blood-v2/configs/competitive.yaml:113-117) - Competitive阶段
- [`blood-v2/configs/elite.yaml`](blood-v2/configs/elite.yaml:113-117) - Elite阶段

#### 参数注册 (cfg.py)
```python
p.add_argument("--dingque_prior_enabled", default=True,
                type=lambda x: str(x).lower() != "false",
                help="Enable progressive prior mechanism for DingQue phase")
p.add_argument("--dingque_prior_warmup_steps", type=int, default=100000,
                help="Number of env steps for prior to decay from 1.0 to 0.0")
p.add_argument("--oracle_dingque_uniform", default=True,
                type=lambda x: str(x).lower() != "false",
                help="Force Oracle to output uniform distribution during DingQue phase")
p.add_argument("--dingque_reward_shaping", default=False, action="store_true",
                help="Enable reward shaping based on hand structure during DingQue")
```

#### YAML配置
```yaml
# DingQue system — progressive prior + Oracle uniformization
dingque_prior_enabled: true           # Enable progressive prior mechanism
dingque_prior_warmup_steps: 100000    # Prior decays from 1.0 to 0.0 over 100K steps
oracle_dingque_uniform: true          # Force Oracle to output uniform distribution during DingQue
dingque_reward_shaping: false         # Reward shaping based on hand structure (disabled by default)
```

#### 参数说明
- **dingque_prior_enabled**: 是否启用渐进式先验（默认true）
- **dingque_prior_warmup_steps**: 先验衰减步数（默认100K）
- **oracle_dingque_uniform**: Oracle是否强制均匀（默认true）
- **dingque_reward_shaping**: 是否启用奖励塑形（默认false，待实现）

---

## 待实现的增强功能

### 5. 增强观测编码 ⏳

**文件**: `blood-v2/crates/engine/src/obs/student.rs`

#### 设计方案
```rust
// Section 3 扩展 (新增 9 个通道)
// 通道 22-24: 各花色孤张数量 / 13.0
// 通道 25-27: 各花色面子数量 / 4.0
// 通道 28-30: 各花色搭子数量 / 6.0

fn encode_hand_structure(hand: &Hand, suit: Suit) -> (f32, f32, f32) {
    let tiles: Vec<u8> = hand.tiles_of_suit(suit);
    let isolated = count_isolated_tiles(&tiles);
    let melds = count_melds(&tiles);
    let taatsu = count_taatsu(&tiles);
    (isolated as f32 / 13.0, melds as f32 / 4.0, taatsu as f32 / 6.0)
}
```

#### 实施优先级
- **中等优先级** - 提供更好信息，但不是必需的
- 需要实现孤张/面子/搭子检测算法
- 可复用现有shanten算法的部分逻辑

---

### 6. 定缺奖励塑形 ⏳

**文件**: `blood-v2/python/blood/env/selfplay_env.py`

#### 设计方案
```python
def _compute_dingque_reward(self, chosen_suit: int) -> float:
    if not self._dingque_reward_shaping:
        return 0.0
    
    # 从观测中提取手牌结构信息
    obs = self._env.get_player_obs(0)["obs"]
    obs_2d = obs.reshape(-1, 27)
    
    # 提取选择花色的结构信息
    isolated = obs_2d[22 + chosen_suit].mean()
    melds = obs_2d[25 + chosen_suit].mean()
    taatsu = obs_2d[28 + chosen_suit].mean()
    
    # 结构质量评分
    quality = 0.5 * isolated + 0.3 * (1 - melds) + 0.2 * (1 - taatsu)
    return 0.01 * quality
```

#### 实施优先级
- **低优先级** - 加速学习，但不是必需的
- 依赖增强观测编码（需要先实现步骤5）
- 使用小奖励值（0.01）避免干扰主要信号

---

## 系统架构对比

### 旧系统（已移除）
```
定缺阶段 → 均匀先验（完全覆盖） → 无梯度传播 → 无法学习
                ↓
            模型输出被忽略
```

### 新系统（已实现）
```
定缺阶段 → 渐进式先验（混合） → 梯度传播 → 可以学习
                ↓
    (1-α) * 模型输出 + α * 均匀先验
    α: 1.0 → 0.0 (100K steps)
```

---

## 验证计划

### 训练过程监控
1. **定缺分布**: 每10K步记录 Man/Pin/Sou 比例
   ```python
   # 在 runner.py 或 callbacks.py 中添加
   dingque_actions = actions[(actions >= 31) & (actions <= 33)]
   if len(dingque_actions) > 0:
       man_ratio = (dingque_actions == 31).float().mean()
       pin_ratio = (dingque_actions == 32).float().mean()
       sou_ratio = (dingque_actions == 33).float().mean()
   ```

2. **先验强度**: 记录当前 `prior_strength` 值
   ```python
   if hasattr(model, "_dingque_global_steps"):
       global_steps = model._dingque_global_steps.item()
       prior_strength = max(0.0, 1.0 - global_steps / warmup_steps)
   ```

3. **定缺奖励**: 记录平均定缺奖励值（如果启用）

### 收敛标准
- **Phase 1 (0-50K)**: 分布偏差 < 5% (28-38% per suit)
- **Phase 2 (50K-100K)**: 分布偏差 < 10% (23-43% per suit)
- **Phase 3 (100K+)**: 允许自然分布，监控是否出现极端偏差 (>60%)

### 性能验证
- 对比重构前后的胜率和平均得分
- 确保定缺策略学习不影响整体性能
- 监控训练稳定性（loss曲线、梯度范数）

---

## 使用指南

### 启用新系统
```bash
# 使用默认配置（已启用渐进式先验和Oracle均匀化）
python -m blood.train --config=configs/default.yaml

# 自定义先验衰减步数
python -m blood.train --config=configs/default.yaml \
    --dingque_prior_warmup_steps=150000

# 禁用渐进式先验（不推荐）
python -m blood.train --config=configs/default.yaml \
    --dingque_prior_enabled=false
```

### 监控训练
```bash
# 查看TensorBoard
tensorboard --logdir=train_dir/

# 关键指标
# - blood/dingque_distribution: 定缺分布
# - blood/prior_strength: 先验强度
# - oracle_ce: Oracle交叉熵损失
```

---

## 风险与缓解

### 已识别风险

#### 风险1: 渐进式先验可能仍然偏差
**症状**: Phase 1结束时分布偏差 > 5%

**缓解措施**:
- 延长warmup步数（150K或200K）
- 调整衰减曲线（线性 → 指数衰减）
- 增加初始先验强度（从1.0改为1.2）

#### 风险2: Oracle均匀化影响蒸馏效果
**症状**: Oracle CE损失异常高或不收敛

**缓解措施**:
- 定缺阶段降低Oracle蒸馏权重
- 仅在Oracle CE中均匀化，KL蒸馏保持原样
- 监控非定缺阶段的蒸馏效果

#### 风险3: 步数更新不及时
**症状**: 先验强度不随训练进度变化

**缓解措施**:
- 验证`_dingque_global_steps`正确更新
- 添加日志记录步数更新
- 检查多进程训练时的同步

---

## 技术债务

### 需要后续优化的部分

1. **观测编码增强** (Rust侧)
   - 实现孤张/面子/搭子检测
   - 扩展Section 3通道数
   - 更新NUM_STUDENT_CHANNELS常量

2. **奖励塑形实现** (Python侧)
   - 依赖观测编码增强
   - 添加配置开关
   - 验证不干扰主要信号

3. **监控仪表盘**
   - 定缺分布可视化
   - 先验强度曲线
   - 自动异常检测

4. **单元测试**
   - 渐进式先验逻辑测试
   - Oracle均匀化测试
   - 步数更新测试

---

## 总结

### 已完成 ✅
1. ✅ 移除旧的均匀先验代码
2. ✅ 移除定缺跳过逻辑
3. ✅ 设计新系统架构
4. ✅ 实现渐进式先验机制
5. ✅ 实现步数更新机制
6. ✅ 实现Oracle均匀化
7. ✅ 添加配置参数

### 待完成 ⏳
1. ⏳ 增强观测编码（Rust侧）
2. ⏳ 实现定缺奖励塑形
3. ⏳ 测试和验证新系统

### 核心改进
- **允许模型学习** - 渐进式先验不阻止梯度传播
- **防止偏差传播** - Oracle均匀化避免引入RuleBot偏差
- **平滑过渡** - 从强先验到自主决策的渐进式学习
- **可配置** - 所有参数可通过配置文件调整

**这不是简单修补，而是完全重新设计的定缺系统。**