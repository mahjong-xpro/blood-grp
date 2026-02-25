# Blood V2 训练日志

---

## 当前训练（2026-02-25 19:19 CST 起）

> 五阶段: warmup → warmup_transition → competitive → competitive_distill → elite

---

### Phase 1: Warmup（已完成 ✅）

> 运行名: `blood_v2_warmup/.summary/0`
> 配置: `configs/warmup.yaml` | 目标: 2M steps | 实际: ~2.01M steps
> 关键设置: `use_rnn: true` (LSTM 512×2), `opponent_mode: rulebot`, `lr: 5e-4`, `lr_adaptive_max: 3e-4`, `gamma: 0.99`

#### 奖励 / 目标

| 指标 | 初始值 | 中间值 (~1M) | 最终值 (2M) | 评估 |
|------|--------|-------------|-------------|------|
| `reward/reward` | -0.22 @0 | -0.22 @1.0M | **-0.20** @1.96M | ⚠️ 负值平稳，见分析 |
| `reward/reward_max` | +2.13 @0 | — | **+0.44** @1.96M | ✅ 正常范围 |
| `reward/reward_min` | -1.16 @0 | — | **-1.13** @1.96M | ✅ 正常范围 |

#### 损失

| 指标 | 初始值 | 中间值 (~1M) | 最终值 | 评估 |
|------|--------|-------------|--------|------|
| `train/loss` | 1.05 @8K | 1.26 @1.0M | **0.50** @2.0M | ✅ 下降 52.6% |
| `train/value_loss` | 1.05 @8K | 1.27 @1.0M | **0.50** @2.0M | ✅ 价值函数收敛 |
| `train/policy_loss` | 0.013 @8K | — | **0.003** @2.0M | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | 初始值 | 中间值 (~1M) | 最终值 | 评估 |
|------|--------|-------------|--------|------|
| `train/entropy` | 1.03 @8K | 0.73 @1.0M | **0.45** @2.0M | ✅ 持续下降，策略收敛 |
| `train/kl_divergence` | 0.007 @25K | — | **0.003** @2.0M | ✅ KL 低，稳定 |
| `train/fraction_clipped` | 10.7% @25K | — | **3.6%** @2.0M | ✅ 大幅下降 |
| `train/grad_norm` | 1.000 @8K | 1.000 @1.0M | **1.000** @2.0M | ⚠️ 全程贴满 max_grad_norm，见分析 |
| `train/actual_lr` | 5e-4 @8K | 3e-4 @1.0M | **3e-4** @2.0M | ✅ 稳定在 lr_adaptive_max（修复生效） |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | ~4915 FPS | ✅ 吞吐量稳定 |
| `len/len` | 20.8 → 19.0 步 | ✅ 正常麻将局长度 |
| `blood/league_pool_size` | 0 → **15** | ✅ 正常递增 |

#### 分析

**LR 修复验证**: `actual_lr` 从 5e-4 起步，稳定在 3e-4（`lr_adaptive_max` 上限），不再攀升到 0.01。✅ 修复生效。

**reward**: sqrt-compressed 尺度下持续为负（-0.22 → -0.20），这是 warmup 阶段对 RuleBot 的正常表现。策略收敛指标全部正常：entropy 1.03→0.45, policy_loss 0.013→0.003。

**grad_norm 全程 1.0**: grad_norm 全程贴满 `max_grad_norm=1.0`。这不是问题——grad_norm clipping 是正常的梯度保护机制，value_loss 仍然正常下降（1.05→0.50）。

---

### Phase 1.5: Warmup Transition（已完成 ✅）

> 运行名: `blood_v2_warmup_transition/.summary/0`
> 配置: `configs/warmup_transition.yaml` | 目标: 500K steps | 实际: ~516K steps
> 核心变化: `gamma: 0.995`, `lr: 2e-4`, `lr_adaptive_max: 3e-4`
> 前置 checkpoint: warmup 最佳 checkpoint（optimizer LR 已重置为 2e-4）

#### 奖励 / 目标

| 指标 | Warmup 结束 | Transition 起始 | Transition 最终 (~500K) | 评估 |
|------|------------|----------------|------------------------|------|
| `reward/reward` | -0.20 | +0.33 @0 | **-0.15** @492K | ✅ 与 warmup 持平 |
| `reward/reward_max` | +0.44 | +4.69 @0 | **+0.41** @492K | ✅ 正常范围 |
| `reward/reward_min` | -1.13 | -1.22 @0 | **-1.00** @492K | ✅ 正常范围 |

#### 损失

| 指标 | Transition 起始 | Transition 最终 | 评估 |
|------|----------------|----------------|------|
| `train/loss` | **3.75** @8K | **0.82** @516K | ✅ 初始尖峰后快速恢复 |
| `train/value_loss` | **3.75** @8K | **0.82** @516K | ✅ 同上 |
| `train/policy_loss` | 0.004 @8K | **0.001** @516K | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | Transition 起始 | Transition 最终 | 评估 |
|------|----------------|----------------|------|
| `train/entropy` | 0.31 @8K | **0.51** @516K | ✅ 略有回升后稳定 |
| `train/kl_divergence` | 0.011 @8K | **0.002** @492K | ✅ 快速降低，稳定 |
| `train/fraction_clipped` | 4.7% @8K | **2.8%** @492K | ✅ 极低 |
| `train/grad_norm` | 1.000 @8K | **0.97** @516K | ✅ 末段开始下降 |
| `train/actual_lr` | **3e-4** @8K | **3e-4** @516K | ✅ 稳定在上限（修复生效） |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | ~4915 FPS | ✅ 吞吐量稳定 |
| `len/len` | 20.5 → 19.0 步 | ✅ 正常 |
| `blood/league_pool_size` | 15 → **18** | ✅ 继承 warmup 池，继续递增 |

#### 过渡质量评估

| 检查项 | 结果 |
|--------|------|
| checkpoint 加载是否成功？ | ✅ 是，架构完全匹配 |
| reward 是否大幅下降？ | ✅ 否，与 warmup 末段持平 (-0.20 → -0.15) |
| KL 是否失控？ | ✅ 否，0.011→0.002 |
| value_loss 是否爆炸？ | ⚠️ 初始尖峰 3.75，快速恢复到 0.82 |
| actual_lr 是否受控？ | ✅ 全程 3e-4（lr_adaptive_max 生效） |
| grad_norm 是否全程贴满？ | ✅ 否，末段降至 0.97 |

#### 分析

**value_loss 初始尖峰 (3.75)**: gamma 从 0.99→0.995 改变了 GAE returns 的尺度，价值函数需要短暂适应。这是正常的超参变化效应，非 bug。快速恢复到 0.82。

---

### Phase 2a: Competitive（进行中 🔄）

> 运行名: `blood_v2_competitive/.summary/0`
> 配置: `configs/competitive.yaml` | 目标: 1M steps | 当前: ~647K steps
> 核心变化: `opponent_mode: selfplay`, `gamma: 0.998`, `lr: 1e-4`, `lr_adaptive_max: 3e-4`, `batch_size: 1024`
> 新增: `reward_rank_bonus: 0.15`, `reward_safe_discard: 0.015`, `shanten_fan_bonus_scale: 0.15`, `adv_clip: 2.0`

#### 奖励 / 目标

| 指标 | 起始值 | 中间值 (~250K) | 当前值 (~647K) | 评估 |
|------|--------|---------------|---------------|------|
| `reward/reward` | +3.08 @0 | -0.07 @205K | **-0.12** @647K | ✅ 自博弈收敛到零和，见分析 |
| `reward/reward_max` | +17.48 @0 | +1.69 @205K | **+1.47** @647K | ✅ 稳定 |
| `reward/reward_min` | -0.75 @0 | -1.37 @205K | **-0.93** @647K | ✅ 正常范围 |

#### 损失

| 指标 | 起始值 | 中间值 (~250K) | 当前值 (~647K) | 评估 |
|------|--------|---------------|---------------|------|
| `train/loss` | 1.86 @8K | 1.25 @205K | **2.02** @647K | ⚠️ V 形走势，见分析 |
| `train/value_loss` | 1.88 @8K | 1.27 @205K | **2.05** @647K | ⚠️ 同上 |
| `train/policy_loss` | 0.004 @8K | 0.002 @205K | **0.003** @647K | ✅ 正常 |

#### 训练稳定性

| 指标 | 起始值 | 中间值 (~250K) | 当前值 (~647K) | 评估 |
|------|--------|---------------|---------------|------|
| `train/entropy` | 0.41 @8K | 0.62 @205K | **0.69** @647K | ✅ entropy schedule 生效（0.01→0.05 线性） |
| `train/kl_divergence` | 0.008 @8K | 0.002 @205K | **0.002** @647K | ✅ 稳定 |
| `train/fraction_clipped` | 5.1% @8K | 4.0% @205K | **7.9%** @647K | ✅ 可控范围 |
| `train/grad_norm` | 1.000 @8K | 1.000 @205K | **1.000** @647K | ⚠️ 全程贴满 |
| `train/actual_lr` | 2e-4 @8K | 2.6e-5 @205K | **8.9e-5** @647K | ⚠️ 先降后升，见分析 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `len/len` | 22.7 → 17.7 步 | ✅ 正常 |
| `blood/league_pool_size` | 18 → **29** | ✅ 持续递增 |

#### 分析

**reward 收敛到零和**: reward 从 +3.08（碾压旧对手）快速回落到 -0.07@250K → -0.12@647K。在自博弈中 reward 趋近零是正常的——对手和自己同步变强，零和博弈的期望 reward 就是 0。轻微负值（-0.12）可能因为 rank_bonus/safe_discard 等额外惩罚信号。

**value_loss V 形走势 (1.88→1.27→2.05)**: 前 250K 步 value_loss 下降（适应自博弈环境），之后反弹上升。可能原因：
1. entropy schedule 从 0.01→0.05 持续提升探索系数，策略变化加大，价值函数追不上
2. league pool 从 18→29，对手多样性增加，returns 方差增大
3. actual_lr 先降到 2.6e-5（过低），后回升到 8.9e-5，LR 波动影响收敛

**actual_lr 先降后升**: KL 在初期较高（0.008），`kl_adaptive_minibatch` 降低 LR 到 2.6e-5。KL 稳定后 LR 回升到 8.9e-5。这个范围（2.6e-5 ~ 2e-4）在 `lr_adaptive_max: 3e-4` 限制内，属于正常自适应行为。但 LR 最低点 2.6e-5 可能过低，导致 250K 后学习速度不足。

**需要关注**: value_loss 是否在剩余 350K 步内开始下降。如果持续上升到 1M 步，可能需要调整 entropy schedule 或 lr_schedule_kl_threshold。

---

### Bug 修复记录

| commit | 描述 |
|--------|------|
| — | manage.sh `find_best_checkpoint()` 路径错误（`checkpoints/` → `train_dir/`） |
| `21c0a1d8` | `--init_checkpoint_path` 未在 argparser 注册 |
| `51dcf87a` | PyTorch 2.6 `weights_only=True` 导致 SF2 checkpoint 加载失败 |
| `cd66d570` | 新增 `manage.sh stop` 命令 |
| `749e58b6` | 跨阶段 `strict=False` 加载（monkey-patch `Learner._load_state`） |
| `00bf21bb` | warmup.yaml `use_rnn: false` → `true`（根本修复架构不匹配） |
| `bf15894a` | 保留 optimizer 状态（SF2 `_load_state` 要求 `checkpoint["optimizer"]`） |
| `ea97f796` | warmup/transition 添加 `lr_adaptive_max: 3e-4` + optimizer LR 重置 + 过时注释修复 |

---

### Phase 2b: Competitive Distill

*待启动*

---

### Phase 3: Elite

*待启动*
