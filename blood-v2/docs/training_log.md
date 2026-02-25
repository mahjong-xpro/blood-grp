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

### Phase 2a: Competitive（已完成 ✅）

> 运行名: `blood_v2_competitive/.summary/0`
> 配置: `configs/competitive.yaml` | 目标: 1M steps | 实际: ~1.016M steps
> 核心变化: `opponent_mode: selfplay`, `gamma: 0.998`, `lr: 1e-4`, `lr_adaptive_max: 3e-4`, `batch_size: 1024`
> 新增: `reward_rank_bonus: 0.15`, `reward_safe_discard: 0.015`, `shanten_fan_bonus_scale: 0.15`, `adv_clip: 2.0`
> entropy schedule: `linear,0.01,0.05,0,500000` — 已在 500K 步达到 0.05 上限

#### 奖励 / 目标

| 指标 | @0 | @100K | @300K | @500K | @700K | @900K | @1M | 评估 |
|------|-----|-------|-------|-------|-------|-------|------|
| `reward/reward` | +3.08 | -0.05 | -0.06 | -0.10 | -0.14 | -0.14 | **-0.12** @1M | ✅ 零和收敛 |
| `reward/reward_max` | +17.48 | +1.90 | +1.65 | +1.89 | +2.02 | +1.57 | **+2.24** @1M | ✅ 稳定 |
| `reward/reward_min` | -0.75 | -1.88 | -1.35 | -1.08 | -1.56 | -1.49 | **-1.92** @1M | ✅ 正常 |

#### 损失

| 指标 | @8K | @100K | @300K | @500K | @700K | @900K | @1M | 评估 |
|------|-----|-------|-------|-------|-------|-------|-----|------|
| `train/loss` | 1.86 | 2.15 | 1.39 | 1.46 | 1.22 | 1.69 | **1.29** | ⚠️ 高波动但末段回落 |
| `train/value_loss` | 1.88 | 2.18 | 1.42 | 1.49 | 1.26 | 1.73 | **1.34** | ⚠️ 同上 |
| `train/policy_loss` | 0.004 | 0.003 | 0.002 | 0.003 | 0.002 | -0.000 | **0.001** | ✅ 正常 |

#### 训练稳定性

| 指标 | @8K | @100K | @300K | @500K | @700K | @900K | @1M | 评估 |
|------|-----|-------|-------|-------|-------|-------|-----|------|
| `train/entropy` | 0.41 | 0.53 | 0.65 | 0.71 | 0.76 | 0.81 | **0.85** | ✅ schedule 生效 |
| `train/kl_divergence` | 0.008 | 0.002 | 0.001 | 0.000 | 0.004 | 0.001 | **0.003** | ✅ 低且稳定 |
| `train/fraction_clipped` | 5.1% | 13.8% | 4.2% | 0.8% | 13.3% | 5.6% | **8.1%** | ⚠️ 高波动 |
| `train/grad_norm` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | ⚠️ 全程贴满 |
| `train/actual_lr` | 2.0e-4 | 1.3e-4 | 1.3e-4 | 5.9e-5 | 1.3e-4 | 5.9e-5 | **2.0e-4** | ⚠️ 大幅波动 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `len/len` | 22.7 → 17.9 步 | ✅ 正常 |
| `blood/league_pool_size` | 18 → **34** | ✅ 持续递增 |
| `blood/sched_exploration_loss_coeff` | 0.012 → **0.050** | ✅ entropy schedule 完成 |
| `blood/elo_current` | 1500 → **1500** | ⚠️ 未变化，见分析 |

#### 🔍 深度分析

##### 1. 训练健康度总评: ⚠️ 有隐患但可接受

整体训练在运行，策略在学习（entropy 上升、reward 零和收敛、league pool 增长），但存在显著的不稳定性。

##### 2. value_loss 高波动（1.88 → 2.18 → 1.42 → 1.49 → 1.26 → 1.73）

value_loss 没有单调下降，而是在 0.91~2.18 之间大幅震荡。这是 competitive 阶段最大的隐患。

**原因分析**:
- 自博弈环境本身就是非平稳的（对手持续变化），value function 需要不断适应
- league pool 从 18→32，对手多样性持续增加，returns 分布不断变化
- entropy schedule 在前 500K 步从 0.01→0.05 线性提升，策略探索性增大，value 预测更难
- `kl_adaptive_minibatch` 导致 LR 在 1.8e-5 ~ 2.0e-4 之间大幅波动（10x 范围），加剧不稳定

**是否正常**: 在自博弈 RL 中，value_loss 波动是常见的。关键看 policy 是否在改善。reward 稳定在 ~0 附近（零和），entropy 持续上升（策略多样性增加），这些都是正面信号。value_loss 的绝对值（0.91~2.18）在 sqrt-compressed reward 尺度下也不算极端。

##### 3. actual_lr 大幅波动（1.8e-5 ~ 2.0e-4）

`kl_adaptive_minibatch` 根据 KL 散度动态调整 LR。KL 在 0.0002~0.008 之间波动，导致 LR 跟着大幅调整。

**问题**: `lr_schedule_kl_threshold: 0.001` 对于 masked action space 的麻将来说可能仍然偏高。当 KL 偶尔跳到 0.004~0.008 时，LR 被大幅降低到 1.8e-5，学习几乎停滞。当 KL 回落到 0.0002 时，LR 又被推高到 2.0e-4。

**建议**: 考虑将 `lr_schedule_kl_threshold` 从 0.001 降到 0.0005，或使用固定 LR schedule（如 cosine decay）替代 `kl_adaptive_minibatch`。

##### 4. policy_loss 偶尔为负

policy_loss 在 @200K (-0.0013)、@400K (-0.0007)、@800K (-0.0009)、@900K (-0.0004) 出现负值。PPO 的 policy_loss 为负意味着 clipped surrogate objective 为正（策略更新方向正确但被 clip 限制）。这在 `ppo_clip_ratio: 0.1`（较紧的 clip）下是正常的。

##### 5. grad_norm 全程 1.0

`max_grad_norm: 1.0` 全程触发 gradient clipping。说明梯度范数始终 > 1.0。这在大模型（256ch/20blocks + LSTM 512×2）+ 多个 loss 项（PPO + aux + oracle distill + oracle CE）的情况下是预期的。不影响训练，但说明模型容量和 loss 复杂度较高。

##### 6. Elo 未变化（1500）— ✅ 已修复

Elo 停留在 1500 是因为训练循环中从未调用 `EloTracker.update_from_game()`。`_log_elo_summaries` 只读取已有评分，不主动触发对局评估。

**修复 (commit 4bfc7472)**: 在 `BloodObserver.on_training_step` 中添加定期 arena 评估。后台线程加载最新 checkpoint，运行 N 局 vs RuleBot，通过 `Arena.evaluate()` 更新 EloTracker。新增 TensorBoard 指标: `blood/elo_current`, `blood/elo_games`, `blood/arena_win_rate`, `blood/arena_avg_rank`, `blood/arena_avg_score`。所有阶段 YAML 已配置 `blood_arena_eval_every`。

##### 7. 结论与建议

**最终状态**: competitive 阶段已完成（1.016M steps）。value_loss 末段回落到 1.34（从 900K 的 1.73 下降），显示价值函数在最后阶段开始收敛。策略在学习，可以进入 Phase 2b。

**Phase 2b 已应用的修复**:
- `lr_schedule_kl_threshold`: 0.001 → 0.0005（减少 masked action space 下的 LR 振荡幅度）
- `lr_adaptive_max`: 3e-4 → 2e-4（防止 KL 低谷时 LR 跳升过高）
- `max_grad_norm`: 1.0 → 2.0（oracle value distill 新增 loss 项，需要更多梯度空间）
- 新增 entropy schedule: cosine 0.05→0.02 over 4M steps（逐步收紧策略）
- 新增 arena eval: 每 200K 步评估 50 局 vs RuleBot（实时跟踪 Elo 变化）

**Phase 2b 监控重点**:
- `blood/oracle_value_head_loss` 是否收敛到 < 0.1
- `train/actual_lr` 振荡幅度是否减小（预期 < 5x 范围）
- `train/value_loss` 是否比 Phase 2a 更稳定
- `blood/elo_current` 是否开始上升

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

### Phase 2b: Competitive Distill（进行中 🔄 ~303K/4M）

> 运行名: `blood_v2_competitive_distill/.summary/0`
> 配置: `configs/competitive_distill.yaml` | 目标: 4M steps | 当前: ~303K steps
> 核心变化: `oracle_value_distill_weight: 0.1`（启用 oracle value 蒸馏）
> 修复: `lr_schedule_kl_threshold: 0.0005`, `lr_adaptive_max: 2e-4`, `max_grad_norm: 2.0`
> entropy schedule: `cosine,0.05,0.02,0,4000000`

#### 奖励 / 目标

| 指标 | @0 | @100K | @200K | @300K | 评估 |
|------|-----|-------|-------|-------|------|
| `reward/reward` | +2.26 | +0.01 | -0.10 | **-0.06** | ✅ 零和收敛（同 competitive） |
| `reward/reward_max` | +11.41 | +2.55 | +1.44 | **+1.46** | ✅ 稳定 |
| `reward/reward_min` | -0.71 | -1.12 | -1.63 | **-1.14** | ✅ 正常 |

#### 损失

| 指标 | @8K | @100K | @200K | @300K | 评估 |
|------|-----|-------|-------|-------|------|
| `train/loss` | 0.91 | 0.75 | 0.51 | **1.17** | ⚠️ 波动，但整体低于 competitive |
| `train/value_loss` | 0.93 | 0.79 | 0.56 | **1.21** | ⚠️ 同上 |
| `train/policy_loss` | 0.002 | -0.001 | -0.000 | **0.001** | ✅ 正常 |

#### 训练稳定性

| 指标 | @8K | @100K | @200K | @300K | 评估 |
|------|-----|-------|-------|-------|------|
| `train/entropy` | 0.58 | 0.91 | 0.98 | **0.94** | ✅ 快速上升后稳定 |
| `train/kl_divergence` | — | 0.000 | 0.000 | **0.002** | ✅ 极低 |
| `train/fraction_clipped` | — | 0.5% | 0.9% | **13.7%** | ⚠️ 最新跳升 |
| `train/grad_norm` | 2.000 | 2.000 | 1.34 | **2.000** | ⚠️ 多数时间贴满 max_grad_norm=2.0 |
| `train/actual_lr` | 1e-4 | ~0 | ~0 | **~0** | 🔴 LR 极低，见分析 |

#### Arena 评估

| 指标 | 值 | 评估 |
|------|-----|------|
| `blood/arena_avg_rank` | **2.49** | ⚠️ 接近随机（2.5），见分析 |
| `blood/arena_avg_score` | **99400** | ⚠️ 略低于初始 100000 |
| `blood/arena_win_rate` | **0.50** | ⚠️ 50% 胜率 vs RuleBot |
| `blood/elo_current` | **1471.7** | ⚠️ 低于初始 1500 |
| `blood/elo_games` | **50** | ✅ 评估已运行 |

#### 🔴 关键问题: actual_lr 接近零

**现象**: `actual_lr` 在 ~100K 步后降到接近 0（约 1e-6 ~ 1e-7 量级），几乎停止学习。

**根因分析**: `kl_adaptive_minibatch` 在 KL 极低（0.0002~0.0003）时持续降低 LR。虽然 `lr_schedule_kl_threshold` 已从 0.001 降到 0.0005，但 KL 仍然远低于阈值，导致 LR 被持续压低。`lr_adaptive_min: 1e-6` 是 SF2 默认下限，LR 可能已经触底。

**影响**: 模型几乎停止学习。value_loss 在 200K 降到 0.56 后反弹到 1.21，说明 oracle value distillation 的新 loss 信号在推动变化，但 LR 太低无法有效优化。

**建议修复**:
1. 提高 `lr_adaptive_min` 从 1e-6 到 1e-5（确保最低 LR 仍有学习能力）
2. 或将 `lr_schedule_kl_threshold` 进一步降到 0.0002
3. 或切换到固定 LR schedule（如 cosine decay 1e-4 → 1e-5 over 4M steps）

#### ⚠️ Arena 表现分析

arena_avg_rank=2.49（接近随机的 2.5）和 win_rate=0.50 说明当前模型 vs RuleBot 表现一般。Elo 从 1500 降到 1471.7。

**可能原因**:
1. LR 接近零导致模型在 distill 阶段几乎没有改善
2. competitive 阶段的自博弈训练可能导致模型过度适应自博弈对手，对 RuleBot 的泛化能力不足
3. arena 评估只运行了 50 局，样本量较小，结果可能有噪声

**需要关注**: 修复 LR 问题后，arena 指标是否改善。

---

### Phase 3: Elite

*待启动*
