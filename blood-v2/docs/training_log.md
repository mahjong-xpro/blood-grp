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

### 优化记录（2026-02-26 00:13 CST）

| 修改 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `lr_adaptive_min` | 2e-5 | **5e-5** | LR 76.6% 时间锁死在下限，策略几乎不更新 |
| `lr_schedule_kl_threshold` | 0.0002 | **0.0005** | 旧值低于实际 KL 均值(0.0004)，导致 LR 被持续压低 |
| `blood_arena_eval_games` | 50 | **100** | 50 局方差过大（95% CI ±14%），提升到 100 局（±10%） |

> 文件: `configs/competitive_distill.yaml` | commit: `25319630`
> 生效方式: `./scripts/manage.sh train distill --resume`

**⚠️ Resume 副作用**: SF2 `--resume` 重置了 `env_steps` 计数器到 0，导致:
1. TensorBoard 历史被覆盖（旧的 2.2M 步数据丢失，仅保留本文档记录）
2. entropy schedule 从 0.05 重新开始（之前已衰减到 0.034）
3. `train_for_env_steps=4M` 将从 0 重新计数，实际总训练量 ~6.2M steps

模型权重和 optimizer 状态已正确恢复（value_loss 起始 ~0.46 与之前最佳值一致，未回到随机水平）。entropy 回升到 0.05 实际上有利于在新 LR 下增加探索。

**优化生效确认**: `actual_lr` 稳定在 **5e-5**（旧值 2e-5），提升 2.5x ✅

---

### Phase 2b-续: Competitive Distill — LR 优化后（进行中 🔄 ~688K/4M 新计数）

> 运行名: `blood_v2_competitive_distill/.summary/0`（step counter 已重置）
> 配置: `configs/competitive_distill.yaml` | 实际总训练量: ~2.2M(旧) + 688K(新) ≈ 2.9M steps
> 关键变更: `lr_adaptive_min: 5e-5`, `lr_schedule_kl_threshold: 0.0005`, `arena_eval_games: 100`
> Elo tracker 状态已继承（起始 elo_games=550，当前 850）

#### 优化效果对比

| 指标 | 优化前 (旧 2.2M 均值) | 优化后 (@688K) | 变化 | 评估 |
|------|----------------------|----------------|------|------|
| `train/actual_lr` | 2e-5 (76.6% 锁死) | **5e-5** (100% 锁死) | ↑ 2.5x | ⚠️ 仍在下限 |
| `train/kl_divergence` | 0.0004 | **0.0016** (均值) | ↑ 4x | ✅ 策略更新更积极 |
| `train/fraction_clipped` | 1.1% | **10.2%** (均值, 峰值 39.7%) | ↑ 9x | ❌ 过高 |
| `train/grad_norm` | 1.50 (均值) | **1.83** (70.2% 贴满 2.0) | ↑ | ❌ 严重 clipping |
| `train/entropy` | 0.92 | **1.03** | ↑ 0.11 | ✅ schedule 重启 |
| `blood/elo_current` | 1441 (末值) | **1446** | ↑ +5 | ⚠️ 几乎无改善 |

#### 奖励 / 目标

| 指标 | @8K | @100K | @200K | @300K | @500K | @688K | 评估 |
|------|-----|-------|-------|-------|-------|-------|------|
| `reward/reward` | +0.62 | -0.25 | -0.21 | -0.19 | -0.04 | **-0.15** | ✅ 零和收敛 |
| `reward/reward_max` | — | — | — | — | — | **+1.64** | ✅ 正常 |
| `reward/reward_min` | — | — | — | — | — | **-1.61** | ✅ 正常 |

#### 损失

| 指标 | @8K | @100K | @200K | @300K | @500K | @688K | 评估 |
|------|-----|-------|-------|-------|-------|-------|------|
| `train/loss` | 0.42 | 0.73 | 0.83 | 1.26 | 0.92 | **0.99** | ⚠️ 波动中 |
| `train/value_loss` | 0.46 | 0.79 | 0.88 | 1.31 | 0.97 | **1.00** | ⚠️ 均值 ~1.0 |
| `train/policy_loss` | — | — | — | — | — | **+0.000** | ✅ 正常 |

#### Arena 评估 📊（100 局/次，Elo 累积，每次分两步记录）

| # | Step (新) | Elo (中间→最终) | Elo Games | Win Rate | Avg Rank | Avg Score | 评估 |
|---|-----------|-----------------|-----------|----------|----------|-----------|------|
| 11 | 213K | —→1498 | 596 | 0.60 | 2.44 | 99440 | ✅ |
| 12 | 221K | —→1517 | 650 | 0.52 | 2.43 | 100380 | ✅ |
| 13 | 418K→426K | 1567→1484 | 697→750 | 0.54 | 2.36 | 101650 | ⚠️ |
| 14 | 623K→631K | 1532→1446 | 812→850 | 0.43 | 2.59 | 100080 | ❌ |

#### 🔍 深度分析（2026-02-26 01:05 CST）

##### 1. 训练健康度总评: ❌ 两个瓶颈阻碍学习

LR 优化（2e-5→5e-5）成功提升了策略更新幅度（KL 4x），但暴露了两个新瓶颈：**grad_norm clipping 过度**和 **PPO clip 过度**。这两个问题限制了模型的有效学习。

##### 2. 瓶颈 A: grad_norm clipping 过度（70.2%）

`max_grad_norm=2.0` 在 70.2% 的步骤触发 clipping，均值 1.83。这意味着大部分梯度更新被截断，模型无法充分利用计算出的梯度方向。

**对比**: 旧 run 中 grad_norm 贴满 2.0 的比例是 28.9%。LR 从 2e-5 提升到 5e-5 后，梯度范数增大，clipping 比例从 29% 跳到 70%。

**影响**: 梯度被截断意味着 value function 和 policy 的更新方向被扭曲，学习效率降低。

##### 3. 瓶颈 B: PPO fraction_clipped 过高（均值 10.2%, 峰值 39.7%）

`ppo_clip_ratio=0.1` 下，10.2% 的 action 概率比被 clip。峰值 39.7% 意味着某些 minibatch 中近 40% 的更新被丢弃。

**对比**: 旧 run 均值 1.1%。LR 提升后策略更新幅度增大，更多更新超出 clip 范围。

**影响**: 过多的 clipping 导致有效梯度信号被丢弃，策略改善速度受限。

##### 4. Elo 中间峰值现象

每次 100 局 arena eval 都出现"中间峰值→最终回落"的模式:
- Eval 13: 1567→1484（-83）
- Eval 14: 1532→1446（-86）

这说明模型在前 ~50 局表现较好，后 ~50 局表现较差。可能原因:
- 100 局内的随机方差（正常）
- 模型对某些 RuleBot 策略模式有优势，对其他模式有劣势
- Elo 系统在小样本下的波动

##### 5. 建议: 放宽 grad_norm 和 clip_ratio

**立即修改** `configs/competitive_distill.yaml`:

```yaml
max_grad_norm: 3.0          # 从 2.0 提升到 3.0，减少梯度截断（当前 70% → 预期 ~30%）
ppo_clip_ratio: 0.15        # 从 0.1 提升到 0.15，减少 PPO clipping（当前 10% → 预期 ~3%）
```

**理由**:
- `max_grad_norm=3.0`: 当前梯度均值 1.83，70% 超过 2.0。提升到 3.0 后，大部分梯度可以完整传递。
- `ppo_clip_ratio=0.15`: 标准 PPO 使用 0.2，当前 0.1 过于保守。0.15 是合理的中间值。

**风险**: 更大的更新幅度可能导致短期 value_loss 波动增大。但当前的过度 clipping 是更大的问题。

**替代方案**: 如果不想同时改两个参数，优先改 `max_grad_norm`（影响更大）。

##### 6. 是否应该继续 Phase 2b 还是进入 Phase 3?

**建议继续 Phase 2b**，原因:
- LR 优化刚生效 ~688K 步，还需要更多时间评估效果
- 放宽 grad_norm/clip_ratio 后可能有显著改善
- Phase 3 的 elite 配置更复杂（RTPA/ISMCE），在基础训练不稳定时进入会更难调试

**进入 Phase 3 的条件**:
- Elo 稳定在 ≥ 1520 且连续 3 次评估不降
- value_loss 均值 < 0.8
- fraction_clipped < 5%

---

### Phase 2b（旧 run 存档）: Competitive Distill — 优化前数据

> 运行名: `blood_v2_competitive_distill/.summary/0`（TensorBoard 数据已被覆盖）
> 配置: `configs/competitive_distill.yaml` | 实际运行: ~2.2M steps
> 核心变化: `oracle_value_distill_weight: 0.1`（启用 oracle value 蒸馏）
> LR 修复: `lr_adaptive_min: 2e-5`, `lr_schedule_kl_threshold: 0.0002`
> entropy schedule: `cosine,0.05,0.02,0,4000000`（当前 0.034）

#### 奖励 / 目标

| 指标 | @0 | @200K | @500K | @800K | @1M | @1.5M | @2M | 评估 |
|------|-----|-------|-------|-------|------|-------|------|------|
| `reward/reward` | +1.13 | -0.10 | -0.09 | -0.21 | -0.05 | -0.10 | **-0.04** | ✅ 零和收敛 |
| `reward/reward_max` | +14.88 | +2.25 | +1.96 | +2.33 | +1.67 | +2.31 | **+1.65** | ✅ 稳定 |
| `reward/reward_min` | -0.96 | -1.38 | -1.49 | -1.69 | -1.32 | -2.06 | **-1.20** | ✅ 正常范围 |

#### 损失

| 指标 | @8K | @200K | @500K | @800K | @1M | @1.5M | @2M | 评估 |
|------|-----|-------|-------|-------|------|-------|------|------|
| `train/loss` | 1.02 | 1.06 | 0.62 | 0.58 | 0.46 | 0.76 | **0.75** | ⚠️ 高波动 0.46~1.38 |
| `train/value_loss` | 1.04 | 1.10 | 0.67 | 0.63 | 0.51 | 0.81 | **0.79** | ⚠️ 高波动，非单调 |
| `train/policy_loss` | 0.003 | 0.001 | -0.000 | -0.001 | -0.001 | +0.002 | **-0.001** | ✅ 正常 |

#### 训练稳定性

| 指标 | @8K | @200K | @500K | @800K | @1M | @1.5M | @2M | 评估 |
|------|-----|-------|-------|-------|------|-------|------|------|
| `train/entropy` | 0.49 | 0.89 | 0.95 | 0.95 | 0.97 | 0.95 | **0.92** | ✅ 稳定 0.87~0.97 |
| `train/kl_divergence` | 0.003 | 0.001 | 0.000 | 0.0004 | 0.0001 | 0.0002 | **0.0007** | ✅ 极低 |
| `train/fraction_clipped` | 8.7% | 1.3% | 0.1% | 1.0% | 0.6% | 0.7% | **2.1%** | ✅ 低 |
| `train/grad_norm` | 2.000 | 1.264 | 1.058 | 2.000 | 1.893 | 0.803 | **0.981** | ⚠️ 波动 0.65~2.0 |
| `train/actual_lr` | 2.6e-5 | 2e-5 | 2e-5 | 2e-5 | 4.5e-5 | 2e-5 | **2e-5** | ⚠️ 偶尔跳升 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `len/len` | 19.4 → 17.8 步 | ✅ 正常 |
| `blood/league_pool_size` | 34 → **50** | ✅ 持续增长 |
| `blood/sched_exploration_loss_coeff` | 0.050 → **0.034** | ✅ cosine 衰减中 |

#### Arena 评估 📊（10 次评估，每次 50 局 vs RuleBot）

> Win Rate / Avg Rank / Avg Score 为该次评估 50 局的统计值。
> Elo 为累积值（基于截至该次评估的所有对局，Elo Games 列显示累积局数）。

| # | Step | Elo | Elo Games | Win Rate | Avg Rank | Avg Score | 评估 |
|---|------|-----|-----------|----------|----------|-----------|------|
| 1 | 213K | 1464 | 50 | 0.60 | 2.44 | 99440 | ✅ |
| 2 | 418K | 1520 | 100 | 0.56 | 2.15 | 104080 | ✅ 最佳 rank |
| 3 | 623K | 1516 | 150 | 0.38 | 2.68 | 100040 | ❌ 低谷 |
| 4 | 827K | 1541 | 200 | 0.52 | 2.50 | 99360 | ✅ Elo 峰值 |
| 5 | 1040K | 1479 | 250 | 0.58 | 2.52 | 99200 | ⚠️ Elo 回落 |
| 6 | 1237K | 1460 | 300 | 0.42 | 2.77 | 97300 | ❌ 最差 rank |
| 7 | 1442K | 1508 | 350 | 0.50 | 2.57 | 98920 | ⚠️ 恢复中 |
| 8 | 1647K | 1536 | 400 | 0.58 | 2.32 | 99260 | ✅ 回升 |
| 9 | 1851K | 1527 | 450 | 0.44 | 2.41 | 99520 | ⚠️ 微降 |
| 10 | 2056K | 1441 | 500 | 0.44 | 2.69 | 97260 | ❌ Elo 新低 |

#### 🔍 深度分析（2026-02-26 00:10 CST 更新）

##### 1. 训练健康度总评: ❌ 学习停滞 + Elo 下降，需要干预

经过 10 次 arena 评估（累计 500 局），Elo 从初始 1500 降到 1441。核心问题是学习率过低导致策略几乎不更新。

##### 2. 核心问题：学习率锁死在下限

**量化分析**:
- `actual_lr` 在 `lr_adaptive_min=2e-5` 的时间占比: **76.6%**（209/273 步）
- 仅 23.4% 的步骤 LR 高于 2e-5，且多数仅到 3e-5（1.5x）
- 最高 LR: 1.01e-4（仅 1 次，@1442K）
- KL 均值仅 0.0004，远低于 `lr_schedule_kl_threshold=0.0002`

**机制失效分析**: `kl_adaptive_minibatch` 的逻辑是：KL > threshold 时降 LR，KL < threshold 时升 LR。但当前 KL 均值 0.0004 虽然高于 threshold 0.0002，但 KL 的波动范围（0.0001~0.003）导致 LR 调整方向频繁翻转，最终被 `lr_adaptive_min=2e-5` 兜底。

**影响**: 策略更新极其保守，模型无法有效学习新策略。这直接导致了 Elo 停滞和下降。

##### 3. Elo 轨迹分析

```
Elo: 1500→1464→1520→1516→1541→1499→1479→1460→1508→1536→1527→1441
                                    ↑ 峰值 @827K                        ↑ 新低 @2056K
```

**前半段（0~827K）**: Elo 上升到峰值 1541，模型在学习。
**后半段（827K~2.1M）**: Elo 震荡下降到 1441，低于初始值。

**Arena win_rate 统计**: 均值 50.9%（9 次评估），6/9 次 ≥ 50%。模型勉强维持在随机水平。

##### 4. value_loss 有改善但波动大

| 区间 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| 前半段 (0~1M) | 0.957 | 0.193 | 0.51~1.41 |
| 后半段 (1M~2.2M) | 0.894 | 0.188 | 0.54~1.43 |
| 最近 10 步 | 0.813 | — | — |

oracle value distillation 在长期上有效（均值从 0.96 降到 0.89），但波动仍然显著（std ~0.19）。

##### 5. grad_norm 频繁触发 clipping

28.9% 的步骤 grad_norm 触发 `max_grad_norm=2.0` clipping。这与 value_loss 的高波动一致——当 value_loss 突然升高时，梯度范数也随之增大。

##### 6. 正面信号

- **entropy 稳定** (均值 0.92, std 0.03): 策略多样性良好，cosine schedule 正常衰减到 0.034
- **KL 极低** (均值 0.0004): 无灾难性遗忘
- **fraction_clipped 低** (均值 1.1%): PPO 更新稳定
- **reward 零和收敛** (均值 -0.05): selfplay 环境正常
- **league_pool_size 增长到 50**: 对手多样性持续增加

##### 7. 优化建议（按优先级排序）

**🔴 P0: 提升学习率（立即实施）**

当前 LR 过低是所有问题的根源。建议修改 `configs/competitive_distill.yaml`:

```yaml
# 方案 A: 提升 KL adaptive 的 LR 范围（推荐）
lr_adaptive_min: 5e-5              # 从 2e-5 提升到 5e-5
lr_adaptive_max: 2e-4              # 保持不变
lr_schedule_kl_threshold: 0.0005   # 从 0.0002 提升到 0.0005，匹配实际 KL 均值

# 方案 B: 改用固定 cosine decay（备选）
# lr_schedule: linear_decay
# learning_rate: 1e-4
# lr_adaptive_min: 2e-5  # 作为 cosine 终点
```

方案 A 保留 KL adaptive 机制但扩大有效范围；方案 B 放弃 adaptive，使用确定性衰减。

**预期效果**: LR 从 2e-5 提升到 5e-5~1e-4，KL 应从 0.0004 上升到 0.001~0.003，策略更新幅度增大 2.5~5x。

**🟡 P1: 混合对手策略（建议实施）**

纯 selfplay 导致对 RuleBot 泛化退化。建议在 selfplay 中混入 RuleBot:

```yaml
opponent_mode: mixed              # 新增模式
opponent_selfplay_ratio: 0.8      # 80% selfplay
opponent_rulebot_ratio: 0.2       # 20% rulebot
```

如果 `mixed` 模式不支持，可以考虑在 Phase 3 的 arena eval 中增加评估频率来更早发现退化。

**🟢 P2: 增加 arena 评估局数**

```yaml
blood_arena_eval_games: 100       # 从 50 提升到 100
```

50 局的 win_rate 95% CI ≈ ±14%，100 局可缩小到 ±10%。

**🟢 P3: 考虑提前进入 Phase 3**

Phase 2b 已完成 2.2M/4M steps（55%），oracle distillation 的 value_loss 已有改善（0.96→0.89）。继续运行的边际收益递减。Phase 3 有 RTPA/ISMCE 等新机制，可能更有效地提升实力。

准入条件评估:
- value_loss 均值 < 1.0: ✅（当前 0.89）
- 训练步数 ≥ 2M: ✅（当前 2.2M）
- Elo 趋势: ❌（仍在下降，但可能是 LR 问题而非模型能力问题）

**建议**: 先实施 P0（提升 LR），观察 2~3 次 arena 评估（~400K steps）。如果 Elo 回升到 ≥ 1520，继续 Phase 2b；如果仍无改善，提前进入 Phase 3。

---

### Phase 3: Elite

*待启动 — 等待 Phase 2b LR 优化后的 arena 评估结果，或在 ~2.6M steps 时提前进入*
