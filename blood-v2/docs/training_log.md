# Blood V2 训练日志

> 训练开始: 2026-02-25 19:19 CST | 五阶段: warmup → warmup_transition → competitive → competitive_distill → elite

---

## 当前状态（2026-02-26 21:30 CST）

```
Phase 1 (2M) → Phase 1.5 (500K) → Phase 2a (1M) → Phase 2b (4.5M) → Phase 3 (4.24M/50M) ⬅️ 当前
总训练量: ~12.2M env steps | Phase 3 进度: 8.5%
```

| 指标 | 当前值 | 评估 |
|------|--------|------|
| Elo | **1505** | ✅ 从 1480 回升 |
| Arena win_rate | **0.65** | 🎉 历史最佳 |
| Arena avg_rank | **2.35** | 🎉 历史最佳 |
| value_loss | **0.98** (分段均值) | ✅ 持续下降 |
| entropy | **0.47** | ⚠️ 持续下降，关注 < 0.40 |
| LR 均值 | 9.8e-5 | ✅ 活跃 |
| grad_norm clip@3.0 | 11.6% | ✅ 健康 |

**结论**: 训练状态良好，奖励优化适应期已结束。继续运行。

**需关注**: entropy 持续下降（0.70→0.47），按当前速率 ~8M 步可能接近 0.35 警戒线。若 Elo 同步上升则无需干预。

---

## Phase 3: Elite（进行中 🔄）

> 配置: `configs/elite.yaml` | 目标: 50M steps | 当前: ~4.24M (8.5%)
> 核心: `gamma: 0.999`, `gae_lambda: 0.97`, RTPA + ISMCE 启用
> 前置修复: `max_grad_norm: 3.0`, `ppo_clip_ratio: 0.15`, `lr_schedule_kl_threshold: 0.002`, `lr_adaptive_min: 5e-5`

#### 训练指标

| 指标 | @500K | @1M | @2M | @3M | @4M | @4.24M | 评估 |
|------|-------|------|------|------|------|--------|------|
| `actual_lr` | 1.69e-4 | 1.33e-4 | 5.0e-5 | 7.5e-5 | 7.5e-5 | **1.33e-4** | ✅ |
| `kl_divergence` | 0.0017 | 0.0018 | 0.0008 | 0.0028 | 0.0025 | **0.0034** | ✅ |
| `fraction_clipped` | 2.1% | 3.5% | 1.2% | 2.5% | 1.6% | **3.5%** | ✅ |
| `grad_norm` | 0.84 | 2.15 | 1.32 | 3.00 | 2.41 | **3.00** | ⚠️ |
| `entropy` | 0.69 | 0.64 | 0.53 | 0.50 | 0.46 | **0.49** | ⚠️ 下降 |
| `value_loss` | 1.04 | 1.14 | 0.75 | 1.08 | 0.93 | **1.08** | ✅ 下降 |
| `reward` | -0.03 | +0.23 | +2.08 | -0.02 | +0.04 | **+0.005** | ✅ 零和 |
| `elo_current` | 1483 | 1537 | 1516 | 1518 | 1535 | **1505** | ✅ |

#### value_loss 分段趋势

| 区间 | 均值 | 标准差 | 评估 |
|------|------|--------|------|
| 0~1M | 1.102 | 0.260 | 初始适应 |
| 1M~2M | 1.078 | 0.236 | ↓ |
| 2M~3M | 0.995 | 0.164 | ↓ |
| 3M~4.24M | **0.979** | **0.187** | ✅ 新低 |

#### Arena 评估（每 ~500K 步，vs RuleBot）

| # | Step | Elo | Win Rate | Avg Rank | Avg Score | 评估 |
|---|------|-----|----------|----------|-----------|------|
| 1 | 524K | 1483 | 0.49 | 2.63 | 98880 | ⚠️ 适应期 |
| 2 | 1032K | 1537 | 0.55 | 2.48 | 99610 | ✅ |
| 3 | 1540K | 1473 | 0.48 | 2.52 | 100030 | ⚠️ |
| 4 | ~2M | 1474 | 0.52 | 2.46 | 101430 | ⚠️ 批量继承 |
| 5 | 2523K | 1506 | 0.41 | 2.55 | 99470 | ⚠️ |
| 6 | 3031K | 1480 | 0.49 | 2.59 | 100190 | ⚠️ |
| 7 | 3539K | 1523 | 0.50 | 2.39 | 101390 | ✅ |
| **8** | **4047K** | **1505** | **0.65** | **2.35** | **101360** | **🎉 最佳** |

#### LR 动态

| 指标 | 全程 | 最近段 (3.34M~4.24M) |
|------|------|---------------------|
| LR 均值 | 1.14e-4 | 9.8e-5 |
| @5e-5 比例 | 24.5% | 28.6% |
| @2e-4 比例 | 22.5% | 6.2% |
| grad_norm clip@3.0 | 13.0% | 11.6% |

#### 分析

**Elo 锯齿上升**: 低谷抬升（1450→1473→1480），峰值维持（1537→1535）。Eval #8 win_rate=0.65 是历史最高，模型实力在提升。

**entropy 持续下降**: 分段均值 0.70→0.57→0.51→0.47。cosine schedule 系数几乎不变（0.020→0.0196），下降来自策略自身收敛。0.47 仍安全，但需监控。

**奖励优化已适应**: score-weighted shaping + safe_discard 降低后，Elo 从 1480 回升到 1505，适应期结束。

---

## Phase 2b: Competitive Distill（已完成 ✅）

> 配置: `configs/competitive_distill.yaml` | 总训练量: ~6.7M steps (含 resume)
> 核心: `oracle_value_distill_weight: 0.1`, selfplay 联赛
> 期间经历两次超参优化（详见附录）

#### 最终指标

| 指标 | 起始 | 最终 | 评估 |
|------|------|------|------|
| `value_loss` | 1.04 | **0.59** | ✅ Oracle 蒸馏有效 |
| `entropy` | 0.49 | **0.87** | ✅ 稳定 |
| `elo_current` | 1441 | **1525** | ✅ 回升 +84 |
| `league_pool_size` | 34 | **50** | ✅ 满池 |

#### 超参优化效果对比

| 指标 | 优化前 | 优化#1 (LR提升) | 优化#1+#2 (grad+clip) |
|------|--------|-----------------|----------------------|
| `actual_lr` | 2e-5 (76.6% 锁死) | 5e-5 (100% 锁死) | 5e-5 (100% 锁死) |
| `kl_divergence` | 0.0004 | 0.0015 | **0.0018** |
| `fraction_clipped` | 1.1% | 9.5% | **4.1%** |
| `grad_norm` clip | 29%@2.0 | 69%@2.0 | **41%@3.0** |
| `value_loss` | 0.89 | 0.93 | **0.81** |
| `elo_current` | 1441 | 1500 | **1525** (峰值 1549) |

#### Arena 评估（合并，关键节点）

| Step | Elo | Win Rate | Avg Rank | 备注 |
|------|-----|----------|----------|------|
| 418K (旧run) | 1520 | 0.56 | 2.15 | 旧 run 最佳 |
| 827K (旧run) | 1541 | 0.52 | 2.50 | 旧 run Elo 峰值 |
| 2056K (旧run) | 1441 | 0.44 | 2.69 | ❌ LR 锁死导致 Elo 新低 |
| 1311K (新run) | 1554 | — | — | 🎉 优化#2 后新高 |
| 2138K (新run) | 1549 | 0.60 | 2.32 | 🎉 最佳 Arena 表现 |
| 2335K (新run) | 1525 | 0.60 | 2.32 | ✅ 最终值 |
#### 关键教训

- KL adaptive LR 在 masked action space 下容易锁死在下限，需要 `kl_threshold` 与实际 KL 均值匹配
- `max_grad_norm` 从 1.0→2.0→3.0 逐步放宽，多 loss 项的大模型需要更多梯度空间
- `ppo_clip_ratio` 从 0.1→0.15，过紧的 clip 在自博弈中丢弃过多有效更新

---

## Phase 2a: Competitive（已完成 ✅）

> 配置: `configs/competitive.yaml` | 实际: ~1.016M steps
> 核心变化: `opponent_mode: selfplay`, `gamma: 0.998`, entropy schedule `linear,0.01,0.05,0,500000`

#### 最终指标

| 指标 | 起始 | 最终 | 评估 |
|------|------|------|------|
| `value_loss` | 1.88 | **1.34** | ⚠️ 高波动但末段回落 |
| `entropy` | 0.41 | **0.85** | ✅ schedule 生效 |
| `grad_norm` | 1.000 | **1.000** | ⚠️ 全程贴满 (max_grad_norm=1.0) |
| `actual_lr` | 2.0e-4 | **2.0e-4** | ⚠️ 大幅波动 1.8e-5~2.0e-4 |
| `league_pool_size` | 18 | **34** | ✅ |

#### 分析

自博弈环境非平稳，value_loss 在 0.91~2.18 间震荡是正常的。策略在学习（entropy 上升、reward 零和收敛）。`kl_adaptive_minibatch` 导致 LR 10x 范围波动，在后续阶段通过调整 `kl_threshold` 解决。

Elo 评估在此阶段修复（commit 4bfc7472），新增 arena 后台评估 + TensorBoard 指标。

---

## Phase 1.5: Warmup Transition（已完成 ✅）

> 配置: `configs/warmup_transition.yaml` | 实际: ~516K steps
> 核心变化: `gamma: 0.995`, `lr: 2e-4`

#### 最终指标

| 指标 | 起始 | 最终 | 评估 |
|------|------|------|------|
| `reward` | +0.33 | **-0.15** | ✅ 与 warmup 持平 |
| `value_loss` | 3.75 (尖峰) | **0.82** | ✅ 快速恢复 |
| `entropy` | 0.31 | **0.51** | ✅ 稳定 |
| `grad_norm` | 1.000 | **0.97** | ✅ 末段开始下降 |

**value_loss 初始尖峰 (3.75)**: gamma 从 0.99→0.995 改变了 GAE returns 尺度，价值函数短暂适应后快速恢复。

---

## Phase 1: Warmup（已完成 ✅）

> 配置: `configs/warmup.yaml` | 实际: ~2.01M steps
> 核心: LSTM 512×2, `opponent_mode: rulebot`, `lr: 5e-4`

#### 最终指标

| 指标 | 起始 | 最终 | 评估 |
|------|------|------|------|
| `reward` | -0.22 | **-0.20** | ✅ 对 RuleBot 正常 |
| `loss` | 1.05 | **0.50** | ✅ 下降 52.6% |
| `entropy` | 1.03 | **0.45** | ✅ 策略收敛 |
| `grad_norm` | 1.000 | **1.000** | ⚠️ 全程贴满 (正常) |
| `actual_lr` | 5e-4 | **3e-4** | ✅ lr_adaptive_max 生效 |
| `league_pool_size` | 0 | **15** | ✅ |

---

## 附录

### 超参数优化历史

| # | 日期 | 阶段 | 参数 | 旧值 | 新值 | 原因 | 效果 |
|---|------|------|------|------|------|------|------|
| 1 | 02-26 00:13 | Phase 2b | `lr_adaptive_min` | 2e-5 | 5e-5 | LR 76.6% 锁死 | KL 3.5x 提升 |
| 1 | 02-26 00:13 | Phase 2b | `lr_schedule_kl_threshold` | 0.0002 | 0.0005 | 低于实际 KL 均值 | LR 提升到 5e-5 |
| 1 | 02-26 00:13 | Phase 2b | `blood_arena_eval_games` | 50 | 100 | 方差过大 | CI ±14%→±10% |
| 2 | 02-26 01:25 | Phase 2b | `ppo_clip_ratio` | 0.1 | 0.15 | 9.5% 更新被丢弃 | 降到 4.1% |
| 2 | 02-26 01:25 | Phase 2b | `max_grad_norm` | 2.0 | 3.0 | 69.2% 梯度截断 | 降到 41% |

> Phase 3 继承了优化 #1+#2 的参数，并进一步调整 `lr_schedule_kl_threshold: 0.002`

### 奖励函数优化（2026-02-26，Phase 3 进行中）

> 详见 `blood-v2/docs/FAN_REWARD_ANALYSIS_2026_02_26.md`

| 组件 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| tsumo_bonus | 固定 0.1 | `0.1 × intensity` | 低番自摸信号过强 |
| deal_in_penalty | 固定 0.05 | `0.05 × intensity` | 低番放铳惩罚过强 |
| rank_bonus | 固定 0.2 | `0.2 × intensity` | 低番局排名信号占比过高 |
| safe_discard | 0.015 | 0.01 | 累积过度防守，加速 entropy 下降 |

> `intensity = clamp(sqrt(|score_delta| / 32000), 0.25, 1.0)` — 1番信号衰减到 ~31%，6番保持 100%

### Bug 修复记录

| commit | 描述 |
|--------|------|
| — | manage.sh `find_best_checkpoint()` 路径错误 |
| `21c0a1d8` | `--init_checkpoint_path` 未在 argparser 注册 |
| `51dcf87a` | PyTorch 2.6 `weights_only=True` 导致 SF2 checkpoint 加载失败 |
| `cd66d570` | 新增 `manage.sh stop` 命令 |
| `749e58b6` | 跨阶段 `strict=False` 加载 |
| `00bf21bb` | warmup.yaml `use_rnn: false` → `true` |
| `bf15894a` | 保留 optimizer 状态 |
| `ea97f796` | warmup/transition `lr_adaptive_max: 3e-4` + optimizer LR 重置 |
| `4bfc7472` | Arena 评估集成到训练循环 |

### 监控阈值参考

| 指标 | 健康范围 | 警报阈值 |
|------|---------|----------|
| `blood/elo_current` | > 1500 | < 1450 连续 3 次 |
| `train/entropy` | 0.40~0.65 | < 0.35 |
| `train/actual_lr` | 均值 > 7e-5 | 持续锁死 > 1M 步 |
| `train/value_loss` | < 1.2 | > 1.5 持续 |
| `train/grad_norm` clip@3.0 | < 30% | > 50% |
| `train/kl_divergence` | 0.001~0.005 | > 0.01 持续 |
| `blood/arena_win_rate` | > 0.50 | < 0.40 连续 3 次 |
