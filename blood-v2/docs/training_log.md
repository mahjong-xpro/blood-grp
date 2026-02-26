# Blood V2 训练日志

> 五阶段: warmup → warmup_transition → competitive → competitive_distill → elite

---

## 当前状态（2026-02-27 02:00 CST）

```
Run 2（架构优化后重训）
Phase 1 (2M) ✅ → Phase 1.5 (524K/500K) 🔄 → Phase 2a → Phase 2b → Phase 3
总训练量: ~2.5M env steps
```

| 指标 | 当前值 | Run 1 同期 | 评估 |
|------|--------|-----------|------|
| reward | **-0.16** (warmup 末) | -0.20 | ✅ 更好 |
| value_loss | **0.80** (warmup 末) | 0.50 | ⚠️ 偏高，新架构收敛较慢 |
| entropy | **0.54** (warmup 末) | 0.45 | ✅ 下降更慢，探索更充分 |
| grad_norm | **0.87** 均值 | 1.000 全程 | ✅ 不再全程贴满 |
| actual_lr | **3e-4** 稳定 | 3e-4 | ✅ |

**架构变更**: `enc_proj_layers: 3`（SpatialPoolingProj 注意力池化），缓解 6912→1024 信息瓶颈。

**结论**: ✅ 新架构训练健康。entropy 下降速率明显变慢（0.54 vs 0.45），直接缓解了 Run 1 中策略过早收敛的核心问题。

**需关注**: warmup_transition 末尾 value_loss 尖峰 6.41（Run 1 同位置 3.75），新架构对 gamma 变化更敏感。预期快速恢复。

---

## Run 2: 架构优化后重训（进行中 🔄）

> 开始: 2026-02-27 | 架构变更: `enc_proj_layers: 3`（SpatialPoolingProj）
> 目的: 缓解 enc_proj 6912→1024 信息瓶颈，改善策略收敛速度

### Phase 1: Warmup（已完成 ✅）

> 配置: `configs/warmup.yaml` | 实际: ~2.01M steps
> 核心: `opponent_mode: rulebot`, `lr: 5e-4`, `max_grad_norm: 1.0`

| 指标 | 起始 | 最终 | Run 1 最终 | 评估 |
|------|------|------|-----------|------|
| `reward` | -0.11 | **-0.16** | -0.20 | ✅ 更好 |
| `value_loss` | 3.05 (尖峰) | **0.80** | 0.50 | ⚠️ 偏高 |
| `entropy` | 1.00 → **0.54** | — | 0.45 | ✅ 下降更慢 |
| `grad_norm` | 1.00 → **0.87** | — | 1.000 | ✅ 改善 |
| `actual_lr` | 3.3e-4 → **3e-4** | — | 3e-4 | ✅ |
| `kl_divergence` | 0.011 → **0.003** | — | — | ✅ |
| `fraction_clipped` | 21% → **4%** | — | — | ✅ |
| `league_pool_size` | 0 → **15** | — | 15 | ✅ |

分段趋势:

| 指标 | Q1 | Q2 | Q3 | Q4 | 评估 |
|------|-----|-----|-----|-----|------|
| value_loss | 0.832 | 0.812 | 0.819 | **0.796** | ✅ 缓慢下降 |
| entropy | 1.014 | 0.773 | 0.574 | **0.564** | ✅ Q3~Q4 企稳 |
| KL | 0.005 | 0.003 | 0.003 | 0.003 | ✅ 稳定 |
| clip% | 8.5% | 4.3% | 3.9% | 4.3% | ✅ 稳定 |

**分析**: 新架构 warmup 表现优于 Run 1。entropy 在 Q3~Q4 企稳在 0.56~0.57，而 Run 1 持续下降到 0.45。value_loss 偏高（0.80 vs 0.50）是因为 SpatialPoolingProj 增加了信息通路，Critic 需要更多时间学习利用额外信息。grad_norm 不再全程贴满 1.0，说明梯度空间更充裕。

---

### Phase 1.5: Warmup Transition（进行中 🔄）

> 配置: `configs/warmup_transition.yaml` | 当前: ~524K steps
> 核心变化: `gamma: 0.995`, `lr: 2e-4`, LSTM 启用

| 指标 | 起始 | 当前 (@524K) | Run 1 最终 | 评估 |
|------|------|-------------|-----------|------|
| `reward` | -0.24 | **+0.18** (末点) | -0.15 | ⚠️ 末点跳变 |
| `value_loss` | 0.69 | **6.41** (尖峰) | 0.82 | ⚠️ 尖峰，见下 |
| `entropy` | 1.02 | **0.59** | 0.51 | ✅ 更高 |
| `grad_norm` | 1.00 | **1.00** | 0.97 | ✅ |
| `actual_lr` | 8.9e-5 | **3e-4** | 3e-4 | ✅ 快速恢复 |
| `kl_divergence` | 0.025 | **0.003** | — | ✅ 冲击后恢复 |
| `fraction_clipped` | 37% | **4%** | — | ✅ 冲击后恢复 |

**value_loss 尖峰 6.41**: gamma 从 0.99→0.995 改变 GAE returns 尺度。Run 1 同位置尖峰 3.75，新架构更大（6.41）是因为更宽的信息通路使 Critic 对尺度变化更敏感。Run 1 从 3.75 快速恢复到 0.82，预期本次也会快速恢复。

**KL/clip 初始冲击**: KL 从 0.025 降到 0.003，fraction_clipped 从 37% 降到 4%，均在 ~200K 步内恢复正常。这是阶段切换的正常现象。

---

## Run 1: 原始架构训练（已归档 📦）

> 训练时间: 2026-02-25 19:19 ~ 2026-02-27 | 架构: `enc_proj_layers: 1`（单层 Linear 6912→1024）
> 总训练量: ~13.7M env steps | 最终阶段: Phase 3 Elite @ 5.73M/50M
> 终止原因: enc_proj 信息瓶颈 + 策略过度收敛，决定架构优化后重训

#### 最终状态

| 指标 | 最终值 | 峰值 | 评估 |
|------|--------|------|------|
| Elo | 1465 | 1537 (@1M) | ⚠️ 回落 |
| Arena win_rate | 0.49 | 0.65 (@4M) | ⚠️ 回落 |
| Arena avg_rank | 2.62 | 2.35 (@4M) | ⚠️ 回落 |
| value_loss | 0.878 | — | ✅ 持续下降 |
| entropy | 0.48 | 0.70 (@0) | ⚠️ 过度收敛 |

#### 关键教训

1. **enc_proj 瓶颈**: 6912→1024 单层压缩丢失过多空间信息，限制了策略精度
2. **entropy 过快下降**: 0.70→0.48 仅 5.73M 步，策略过早收敛导致探索不足
3. **Critic-Actor 分离**: value_loss 持续改善但 Elo 下降，Critic 学到了但 Actor 用不上
4. **自博弈池过拟合**: 对自己历史版本过度适应，对 RuleBot 泛化下降
5. **KL adaptive LR 振荡**: 尖峰频繁导致 LR 34.8% 时间锁在 5e-5 下限

#### Phase 3 Elite Arena 评估（完整记录）

| # | Step | Elo | Win Rate | Avg Rank | Avg Score |
|---|------|-----|----------|----------|-----------|
| 1 | 524K | 1483 | 0.49 | 2.63 | 98880 |
| 2 | 1032K | 1537 | 0.55 | 2.48 | 99610 |
| 3 | 1540K | 1473 | 0.48 | 2.52 | 100030 |
| 4 | ~2M | 1474 | 0.52 | 2.46 | 101430 |
| 5 | 2523K | 1506 | 0.41 | 2.55 | 99470 |
| 6 | 3031K | 1480 | 0.49 | 2.59 | 100190 |
| 7 | 3539K | 1523 | 0.50 | 2.39 | 101390 |
| 8 | 4047K | 1505 | 0.65 | 2.35 | 101360 |
| 9 | 4555K | 1510 | 0.47 | 2.47 | 100560 |
| 10 | 5063K | 1520 | 0.50 | 2.54 | 99710 |
| 11 | 5571K | 1465 | 0.49 | 2.62 | 99600 |

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
| 3 | 02-27 | 全局 | `enc_proj_layers` | 1 | 3 | enc_proj 信息瓶颈 | Run 2 重训 |

### 奖励函数优化（2026-02-26）

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
| `dc211495` | warmup_dq_bonus 死代码修复 + get_agent_suit_counts() API |

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
