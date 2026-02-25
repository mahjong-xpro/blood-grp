# Blood V2 训练日志

---

## 第四轮训练（2026-02-25 18:40 CST）— 修复后重跑

> 修复 warmup `use_rnn: false` 导致的跨阶段架构不匹配，改为 `use_rnn: true`。
> 五阶段: warmup → warmup_transition → competitive → competitive_distill → elite

---

### Phase 1: Warmup（已完成 ✅）

> 采集时间: 2026-02-25 18:40 CST | 运行名: `blood_v2_warmup/.summary/0`
> 配置: `configs/warmup.yaml` | 目标: 2M steps | 实际: ~2.01M steps
> 关键设置: `use_rnn: true` (LSTM 512×2), `opponent_mode: rulebot`, `lr: 5e-4`, `gamma: 0.99`

#### 奖励 / 目标

| 指标 | 初始值 | 中间值 (~1M) | 最终值 (2M) | 评估 |
|------|--------|-------------|-------------|------|
| `reward/reward` | -0.10 @0 | -0.22 @1.0M | **-0.21** @2.01M | ⚠️ 负值震荡，见分析 |
| `reward/reward_max` | — | — | — | — |
| `reward/reward_min` | — | — | — | — |

#### 损失

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/loss` | 2.85 @8K | **1.01** @2.0M | ✅ 下降 64.6% |
| `train/value_loss` | 2.84 @8K | **1.01** @2.0M | ✅ 价值函数收敛 |
| `train/policy_loss` | 0.017 @8K | **0.001** @2.0M | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/entropy` | 1.07 @8K | **0.66** @2.0M | ✅ 持续下降，策略收敛 |
| `train/kl_divergence` | 0.007 @41K | **0.002** @2.0M | ✅ KL 极低，非常稳定 |
| `train/fraction_clipped` | 15.1% @41K | **2.0%** @2.0M | ✅ 大幅下降 |
| `train/grad_norm` | 1.000 @8K | **0.15** @2.0M | ✅ 末段梯度极低，模型充分收敛 |
| `train/actual_lr` | 3.3e-4 @8K | **0.010** @2.0M | ⚠️ LR 攀升至上限 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | ~4915 FPS | ✅ 吞吐量稳定 |
| `len/len` | 21.7 → 19.4 步 | ✅ 正常麻将局长度 |
| `blood/league_pool_size` | 0 → **15** | ✅ 正常递增 |

#### 分析

reward 在 sqrt-compressed 尺度下持续为负（-0.10 → -0.21），与上一轮（无 LSTM）表现一致。策略收敛指标全部正常：entropy 1.07→0.66, policy_loss 0.017→0.001, grad_norm 1.0→0.15（比上轮的 0.32 更低，说明 LSTM 模型收敛更彻底）。KL 极低（0.002），导致 `kl_adaptive_minibatch` 持续提升 LR 至 0.01 上限。

---

### Phase 1.5: Warmup Transition（已完成 ✅）

> 采集时间: 2026-02-25 18:43 CST | 运行名: `blood_v2_warmup_transition/.summary/0`
> 配置: `configs/warmup_transition.yaml` | 目标: 500K steps | 实际: ~500K steps
> 核心变化: `gamma: 0.995`, `lr: 2e-4`（LSTM 已在 warmup 启用，本阶段继续微调）
> 前置 checkpoint: warmup 最佳 checkpoint（通过 `--init_checkpoint_path` 链式加载，含 optimizer 状态）

#### 奖励 / 目标

| 指标 | Warmup 结束 | Transition 起始 | Transition 最终 (~500K) | 评估 |
|------|------------|----------------|------------------------|------|
| `reward/reward` | -0.21 | +0.19 @0 | **-0.16** @492K | ✅ 与 warmup 持平 |
| `reward/reward_max` | — | +3.88 @0 | **+0.88** @492K | ✅ 正常范围 |
| `reward/reward_min` | — | -0.75 @0 | **-0.88** @492K | ✅ 与 warmup 一致 |

#### 损失

| 指标 | Transition 起始 | Transition 最终 | 评估 |
|------|----------------|----------------|------|
| `train/loss` | **7.06** @8K | **0.90** @500K | ⚠️ 初始尖峰后快速恢复，见分析 |
| `train/value_loss` | **7.06** @8K | **0.90** @500K | ⚠️ 同上 |
| `train/policy_loss` | ~0 @8K | **0.003** @500K | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | Transition 起始 | Transition 最终 | 评估 |
|------|----------------|----------------|------|
| `train/entropy` | 0.45 @8K | **0.60** @500K | ✅ 略有回升后稳定 |
| `train/kl_divergence` | 0.002 @98K | **0.002** @500K | ✅ 极低，非常稳定 |
| `train/fraction_clipped` | 2.0% @98K | **3.9%** @500K | ✅ 极低 |
| `train/grad_norm` | 1.000 @8K | **0.05** @500K | ✅ 极低，模型充分收敛 |
| `train/actual_lr` | 0.010 @8K | **0.010** @500K | ⚠️ 全程保持上限（继承自 warmup） |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | ~4915–5734 FPS | ✅ 吞吐量稳定 |
| `len/len` | 18.7 → 18.9 步 | ✅ 正常 |
| `blood/league_pool_size` | 15 → **18** | ✅ 继承 warmup 池，继续递增 |

#### 过渡质量评估

| 检查项 | 结果 |
|--------|------|
| checkpoint 加载是否成功？ | ✅ 是，架构完全匹配（warmup 已启用 LSTM） |
| reward 是否大幅下降？ | ✅ 否，与 warmup 末段持平 (-0.21 → -0.16) |
| KL 是否失控？ | ✅ 否，全程 0.002 |
| value_loss 是否爆炸？ | ⚠️ 初始尖峰 7.06，但 32K 步内恢复到 0.92，见分析 |
| grad_norm 是否全程贴满？ | ✅ 否，快速降至 0.05 |

#### ⚠️ 异常分析: value_loss 初始尖峰 (7.06)

**现象**: transition 起始 value_loss 从 warmup 末段的 1.01 跳升至 7.06，然后在 ~32K 步内恢复到 0.92。

**根因**: warmup.yaml 和 warmup_transition.yaml 均未设置 `lr_adaptive_max`，导致 `kl_adaptive_minibatch` 在 KL 极低时将 LR 推到 SF2 默认上限 0.01。`_seed_init_checkpoint` 保留了 optimizer 状态（含 LR=0.01 和 Adam 动量），新阶段继承了这个过高的 LR。高 LR + gamma 变化导致价值函数短暂震荡。

**已修复**（见下方训练日志分析修复）:
1. warmup.yaml / warmup_transition.yaml 添加 `lr_adaptive_max: 3e-4`
2. `_seed_init_checkpoint` 新增 optimizer LR 重置逻辑

---

### Bug 修复记录（第四轮）

| 时间 | commit | 描述 |
|------|--------|------|
| 16:24 | — | manage.sh `find_best_checkpoint()` 路径错误（`checkpoints/` → `train_dir/`） |
| 16:54 | `21c0a1d8` | `--init_checkpoint_path` 未在 argparser 注册 |
| 17:25 | `51dcf87a` | PyTorch 2.6 `weights_only=True` 导致 SF2 checkpoint 加载失败 |
| 17:35 | `cd66d570` | 新增 `manage.sh stop` 命令 |
| 18:00 | `749e58b6` | 跨阶段 `strict=False` 加载（monkey-patch `Learner._load_state`） |
| 18:06 | `00bf21bb` | warmup.yaml `use_rnn: false` → `true`（根本修复架构不匹配） |
| 18:22 | `bf15894a` | 保留 optimizer 状态（SF2 `_load_state` 要求 `checkpoint["optimizer"]`） |

---

### 训练日志分析修复（2026-02-25 18:53 CST）

基于 Phase 1 + Phase 1.5 日志分析，发现并修复以下问题:

| # | 严重度 | 问题 | 修复 |
|---|--------|------|------|
| 1 | 🔴 高 | warmup/transition 缺少 `lr_adaptive_max`，LR 被推到 0.01 上限，导致跨阶段 value_loss 尖峰 | `warmup.yaml` / `warmup_transition.yaml` 添加 `lr_adaptive_max: 3e-4` |
| 2 | 🔴 高 | `_seed_init_checkpoint` 继承 optimizer LR=0.01，新阶段首步 LR 过高 | `runner.py` 新增 optimizer param_groups LR 重置为 `cfg.learning_rate` |
| 3 | 🟡 中 | `_seed_init_checkpoint` docstring 说 "strip optimizer state" 但实际保留 | 修正 docstring 和日志消息 |
| 4 | 🟡 中 | `manage.sh` 注释说 warmup "无 LSTM"，与实际 `use_rnn: true` 矛盾 | 更新 manage.sh 所有 LSTM 相关注释 |
| 5 | 🟡 中 | `warmup_transition.yaml` 过渡策略注释描述旧方案（warmup 无 LSTM） | 更新注释反映当前配置 |

**影响评估**: 问题 1-2 会导致每次跨阶段过渡时出现 value_loss 尖峰（本轮 7.06），虽然 32K 步内可恢复，但在 competitive 阶段（自博弈 + 更敏感的策略）可能导致更严重的不稳定。建议在启动 Phase 2a 前重跑 warmup + transition（或至少从 transition 重跑）。

---

### Phase 2a: Competitive

*待启动 — warmup_transition checkpoint 已就绪，建议先用修复后的配置重跑 warmup+transition*

---

### Phase 2b: Competitive Distill

*待启动*

---

### Phase 3: Elite

*待启动*
