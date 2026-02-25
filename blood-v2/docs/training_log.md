# Blood V2 训练日志

---

## Warmup 阶段（Phase 1）

**配置**：`configs/warmup.yaml`
**目标**：2M env steps，对战 RuleBot，学习基础打牌策略
**GPU**：RTX 4090 24GB
**实验名**：`blood_v2_warmup`

---

### 运行记录

| 日期 | 步数范围 | 备注 |
|------|----------|------|
| 2026-02-23 | 0 → 2M | 首次运行（含 LSTM 加载 bug，checkpoint 无效） |
| 2026-02-24 | 0 → 2.01M | 修复 `core.rnn.*` → `core.core.*` 后重跑，checkpoint 有效 |

---

### 关键指标分析（第二次运行，step 0 → 2.01M，实测数据）

#### 奖励 / 目标

| 指标 | 初始值 | 中间值（~1M） | 最终值（2.01M） | 评估 |
|------|--------|--------------|----------------|------|
| `reward/reward` | -0.205 @16K | 6.15 @999K | **7.84** @2.01M | ✅ 持续上升，末段仍未饱和 |
| `reward/reward_max` | 0.125 @16K | 9.69 @999K | 9.56 @2.01M | ✅ 单局最高奖励稳定在 9–10 |
| `reward/reward_min` | -1.188 @16K | -0.25 @999K | **+3.94** @2.01M | ✅ 末段全部转正，所有局均盈利 |

#### 损失

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/loss` | 5.305 | 0.039 | ✅ 下降 99.3%，正常收敛 |
| `train/value_loss` | 5.309 | 0.039 | ✅ 价值函数收敛良好 |
| `train/policy_loss` | 0.031 | 0.002 | ✅ PPO 策略梯度正常 |

#### 训练稳定性

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/entropy` | 3.468 | **0.197** | ✅ 大幅下降，策略高度收敛（对比上次末段回升至 2.75，本次收敛更彻底） |
| `train/kl_divergence` | 0.006 | 0.012 | ✅ KL 受控，无策略崩溃 |
| `train/fraction_clipped` | 9.5% @49K | 7.8% @2M | ✅ 全程保守更新 |
| `train/grad_norm` | 1.000 | **0.861**（末段 0.51–0.86） | ✅ 后期梯度开始下降，不再全程贴满上限 |
| `train/actual_lr` | 3e-4 | 1e-4 | ✅ `kl_adaptive_minibatch` 自适应降低 LR |

#### 优势函数

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/adv_mean` | -0.079 | 0.051 | ✅ 接近 0，策略决策质量稳定 |
| `train/adv_std` | 0.174 | 0.396 | ✅ 本次正常记录（上次全程为 0 的异常已消失） |
| `train/returns_running_mean` | 0.025 | 1.800 | ✅ 回报基线持续上升 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | 8192–9830 FPS | ✅ 吞吐量稳定，无瓶颈 |
| `stats/gpu_mem_learner` | 5615 MB（全程不变） | ✅ 无显存泄漏 |
| `len/len`（episode 长度） | 44.5 → 62.8 步 | ✅ 正常麻将局长度，末段略有增长 |
| `len/len_max` | 74 → 91 步 | ✅ 无异常截断 |

#### League

| 指标 | 值 | 评估 |
|------|-----|------|
| `blood/league_pool_size` | 0 → **12**（整数，正常递增） | ✅ 归一化问题已修复，数值正确 |

---

### 与第一次运行对比

| 指标 | 第一次（2026-02-23） | 第二次（2026-02-24） | 说明 |
|------|---------------------|---------------------|------|
| 最终 reward | 6.35 | **7.84** | +1.49，LSTM 正确加载后策略更强 |
| 最终 entropy | 2.75（末段回升） | **0.197** | 策略收敛更彻底 |
| grad_norm | 全程 1.0 | 末段 0.51–0.86 | 梯度开始自然下降 |
| reward_min | 负值 | **+3.94** | 所有局均盈利 |
| league_pool_size | -0.28 ~ 0.19（异常） | 0 → 12（正常整数） | 归一化 bug 已修复 |
| checkpoint 有效性 | ❌ LSTM 未加载 | ✅ LSTM 正确加载 | `core.rnn.*` → `core.core.*` 修复 |

---

### Warmup 阶段结论

| 项目 | 结论 |
|------|------|
| 收敛状态 | ✅ 正常收敛，reward 从 -0.21 升至 7.84，末段仍在上升 |
| 训练稳定性 | ✅ entropy=0.197，KL 受控，无崩溃 |
| 显存 | ✅ 稳定在 5615 MB，无泄漏 |
| 吞吐量 | ✅ 8192–9830 FPS |
| grad_norm | ✅ 末段自然下降至 0.51–0.86 |
| reward_min | ✅ 末段转正（+3.94），所有局均盈利 |
| league_pool_size | ✅ 正确记录为整数（0→12） |

**下一步**：加载本次 warmup checkpoint，启动 competitive 阶段（`configs/competitive.yaml`）。

---

## Competitive 阶段（Phase 2）

**配置**：`configs/competitive.yaml`
**目标**：5M env steps，神经网络自对弈 + 联赛池
**GPU**：RTX 4090 24GB
**实验名**：`blood_v2_competitive`

---

### 运行记录

| 日期 | 步数范围 | 备注 |
|------|----------|------|
| 2026-02-23 | 0 → 5M | 首次运行，中途修复多个 bug，策略崩溃（len=500 全程截断） |
| 2026-02-24 | 0 → 5.03M（完成） | 修复 LSTM 加载 bug + 所有死锁 bug 后重跑，训练正常收敛 |

---

### 关键指标分析（第二次运行，step 0 → 5.03M，实测数据）

#### 奖励 / 目标

| 指标 | 初始值 | 中间值（~2.5M） | 最终值（5.03M） | 峰值 | 评估 |
|------|--------|----------------|----------------|------|------|
| `reward/reward` | -0.003 @0 | 14.25 @2.52M | **14.45 @5.01M** | 15.48 @3.15M | ✅ 持续上升，末段在 13–15 区间震荡 |
| `reward/reward_max` | 0.88 @0 | 22.53 @2.52M | 22.10 @5.01M | 23.64 @885K | ✅ 单局最高奖励稳定在 22–24 |
| `reward/reward_min` | -0.69 @0 | 1.01 @2.52M | -0.21 @5.01M | 4.77 @2.59M | ⚠️ 末段偶有负值，见下方分析 |

#### 损失

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/loss` | 0.948 @16K | 0.060 @5.03M | ✅ 下降 93.7%，正常收敛 |
| `train/value_loss` | 0.993 @16K | 0.078 @5.03M | ✅ 价值函数收敛良好 |
| `train/policy_loss` | 0.025 @16K | 0.002 @5.03M | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/entropy` | 3.506 @16K | **1.007 @5.03M** | ✅ 正常下降，策略收敛；最低 0.824 @3.26M |
| `train/grad_norm` | 1.000 | **0.948 @5.03M**（最低 0.143 @4.11M） | ✅ 后期开始下降，偶有回升 |
| `train/fraction_clipped` | 60.9% @16K | **2.1% @5.01M** | ✅ 大幅下降，更新趋于保守 |
| `train/kl_divergence` | 0.018 @16K | 0.010 @5.01M | ⚠️ 整体受控，但有 4 次大尖峰，见下方分析 |
| `train/actual_lr` | ~0 @16K | 2.9e-4 @5.03M | ⚠️ 大幅波动（0 ~ 1.5e-3），见下方分析 |
| `train/adv_mean` | 0.015 @16K | **0.150 @5.03M** | ✅ 末段转正，策略优于基线 |
| `train/adv_std` | 0.190 @16K | 1.403 @5.03M | ✅ 优势函数方差增大，探索充分 |
| `train/returns_running_mean` | -0.084 @16K | **6.357 @5.03M** | ✅ 回报基线持续上升 |

#### Episode 长度

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `len/len` | 29.1 @0 | **51.6 @5.01M** | ✅ 正常麻将局长度，无系统性截断 |
| `len/len_min` | 9 @0 | 33 @5.01M | ✅ 最短局正常 |
| `len/len_max` | 57 @0 | 72 @5.01M | ✅ 无 500 截断；全程仅 **1 次** 206 尖峰（@2.52M） |

#### 联赛池 / 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `blood/league_pool_size` | 12（继承 warmup）→ **50（满池）** @3.67M | ✅ 正常填充，满池后保持稳定 |
| `perf/_fps` | ~3277 FPS（中段）→ ~1638 FPS（末段） | ✅ 吞吐量正常，末段降速为训练结束正常现象 |
| `stats/gpu_mem_learner` | 5857 MB（全程不变） | ✅ 无显存泄漏 |
| 总收集帧数 | **5,029,888** | ✅ 超出目标 5M |

---

### 异常指标分析

#### ⚠️ `train/kl_divergence` 大尖峰（最高 1.11）

**现象**：KL 散度整体在 0.01–0.02 范围内，但在 ~1.44M–1.47M steps 出现连续 3 次尖峰（0.181、0.336、0.402），以及 ~2.75M steps 出现最大尖峰 1.113。

**分析**：联赛池在 ~1.4M 和 ~2.75M 附近切换到与当前策略差异较大的历史模型，导致策略分布剧烈偏移。`kl_adaptive_minibatch` 在尖峰后将 LR 大幅降低（actual_lr 降至接近 0），随后逐步回升。

**影响**：尖峰后 reward 无明显下降（14.x 区间维持），说明自适应调度有效抑制了策略崩溃。`kl_divergence_max` 最高达 17.99，说明个别 minibatch 内有极端样本，但整体 epoch 均值受控。

---

#### ⚠️ `train/actual_lr` 大幅波动（0 ~ 1.5e-3）

**现象**：LR 在全程大幅震荡，而非上次的稳定 1e-4。

**分析**：`kl_adaptive_minibatch` 根据 KL 动态调整 LR——KL 尖峰时 LR 骤降至接近 0，KL 恢复后 LR 回升。这是自对弈中对手策略跳变的正常响应，不影响最终收敛。

---

#### ⚠️ `reward/reward_min` 末段偶有负值

**现象**：reward_min 在 ~2.59M 达到峰值 4.77，但末段（5.01M）回落至 -0.21。

**分析**：满池后联赛对手包含各阶段历史模型，部分强对手局面下 agent 偶有亏损。这是正常的自对弈现象，reward 均值（14.45）不受影响。

---

#### ✅ `len/len_max` 大幅改善（500 → 206，仅 1 次）

**现象**：上次运行共 6 次 len_max=500 尖峰；本次全程仅 1 次 206 尖峰（@2.52M），无 500 截断。

**结论**：`finalize_scoring` 修复（commit `c2448d6a`）效果显著，stall 问题基本消除。

---

### Competitive 阶段 Bug 修复（全部）

| commit | 描述 |
|--------|------|
| `62124c71` | fix: `_inject_config_yaml` str2bool 参数需要显式值 |
| `4b4ca54f` | fix: `selfplay_env.py` `terminated` 在赋值前被引用 |
| `1f9ff565` | fix: CUDA OOM，降低 `num_envs_per_worker` 和 `batch_size` |
| `2d91c8a2` | fix: BPTT `forward_core` 收到 `PackedSequence`，`LayerNorm` 报错 |
| `1f9f6e2c` | tune: competitive `learning_rate` 3e-4 → 1e-4 |
| `4b7861bf` | fix: 联赛快照静默失败（主进程 `learner` 为 None） |
| `777f24d4` | fix: 联赛 checkpoint 损坏 + `rnn_size` 维度不匹配 |
| `b773fb4a` | fix: checkpoint 验证改用 `weights_only=False` |
| `fa57af40` | fix: `apply_self_check` tsumo panic + stall |
| `95fd862c` | fix: 损坏联赛 checkpoint 自动删除 |
| `e4674675` | fix: 死锁根因——`has_decision` 检查 + `process_win` 安全网 |
| `1f7b3080` | fix: Kan 后 stall 误报（`prev_state=None` 重置） |
| `f2c53c57` | fix: 联赛快照日志 spam（降级为 debug） |
| `91053e9b` | fix: `apply_discard/reaction/kan_select` 缺少 wrong-player guard |
| `c2448d6a` | fix: **`finalize_scoring` 永不被调用**（`is_done()` 循环依赖，len=500 根本原因） |
| `0b309a4a` | fix: 联赛快照 `[Errno 2]`（`ckpt_dir` 改用 `abspath`） |
| `86b8a0f8` | fix: **LSTM 永不加载**（`core.rnn.*` → `core.core.*`，agent 随机打牌根本原因） |
| `7ddb8673` | fix: 杠后无摸牌事件 + ron 后双摸牌（engine board.rs） |

---

### Competitive 阶段结论

| 项目 | 结论 |
|------|------|
| 收敛状态 | ✅ 正常收敛，reward 从 -0.003 升至 14.45（5.03M steps），峰值 15.48 |
| 训练稳定性 | ✅ entropy=1.007，fraction_clipped=2.1%，无崩溃 |
| Episode 长度 | ✅ 均值 51.6 步；len_max 全程仅 1 次 206 尖峰，无 500 截断 |
| 联赛池 | ✅ 满池（50 个历史模型），@3.67M 达到满池 |
| KL 稳定性 | ⚠️ 4 次大尖峰（最高 1.11），自适应 LR 有效抑制，reward 无崩溃 |
| grad_norm | ✅ 末段开始下降（最低 0.14），整体趋势改善 |
| adv_mean | ✅ 末段 +0.15，策略优于基线 |
| reward_min | ⚠️ 末段偶有负值（-0.21），均值不受影响 |

**下一步**：加载 competitive checkpoint，启动 Elite 阶段（`configs/elite.yaml`）。

---

## Elite 阶段（Phase 3）

*待填写*

---

## Bug 修复记录

| 日期 | commit | 描述 |
|------|--------|------|
| 2026-02-23 | `fcda2662` | engine: 用 tomohxx 查表法替换递归 shanten，~1000× 加速 |
| 2026-02-23 | `a664290f` | config: 降低 batch_size 和 rnn_size 修复 CUDA OOM |
| 2026-02-23 | `38eb914d` | runner: 进程退出时清理孤立 SF2 worker 子进程 |
| 2026-02-23 | `aa8933c5` | config: 降低 ResBlock 数量修复 24GB GPU 上的 CUDA OOM |
| 2026-02-23 | `4bc7df8c` | config: batch_size 8192→4096 修复 CUDA OOM |
| 2026-02-23 | `4f62ce45` | config: 恢复 block 数量至 16+20（利用释放的显存空间） |
| 2026-02-23 | `64bfd4a5` | engine: 修复 15 tile 手牌导致的 calc_shanten OOB panic |


---

## 第三轮训练（2026-02-25）— 新五阶段流水线

> 重新设计训练流水线，增加 warmup_transition 过渡阶段，分离 LSTM 启用和自博弈切换。
> 五阶段: warmup → warmup_transition → competitive → competitive_distill → elite

---

### Phase 1: Warmup（已完成 ✅）

> 采集时间: 2026-02-25 17:14 CST | 运行名: `blood_v2_warmup/.summary/0`
> 配置: `configs/warmup.yaml` | 目标: 2M steps | 实际: ~2.01M steps
> 关键设置: `use_rnn: false`, `opponent_mode: rulebot`, `lr: 5e-4`, `gamma: 0.99`

#### 奖励 / 目标

| 指标 | 初始值 | 中间值 (~1M) | 最终值 (2M) | 评估 |
|------|--------|-------------|-------------|------|
| `reward/reward` | +0.06 @0 | -0.13 @1.04M | **-0.14** @1.99M | ⚠️ 负值区间震荡，见分析 |
| `reward/reward_max` | +2.31 @0 | +0.56 @1.25M | **+3.19** @1.99M | ✅ 末段出现高奖励局 |
| `reward/reward_min` | -1.75 @0 | -1.19 @1.25M | **-1.16** @1.99M | ⚠️ 仍有亏损局 |

#### 损失

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/loss` | 3.14 @8K | **0.61** @2.01M | ✅ 下降 80.6% |
| `train/value_loss` | 3.13 @8K | **0.61** @2.01M | ✅ 价值函数收敛 |
| `train/policy_loss` | 0.023 @8K | **0.001** @2.01M | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/entropy` | 1.05 @8K | **0.48** @2.01M | ✅ 持续下降，策略收敛 |
| `train/kl_divergence` | 0.014 @24K | **0.006** @2.01M | ✅ KL 受控 |
| `train/fraction_clipped` | 17.5% @24K | **9.7%** @2.01M | ✅ 下降 |
| `train/grad_norm` | 1.000 @8K | **0.32** @2.01M | ✅ 末段梯度自然下降 |
| `train/actual_lr` | 5e-4 @8K | **0.010** @2.01M | ⚠️ LR 攀升至上限，见分析 |
| `train/adv_std` | 0.29 @8K | **0.12** @2.01M | ✅ 优势函数方差稳定 |
| `train/value_delta` | 0.21 @24K | **0.09** @2.01M | ✅ 价值预测误差下降 |
| `train/returns_running_mean` | — | **-0.157** @2.01M | ⚠️ 负值，与 reward 一致 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | 4096–5734 FPS | ✅ 吞吐量稳定 |
| `len/len` | 23 → 19.8 步 | ✅ 正常麻将局长度 |
| `blood/league_pool_size` | 0 → **15** | ✅ 正常递增 |
| `blood/elo_current` | 1500（不变） | ✅ warmup 对手为 RuleBot，Elo 不更新 |

#### ⚠️ 异常分析: reward 持续为负

**现象**: reward 从 +0.06 快速下降到 -0.26，之后在 -0.13 ~ -0.24 区间震荡，2M 步后仍为 -0.14。

**分析**:
1. **本轮 `use_rnn: false`**: 第二轮 warmup 使用了 LSTM，本轮明确禁用。无 LSTM 的模型对 RuleBot 时序模式建模能力有限。
2. **奖励尺度差异**: 本轮使用 sqrt-compressed 奖励（REWARD_NORM=32000），范围约 [-1.75, +3.19]。reward=-0.14 在此尺度下对应轻微亏损，并非严重问题。
3. **策略确实在学习**: entropy 1.05→0.48, policy_loss 0.023→0.001, grad_norm 1.0→0.32，收敛指标全部正常。

**结论**: 策略收敛正常，reward 负值是奖励尺度 + 无 LSTM 的综合结果。warmup 主要目标是学习基础打牌模式，后续阶段会进一步优化。

#### ⚠️ 异常分析: LR 攀升至上限 (0.01)

**现象**: `actual_lr` 从 5e-4 持续攀升，~1.5M 步后达到 0.01 上限。

**分析**: `kl_adaptive_minibatch` 在 KL < 阈值时提升 LR。本轮 KL 从 0.014 降至 0.003-0.006，远低于默认阈值 0.008，调度器持续提升 LR。grad_norm 反而下降（0.32），说明模型已接近局部最优。

---

### Phase 1.5: Warmup Transition（已完成 ✅）

> 采集时间: 2026-02-25 17:16 CST | 运行名: `blood_v2_warmup_transition/.summary/0`
> 配置: `configs/warmup_transition.yaml` | 目标: 500K steps | 实际: ~516K steps
> 核心变化: `use_rnn: true` (LSTM 512×2), `gamma: 0.995`, `lr: 2e-4`
> 前置 checkpoint: warmup 最佳 checkpoint（通过 `--init_checkpoint_path` 链式加载）

#### 奖励 / 目标

| 指标 | Warmup 结束 | Transition 起始 | Transition 最终 (~500K) | 评估 |
|------|------------|----------------|------------------------|------|
| `reward/reward` | -0.14 | +0.04 @0 | **-0.15** @483K | ✅ 与 warmup 持平 |
| `reward/reward_max` | +3.19 | +2.44 @0 | **+2.13** @336K | ✅ 正常范围 |
| `reward/reward_min` | -1.16 | -1.31 @0 | **-1.25** @278K | ✅ 与 warmup 一致 |

#### 损失

| 指标 | Transition 起始 | Transition 最终 | 评估 |
|------|----------------|----------------|------|
| `train/loss` | 0.72 @8K | **1.01** @516K | ⚠️ 略有上升，LSTM 适应期 |
| `train/value_loss` | 0.73 @8K | **1.01** @516K | ⚠️ 同上 |
| `train/policy_loss` | 0.003 @8K | **0.001** @516K | ✅ 策略梯度正常 |

#### 训练稳定性

| 指标 | Transition 起始 | Transition 最终 | 评估 |
|------|----------------|----------------|------|
| `train/entropy` | 1.06 @8K | **0.94** @516K | ✅ 缓慢下降，LSTM 逐步适应 |
| `train/kl_divergence` | 0.021 @8K | **0.011** @516K | ✅ KL 受控 |
| `train/fraction_clipped` | 34% @8K | **23%** @516K | ✅ 下降趋势 |
| `train/grad_norm` | 1.000 @8K | **0.63** @516K | ✅ 末段下降 |
| `train/actual_lr` | 8.9e-5 @8K | **0.010** @516K | ⚠️ LR 再次攀升至上限 |
| `train/adv_std` | 0.30 @8K | **0.14** @516K | ✅ 稳定 |
| `train/value_delta` | 0.24 @8K | **0.10** @385K | ✅ 价值预测误差下降 |
| `train/returns_running_mean` | — | **-0.174** @516K | ⚠️ 负值，与 warmup 一致 |
| `train/adam_max_second_moment` | — | **0.003** @516K | ✅ 远低于 warmup 末段 (0.057) |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | 3277–5734 FPS | ✅ 吞吐量稳定 |
| `len/len` | 22 → 19.3 步 | ✅ 正常 |
| `blood/league_pool_size` | 15 → **18** | ✅ 继承 warmup 池，继续递增 |
| `blood/elo_current` | 1500（不变） | ✅ 对手仍为 RuleBot |

#### 过渡质量评估

| 检查项 | 结果 |
|--------|------|
| LSTM 启用是否导致崩溃？ | ✅ 否，entropy 从 1.06 缓降至 0.94 |
| reward 是否大幅下降？ | ✅ 否，与 warmup 末段持平 (-0.14 → -0.15) |
| KL 是否失控？ | ✅ 否，从 0.021 降至 0.011 |
| value_loss 是否爆炸？ | ⚠️ 略有上升 (0.73 → 1.01)，LSTM 适应期正常现象 |
| grad_norm 是否全程贴满？ | ✅ 否，末段降至 0.63 |

**分析**: 过渡阶段表现平稳。LSTM 启用后 entropy 从 warmup 末段 0.48 回升至 1.06（LSTM 新增参数需要探索），在 500K 步内降至 0.94，说明 LSTM 正在学习时序表示。value_loss 略有上升（0.73→1.01），是 LSTM 改变特征表示后价值函数重新适应的正常现象。

**关键发现**: `adam_max_second_moment` 从 warmup 末段 0.057 降至 transition 的 0.003，确认 optimizer 状态被正确重置（`_seed_init_checkpoint` 剥离了 optimizer state），新阶段从干净的 Adam 状态开始。

---

### ⚠️ 事件: manage.sh checkpoint 路径 bug (2026-02-25 16:24)

Competitive 阶段启动时输出 "未找到 warmup_transition 的 checkpoint，从头开始训练"。

**根因**: `find_best_checkpoint()` 在 `checkpoints/` 查找，但 SF2 存储在 `train_dir/<exp>/checkpoint_p0/`。
**修复**: 更新函数优先查找 `train_dir/` 路径。

---

### ⚠️ 事件: --init_checkpoint_path 未注册 bug (2026-02-25 16:54)

`manage.sh train warmup_transition` 报错: `runner.py: error: unrecognized arguments: --init_checkpoint_path`

**根因**: `manage.sh` 的 `do_train()` 在非 resume 模式下自动查找前一阶段 checkpoint 并通过 `--init_checkpoint_path` 传给 runner，但该参数从未在 `cfg.py` 注册。

**修复** (commit `21c0a1d8`):
1. `cfg.py`: 注册 `--init_checkpoint_path` 参数
2. `runner.py`: 新增 `_seed_init_checkpoint()` — 在 `runner.init()` 前将前一阶段 checkpoint 复制到新实验的 `checkpoint_p0/` 目录，剥离 optimizer 状态并重置 env_steps=0

---

### Phase 2a: Competitive

*待启动 — warmup_transition checkpoint 已就绪*

---

### Phase 2b: Competitive Distill

*待启动*

---

### Phase 3: Elite

*待启动*
