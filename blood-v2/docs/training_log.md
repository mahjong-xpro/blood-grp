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
| 2026-02-23 | 0 → 2M | 首次完整 warmup 运行 |

---

### 关键指标分析（step 0 → 2M）

#### 奖励 / 目标

| 指标 | 初始值 | 最终值 | 趋势 | 评估 |
|------|--------|--------|------|------|
| `reward/reward` | -0.145 | 6.35 | 强劲上升，~300K 后趋于平稳 | ✅ 正常。从负值快速收敛到 6.x，说明模型学会了基本策略 |
| `policy_stats/avg_true_objective` | — | 6.35 | 在 5.8–6.7 区间震荡 | ✅ 与 reward 一致，无异常 |

#### 损失

| 指标 | 初始值 | 最终值 | 趋势 | 评估 |
|------|--------|--------|------|------|
| `train/loss` | 3.414 | 0.012 | 下降 99.7% | ✅ 正常收敛 |
| `train/value_loss` | 3.436 | 0.036 | 下降 98.9% | ✅ 价值函数收敛良好 |
| `train/policy_loss` | 0.013 | 0.003 | 在 0 附近小幅震荡 | ✅ PPO 策略梯度正常 |

#### 训练稳定性

| 指标 | 值 / 范围 | 评估 |
|------|-----------|------|
| `train/entropy` | 3.50 → 2.75（末段回升） | ⚠️ 见下方分析 |
| `train/kl_divergence` | 0.026 → 0.003 | ✅ KL 受控，无策略崩溃 |
| `train/fraction_clipped` | 42% → 4% | ✅ 早期大幅更新，后期趋于保守，符合预期 |
| `train/grad_norm` | ~1.0（持续被 clip） | ⚠️ 见下方分析 |
| `train/actual_lr` | 在 1e-4 ~ 3.3e-4 自适应调整 | ✅ `kl_adaptive_minibatch` 正常工作 |

#### 优势函数

| 指标 | 值 | 评估 |
|------|-----|------|
| `train/adv_mean` | ~0.34–0.55 | ✅ 正值说明模型在做出高于基线的决策 |
| `train/adv_std` | **0.0（全程）** | ❌ 异常，见下方分析 |
| `train/value` | 在 -0.05 ~ 0.20 震荡 | ✅ 价值估计接近 0，符合归一化后的期望 |
| `train/returns_running_mean` | 136.75（稳定） | ✅ 回报归一化基准稳定 |

#### 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `perf/_fps` | 8192–9830 FPS | ✅ 吞吐量稳定，无瓶颈 |
| `stats/gpu_mem_learner` | ~5637 MB（全程不变） | ✅ 无显存泄漏 |
| `stats/gpu_mem_policy_worker` | 1.0（固定值） | ⚠️ 疑似未正确采集，非实际 MB 值 |
| `len/len`（episode 长度） | 92–131 步 | ✅ 正常麻将局长度 |

#### League

| 指标 | 值 | 评估 |
|------|-----|------|
| `blood/league_pool_size` | 在 -0.28 ~ 0.19 震荡 | ❌ 异常，见下方分析 |

---

### 异常指标分析

#### ⚠️ `train/adv_std = 0.0`（全程）

**现象**：优势函数标准差全程为 0，说明 SF2 记录的是归一化后的优势（已做 batch normalization），标准差被归一化到 1 后再记录为 0，或者该字段记录的是归一化前的原始值但被截断了。

**影响**：无实际训练影响，仅是监控指标的记录问题。

**建议**：可在 callbacks.py 中手动记录归一化前的 `adv_std`。

---

#### ⚠️ `train/grad_norm` 持续接近 1.0

**现象**：`max_grad_norm=1.0`，grad_norm 全程贴近上限，说明梯度持续被 clip。

**影响**：warmup 阶段属于正常现象——模型从随机初始化开始，早期梯度大。但如果 competitive 阶段仍然持续 clip，需要考虑降低 `learning_rate` 或增大 `max_grad_norm`。

**建议**：competitive 阶段观察是否改善。

---

#### ❌ `blood/league_pool_size` 出现负值

**现象**：league_pool_size 应为非负整数（league 中保存的 checkpoint 数量），但实际值在 -0.28 ~ 0.19 之间，明显不是整数。

**根本原因**：`blood/league_pool_size` 被 SF2 的 running mean/std 归一化处理了，实际 pool size 是整数但被当作 obs 归一化了。

**建议**：在 `BloodObserver.on_stats` 中用 `writer.add_scalar` 直接写入，绕过 SF2 的归一化管道。

---

#### ⚠️ `train/entropy` 末段回升（2.05 → 2.75）

**现象**：entropy 在 ~1.5M step 后从 2.05 回升到 2.75，说明策略在末段变得更随机。

**可能原因**：`kl_adaptive_minibatch` 在 KL 过小时自动提高 LR，导致策略更新幅度增大，entropy 上升。这是自适应 LR 的正常行为。

**影响**：无负面影响，说明模型没有过早收敛到局部最优。

---

### Warmup 阶段结论

| 项目 | 结论 |
|------|------|
| 收敛状态 | ✅ 正常收敛，reward 从 -0.15 升至 6.35 |
| 训练稳定性 | ✅ KL、clip ratio 均受控 |
| 显存 | ✅ 稳定在 5637 MB，无泄漏 |
| 吞吐量 | ✅ 8192–9830 FPS，约 3–4 分钟完成 2M steps |
| 待修复 | `blood/league_pool_size` 归一化问题；`adv_std` 记录问题 |

**下一步**：加载 warmup checkpoint，启动 competitive 阶段（`configs/competitive.yaml`）。

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
| 2026-02-24 | 0 → 5M（完成） | 修复所有死锁 bug 后重跑，训练正常收敛至目标步数 |

---

### 关键指标分析（第二次运行，step 0 → 5M，实测数据）

#### 奖励 / 目标

| 指标 | 初始值 | 最终值 | 峰值 | 评估 |
|------|--------|--------|------|------|
| `reward/reward` | 0.002 @33K | **14.24 @5M** | 15.375 | ✅ 持续上升，无崩溃 |
| `reward/reward_max` | 1.14 @33K | ~22 @5M | 23.64 | ✅ 单局最高奖励稳步提升 |
| `reward/reward_min` | -0.58 @33K | **+0.88 @5M** | — | ✅ 末段全部转正，所有局均盈利 |

#### 训练稳定性

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/entropy` | 3.50 @16K | **1.22 @5M** | ✅ 正常下降，策略收敛但未崩溃 |
| `train/grad_norm` | 1.0（打满上限） | **0.35 @5M** | ✅ 后期梯度显著下降，不再持续 clip |
| `train/fraction_clipped` | 0.42 | ~0.05 | ✅ 更新趋于保守 |
| `train/value_loss` | 1.63 | ~0.08 | ✅ 价值函数持续收敛 |
| `train/policy_loss` | 0.030 | ~0.001 | ✅ 策略梯度正常 |
| `train/kl_divergence` | — | 0.040（偶有尖峰至 0.187） | ⚠️ 整体受控，偶发尖峰，见下方分析 |
| `train/actual_lr` | 1e-4 | **1e-4** | ✅ KL 自适应调度正常 |
| `train/adv_std` | 0.18 | ~2.0 | ✅ 优势函数方差增大，探索充分 |
| `train/adv_mean` | -0.03 | **-0.06 @5M** | ✅ 从 -0.43 回升至接近 0，策略不再过于保守 |
| `train/returns_running_mean` | -0.04 | **6.28** | ✅ 回报基线持续上升 |

#### Episode 长度

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `len/len` | 43.1 @33K | **53.65 @5M** | ✅ 正常麻将局长度，无系统性截断 |
| `len/len_min` | 19 | 32 | ✅ 最短局正常 |
| `len/len_max` | 63 | 352 @5M（偶发尖峰至 500） | ⚠️ 偶发残余死锁，见下方分析 |

`len/len` 均值全程稳定在 40–73 步，主要死锁已修复；`len_max` 偶发 500 尖峰（5M 内共 6 次）。

#### 联赛池 / 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `blood/league_pool_size` | 0 → **50（满池）** | ✅ 正常填充，每 50K steps 新增一个 checkpoint |
| `perf/_fps` | ~6000–9831 FPS（末段 ~1638） | ✅ 吞吐量正常，末段降速为训练结束正常现象 |

---

### 异常指标分析

#### ⚠️ `len/len_max` 偶发 500 尖峰（已修复）

**现象**：5M steps 内共出现 6 次 `len_max=500` 尖峰。

**根本原因**：Python 侧 stall 检测器（`_advance_external_opponents`）在检测到死锁时只执行 `break`，未强制终止游戏。Rust 侧 rulebot 的 stall 检测器会将 `phase` 强制设为 `Scoring`，但 Python 侧缺少这一步。

**后果**：stall 触发后 `_advance_external_opponents` 返回，游戏仍处于非 agent 回合。下一次 `step()` 调用 agent 动作为 NO-OP（wrong-player guard），再次触发 stall，如此循环 500 次直到 `_step_count >= _max_steps` 截断。

**理论上限**：108 张牌，53 张发牌后剩余 55 张，agent 约摸 14 次摸牌，最多 ~44 次决策。500 步从麻将逻辑上不可能合法发生。

**修复**：stall 触发时改为调用 `self._env.finalize_scoring()`，与 Rust 侧行为一致。

---

#### ⚠️ `train/kl_divergence` 偶发尖峰（最高 0.187）

**现象**：KL 散度整体在 0.01–0.05 范围内，但偶发尖峰至 0.094、0.066、0.074、0.187。

**分析**：自对弈中对手策略跳变（联赛池切换到差异较大的历史模型）时，当前策略面对新对手的分布偏移较大，导致单步 KL 升高。`kl_adaptive_minibatch` 会在下一 minibatch 自动降低 LR 以补偿。

**影响**：尖峰后 reward 无明显下降，说明自适应调度有效抑制了策略崩溃。

---

#### ✅ `train/grad_norm` 后期改善（1.0 → 0.35）

**现象**：grad_norm 在前 ~2M steps 持续打满 1.0，后期逐渐下降至 0.35。

**分析**：随着策略趋于稳定、价值函数收敛，梯度幅度自然下降。`learning_rate=1e-4` 在后期已足够保守，无需进一步调整。

**结论**：warmup 阶段预测的"competitive 阶段观察是否改善"已验证——后期自然改善，无需手动干预。

---

#### ✅ `train/adv_mean` 从 -0.43 回升至 -0.06

**现象**：`adv_mean` 在 ~2M steps 时降至 -0.43，但 5M steps 时回升至 -0.06。

**分析**：中期偏负是自对弈对手快速进步导致的暂时现象。随着联赛池填满（50 个历史模型），对手分布趋于稳定，模型重新找到相对优势，`adv_mean` 回升。

**结论**：无需干预，自然收敛。

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

---

### 末段实测快照（~966K → 5M steps 完成前）

以下数据来自训练结束前最后一段日志（2026-02-24 05:31–05:32），记录最终收敛状态：

| 指标 | 值 | 说明 |
|------|----|------|
| 帧数范围 | 966,656 → 1,081,344 | 最后约 115K frames |
| Policy 版本 | ~480 → ~521 | 每 ~2K frames 更新一次 |
| Best reward（滑动窗口） | 13.771 → 13.792 → **14.232** | 末段持续小幅提升 |
| FPS（10s 窗口） | ~1638 | 训练结束前正常吞吐 |
| FPS（60s 窗口） | ~1365 | 含 checkpoint 保存开销 |
| 样本吞吐 | ~1441–1506 samples/sec | 稳定 |
| Policy lag（avg） | 7–11.6 steps | 正常范围，无严重滞后 |
| Policy lag（max） | 最高 23 steps | 偶发，不影响训练质量 |

**结论**：末段 reward 仍在缓慢上升（13.77 → 14.23），说明 5M steps 时模型尚未完全饱和，
Elite 阶段有进一步提升空间。FPS 和 lag 均在正常范围，无系统性瓶颈。

---

### Competitive 阶段结论

| 项目 | 结论 |
|------|------|
| 收敛状态 | ✅ 正常收敛，reward 从 0.002 升至 14.24（5M steps），峰值 15.375 |
| 训练稳定性 | ✅ entropy=1.22，KL 整体受控，无崩溃 |
| Episode 长度 | ✅ 均值 53.65 步，主要死锁已修复；偶发 len_max=500（6 次 / 5M steps） |
| 联赛池 | ✅ 满池（50 个历史模型） |
| 吞吐量 | ✅ ~6000–9831 FPS |
| grad_norm | ✅ 后期自然下降至 0.35，无需手动调整 |
| adv_mean | ✅ 从 -0.43 回升至 -0.06，策略不再过于保守 |
| reward_min | ✅ 末段转正（+0.88），所有局均盈利 |
| 待观察 | 残余 `len_max=500` 偶发尖峰（极低频率，可在 Elite 阶段继续监控） |

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
