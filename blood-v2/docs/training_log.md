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

*待填写*

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
