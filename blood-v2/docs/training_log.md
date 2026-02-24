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
| 2026-02-23 | 0 → 5M | 首次 competitive 运行，中途修复多个 bug |
| 2026-02-24 | 0 → 5M（重跑） | 实际 TensorBoard 数据更新（见下方） |

---

### 关键指标分析（step 0 → 5M，实测数据）

#### 奖励 / 目标

| 指标 | 初始值 | 峰值 | 最终值 | 评估 |
|------|--------|------|--------|------|
| `reward/reward` | 0.0 @65K | **10.22 @917K** | 0.0 @5M | ❌ 策略崩溃，见下方分析 |
| `reward/reward_max` | 0.0 | 22.10 @1.44M | 0.0 | ❌ 同上 |
| `reward/reward_min` | -0.59 | +0.26 @1.1M | 0.0 | ❌ 同上 |
| `policy_stats/avg_true_objective` | 0.0 | 10.22 @917K | 0.0 | ❌ 与 reward 一致 |

奖励在 ~917K steps 达到峰值 10.22，随后持续下滑，~1.87M steps 后归零并保持至训练结束（5M steps）。

#### 训练稳定性

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `train/entropy` | 3.50 @16K | **0.0 @1.59M 起** | ❌ 策略熵崩溃，模型输出近似确定性动作 |
| `train/grad_norm` | 1.0（打满上限） | 0.0008 | ❌ 梯度几乎为零，模型停止学习 |
| `train/fraction_clipped` | 0.64 | 0.0（偶发 0.79 spike @4.01M） | ⚠️ 正常为 0 但有异常 spike |
| `train/value_loss` | 1.01 | 0.0 | ❌ 价值函数无更新 |
| `train/policy_loss` | 0.021 | ~0.0（spike 0.21 @4.01M） | ❌ 策略无更新 |
| `train/kl_divergence` | 0.020 | 0.0（spike 57.2 @4.01M） | ❌ 异常 spike，见下方分析 |
| `train/actual_lr` | — | 0.01（KL 自适应） | ✅ 调度器正常 |

#### Episode 长度

| 指标 | 初始值 | 最终值 | 评估 |
|------|--------|--------|------|
| `len/len` | 107.7 @65K | 500.0 @5M | ❌ 全部截断，游戏从不正常结束 |
| `len/len_min` | 52 | 500 | ❌ 最短局也被截断 |
| `len/len_max` | 148 | 500 | ❌ 同上 |

`len/len` 在 **327K steps** 时首次全部达到 500（截断上限），此后再未恢复正常。

#### 联赛池 / 系统性能

| 指标 | 值 | 评估 |
|------|-----|------|
| `blood/league_pool_size` | 0 → 50（满池） | ✅ 正常填充 |
| `perf/_fps` | 8192–9831 FPS | ✅ 吞吐量正常 |
| `stats/gpu_mem_learner` | 5856 MB（全程不变） | ✅ 无显存泄漏 |

---

### 异常指标分析

#### ❌ 策略崩溃（Policy Collapse）

**现象**：
- `train/entropy` 在 ~1.59M steps 降至 0，此后全程为 0
- `reward/reward` 在 ~1.87M steps 归零，此后全程为 0
- `len/len` 全部截断（500 steps），游戏从不结束
- `train/grad_norm` 降至 ~0.001，模型停止更新

**根本原因**：自对弈中双方策略同步退化为"不打牌"（全部 Pass 或随机打牌），导致游戏永远无法结束，奖励信号消失，梯度归零，形成正反馈崩溃循环。

**触发时机**：`len/len` 在 327K steps 首次全部截断，说明崩溃在早期就已开始，entropy 和 reward 的崩溃是滞后表现。

**可能诱因**：
1. `max_steps=500` 过小——正常麻将局约 92–131 步，但自对弈双方策略不成熟时局面会拖长，500 步不够
2. 联赛池早期为空，对手是随机策略，模型学到了对抗随机策略的"不作为"均衡
3. 缺乏足够的进攻性奖励（`reward_rank_bonus=0`），模型没有动力赢牌

#### ⚠️ 异常 Spike @4.01M steps

**现象**：在 step ~4.01M 出现单点异常：`kl_divergence=57.2`、`fraction_clipped=0.79`、`policy_loss=0.21`，前后均为 0。

**原因**：疑似加载了一个损坏的联赛 checkpoint（`checkpoint_4325376.pth`，已在 2026-02-24 修复自动删除逻辑），导致对手策略突变，产生一次大梯度更新，但随即被 PPO clip 压制，未能恢复训练。

---

### Competitive 阶段 Bug 修复

| commit | 描述 |
|--------|------|
| `62124c71` | fix: `_inject_config_yaml` str2bool 参数需要显式值（`--use_rnn True`） |
| `4b4ca54f` | fix: `selfplay_env.py` 中 `terminated` 在赋值前被引用（UnboundLocalError） |
| `1f9ff565` | fix: competitive/elite 阶段 CUDA OOM，降低 `num_envs_per_worker` 和 `batch_size` |
| `2d91c8a2` | fix: BPTT 训练时 `forward_core` 收到 `PackedSequence`，`LayerNorm` 报错 |
| `1f9f6e2c` | tune: competitive `learning_rate` 3e-4 → 1e-4（`fraction_clipped` 达 40%） |
| `4b7861bf` | fix: 联赛快照静默失败（`LearnerWorker.learner` 在主进程为 None） |
| `777f24d4` | fix: 联赛 checkpoint 损坏（竞态条件）+ `rnn_size` 检测失败导致维度不匹配 |
| `b773fb4a` | fix: checkpoint 验证改用 `weights_only=False`（SF2 包含 numpy 标量） |
| `fa57af40` | fix: `apply_self_check` tsumo panic + stall（wrong-player guard + `_ => Discard`） |
| `95fd862c` | fix: 损坏联赛 checkpoint 自动删除（`load()` 返回 bool，失败时 unlink） |

---

### Competitive 阶段结论

| 项目 | 结论 |
|------|------|
| 收敛状态 | ❌ 策略崩溃，reward 峰值 10.22 @917K，之后归零 |
| 训练稳定性 | ❌ entropy=0，grad_norm≈0，模型停止学习 |
| 联赛池 | ✅ 满池（50 个历史模型） |
| 吞吐量 | ✅ ~8192–9831 FPS |
| 根本问题 | 自对弈策略崩溃，需要重跑并修复崩溃诱因 |

**下一步（Competitive 重跑）**：
1. 增大 `max_steps`（500 → 1000）防止过早截断
2. 开启 `reward_rank_bonus`（建议 0.3）激励进攻
3. 从 warmup checkpoint 重新启动 competitive 阶段
4. 监控 `len/len` 和 `train/entropy`，若 327K steps 内 len 全部达到上限则立即停止调查

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
