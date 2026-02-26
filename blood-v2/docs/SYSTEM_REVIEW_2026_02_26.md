# Blood-V2 系统评审报告

> 评审日期: 2026-02-26 | 阶段: Phase 3 Elite @ 4.24M / 50M steps
> 前置: 基于 2026-02-24 评审（64 项全部完成）后的运行中系统复审
> 范围: Rust 引擎、神经网络、环境、训练、评估、配置 全子系统

---

## 一、执行摘要

系统整体健康，64 项初始评审问题已全部修复并经过 ~12.2M 步训练验证。本次复审发现 **3 项需修复**、**3 项需关注**、**2 项已验证无误**。

| 级别 | 数量 | 说明 |
|------|------|------|
| S2 需修复 | 3 | cfg.py 默认值不匹配、CLAUDE.md 参数量过估、adv_clip 跨阶段跳变 |
| S3 需关注 | 3 | Oracle CE adv_std 地板值、entropy 持续下降、warmup_dq_bonus 死代码已修复 |
| ✅ 已验证 | 2 | calc_max_hand_score is_ron=false、_score_delta_to_fan tsumo 处理 |

---

## 二、发现详情

### F1 [S2] cfg.py `blood_num_tile_attn_layers` 默认值不匹配

**位置**: `python/blood/cfg.py:23`

```python
p.add_argument("--blood_num_tile_attn_layers", type=int, default=2, ...)
```

所有生产 YAML 配置均设为 `4`（default.yaml:35, warmup.yaml:38, elite.yaml:42 等）。argparse 默认值为 `2`。

**风险**: 不加载 YAML 的代码路径（测试、notebook、快速实验）会静默构建 2-segment 模型，与训练模型不兼容。

**建议**: 将 cfg.py 默认值改为 `4`，与生产配置一致。

---

### F2 [S2] CLAUDE.md 参数量过估（~37M → ~18-21M）

**位置**: `/CLAUDE.md` 神经网络描述

CLAUDE.md 声称模型约 37M 参数，实际估算：

| 组件 | 参数量 |
|------|--------|
| Stem (SuitAwareConv1d 470→256) | ~361K |
| BottleneckBlock ×20 | ~2.3M |
| TileAttention ×4 | ~1.08M |
| enc_proj (6912→1024) | ~7.1M |
| LSTM (2层, 512-dim) | ~6.3M |
| Actor/Critic + 辅助头 | ~1-3M |
| **学生模型总计** | **~18-21M** |

Oracle 编码器共享大部分结构（额外 52 通道输入），加上 Oracle 价值头，完整系统约 25-28M。37M 为过估。

**建议**: 更新 CLAUDE.md 参数量描述为 ~20M（学生模型）。

---

### F3 [S2] adv_clip 跨阶段跳变 2.0 → 5.0

**位置**: `configs/competitive_distill.yaml:115` vs `configs/elite.yaml:118`

| 阶段 | adv_clip |
|------|----------|
| competitive_distill (Phase 2b) | 2.0 |
| elite (Phase 3 初始) | 5.0 |
| elite (Phase 3 @40M) | 3.0 (线性退火) |

Phase 2b→3 过渡时 adv_clip 从 2.0 跳到 5.0（2.5 倍），可能导致初期优势估计波动。

**当前影响**: Phase 3 已运行 4.24M 步，Elo 从适应期恢复到 1505，Arena win_rate=0.65（历史最佳）。跳变的冲击已被吸收。

**建议**: 下次训练可考虑 elite 初始 adv_clip 设为 3.0，schedule 改为 `3.0→2.0`，减少跨阶段冲击。当前训练无需调整。

---

### F4 [S3] Oracle CE adv_std 地板值过小

**位置**: `python/blood/training/losses.py:150`

```python
adv_std = max(advantages.detach().std().item(), 1e-4)
adv_w = F.softmax(advantages.detach() / adv_std, dim=0)
```

当优势方差接近零时，`1e-4` 地板使 softmax 输入被放大 ~10000 倍，产生近似 one-hot 的权重分布。

**当前影响**: 训练中 value_loss 持续下降（分段均值 1.102→0.979），未观察到不稳定。但在极端情况下可能放大噪声。

**建议**: 可将地板值提高到 `1e-2`，或改用 `max(adv_std, 0.01 * adv_mean.abs())`。低优先级。

---

### F5 [S3] entropy 持续下降趋势

**数据来源**: TensorBoard `train/entropy`

| 区间 | entropy 均值 |
|------|-------------|
| 0~1M | 0.70 |
| 1M~2M | 0.57 |
| 2M~3M | 0.51 |
| 3M~4.24M | 0.47 |

cosine schedule 系数几乎不变（0.020→0.0196），下降来自策略自身收敛。当前 0.47 仍在安全范围（阈值 0.35），但按当前速率 ~8M 步可能接近警戒线。

**建议**: 若 Elo 同步上升则无需干预。若 entropy < 0.40 且 Elo 停滞，考虑临时提高熵系数或增加 entropy bonus。

---

### F6 [S3] warmup_dq_bonus 死代码（已修复）

**位置**: `python/blood/env/selfplay_env.py:532`

`warmup_dq_bonus` 在 cfg.py 中定义（default=0.05），在 selfplay_env.py 中读取，但从未在奖励计算中使用。

**状态**: 已在本次评审中修复。新增 `get_agent_suit_counts()` Rust API（`crates/pybind/src/env.rs`），在 `_compute_shaping_reward()` 中激活定缺奖励逻辑。仅在 `warmup_reward_shaping: true` 阶段生效。

---

### V1 [✅] calc_max_hand_score is_ron=false — 设计正确

**位置**: `crates/engine/src/state/board.rs:888`

```rust
is_ron: false, // assume tsumo for max
```

用于花猪罚分计算，`is_ron=false`（自摸）可获得更高番数（门清自摸 +1），从而最大化罚分。注释明确标注意图。

---

### V2 [✅] _score_delta_to_fan tsumo 处理 — 逻辑正确

**位置**: `python/blood/env/selfplay_env.py:30-49`

```python
for divisor in (3, 1):  # tsumo (3 payers) first, then ron (1 payer)
    per_payer = delta / divisor
```

正确处理自摸 3 人支付和荣和 1 人支付。故意省略 `divisor=2` 以避免 2 番荣和被误判为 1 番自摸。

---

## 三、子系统健康度

| 子系统 | 状态 | 说明 |
|--------|------|------|
| Rust 引擎 | ✅ 优秀 | 7 阶段状态机、470ch 学生/522ch Oracle 观测、SP Table、ISMCE 搜索均正常 |
| 神经网络 | ✅ 良好 | 4-segment 循环架构 + 2 层 LSTM，cfg.py 默认值需修复 |
| 环境 | ✅ 良好 | selfplay + 联赛系统运行稳定，warmup_dq_bonus 已激活 |
| 训练 | ✅ 良好 | PPO + Oracle 蒸馏 + 动态调度，adv_clip 跳变已被吸收 |
| 评估 | ✅ 良好 | RTPA + ISMCE + Arena 评估正常，Elo 追踪稳定 |
| 配置 | ⚠️ 注意 | cfg.py 默认值与 YAML 不一致（F1） |

---

## 四、训练状态快照（@4.24M / 50M）

| 指标 | 当前值 | 评估 |
|------|--------|------|
| Elo | 1505 | ✅ 从 1480 回升 |
| Arena win_rate | 0.65 | 🎉 历史最佳 |
| Arena avg_rank | 2.35 | 🎉 历史最佳 |
| value_loss | 0.98 (分段均值) | ✅ 持续下降 |
| entropy | 0.47 | ⚠️ 持续下降，关注 < 0.40 |
| grad_norm clip@3.0 | 11.6% | ✅ 健康 |

详见 `training_log.md`。

---

## 五、与前次评审对比

| 维度 | 2026-02-24 评审 | 本次复审 |
|------|----------------|---------|
| 发现总数 | 64 项 | 8 项（3 S2 + 3 S3 + 2 验证） |
| S0 严重 | 4 项（ISMCE 失效等） | 0 项 |
| 系统状态 | 未训练，需从头开始 | Phase 3 运行中，Elo 1505 |
| 核心瓶颈 | SP Table 退化、评估管道失效、训练规模不足 | 无核心瓶颈，仅配置细节 |

---

## 六、行动项

| # | 级别 | 行动 | 优先级 |
|---|------|------|--------|
| 1 | S2 | cfg.py `blood_num_tile_attn_layers` 默认值改为 4 | 高 — 防止测试/实验构建错误模型 |
| 2 | S2 | CLAUDE.md 参数量从 ~37M 更正为 ~20M | 中 — 文档准确性 |
| 3 | S2 | 记录 adv_clip 跳变为已知设计决策 | 低 — 当前训练无需调整 |
| 4 | S3 | 监控 entropy，< 0.40 时评估是否干预 | 持续 |
| 5 | S3 | Oracle CE adv_std 地板值可提高到 1e-2 | 低 — 未观察到问题 |
