# 血战到底 AI 全面系统审查

> 审查日期：2026-02-15  
> 审查范围：配置文件、训练日志、神经网络、奖励系统、番数系统  
> 审查目标：评估系统能否达到超人类水平

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [番数与计分系统](#2-番数与计分系统)
3. [观测空间](#3-观测空间)
4. [神经网络容量](#4-神经网络容量)
5. [奖励信号](#5-奖励信号)
6. [训练方法论](#6-训练方法论)
7. [系统正确性审查](#7-系统正确性审查)
8. [关键瓶颈排序](#8-关键瓶颈排序)
9. [综合评估与路线图](#9-综合评估与路线图)

---

## 1. 系统架构总览

| 组件 | 规格 | 评分 |
|------|------|------|
| 观测空间 | Student: (423, 27), Oracle: (121, 27) | A |
| 动作空间 | 34 (27 弃牌 + pon/kan/agari/pass + 3 定缺) | A |
| Student 网络 | 30-block ResNet + SuitAwareConv + ChannelAttention, **10.5M params** | B+ |
| Oracle 网络 | 15-block ResNet (完美信息), **6.6M params** | B+ |
| DQN | Dueling DQN (独立 V/A 流, 512 隐藏层), **1.1M params** | B+ |
| 辅助任务 | AuxNet: next_rank + opp_wait (3×27) + ding_que, **0.6M params** | B |
| 奖励信号 | 纯 score_diff / 10000 | A- |
| 番数系统 | 17 种番型, 5 番封顶, FanConfig 可配置 | A |
| 训练方法 | DQN + Self-play + Target Network + Oracle Guiding + TD(λ) | B |

---

## 2. 番数与计分系统

**评分: A — 完整且正确**

### 已实现的 17 种番型

| 番型 | 番数 | 类型 | 实现位置 |
|------|------|------|---------|
| 平胡 (PingHu) | +1 基础 | 必含 | `agari.rs:374` |
| 自摸 (Tsumo) | +1 | 和牌方式 | `agari.rs:376` |
| 门清 (MenQing) | +1 | 和牌方式 | `agari.rs:378` (可配置) |
| 七对 (QiDui) | +2 | 手牌结构 | `agari.rs` (与碰碰胡互斥) |
| 碰碰胡 (ToiToi) | +1 | 手牌结构 | `agari.rs` (与七对互斥) |
| 金钩钓 (JinGouDiao) | +1 | 手牌结构 | `agari.rs` |
| 清一色 (QingYiSe) | +2 | 花色 | `agari.rs` |
| 带幺九 (DaiYaoJiu) | +3 | 花色 | `agari.rs` (可配置, 与断幺九互斥) |
| 断幺九 (DuanYaoJiu) | +1 | 花色 | `agari.rs` (可配置, 与带幺九互斥) |
| 一条龙 (YiTiaoLong) | +1 | 特殊 | `agari.rs` (可配置) |
| 夹心五 (JiaXinWu) | +1 | 特殊 | `agari.rs` (可配置) |
| 四归一/根 (SiGuiYi) | +1/根 | 根 | `agari.rs` (可叠加多根) |
| 杠上花 (GangShangHua) | +1 | 杠相关 | `agari.rs:433` |
| 杠上炮 (GangShangPao) | +1 | 杠相关 | `agari.rs:434` |
| 抢杠 (Chankan) | +1 | 杠相关 | `agari.rs:435` |
| 海底 (Haidi) | +1 | 特殊 | `agari.rs:377` (可配置) |
| 天胡/地胡 (TianHu/DiHu) | 直接 5 | 特殊 | `agari.rs:437-440` (可配置) |

### 计分公式

```
点数 = 1000 × 2^(番数-1)，min(fan, 5) 封顶
```

| 番数 | 点数 | 说明 |
|------|------|------|
| 1 番 | 1,000 | 基础 (平胡) |
| 2 番 | 2,000 | |
| 3 番 | 4,000 | |
| 4 番 | 8,000 | |
| 5 番 | 16,000 | **封顶** |

### 杠即时支付 (独立于番数计分)

| 杠类型 | 支付方 | 金额 |
|--------|--------|------|
| 明杠 | 放杠者 → 杠者 | 2,000 |
| 暗杠 | 每位未和牌者 → 杠者 | 2,000 × N |
| 补杠 (及时雨) | 每位未和牌者 → 杠者 | 1,000 × N |
| 补杠 (非及时雨) | 无即时支付 | 0 |

### 验证结论

- 番数叠加与互斥逻辑正确
- 5 番封顶在两处执行（`agari.rs` + `point.rs`），双重保险
- 杠即时支付与番数独立计算，抢杠退款机制完善
- `FanConfig` 7 个可配置开关，支持多规则训练

---

## 3. 观测空间

**评分: A — 信息极其丰富**

### Student 观测: (423, 27)

| Section | Channels | 内容 |
|---------|----------|------|
| 手牌 | 5 | 手牌计数 (4ch) + 最后摸牌 (1ch) |
| 场况 | 10 | 分数 (4ch) + 排名 (4ch) + 局 (1ch) + 庄家 (1ch) |
| 定缺 | 17 | 自家花色/完成/剩余 (5ch) + 对手花色 (9ch) + 对手和牌 (3ch) |
| 局面 | 5 | 剩余牌 + 禁手 + 临时振听 + 杠标记 + 场上杠数 |
| 自家牌河 | 38 | 最近 18 张位置编码 (36ch) + 指数衰减概览 (2ch) |
| 对手牌河 | 111 | 3 人 × (18×2 + 衰减) |
| 可见牌 | 53 | 牌河概览 (16ch) + 副露概览 (32ch) + 暗杠 (4ch) + 可见比 (1ch) |
| 防守 | 9 | 对手花色倾向 (3×3) |
| 导出特征 | 8 | 剩余牌/门清/副露数/巡目/受理数/对手副露数 |
| 手牌分析 | 7 | 听牌 (1ch) + 向听 one-hot (5ch) + 杠选择 (1ch) |
| 动作上下文 | 11 | 最后牌河牌 + 弃牌候选 + 碰杠和 + 当前 Ron 番数 |
| **SP Table** | **~100** | **28 巡 × (听牌概率/和牌概率/期望值) + 受理牌 + 最大 EV** |
| FanConfig | 7 | 7 个番种开关 (multi-rule 训练预留) |

**SP Table 是关键优势**：内置单人最优弃牌分析（听牌概率、和牌概率、期望值按巡展开），相当于在观测中内嵌了"牌理计算器"，大幅降低网络学习负担。

### Oracle 观测: (121, 27)

| Section | Channels | 内容 |
|---------|----------|------|
| 对手手牌 | 12 | 3 人 × 手牌计数 (4ch) |
| 对手定缺 | 9 | 3 人 × 花色 one-hot (3ch) |
| 对手向听 | 24 | 3 人 × (向听 one-hot 7ch + rescale 1ch) |
| 对手听牌 | 3 | 3 人 × 听牌 (1ch) |
| 对手定缺完成 | 3 | 3 人 × 完成标记 (1ch) |
| 剩余牌墙 | ~56 | 每张牌 1 channel (tile one-hot) |
| **合计** | **~121** | **真正的完美信息** |

### 潜在改进空间

1. 缺少显式的"对手危险牌预测"通道（由 opp_wait AuxNet 隐式提供）
2. 缺少"自家当前手牌番数结构"通道（SP Table 的 EV 隐含了部分信息）
3. 对手牌河的时序信息可进一步丰富（如碰后第一张弃牌的特殊标记）

---

## 4. 神经网络容量

**评分: B+ — 对血战到底基本充足**

### 参数量

| 模型 | 参数量 | 说明 |
|------|--------|------|
| Student Brain | 8,876,264 | 30-block ResNet, 192ch |
| DQN | 1,067,555 | Dueling V/A 独立 512 隐藏层 |
| AuxNet | 569,944 | 1 层 512 隐藏层 |
| **Student 合计** | **10,513,763** | |
| Oracle Brain | 5,544,500 | 15-block ResNet, 192ch |
| Oracle DQN | 1,067,555 | 同 Student DQN |
| **全部合计** | **17,125,818** | |

### 架构特点

- **SuitAwareConv1d**：阻止跨花色卷积（万↔筒、筒↔条边界），kernel=3 时自然隔离
- **ChannelAttention**：每个 ResBlock 内提供花色间信息交换
- **Pre-activation ResNet**：BN → Mish → Conv 顺序，梯度流更平滑
- **Dueling DQN**：V/A 流独立，A 输出层初始化为零（初始 Q ≈ V，稳定训练）
- **最终 Conv 64ch**：192→64 (3x 压缩) → Flatten(1728) → Linear(1024)

### 对比参考

| 系统 | 参数量 | 牌组 | 结果 |
|------|--------|------|------|
| Mortal (日麻) | ~12M | 136 张 (含字牌) | 接近顶级人类 |
| Suphx (微软) | ~60M | 136 张 (含字牌) | 超人类水平 |
| **本系统** | **10.5M** | **108 张 (无字牌)** | **训练中** |

### 潜在改进

- **宽度 vs 深度**：30 blocks 可能已饱和，增大宽度 (192→256) 可能收益更大
- **DQN V/A 流**：各只有 1 层 512 维，复杂场景的 Q-value 估计可能受限
- **AuxNet**：opp_wait (3×27=81 维输出) 是高维预测任务，1 层 512 可能不够

---

## 5. 奖励信号

**评分: A- — 理论最优但信号稀疏**

### 当前配置

```toml
# 纯 score_diff 驱动，无额外奖励塑形
rank_bonus_enabled = false
action_bonus_enabled = false
agari_bonus = 0.0
houjuu_penalty = 0.0
```

奖励值 = `score_diff / 10000`（每局 kyoku 结束时计算）

### 优势

- **直接对齐游戏目标**：最大化得分差 = 最大化胜率
- **零和博弈**：4 人分数零和，score_diff 自然捕获竞争关系
- **无 reward hacking 风险**：没有人工奖励偏差
- **杠支付已包含**：杠的即时支付反映在 score_diff 中

### 劣势

- **极度稀疏**：每局 20+ 个决策点共享同一个终局奖励
- **高方差**：番数指数增长（1000→16000），同手牌自摸 vs 荣和差 3x
- **信用分配困难**：好的防守（不放铳）完全没有直接奖励
- **无中间信号**：杠的即时支付虽影响 score_diff，但不作为单步 reward

### Target Network + TD(λ) 的缓解效果

- TD(λ=0.95) 混合 95% MC + 5% 1-step TD bootstrap
- V_target(s') 提供中间状态价值估计
- 但 V_target 仍在收敛中（Target Network 仅训练约 60k 步）

---

## 6. 训练方法论

**评分: B — 可行但有改进空间**

### 6.1 DQN vs Policy Gradient

| 维度 | DQN (当前) | PPO/A2C |
|------|-----------|---------|
| Off-policy | ✅ 可复用数据 | ❌ On-policy |
| 信用分配 | Q(s,a) 直接估计 | GAE + Baseline 更精确 |
| 策略表达 | 隐式 (argmax Q) | 显式概率分布 |
| 梯度稳定性 | MSE loss 平滑 | Clipped objective |
| 探索 | Boltzmann (ε=0.05, T=0.10) | 策略熵自然探索 |

DQN 用于麻将 AI 是可行的（Mortal 在日麻上验证），但 Boltzmann 探索强度低，接近最优时可能陷入局部最优。

### 6.2 Self-play

```toml
# 对手池配置
[baseline.pool]
enabled = true
newest_weight = 3.0     # 最新检查点 3x 权重
baseline_update_threshold = 3.2  # avg_pt > 3.2 时更新
```

- 维护 5-8 个历史 checkpoint 的对手池
- avg_ranking ≈ 2.50 时达到自我平衡
- 缺乏外部强力对手提供突破压力

### 6.3 Oracle Guiding

```toml
[oracle]
enabled = true
distill_weight = 0.0      # 当前：Oracle 自学阶段
distill_temperature = 2.0  # 软化分布，传递 dark knowledge
oracle_dqn_weight = 0.5    # 减少梯度占比
```

Oracle 看到完美信息（对手手牌 + 牌墙），理论上能学到最优策略，再通过 KL 蒸馏传递给 Student。当前 distill_weight=0.0（Oracle 自学中），待 oracle_dqn_loss 收敛后逐步开启。

### 6.4 过手策略 (Pass-on-Agari)

**经审查确认**：模型已可自由学习过手策略。

- 训练时 `enable_rule_based_agari_guard = False`（MortalEngine 默认值）
- 测试时虽启用 guard，但 `rule_based_agari()` 始终返回 `can_agari()` → guard 永不触发
- 模型可在训练中选择 pass(30) 而非 agari(29)，数据正确记录
- 血战到底规则已实现"过手加番可胡"机制

`rule_based_agari()` 当前为 no-op 安全网（总是允许和牌），有 TODO 计划添加启发式逻辑。

---

## 7. 系统正确性审查

### 7.1 已确认正确

| 项目 | 状态 | 说明 |
|------|------|------|
| 番数计算 (17 种) | ✅ | 互斥/叠加逻辑正确 |
| 计分公式 | ✅ | `1000 × 2^(fan-1)`，5 番封顶 |
| 杠即时支付 | ✅ | 明杠/暗杠/补杠/抢杠退款 |
| 过手规则 | ✅ | 加番可胡实现 |
| 观测编码 | ✅ | 断言验证 `idx == obs_shape` |
| SuitAwareConv1d | ✅ | 防跨花色卷积 |
| Dueling DQN | ✅ | V/A 流独立，A 初始化零 |
| BUG-5 修复 | ✅ | TD(λ) 双重衰减已修复 |
| ISSUE-1 修复 | ✅ | 定缺步排除蒸馏 |
| ISSUE-2 修复 | ✅ | oracle_dqn_weight 1.0→0.5 |
| ISSUE-3 修复 | ✅ | distill_temperature 1.0→2.0 |
| ISSUE-5 修复 | ✅ | opt_step_every 1→2 |
| ding_que 配置 | ✅ | ISSUE-1 修复后 weight 恢复 0.01 |
| 过手策略 | ✅ | 模型训练时自由选择，非被 agari_guard 锁死 |

### 7.2 已知的潜在改进点

**P1: Oracle 与 Student 共享 Optimizer**

所有 5 个模型共享同一个 AdamW 和 `clip_grad_norm_`。Oracle 梯度直接影响 Student 的有效 step size。虽然 `oracle_dqn_weight=0.5` 缓解了梯度竞争，但两个模型学习目标不同（Student: 不完全信息, Oracle: 完全信息），共享优化器会导致梯度方向冲突。

- 严重程度：中
- 建议：为 Oracle 使用独立 Optimizer

**P2: TD(1-step) 的 `imm_reward` 仅在 done 出现**

```python
if is_done:
    imm_reward_arr[i] = kyoku_rewards[at_kyoku_np[i]]
else:
    imm_reward_arr[i] = 0.0
```

非 done 步骤即时奖励恒为 0。数学上正确（终端奖励模型），但 TD 效率受限——所有中间步骤只能通过 V(s') 传递信号。

- 严重程度：低
- 建议：如果引入中间奖励（如杠支付作为 step reward），需修改此处

**P3: `rule_based_agari()` 是 no-op**

当前实现等价于 `return can_agari()`，guard 永远不触发。有 TODO 计划添加启发式（番数/剩余牌/活跃玩家），但当前无害。

- 严重程度：低（无害，模型自由决策）
- 建议：未来可添加推理时安全网

---

## 8. 关键瓶颈排序

### 🔴 瓶颈 1：无推理时搜索 (Critical Gap)

**现状**：模型仅通过单次前向传播决策。

**影响**：
- 不完全信息博弈中，单次前向传播无法最优推断隐藏信息
- 人类高手在关键决策点"读牌"本质上是搜索过程
- AlphaGo Zero 证明搜索能在网络已很强时仍带来巨大提升

**建议**：实现 ISMCE (Information Set Monte Carlo Evaluation)

**预计提升**：avg_ranking +0.10 ~ +0.30

### 🔴 瓶颈 2：信用分配不精确 (Significant Gap)

**现状**：
- 奖励仅在局结束时产生（score_diff）
- 一局平均 20+ 个决策点共享同一结果
- Target Network + TD(λ=0.95) 部分缓解，但 V(s') 仍在收敛

**影响**：
- 防守（不放铳）无直接奖励
- 高番构建需连续多步正确弃牌，每步只得到部分信用
- 杠/碰时机高度依赖后续走势

**建议**：
- 等 Target Network 充分收敛（200k+ 步）后评估
- 考虑引入轻量中间奖励（听牌进度、杠即时支付）
- 长期考虑 PPO/GAE 替代 DQN

**预计提升**：+0.05 ~ +0.15

### 🟡 瓶颈 3：对手建模不显式 (Moderate Gap)

**现状**：
- 观测含对手牌河 (111ch) + 花色倾向 (9ch)
- opp_wait AuxNet 预测对手听牌
- 但无"对手手牌信念分布"

**影响**：
- 模型无法回答"对手手里有几张 5万"
- AuxNet 是二值预测（听/不听），缺概率信息
- 防守决策依赖对隐藏信息的推断质量

**建议**：
- Oracle Guiding 是最佳解法（通过蒸馏间接传递）
- 或增加 AuxNet 输出维度（对手手牌概率分布预测）

**预计提升**：+0.05 ~ +0.10

### 🟡 瓶颈 4：Self-play 天花板 (Moderate Gap)

**现状**：avg_ranking ≈ 2.50 (对自己下 = 随机水平)

**影响**：
- 无外部强力对手提供进步压力
- 对手池都是自己的历史版本，策略空间有限

**建议**：
- Population-based Training (PBT)
- 或引入规则 AI 作为多样化对手

**预计提升**：+0.05 ~ +0.15

### 🟡 瓶颈 5：Oracle 梯度竞争 (Moderate Gap)

**现状**：5 个模型共享 Optimizer + clip_grad_norm_

**建议**：Oracle 独立 Optimizer，实现简单

**预计提升**：+0.02 ~ +0.05

### 🟢 瓶颈 6：网络容量 (Minor Gap)

**现状**：10.5M params 对 108 张牌基本足够

**建议**：其他瓶颈解决后，再考虑增大宽度 (192→256)

**预计提升**：+0.01 ~ +0.03

---

## 9. 综合评估与路线图

### 当前水平定位

```
初学者 ← ── ── ── 中级 ── ── ── 高级 ── ── ── 职业 ── ── ── 超人类
                                            ↑
                                         当前位置
                                      (avg_rank ≈ 2.50)
```

### 能否达到超人类水平？

**结论：架构上可行，但需要若干关键改进。**

**有利因素**：
1. 血战到底比日麻简单（108 张牌、无字牌、无立直/宝牌/流局）
2. 观测空间极其丰富（423ch + SP Table），信息量足够
3. Mortal 用类似架构在日麻上接近顶级人类 → 血战到底更容易突破
4. 番数/规则实现完整且正确
5. Oracle Guiding 方向正确（完美→不完全信息蒸馏）

**需要改进的因素**：
1. 无推理时搜索 = 无法深度推理
2. 信用分配在长序列上不精确
3. Self-play 已到达平衡点
4. Oracle 与 Student 梯度竞争

### 优先级路线图

| 优先级 | 改进项 | 实现难度 | 预计 rank 提升 | 阶段 |
|--------|--------|---------|---------------|------|
| **P1** | 稳定 Oracle → 开启蒸馏 | 中 | +0.05~0.15 | 当前进行中 |
| **P2** | Oracle 独立 Optimizer | 低 | +0.02~0.05 | Phase 7 |
| **P3** | 实现 ISMCE 推理时搜索 | 高 | +0.10~0.30 | Phase 8 |
| **P4** | 引入轻量中间奖励 | 中 | +0.03~0.08 | Phase 8 |
| **P5** | PBT / 多样化对手 | 高 | +0.05~0.15 | Phase 9 |
| **P6** | 增大网络宽度 (192→256) | 中 | +0.01~0.03 | Phase 9 |
| **P7** | PPO 替代 DQN | 极高 | 不确定 | 长期评估 |

### 达到超人类的预估

如果 P1-P5 全部完成，理论上 avg_ranking 可提升 0.25-0.75，从 2.50 降至 1.75-2.25 区间。配合充分训练步数（1M+ steps），达到超人类水平是可行的。

---

## 附录：关键文件索引

| 文件 | 说明 |
|------|------|
| `mortal/config.toml` | 训练配置 |
| `mortal/train.py` | 训练循环 |
| `mortal/model.py` | 神经网络定义 |
| `mortal/dataloader.py` | 数据加载与奖励计算 |
| `mortal/reward_calculator.py` | 奖励计算器 |
| `mortal/player.py` | Self-play 管理 |
| `mortal/engine.py` | 推理引擎 |
| `libblood/src/algo/agari.rs` | 番数计算 |
| `libblood/src/algo/point.rs` | 点数计算 |
| `libblood/src/state/obs_repr.rs` | 观测编码 |
| `libblood/src/dataset/invisible.rs` | Oracle 观测编码 |
| `libblood/src/agent/mortal.rs` | Rust 侧 Agent |
| `libblood/src/arena/board.rs` | 游戏引擎/杠支付 |
| `rules.md` | 完整规则文档 |
| `TRAINING_PLAN.md` | 训练计划与历史 |
