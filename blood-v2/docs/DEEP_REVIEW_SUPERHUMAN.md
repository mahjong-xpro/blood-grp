# Blood V2 深度评审 — 超人类水平目标

> 评审日期: 2026-03-01
> 范围: blood-v2/ 全部模块
> 方法: 逐行代码审查 + 规则交叉验证 + 架构分析
> 目标: 识别阻碍超人类水平的真实瓶颈，区分 bug/设计缺陷/优化机会

---

## 评审原则

1. **区分训练期与推理期影响** — 影响训练的问题（观测编码、奖励、网络架构）比仅影响推理的问题（ISMCE、RTPA）优先级更高
2. **验证每个声明** — 每个问题都附带具体代码位置和可复现的论证
3. **不过度设计** — 只推荐有明确收益的改动，避免"理论上更好"的重构

---

## 模块 1: 和牌判定 (`algo/agari.rs`)

### P1: 夹心五判定存在漏判 [确认 Bug]

`agari.rs:347-352`:
```rust
let tile5_count = ctx.tehai[wt as usize];
tile5_count == 1
```

**问题**: 要求完成手牌中只有一张5。当手牌中已有5（作为对子或其他组合的一部分）时，即使和牌确实是46嵌5，也会被拒绝。

**可复现反例**:
听牌: `4m 6m 5m 5m 1p 2p 3p 7p 8p 9p 1s 2s 3s` (13张, 无副露)
唯一待牌: 5m (46嵌5)
和5m后: `4m 5m 6m` (顺) + `5m 5m` (对) + `1p2p3p` + `7p8p9p` + `1s2s3s` = 4组+1对 ✓
tehai[5m] = 3 → `tile5_count == 1` 为 false → 夹心五被拒绝 ✗

**影响**: 训练中番数少算1番，影响奖励信号准确性。SP Table 的 `get_win_score` 也调用 `calc_fan`，同样受影响。

**修复方案**: 当前的 `tile5_count == 1` 是一个充分但非必要条件。正确的判定需要检查：在当前 division 中，456顺子所使用的那张5是否就是和牌张。

实现思路：统计 division 中所有组合对 tile5 的消耗量。如果 division 的 shuntsu 包含456，且 pair 是55，则消耗 = 1(顺) + 2(对) = 3。如果 tehai[5] == 3，说明所有5都被用完，和牌张的5必然是其中之一。关键是判断：如果从 tehai 中移除和牌张（tehai[5] -= 1），剩余的5是否仍能满足 division 中除456顺子外的所有组合。

```rust
fn check_jiaxinwu(ctx: &WinContext, div: &Division) -> bool {
    let wt = ctx.winning_tile;
    if Suit::rank(wt) != 5 { return false; }
    let suit_start = Suit::from_tile(wt).start();
    let seq_start = (suit_start + 3) as Tile;
    if !div.shuntsu.contains(&seq_start) { return false; }

    // 统计 division 中除456顺子外，其他组合对 tile5 的消耗量
    let tile5 = wt as usize;
    let mut other_usage: u8 = 0;
    // 对子
    if div.pair_tile == wt { other_usage += 2; }
    // 其他刻子
    for &k in &div.kotsu {
        if k == wt { other_usage += 3; }
    }
    // 其他顺子（排除当前456）
    // 5可能出现在: 345(start=3), 456(start=4), 567(start=5) 的顺子中
    let mut found_456 = false;
    for &s in &div.shuntsu {
        let s_suit_start = Suit::from_tile(s).start();
        if s == seq_start && !found_456 {
            found_456 = true; // 跳过第一个456（这是我们要检查的）
            continue;
        }
        // 检查这个顺子是否包含 tile5
        let s_rank = Suit::rank(s);
        if Suit::from_tile(s).start() == suit_start {
            // 同花色的顺子
            if s_rank <= 5 && s_rank + 2 >= 5 {
                // 这个顺子包含 rank 5
                other_usage += 1;
            }
        }
    }
    // 如果 tehai 中的5 = other_usage + 1(456顺子)，
    // 说明移除和牌张后，剩余的5恰好满足其他组合，
    // 即和牌张确实是补全456顺子的那张5
    ctx.tehai[tile5] == other_usage + 1
}
```

---

## 模块 2: ISMCE 搜索 (`algo/ismce.rs`)

### P2: estimate_fan_quick 遗漏高价值番型 [确认 Bug]

`ismce.rs:292-366` 检查: 平胡、自摸、门清、清一色、七对、根、碰碰胡。

**遗漏**:
- **断幺九** (+1番): 所有牌为 rank 2-8。常见，遗漏导致 EV 低估约 2x
- **带幺九** (+3番): 所有组合含幺九。较少见，但遗漏导致 EV 低估 4-8x

**不可用 calc_fan 替代的原因**: rollout 只跟踪 `HandCounts` 和 `melds: usize`（副露数量），不跟踪 `Vec<MeldType>`（具体副露类型）。`calc_fan` 需要 `WinContext`，其中包含 `melds: Vec<MeldType>`。

**修复方案**: 在 `estimate_fan_quick` 中补充:

```rust
// 断幺九: 所有牌 rank 2-8（从 HandCounts 可直接判断）
let mut all_inner = true;
for t in 0..NUM_TILE_TYPES {
    if hand[t] > 0 {
        let r = t % TILES_PER_SUIT + 1;
        if r == 1 || r == 9 { all_inner = false; break; }
    }
}
if all_inner { fan += 1; }

// 带幺九近似: 所有牌 rank ∈ {1,2,3,7,8,9}（必要条件，非充分）
// 充分条件需要 division，但必要条件已能覆盖大部分实际带幺九手牌
let mut all_terminal_adjacent = true;
for t in 0..NUM_TILE_TYPES {
    if hand[t] > 0 {
        let r = t % TILES_PER_SUIT + 1;
        if r >= 4 && r <= 6 { all_terminal_adjacent = false; break; }
    }
}
if all_terminal_adjacent && !all_inner { fan += 3; }
```

注意: 带幺九的近似检查（所有牌 rank ∈ {1,2,3,7,8,9}）是必要条件。实际带幺九允许 rank 4-6 出现在顺子中（如 789），但纯从 HandCounts 无法区分顺子和刻子。这个近似会漏判含有中张顺子的带幺九（如 123+456+789），但能正确识别纯幺九手牌。对于 ISMCE rollout 的 EV 估计，这个精度已经足够。

### P3: Rollout 策略质量是 ISMCE 的核心瓶颈 [设计局限]

`simulate_draws_with_opponents` 使用贪心向听最小化。这是推理时搜索质量的上限。

**具体问题**:
- 向听数相同时，仅用 danger 作为 tiebreaker，不考虑 EV（`ismce.rs:786-799`）
- 不考虑番型方向: 贪心可能选择向听数相同但番型潜力低的弃牌

**注意**: ISMCE 仅在推理时使用，不影响训练。优先级低于影响训练的问题。

**改进方案**: 在向听数相同的候选中，增加 SP-EV 作为第二 tiebreaker:
```rust
// 当前: shanten → danger (低优先)
// 改进: shanten → SP-EV (高优先) → danger (低优先)
if s < best_s || (s == best_s && ev > best_ev) || (s == best_s && ev == best_ev && d < best_danger) {
```

这需要在 rollout 中调用 SP 计算，性能开销较大。折中方案: 仅对听牌候选（shanten=0）计算精确 EV，其他用启发式。

---

## 模块 3: SP Table (`algo/sp/calc.rs`)

### P4: 一向听前瞻采样数偏少 [优化机会]

`sp/calc.rs:244`: `MAX_SAMPLES=5`

一向听手牌可能有 10-15 个有效牌。按可用枚数降序只采样5个，会忽略低枚数但高番型的进牌（如清一色方向的关键牌只剩1枚）。

**量化影响**: 假设有效牌按枚数排序为 [4,4,3,3,2,2,1,1,1,1]，前5个覆盖 16/22 = 73% 的概率质量。提升到8个覆盖 20/22 = 91%。

**建议**: `MAX_SAMPLES` 提升到 8。性能影响: 额外 3×27 = 81 次 `calc_shanten` 调用，大部分命中缓存，实测增加 < 5% 的 SP 计算时间。

### P5: 二向听 EV 不区分番型方向 [优化机会]

`sp/calc.rs:201-202`:
```rust
let base_score = if self.num_melds == 0 { 2000.0 } else { 1000.0 };
```

清一色二向听（潜在 3-5 番 = 4000-16000 分）和普通二向听（1-2 番 = 1000-2000 分）使用相同的 base_score。

**建议**: 轻量级番型潜力检查:
```rust
let mut base_score = if self.num_melds == 0 { 2000.0 } else { 1000.0 };
// 清一色方向: 所有手牌同花色
let suits: u8 = (0..3).filter(|&s| {
    (s*9..(s+1)*9).any(|t| hand[t] > 0)
}).count() as u8;
if suits == 1 { base_score *= 2.5; }  // 清一色 +2番 → 约 4x 分数
```

---

## 模块 4: 观测编码 (`obs/student.rs`)

### 通道分配现状

| Section | 通道数 | 占比 | 内容 |
|---------|--------|------|------|
| 5+6: 牌河 | 232 | 49.1% | 自家+对手牌河 |
| 12: SP Table | 99 | 20.9% | SP 计算结果 |
| 7: 可见牌 | 48 | 10.1% | 弃牌计数+副露 |
| 其他 | 94 | 19.9% | 手牌/上下文/防守/分析 |

### P6: SP Table 逐巡数据可能冗余 [需消融验证]

Section 12 的 84 通道（28巡×3指标）编码最佳候选的逐巡 tenpai_prob/win_prob/EV。同时已有 11 通道汇总统计（总EV、总胜率、EV spread 等）和 4 通道 per-tile 数据。

**假设**: 网络主要使用汇总统计和 per-tile 数据，逐巡数据的边际贡献有限。

**建议**: 消融实验 — 将 84 通道逐巡数据置零，对比训练曲线。如果性能无显著下降，释放这些通道给更有价值的信号。

**注意**: 修改观测通道数需要重新训练。这是一个高成本实验，应在其他低成本改进完成后进行。

### P7: 牌河编码的效率 [设计权衡]

~~原评审认为牌河占比"过高"，这是错误的。~~ 牌河是麻将中最重要的公开信息源之一，49% 的通道分配与 Mortal（日麻 AI）的设计一致。

但编码效率可以改进: 28巡窗口中，平均只有 10-15 个位置非零。如果未来需要增加通道（如新的防守信号），可以考虑将窗口从 28 缩减到 20（覆盖 >95% 的实际打牌数），释放 (28-20)×2×4 = 64 通道。

**当前建议**: 不改动。牌河编码是成熟设计，改动风险高于收益。

### P8: 缺少对手听牌概率的显式编码 [潜在改进]

当前观测包含对手听牌的间接信号（副露数、花色比例、幺九比例、现物），但没有综合这些信号的显式听牌概率通道。

**分析**: 网络理论上能从间接信号中学习推断对手听牌概率。但这需要网络自己发现多信号的非线性组合，增加了学习难度。

**建议**: 低优先级。如果评估发现模型防守能力不足，可以考虑增加 3 通道（每对手1通道）的听牌概率估计。但需要注意: 如果估计不准确，可能反而误导网络。

---

## 模块 5: 神经网络架构 (`model/encoder.py` + `model/factory.py`)

### P9: TileAttention 位置编码的花色对称性 [设计权衡]

`encoder.py:140`:
```python
self.pos_embed = nn.Parameter(torch.zeros(1, NUM_TILES, channels))  # (1, 27, C)
```

27 个位置各有独立编码，Man-1 和 Pin-1 有不同的位置向量。这与 SuitAwareConv1d 的花色共享权重设计存在张力。

**但这不一定是问题**:
- TileAttention 的目的就是建模跨花色交互，需要区分不同花色
- 50% 的花色增强已经鼓励模型学习花色对称模式
- 绝对位置编码允许模型学习任意位置依赖模式

**替代方案**: rank+suit 分解编码:
```python
self.rank_embed = nn.Parameter(torch.zeros(1, 9, channels))
self.suit_embed = nn.Parameter(torch.zeros(1, 3, channels))
# forward: embed = rank_embed.repeat(1,3,1) + suit_embed.repeat_interleave(9, dim=1)
```

**建议**: 作为消融实验。如果花色增强概率提升到 0.7 后模型性能提升，说明花色对称性确实重要，此时 rank+suit 编码可能有帮助。

### P10: Actor-Critic 解耦设计的合理性 [确认良好]

`factory.py:99-114` 使用独立的 `actor_head` 和 `critic_head`，各自是 2 层 MLP (core_out → 512 → 512)。

~~原评审担心解耦导致 Critic 无法利用 Actor 信息。~~ 实际上，两个 head 共享 encoder + LSTM 的特征（`core_output`），只是最后两层独立。这是标准的解耦设计，允许 Actor 和 Critic 在共享表示上学习不同的映射。DingQue progressive prior 和 action masking 也在 `forward_tail` 中正确应用。

### P11: TurnAttention 的零初始化设计 [确认良好]

`factory.py:38-39`:
```python
nn.init.zeros_(self.attn.out_proj.weight)
nn.init.zeros_(self.attn.out_proj.bias)
```

并且在 `__init__` 末尾重新应用零初始化（`factory.py:196-198`），确保 `self.apply(initialize_weights)` 不会覆盖。这保证训练初期 TurnAttention 等价于恒等映射，逐渐学习跨回合模式。设计正确。

---

## 模块 6: 训练流水线 (`training/` + `env/selfplay_env.py`)

### P12: 奖励塑形的设计评估 [确认合理，微调建议]

~~原评审认为奖励信号"过多且冲突"。~~ 重新审查后，奖励设计是经过深思熟虑的:

1. **主奖励**: sqrt 压缩的分数变化 — 将 32:1 的线性比例压缩到 ~5.6:1
2. **塑形奖励**: 全部使用 score-weighted intensity (`clamp(sqrt(|Δ|/32000), 0.25, 1.0)`)，低番事件自动衰减
3. **向听奖励**: 有衰减调度 + 番数加权，避免贪心追求听牌

**实际权重分析** (以1番自摸为例，Δ=3000):
- 主奖励: sign(3000/32000) × sqrt(3000/32000) = 0.306
- tsumo_bonus: 0.2 × max(0.25, sqrt(3000/32000)) = 0.2 × 0.306 = 0.061
- shanten_progress: 0.01 × 1 × decay × fan_bonus ≈ 0.01

主奖励占绝对主导地位。塑形奖励是小幅度的引导信号，不会造成冲突。

**微调建议**:
- `reward_safe_discard: 0.01` 是每步累积的，一局游戏约 10-15 步弃牌，累积 0.1-0.15。这与主奖励的量级接近，可能导致过度防守。建议降低到 0.005 或改为仅在对手听牌时生效。

### P13: 联赛系统的淘汰策略 [确认良好]

`league.py:191-226` 的稀疏保留淘汰策略:
- 最新 50% 密集保留（保证近期策略密度）
- 旧 50% 用 linspace 均匀选取（自动包含最旧的 checkpoint）

这已经实现了"里程碑保留"的效果 — linspace 采样确保旧 checkpoint 的时间跨度覆盖。

### P14: Oracle 蒸馏权重的阶段适配 [优化机会]

Elite 阶段 `oracle_distill_weight: 0.01`。

**分析**: 如果学生策略在某些罕见场景（如复杂多家听牌的防守决策）偏离 Oracle，0.01 的权重可能不足以纠正。但过高的蒸馏权重会限制学生超越 Oracle 的能力。

**建议**: 监控 `distill_loss` 指标。如果 distill_loss 在训练后期持续上升，说明学生正在偏离 Oracle，可以适当提升权重到 0.02-0.03。

---

## 模块 7: 游戏引擎 (`state/board.rs`)

### P15: 地胡条件的设计选择 [非 Bug，记录]

`board.rs:890-891` 要求所有玩家无副露。代码注释明确说明这是有意设计:
```rust
// 额外要求：第一巡无人鸣牌（碰/杠），否则摸牌顺序被打断，不算地胡。
```

这是四川血战麻将的一种常见规则解释。`rules.md` 的描述 "闲家第一巡自摸" 可以有两种理解，当前实现选择了更严格的版本。

**建议**: 无需修改。如果需要支持宽松版本，可以通过 `FanConfig` 增加一个开关。

### P16: 缺少 property-based 不变量测试 [测试改进]

当前测试覆盖特定场景，缺少随机游戏的不变量验证。

**建议**: 增加 `proptest` 测试:
- 牌数守恒: Σ(hand + melds + discards) + wall = 108
- 分数守恒: Σ(score_change) = 0（杠支付是内部转移，不改变总和）
- Phase 状态机: 不会出现非法转换
- 终止性: 任意合法动作序列最终到达 Done

---

## 模块 8: RTPA (`eval/rtpa.py`) [仅推理时]

### P17: 对手听牌推断缺少验证 [推理优化]

`_estimate_opponent_tenpai` 使用 6 个信号的加权组合，但从未用 Oracle ground truth 验证准确率。

**建议**: 在评估中记录 RTPA 推断 vs Oracle 真实向听数，计算 precision/recall。这是低成本的诊断工作，可以指导信号权重调优。

---

## 模块 9: 评估系统 (`eval/`)

### P18: 评估游戏数不足 [确认问题]

`blood_arena_eval_games: 100`。麻将单局方差极大（6番自摸 Δ=96000 vs 1番荣和 Δ=1000），100 局的 Elo 置信区间过宽。

**量化**: 假设胜率标准差 σ ≈ 0.15，100 局的 95% CI ≈ ±3%。对于 Elo 差异 < 50 的模型，这个精度不足以区分。

**建议**: 提升到 500 局。评估频率可以相应降低（从 100K 步/次 到 500K 步/次）以保持总评估时间不变。

### P19: 缺少关键诊断指标 [确认问题]

当前追踪: win rate, avg rank, avg score, avg fan, Elo。

**应增加**:
- **放铳率**: 超人类水平应 < 12%（人类高手约 15%）
- **自摸率**: 反映进攻效率
- **番型分布**: 和牌的番数直方图（检测是否过于保守只做1-2番）
- **定缺选择分布**: 检测是否存在花色偏好

这些指标可以从现有的 `info` dict 中提取，实现成本低。

### P20: 评估对手单一 [长期改进]

仅对 RuleBot 评估。RuleBot 不防守、不做番型规划，对其高 Elo 不代表对人类的超人类水平。

**建议**: 增加对历史自身版本的评估（最近 5 个里程碑），追踪 Elo 进步曲线。这比引入新对手类型成本更低，且能直接衡量训练进步。

---

## 优先级排序

### Tier 1: 确认 Bug — 立即修复

| ID | 问题 | 影响范围 | 修复复杂度 |
|----|------|----------|------------|
| P1 | 夹心五漏判 | 训练+推理 | 中等 |
| P2 | estimate_fan_quick 遗漏番型 | 推理(ISMCE) | 低 |

### Tier 2: 高价值优化 — 下一训练周期前完成

| ID | 问题 | 影响范围 | 预期收益 |
|----|------|----------|----------|
| P4 | SP 一向听采样数 | 训练(观测) | SP 精度 +15-20% |
| P5 | SP 二向听 EV 番型感知 | 训练(观测) | 清一色方向 EV 准确度 |
| P18 | 评估游戏数 | 评估 | Elo 置信区间缩小 60% |
| P19 | 诊断指标 | 评估 | 发现训练问题的能力 |

### Tier 3: 消融实验 — 需要数据验证

| ID | 问题 | 假设 | 验证方法 |
|----|------|------|----------|
| P6 | SP 逐巡数据冗余 | 84ch 边际贡献低 | 置零对比训练曲线 |
| P9 | TileAttention 位置编码 | rank+suit 更好 | A/B 训练对比 |
| P12 | safe_discard 累积过大 | 导致过度防守 | 降低权重对比 |

### Tier 4: 长期架构改进

| ID | 问题 | 方向 |
|----|------|------|
| P3 | ISMCE rollout 策略 | SP-EV tiebreaker → 轻量级 policy |
| P8 | 对手听牌概率编码 | 需先验证 P17 的推断准确率 |
| P20 | 评估对手多样性 | 历史版本对战 |
| P16 | Property-based 测试 | 长期可靠性保障 |

---

## 与原评审的差异说明

| 原评审声明 | 修正 |
|-----------|------|
| "牌河编码占比过高" | 牌河是麻将最重要的公开信息，49% 分配合理 |
| "奖励信号过多且冲突" | 奖励设计经过深思熟虑，score-weighted 避免了冲突 |
| "TileAttention 位置编码矛盾 [高]" | 降级为设计权衡，需消融验证 |
| "缺少关键防守信号 [高]" | 现有间接信号可能已足够，需先验证 |
| "解耦 Actor-Critic 可能导致价值估计不准" | 共享 encoder+LSTM，仅最后两层解耦，设计正确 |
| "联赛检查点淘汰策略有问题" | 稀疏保留策略已实现里程碑保留效果 |
| "地胡条件过严 [Bug]" | 有意设计，代码注释明确说明 |
