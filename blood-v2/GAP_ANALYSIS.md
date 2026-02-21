# blood-v2 深度缺口分析

> 逐模块对比 V2 当前实现与 V1（423ch 观测 + 17 番 + 10.5M 网络）及 rules.md 权威规则，  
> 按 **严重程度** 分为 🔴 Critical / 🟡 Major / 🟢 Minor。

---

## 目录

1. [番型系统 (win.rs)](#1-番型系统-winrs)
2. [游戏引擎 (game.rs)](#2-游戏引擎-gamers)
3. [特征工程 / 观测空间 (obs.rs)](#3-特征工程--观测空间-obsrs)
4. [奖励系统](#4-奖励系统)
5. [神经网络 (encoder.py)](#5-神经网络-encoderpy)
6. [环境封装 (pybind.rs / env.py)](#6-环境封装-pybindrs--envpy)
7. [优先级修复路线图](#7-优先级修复路线图)

---

## 1. 番型系统 (win.rs)

### 1.1 缺失番型

| 番型 | rules.md | V1 | V2 | 严重度 |
|------|----------|----|----|--------|
| 一条龙 (YiTiaoLong) | +1番：同花色 123+456+789 | ✅ | ❌ 未实现 | 🟡 |
| 杠上炮 (GangShangPao) | +1番：杠后弃牌被荣和 | ✅ | ❌ 未实现 | 🟡 |
| 抢杠 (Chankan) | +1番：别人加杠时抢胡 | ✅ | ❌ 未实现 | 🟡 |
| 天胡/地胡 (TianHu/DiHu) | 直接 5 番封顶 | ✅ | ❌ 未实现 | 🟢 |

**影响**：4 种番型缺失导致引擎计分不完整。一条龙在实战中出现频率中等，杠上炮/抢杠影响杠的策略价值评估。

### 1.2 逻辑错误

#### 🔴 金钩钓 (JinGouDiao) 番数错误

**当前**（win.rs:121-127）：

```rust
if num_melds == 4 && total_tiles(hand) == 2 {
    result.jin_gou_diao = true;
    result.toi_toi = false; // 互斥 ← 这是错的
    fan += 2;               // ← 应该是 +1
    if fan > 1 { fan -= 1; }
}
```

**rules.md 规定**：
- 金钩钓 = **+1 番**（不是 +2）
- 金钩钓与碰碰胡**不互斥**，常见合计：平胡1 + 碰碰胡1 + 金钩钓1 = 3番
- 当前代码强制 `toi_toi = false` 后又试图 `-1` 补偿，逻辑混乱

**正确逻辑**：金钩钓 +1，碰碰胡 +1，两者共存。

#### 🔴 门清 (Menzen) 条件判断错误

**当前**（win.rs:93-105）：

```rust
if is_menzen && !ctx.melds.is_empty() || (ctx.melds.is_empty() && ctx.is_tsumo) {
    if ctx.melds.iter().all(|m| matches!(m, MeldType::AnKan(_))) {
        result.menzen = true; fan += 1;
    }
} else if ctx.melds.is_empty() {
    result.menzen = true; fan += 1;
}
```

**问题**：
- 运算符优先级歧义：`&&` 优先于 `||`，条件实际为 `(is_menzen && !ctx.melds.is_empty()) || (ctx.melds.is_empty() && ctx.is_tsumo)`
- 当 `melds.is_empty() && !is_tsumo`（荣和，无副露）→ 走 else 分支 → 正确判 menzen
- 当 `melds.is_empty() && is_tsumo` → 走 if 分支 → 内部再检查 melds 全是 AnKan → 空列表 `.all()` 返回 true → 正确
- 但当只有 AnKan 且 `!is_tsumo`（荣和有暗杠）→ `is_menzen=true && !melds.is_empty()=true` → 进入 if → 正确
- **实际上偶然正确**，但代码逻辑极度混乱，应重写。

**正确定义**：门清 = 没有碰(Pon)、没有明杠(MinKan)、没有加杠(KaKan)。暗杠不破门清。与自摸/荣和无关。

#### 🔴 带幺九 (DaiYaoJiu) 实现过度简化

**当前**（win.rs:262-280）：

```rust
fn is_dai_yao_jiu(hand: &HandCounts, melds: &[MeldType]) -> bool {
    for m in melds { // 只检查刻子/杠的牌是否是1或9
        let rank = Suit::rank(m.tile());
        if rank != 1 && rank != 9 { return false; }
    }
    for (i, &c) in hand.iter().enumerate() { // 所有手牌必须是1或9
        if c > 0 {
            let rank = Suit::rank(i as u8);
            if rank != 1 && rank != 9 { return false; }
        }
    }
    true
}
```

**问题**：当前实现要求**所有牌**都是 1 或 9，但 rules.md 定义是"**每个组合**都包含 1 或 9"。例如：
- `1-2-3` 顺子是合法的（包含 1）
- `7-8-9` 顺子是合法的（包含 9）
- 当前代码会拒绝 `1-2-3` 因为 2 和 3 不是 terminal

这实际上把带幺九变成了"纯幺九"（只有 1 和 9），严重限制了合法范围。正确实现需要对手牌做面子分解后逐组检查。

### 1.3 缺失特性

| 特性 | 说明 | 严重度 |
|------|------|--------|
| 番型互斥表 | 带幺九↔断幺九互斥，七对↔碰碰胡互斥等缺少系统化处理 | 🟡 |
| 最优分解 | 同一手牌可能有多种分解，应取番数最高的 | 🟡 |
| FanConfig | V1 支持 7 个可配置开关（门清/断幺/带幺九/一条龙/夹心五/海底/天地胡），V2 无 | 🟢 |

---

## 2. 游戏引擎 (game.rs)

### 2.1 规则缺失

| 规则 | rules.md | V2 | 严重度 |
|------|----------|----|----|
| 杠即时支付 | 明杠2000, 暗杠2000×N, 及时雨补杠1000×N | ❌ 完全未实现 | 🔴 |
| 抢杠胡 (Chankan) | 加杠时其他人可胡 | ❌ TODO (game.rs:307) | 🟡 |
| 过手规则 | 过手加番可胡，同番不可再荣和 | ❌ 未实现 | 🟡 |
| 振听 (Furiten) | 临时振听 + 永久振听 | ❌ 未实现 | 🔴 |
| 初始分数 | 每人 60000 分 | ❌ 初始 0 分 | 🟡 |

#### 🔴 杠即时支付未实现

rules.md 明确规定杠产生独立于和牌的即时支付：

| 杠类型 | 支付方 | 金额 |
|--------|--------|------|
| 明杠 | 放杠者 → 杠者 | 2,000 |
| 暗杠 | 每位未和牌者 → 杠者 | 2,000 × N |
| 及时雨补杠 | 每位未和牌者 → 杠者 | 1,000 × N |
| 非及时雨补杠 | 无 | 0 |

当前 game.rs 中 `apply_self_check(AnKan)` 和 `apply_reactions(MinKan)` 只处理牌面操作，**没有任何分数变动**。这导致：
- 杠的策略价值被严重低估
- reward signal 缺少杠的即时收益/损失
- AI 无法学习正确的杠决策

#### 🔴 振听 (Furiten) 未实现

振听是麻将核心规则：
- **临时振听**：过手后在本巡内不能荣和
- **牌河振听**：自己打过的牌出现在听牌中时不能荣和

当前 `get_reaction_actions_for()` 完全不检查振听。这导致 AI 可能学到在非法情况下荣和。

### 2.2 代码 Bug

#### 🔴 process_win 中 winning_tile 错误

```rust
fn process_win(&mut self, winner: usize, loser: Option<usize>) {
    let ctx = WinContext {
        winning_tile: if let Some((_discarder, tile)) = self.last_discard {
            tile
        } else {
            0 // ← 自摸时 winning_tile 恒为 0（错误！）
        },
        ...
    };
```

自摸时 `last_discard` 为 `None` 或指向之前的弃牌，而不是刚摸到的牌。需要新增 `last_drawn_tile: Option<Tile>` 字段来追踪最后摸到的牌。

**影响**：夹心五判断依赖 `winning_tile`，自摸时永远判 tile=0（1万），导致夹心五在自摸场景下的判断完全错误。

#### 🟡 check_da_jiao 赔付金额简化

```rust
let penalty = 1000i32; // simplified: max hand value of tenpai hand
```

rules.md 规定：未听牌者赔付给听牌者该听牌者**手牌能和的最大番数对应分数**。当前固定 1000 分，应根据听牌者实际手牌计算最大可能番数。

#### 🟡 test_full_pass_cycle 测试失败

game.rs:271 的 `draw_tile()` 断言失败。`apply_reactions` 全部 pass 后调用 `draw_tile()`，但此时 phase 已被 `advance_to_next_player()` 改变。状态机转换有 bug。

---

## 3. 特征工程 / 观测空间 (obs.rs)

### V2 vs V1 通道对比

| 特征组 | V1 通道数 | V2 通道数 | 差值 | 严重度 |
|--------|----------|----------|------|--------|
| 手牌 | 5 (4 + last_tsumo) | 4 | -1 | 🟡 |
| 场况 | 10 (分数+排名+庄家) | 11 (diff+turn+dealer) | +1 | ✅ |
| 定缺 | 17 (含完成度/剩余) | 15 (缺完成度) | -2 | 🟡 |
| 局面状态 | 5 (振听/杠标记等) | 2 (progress+shanten) | -3 | 🔴 |
| 自家牌河 | 38 (时序编码) | 0 | **-38** | 🔴 |
| 对手牌河 | 111 (时序编码) | 12 (仅计数) | **-99** | 🔴 |
| 可见牌 | 53 (详细分类) | 4 (仅计数) | **-49** | 🔴 |
| 防守 | 9 (花色倾向) | 0 | **-9** | 🟡 |
| 导出特征 | 8 (menzen/acc等) | 0 | **-8** | 🟡 |
| 手牌分析 | 7 (waits/shanten) | 1 (shanten only) | **-6** | 🔴 |
| 动作上下文 | 11 (详细) | 1 (trigger tile) | **-10** | 🔴 |
| **SP Table** | **~100** | **0** | **-100** | 🔴🔴 |
| FanConfig | 7 | 0 | -7 | 🟢 |
| RTPA 预留 | 0 | 2 (空占位) | +2 | — |
| **合计** | **~423** | **67** | **-356** | |

### 3.1 最关键缺失：SP Table (~100ch)

V1 的 SP Table（Single Player Tables）是**最重要的特征**，被 SYSTEM_REVIEW.md 称为"关键优势"。它为每个弃牌候选计算：

| SP 通道 | 内容 | 作用 |
|---------|------|------|
| tenpai_probs × 28巡 | 每巡达到听牌的概率 | 选择最快听牌的弃牌 |
| win_probs × 28巡 | 每巡和牌的概率 | 评估和牌期望 |
| exp_values × 28巡 | 每巡期望收益 | 综合攻防决策 |
| required_tiles × 27 | 哪些牌能改善手牌 | 知道什么牌有用 |
| max_ev | 当前最大期望值 | 全局价值估计 |

**没有 SP Table 的后果**：
- 网络必须**从零学习**哪张牌该打、哪张牌有用 → 需要极大量训练
- 相当于让 AI 自己发明"牌理计算器" → V1 直接内嵌了
- 弃牌决策质量会显著下降

**建议**：在 `engine/` 中实现 SP 计算模块，在 obs 编码中加入 ~100 通道。

### 3.2 缺失：牌河时序编码

V2 只编码对手弃牌**计数**（每家 4 通道 = 弃了几张），完全丢失**时序信息**。

V1 编码的信息：
- 最近 18 张弃牌的**位置编码**（每张 2ch：哪张牌 + 是否摸切）
- 指数衰减概览（最近的牌权重高）

**为什么时序重要**：
- 对手先打 1万 后打 9万 vs 先打 9万 后打 1万 → 推断的定缺/听牌完全不同
- 碰后的第一张弃牌高度暴露手牌结构
- 早期弃牌 vs 后期弃牌的危险度不同

### 3.3 缺失：听牌/等待信息

V2 只有 1 个通道编码向听数的标量值。V1 有：
- **waits (1ch)**: 具体等待哪些牌（27 维 bool）
- **shanten (5ch)**: one-hot 精确编码 0-4
- **acceptance count**: 有效残枚总数（和牌概率的核心指标）

### 3.4 缺失：动作上下文

V2 在 Reaction 阶段只编码触发牌 (1ch)。V1 编码 (11ch)：
- 触发牌
- 弃牌候选（哪些牌可以打）
- 保持向听的弃牌
- 改善向听的弃牌
- 无条件听牌弃牌
- 碰/杠/和标志
- **当前 Ron 番数**（关键：AI 需要知道这把荣和值多少番来决定是否 pass）

### 3.5 缺失：防守特征

V2 完全没有防守相关特征：
- **对手花色倾向 (9ch)**：从弃牌比例推断定缺
- **Wall remaining per tile (1ch)**：每张具体牌剩余多少张（关键防守信息）
- **Forbidden tiles (1ch)**：振听标记
- **Menzen flag (1ch)**：是否门清

### 3.6 RTPA 通道是空占位

obs.rs:186-188 的 RTPA 2 通道从不写入值，浪费空间。

---

## 4. 奖励系统

### 4.1 当前问题

| 问题 | 说明 | 严重度 |
|------|------|--------|
| 杠支付不计入 reward | 引擎未实现杠即时支付，reward 完全缺少杠收益 | 🔴 |
| 终局 reward 非增量 | `get_rewards()` 返回累计分数而非增量Δ | 🟡 |
| 无中间 reward | 每次胡牌事件可产生 reward，但当前只有终局一次性结算 | 🟡 |
| 初始分数为 0 | 不影响 Δscore 但影响排名计算 | 🟢 |

### 4.2 reward 计算 bug

**pybind.rs 中的 reward 逻辑**：

```rust
let current_score = self.state.players[self.player_id].score;
let reward = (current_score - self.prev_score) as f32 / 16000.0;
self.prev_score = current_score;
```

这个增量计算本身是正确的。但因为杠即时支付未实现，所有杠产生的分数变动（最高 2000×3=6000 分/次暗杠）都不会出现在 reward 中。

### 4.3 缺失的 reward 信号

| 信号 | 说明 | 建议 |
|------|------|------|
| 杠即时支付 | 明杠/暗杠/补杠产生的即时分数 | 引擎实现后自动纳入 Δscore |
| 查花猪/查大叫赔付 | 终局惩罚已实现但可能金额不准 | 修正 da_jiao 赔付金额 |
| 热身期行为奖励 | 定缺清空+0.05，胡牌+0.1，放铳-0.05 | sf_blood/reward.py 已有框架但未集成到 env.py |

---

## 5. 神经网络 (encoder.py)

### 5.1 架构问题

| 问题 | 说明 | 严重度 |
|------|------|--------|
| 参数量偏小 | ~3.5M（V1: 10.5M, Suphx: 60M）| 🟡 |
| 无辅助任务头 | V1 有 AuxNet 预测对手听牌/定缺/下一张排 | 🟡 |
| GlobalAvgPool 丢信息 | 每花色 9 张取平均 → 丢失哪个 rank 重要 | 🟡 |
| 无记忆机制 | 无 LSTM/GRU，无法建模对局历史 | 🟢 |
| 共享 Encoder | actor_critic_share_weights=True 可能导致目标冲突 | 🟢 |

### 5.2 缺失：辅助任务 (Auxiliary Heads)

V1 的 AuxNet 训练以下辅助任务，强化主干表征学习：

| 辅助任务 | 输出维度 | 作用 |
|---------|---------|------|
| 对手听牌预测 (opp_wait) | 3×27=81 | 防守核心：预测对手在等什么牌 |
| 对手定缺预测 (ding_que) | 3×3=9 | 推断对手缺什么门 |
| 下一张排预测 (next_rank) | — | 预测对手可能打什么 |

研究表明辅助任务可提升主任务表现 5-15%。

### 5.3 GlobalAvgPool 的信息损失

当前 Encoder 最终做 `mean(dim=2)` 对每花色 9 个位置取均值：

```python
for i in range(3):
    pools.append(x[:, :, start:end].mean(dim=2))
```

这丢失了"哪个 rank 特别重要"的信息。例如无法区分"有很多 5 万"和"有很多 1 万"。

**建议**：保留 Flatten（256×27=6912）或使用 attention pooling 代替 mean pool。

### 5.4 Action Masking 集成

Sample Factory 支持 action masking，但当前 `BloodMahjongEnv.observation_space` 中 mask 的 dtype 是 `float32`：

```python
"mask": spaces.Box(low=0.0, high=1.0, shape=(38,), dtype=np.float32),
```

SF 期望 mask key 使用 `bool` 类型或在自定义 Encoder 中手动处理。需确认 SF 的 mask 机制是否正确被触发。

---

## 6. 环境封装 (pybind.rs / env.py)

### 6.1 env.py 问题

| 问题 | 说明 | 严重度 |
|------|------|--------|
| opponent 决策同步 | step() 中其他玩家用规则 Bot，但多策略 self-play 时应由 SF 控制 | 🟡 |
| info 缺少 oracle_obs | Oracle 蒸馏需要在 info 中传递 oracle obs | 🟡 |
| 无花色增强集成 | augment.py 存在但未在 env 中使用 | 🟢 |
| reward shaping 未集成 | reward.py 存在但 env.step() 直接返回引擎 reward | 🟢 |

### 6.2 多智能体 Self-play 架构

当前设计为"单玩家视角 + 内置 Bot"，这在热身期 OK，但进入竞争期后需要让 SF 的多个策略在同一桌对弈。当前架构不支持这个——其他 3 个玩家的决策在 Rust 端硬编码为 Rule Bot。

**需要两种模式**：
1. **Bot 模式**（Phase C）：当前模式，3 对手是 Rust Rule Bot
2. **Multi-agent 模式**（Phase D+）：4 个 env 共享一个底层 Game，各控制 1 个玩家，通过 SF 多策略分配

---

## 7. 优先级修复路线图

### 🔴 P0: Must Fix (阻塞训练)

| # | 缺口 | 影响 | 工作量 |
|---|------|------|--------|
| 1 | **杠即时支付** (game.rs) | reward 完全缺少杠收益，AI 无法学习杠策略 | 2h |
| 2 | **winning_tile 自摸 bug** (game.rs:410) | 自摸时番型计算全错（夹心五/杠上花等） | 1h |
| 3 | **振听** (game.rs) | 无振听 = 可以非法荣和 | 4h |
| 4 | **金钩钓番数** (win.rs:121) | +2→+1，去掉碰碰胡互斥 | 0.5h |
| 5 | **带幺九过度简化** (win.rs:262) | 当前实现是纯幺九，严重缩小合法范围 | 3h |
| 6 | **门清逻辑重写** (win.rs:93) | 当前偶然正确但不可维护 | 1h |
| 7 | **FSM 状态转换 bug** (test_full_pass_cycle) | 导致牌局在特定情况下 panic | 2h |

### 🟡 P1: Important (影响训练质量)

| # | 缺口 | 影响 | 工作量 |
|---|------|------|--------|
| 8 | **牌河时序编码** (obs.rs) | 丢失对手行为的关键时序信息 | 4h |
| 9 | **听牌/waits 编码** (obs.rs) | 网络不知道自己在等什么 | 1h |
| 10 | **动作上下文** (obs.rs) | 网络不知道 Ron 值多少番 | 2h |
| 11 | **Wall remaining per tile** (obs.rs) | 无法判断某张牌还剩几张 | 1h |
| 12 | **防守特征** (obs.rs) | 对手花色倾向、menzen 等 | 2h |
| 13 | **一条龙 + 杠上炮 + 抢杠番型** | 3 种中频番型缺失 | 4h |
| 14 | **辅助任务头** (encoder.py) | 对手听牌预测等改善表征 | 4h |
| 15 | **da_jiao 赔付金额** (game.rs:534) | 固定 1000 应改为实际手牌计算 | 2h |
| 16 | **过手规则** (game.rs) | 过手加番可胡机制 | 3h |
| 17 | **last_tsumo 编码** (obs.rs) | 网络不知道自己刚摸了什么 | 0.5h |

### 🟢 P2: Enhancement (提升上限)

| # | 缺口 | 影响 | 工作量 |
|---|------|------|--------|
| 18 | **SP Table** (engine/) | V1 的核心优势，~100 通道 | 2 周 |
| 19 | **Multi-agent env** | 竞争期 self-play 必须 | 1 周 |
| 20 | **Attention pooling** 替代 GlobalAvgPool | 保留 rank 位置信息 | 2h |
| 21 | **初始分数 60000** | 影响排名和收益感知 | 0.5h |
| 22 | **天胡/地胡** | 极罕见但满番 | 1h |
| 23 | **FanConfig 多规则** | 提升泛化性 | 2h |
| 24 | **reward shaping 集成** | 热身期辅助奖励 | 1h |
| 25 | **Oracle obs 传递** | 蒸馏阶段必须 | 1h |

---

## 修复后的目标通道数

当前 67ch → 修复后预估 **~180ch**（不含 SP Table）或 **~280ch**（含 SP Table）：

| 特征组 | 当前 | 修复后 | 增量 |
|--------|------|--------|------|
| 手牌 | 4 | 5 (+last_tsumo) | +1 |
| 场况 | 11 | 11 | 0 |
| 定缺 | 15 | 17 (+完成度/剩余) | +2 |
| 局面状态 | 2 | 5 (+振听/杠标记/menzen) | +3 |
| 自家牌河 | 0 | 38 (时序编码) | +38 |
| 对手牌河 | 12 | 39 (时序+衰减) | +27 |
| 可见牌 | 4 | 10 (+per-tile remaining) | +6 |
| 防守 | 0 | 9 (花色倾向) | +9 |
| 导出特征 | 0 | 8 | +8 |
| 手牌分析 | 1 | 7 (waits+shanten+acceptance) | +6 |
| 动作上下文 | 1 | 11 | +10 |
| SP Table (可选) | 0 | ~100 | +100 |
| **合计** | **67** | **~180 / ~280** | |

---

> **结论**：V2 引擎的骨架（FSM/牌型/洗牌）基本正确，但 **番型计算有 3 个逻辑错误，杠支付完全缺失，观测空间仅为 V1 的 16%**。在进入 PPO 训练前，P0 的 7 项必须全部修复，否则 AI 在错误规则下训练会产生系统性偏差。
