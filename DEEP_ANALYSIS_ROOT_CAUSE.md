# 深度分析：定缺相关 Bug 根因与非定缺潜在问题

> 生成时间：2026-01-28  
> 目的：解释为何 bug 集中在定缺、定缺的架构根因、以及非定缺模块的潜在问题

---

## 一、为何 Bug 几乎全部围绕定缺？

### 1.1 根本原因：定缺是一个「插入阶段」

游戏主循环原本是**单阶段**设计：

```
发牌(haipai) → 摸牌(Tsumo) → 打牌/鸣牌/和牌(Dahai/Pon/Kan/Hora) → 循环直到结束
```

定缺规则要求在**打牌之前**增加一个**独立阶段**：

```
发牌(haipai) → 【定缺选择阶段】→ 摸牌(Tsumo) → 打牌/鸣牌/和牌 → ...
```

因此定缺不是「多一个事件类型」，而是**多了一整段状态机**。这段状态机被**硬塞进**原有 `step()` / `poll()` / `commit()` 流程，导致：

- **谁需要行动** 的判定出现两套逻辑（见下节）
- **反应校验** 的时机和内容在定缺阶段与正常阶段不一致
- **阶段切换**（所有玩家选完定缺 → 第一次摸牌）在一次 `step()` 里完成，容易与上层 `poll/commit` 的假设冲突

所以 bug 集中在定缺，是因为**定缺是唯一一个「多出来的阶段」**，和原有架构的接缝最多。

### 1.2 双源「谁需要行动」导致分支泛滥

| 阶段       | 谁需要行动？           | 数据来源                          |
|------------|------------------------|-----------------------------------|
| 定缺阶段   | 还没选定缺的玩家       | `BoardState::ding_que_selected`   |
| 正常游戏   | 能打牌/鸣牌/和牌的玩家 | `PlayerState::last_cans().can_act()` |

因此 **Game 层** 必须写成分支：

```rust
// game.rs poll() 和 commit()
let needs_reaction = if self.board.is_ding_que_phase() {
    !self.board.ding_que_selected(player_id)
} else {
    state.last_cans().can_act()
};
```

带来的问题：

- 任何只依赖 `can_act()` 的逻辑（如观测编码、统计、日志）在定缺阶段都会错，除非再写一遍「如果在定缺阶段则看 ding_que_selected」。
- `PlayerState` 不知道「当前是不是定缺阶段」，它只有 `can_ding_que`。所以「能不能打牌」在定缺阶段本应由 Board 的 phase 决定，但校验却在 `PlayerState::validate_reaction()` 里用 `can_discard` 做，容易产生**未选定缺却能打牌**的漏洞（或相反，误拒合法动作）。

结论：**定缺在「谁需要行动」上引入了第二套数据源，且没有统一抽象**，导致所有依赖「当前该谁动」的代码都要特殊处理定缺，bug 面大。

### 1.3 step() 的分段结构把定缺挂在入口处

`board.rs` 的 `step(reactions)` 大致结构是：

1. `agari_count >= 3` → 直接 `Poll::End`
2. `tiles_left == 56 && !ding_que_phase` → `haipai()`，返回 `InGame`
3. **`ding_que_phase`** → 只处理定缺，**不**做 `validate_reaction`，直接返回 `InGame` 或继续
4. **之后**才 `validate_reaction`，再处理 Hora / None / Dahai / Pon / Kan 等

因此：

- 定缺阶段**不会**对 reactions 做 `validate_reaction`。若 Agent 在定缺阶段误发 `Dahai`，不会在这里报错，而是被当成「非 DingQue」走自动选定缺逻辑。
- 一旦**所有玩家都选完定缺**，同一轮 `step()` 里会：清掉 `ding_que_phase`、发第一次 Tsumo、返回 `InGame`。上层下一轮用「对第一次 Tsumo 的反应」再调 `step()`。若上层或 Agent 仍以为还在定缺阶段，就会产生**状态理解不一致**（你之前报告里的竞态/时序问题都与此有关）。

也就是说：**定缺既绕过了统一校验，又把阶段切换和第一次摸牌绑在同一次 step 里**，这是定缺相关 bug 的集中点。

### 1.4 定缺在代码中的扩散范围

定缺相关状态/逻辑分布在多个模块（grep 统计约 250+ 处引用）：

| 模块 | 作用 |
|------|------|
| `arena/board.rs` | `ding_que_phase`、`ding_que_selected`、step 内定缺分支、流局花猪/查大叫 |
| `arena/game.rs` | poll/commit 里「谁需要 reaction」的定缺分支 |
| `state/update.rs` | `can_ding_que`、`ding_que()`、打牌/听牌/杠与定缺的联动 |
| `state/action.rs` | 打牌前的定缺校验、未选定缺时是否允许打牌（当前为注释掉的 ensure） |
| `state/agent_helper.rs` | 可打牌候选与定缺规则 |
| `state/obs_repr.rs` | 定缺相关特征、`tiles_left==56 && ding_que.is_none()` 的 can_ding_que 推断 |
| `algo/agari.rs` | 和牌时是否还捏着定缺花色（花猪） |
| `algo/shanten.rs` | 听牌计算考虑定缺 |
| `algo/sp/*` | SP 与定缺 | 

任何一处对「当前阶段」或「定缺是否已选」的理解不一致，都会变成 bug；所以**从现象上看**，bug 几乎都围绕定缺。

---

## 二、定缺问题的架构根因归纳

1. **阶段单一数据源缺失**  
   没有「当前阶段」的单一权威（例如 enum: `DingQue | NormalPlay`），Board 用 `ding_que_phase`，Player 用 `can_ding_que`/`ding_que`，两处可能不同步。

2. **「谁需要行动」未抽象**  
   Game 层用 if 分支拼出 `needs_reaction`，而不是由 Board/State 提供统一的「当前需要反应的玩家集合」，导致分支重复、易错。

3. **校验与阶段绑定不清晰**  
   定缺阶段应只接受 `DingQue`（或显式「跳过」），其他事件应统一在此处拒绝或转换，而不是跳过 `validate_reaction` 再在别处用 `ensure` 崩掉。

4. **阶段切换原子性不足**  
   「所有人选完定缺 → 关 phase → 第一次 Tsumo」在同一 `step()` 内完成，但对外仍是一次 `InGame`，没有单独的「阶段切换」事件或状态，不利于上层/日志/回放一致理解。

---

## 三、非定缺相关潜在问题（简要）

以下是在**非定缺**逻辑里值得后续重点排查或加固的点，不是完整 bug 列表。

### 3.1 和牌人数与游戏结束

- `agari_count` 与 `players_agari` 在 `handle_hora` 中更新；多处用 `agari_count >= 3` 或 `players_agari` 做判断。需确保：多人和牌时（Multi-Ron）每次 `handle_hora` 只加一次，且 3 人和牌后不再处理新 Hora。
- 已有 `bail!` 在「无人可行动却未结束」等情况，有助于发现不一致，但若存在漏判路径，仍可能进入异常状态。

### 3.2 tiles_left 与 yama 的同步

- `BoardState::tiles_left` 与 `board.yama.len()` 在多处用 `assert_eq!` 绑定，设计清晰。
- `PlayerState` 各自持有一份 `tiles_left`，在 `broadcast(Tsumo)` 时每个玩家都会在 `tsumo()` 里对**自家**状态做 `tiles_left -= 1`（且仅在 actor 为自己时继续后续逻辑），因此 4 份会一起减，保持一致。若未来有「只对部分玩家 broadcast」的路径，这里可能出错。

### 3.3 抢杠（Chankan）与杠的结算

- `handle_hora` 里有两段与抢杠相关的逻辑：  
  - 前面一段（约 491–527 行）：在标记和牌前，对 Chankan 做「杠钱退款」并写 `kyoku_deltas`，然后清 `last_kan_revenue`。  
  - 后面一段（约 605–632 行）：在 Ron 的 deltas 里再做一次「杠钱退款」，但条件包含 `last_kan_revenue > 0`，因前段已清零，不会重复退款。  
- 逻辑上未发现重复退款，但两段职责接近（都是「抢杠时撤销当杠的即时支付」），长期维护容易改错，建议合并或加注释明确「只执行一次」。

### 3.4 流局与花猪/查大叫

- `exhaustive_ryukyoku` 中依赖 `players_agari`、`ding_que`、`shanten` 等；若存在「未选定缺就进流局」的异常路径，`players_with_ding_que == 0` 的 warn 会触发，但不会阻止结算。可考虑在流局入口强制保证「正常对局下所有人已选定缺」，否则视为异常并打日志或断言。

### 3.5 多 Ron（Yi Pao Duo Xiang）后的下一动

- Ron 后 `tsumo_actor` 从「弃牌家下家」起跳，跳过已和牌者；若 3 人已和牌会 `Poll::End`。需确认：多人同时 Ron 时，`handle_hora` 的调用顺序是否与规则一致（例如按座次），以及 `last_kan_revenue` / `last_kan_actor` 在多人 Ron 下是否只被正确使用一次（当前注释说明首和者拿转移，逻辑上依赖 `last_kan_revenue` 被及时清零）。

---

## 四、建议的改进方向（针对定缺）

1. **引入显式阶段类型**  
   用 `enum Phase { DingQue, NormalPlay }`（或带更多子状态）作为 Board 的单一来源，`ding_que_phase` 改为从该 enum 派生，避免布尔 + 数组分散表达。

2. **统一「需要反应的玩家」**  
   在 Board 或 Game 提供类似 `fn players_needing_reaction(&self) -> impl Iterator<Item = usize>`，内部根据当前阶段决定用 `ding_que_selected` 还是 `can_act()`，poll/commit 只依赖这一接口，减少分支。

3. **定缺阶段显式校验**  
   在 `step()` 的定缺分支里：若当前是定缺阶段，只接受 `DingQue` 或约定的「默认」；对其他事件类型直接返回错误或统一转成自动选定缺并打日志，而不是跳过校验。

4. **阶段切换可观测**  
   考虑在「所有人选完定缺 → 关 phase → 第一次 Tsumo」时写入一条内部事件或日志（例如 `PhaseChange::DingQueComplete`），便于调试、回放和测试断言。

5. **未选定缺禁止打牌**  
   在 `validate_reaction` 中，对 `Dahai` 若 `ding_que.is_none()` 且当前局为正常规则（非测试），应 `ensure` 拒绝，除非 Board 明确传入「允许未选定缺打牌」的测试/兼容标志。

---

## 五、小结

- **为什么 bug 几乎全是定缺？**  
  因为定缺是**额外插入的一整段阶段**，和原有「单阶段」主循环接缝多；且「谁需要行动」存在**双源**（Board 的 ding_que 状态 vs Player 的 can_act），没有统一抽象，导致分支多、易漏、难维护。

- **根因**  
  阶段表达分散、行动者判定不统一、定缺阶段绕过统一校验、阶段切换与第一次摸牌绑在同一次 step 且缺少显式阶段切换表示。

- **非定缺**  
  也有潜在风险点（和牌结束、tiles_left 同步、抢杠两段逻辑、流局前提、多 Ron 顺序），但当前代码已有较多断言和注释，问题密度低于定缺；建议在修定缺的同时逐步把上述点纳入测试或小重构。

---

**文档版本**: 2026-01-28  
**关联**: `DEEP_BUG_ANALYSIS_V2.md`（具体 bug 列表与修复建议）
