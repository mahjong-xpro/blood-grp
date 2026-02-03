# 血战到底麻将 AI — 规则、奖励与 Bug 深度分析

本文档对血战到底麻将 AI 的规则实现、奖励/计分系统及已发现/潜在 Bug 做集中梳理与结论。

---

## 一、规则实现总览

### 1.1 规则文档与代码对应

| 规则项 | 文档 (rules.md) | 实现位置 | 结论 |
|--------|-----------------|----------|------|
| 牌组 108 张、3 花色 | ✅ | `board.rs` UNSHUFFLED, `agari` 27 维 | 一致 |
| 初始 13 张/人、庄家补 1 张 | ✅ | `board.rs` haipai + 首次 Tsumo | 一致 |
| 定缺（必须缺一门） | ✅ | `ding_que.rs`, `update.rs` DingQue, `board.rs` ding_que_phase | 一致 |
| 定缺完成前不能打牌 | ✅ | poll/step 中 ding_que_phase 控制 | 一致 |
| 有定缺花色必须先打定缺 | ✅ | `ding_que::discard_allowed`, `update.rs` validate | 一致 |
| 花猪不能和牌 | ✅ | `ding_que::can_agari`, `agari.rs` has_yaku/agari | ⚠️ 见 Bug#2 |
| 和牌型 4 组+1 对 / 七对 | ✅ | `agari.rs` AGARI_TABLE, has_chitoi | 一致 |
| 番数：平胡 1、自摸 1、七对 2、碰碰胡 1、金钩钓 1、清一色 2、带幺九 3、根 1/根、杠上花/杠上炮/抢杠 1、海底 1、天/地胡 5 番封顶 | ✅ | `agari.rs` agari() | 一致（金钩钓注释写 +2 实为 +1，见下） |
| 5 番封顶、点数 1000×2^(番-1) | ✅ | `point.rs` calc_from_fan, agari fan.min(5) | 一致 |
| 自摸：未和牌者付、已和牌者不付（自摸3家/2家/1家）；荣和放铳者付；无庄家加成 | ✅ | `board.rs` handle_hora | 一致 |
| 流局：查花猪、查大叫、退税 | ✅ | `board.rs` exhaustive_ryukyoku | 一致 |
| 抢杠：加杠不成立、MinKan 回滚为 Pon、刮风退款 | ✅ | `board.rs` handle_hora chankan 分支, CHANKAN_STATE_FIX_ANALYSIS.md | 一致 |

### 1.2 番数实现细节（agari.rs）

- **平胡**：基础 1 番，始终加。
- **自摸**：`!is_ron` 时 +1。
- **海底**：`is_haidi` +1。
- **七对**：`div.has_chitoi` 时 +2，该分支不与碰碰胡/金钩钓叠加。
- **碰碰胡**：4 刻无顺 +1。
- **金钩钓**：4 副露 + 单钓（将牌为和牌牌）再 +1；**注释写「+2番」实为代码 `div_fan += 1`，与 rules.md 的 +1 一致，仅注释错误。**
- **清一色 / 带幺九**：按手牌+副露检查，正确。
- **根**：total_counts 中某牌 4 张且非 exclude_gen_tile 则 +1/根；抢杠时 exclude_gen_tile 用于和牌方计算，当前和牌方未用 exclude（见 agent_helper），即和牌方按正常根数计番，符合「被抢杠者根不成立」的语义。
- **杠上花 / 杠上炮 / 抢杠**：is_after_kan / is_kan_discard / is_chankan 互斥使用，正确。
- **天胡/地胡**：fan=5 封顶，正确。

---

## 二、奖励与计分系统

### 2.1 局内计分（libblood）

- **点数**：`Point::calc_from_fan(fan)`，`fan = fan.min(5)`，`point = 1000 * 2^(fan-1)`，ron/tsumo_ko 同值，无庄家区别。
- **自摸**：仅**未和牌**的玩家各付 `point.tsumo_ko`（已和牌者不付）；和牌者得 `付家数 × point.tsumo_ko`（自摸3家=3×、自摸2家=2×、自摸1家=1×）。见 `board.rs` handle_hora Tsumo 分支 `!self.players_agari[i]` 条件。
- **荣和**：放铳者付 `point.ron`，和牌者得 `point.ron`。
- **杠**：暗杠/明杠/加杠即时收 2000/2000/1000（每家），记录在 `gang_history`；流局时花猪/未听牌者退税；抢杠时加杠者刮风退款并 MinKan→Pon 回滚。

### 2.2 流局（exhaustive_ryukyoku）

- **查花猪**：有定缺且手牌（当前仅 tehai）仍含定缺花色 → 花猪；花猪向每位「非花猪且未和牌」者付 16000。**花猪判定未含副露，见 Bug#2。**
- **查大叫**：听牌者按理论最大番（Ron 计，不含海底等）得到「点数」；未听牌者向每位听牌者付该点数。
- **退税**：花猪或未听牌者将本局杠收入按记录原路退回。

### 2.3 训练侧奖励（mortal）

- **RewardCalculator.calc_delta_points**：用 `scores_history`（每局开始分数）与 `final_scores` 构造该玩家的分数序列，再作一阶差分得到每步 delta。若 Rust 端传入的 scores 与最终结算一致，则 delta 即每局/每步的得分变化，逻辑正确。

---

## 三、已发现与潜在 Bug

### Bug #1：SP 计算器自摸计分逻辑错误（已修复）

**位置**：`libblood/src/algo/sp/calc.rs`，`get_score()`（用于自摸期望分）。

**规则（rules.md §支付规则）**：  
- **自摸**：其他 3 名玩家各付「和牌点数」；和牌者得 3×和牌点数。  
- **和牌点数**：1000×2^(番-1)。自摸至少 2 番（平胡+自摸），故最低 2 番 = 2000 点/家 → 每家付 2000，和牌者得 **6000**。

**正确语义**：公式算出的 `base_points` = **和牌点数 = 每家应付**，不是和牌者总得点。  
- 对局实现（`board.rs` handle_hora、`point.rs`）：`point.tsumo_ko` = 和牌点数，每家付 `tsumo_ko`，和牌者得 `tsumo_ko * 3`。  
- 因此 SP 中自摸应为：和牌者得 `base_points * 3`，三家各付 `base_points`，**不应再除以 3**。

**错误逻辑（原代码）**：  
把 `base_points` 当成「和牌者总得点」，再除以 3 得到「每家应付」：
```rust
let points_per_player = base_points / 3;  // 错误：把 base_points 当总得点
scores = [base_points, -points_per_player, ...];  // 和牌者写 base_points 也错
```
- 2 番时：base_points=2000，原代码变成每家付 666、和牌者得 2000 → 实际应为每家付 2000、和牌者得 6000。  
- 既违反规则（自摸最低应为 6000），又因整除导致零和破坏（999 vs 1000 等）。

**修复**（已做）：  
- 和牌者得 `base_points * 3`，三家各付 `base_points`。  
- 与 `point.rs` 的 `tsumo_ko` / `tsumo_total` 及 `board.rs` 的自摸结算一致；自摸最低 2 番 = 6000（3 家付时），每家 2000。  
- **说明**：SP 的 `get_score()` 固定按「3 家付」算（最大自摸收益）；实际对局中 `board.rs` 按未和牌人数计，已和牌者不付（自摸 2 家 = 2×和牌点数，自摸 1 家 = 1×）。

---

### Bug #2：定缺/花猪未考虑副露（潜在规则偏差）

**位置**：  
- `libblood/src/ding_que.rs`：`has_ding_que_tiles(tehai, ding_que)`、`can_agari(tehai, ding_que)` 只查 `tehai`。  
- `libblood/src/state/player_state.rs`：`check_ding_que_complete()` 只查 `self.tehai`。

**规则**：四川规则中「手牌」通常包含副露（碰/杠）；「手牌中还有缺门花色的牌」应包含明牌中的定缺花色。

**问题**：  
1. **和牌**：若定缺万，仅副露中有万（如碰 1m），tehai 无万，`can_agari` 会返回 true，可能允许「花猪和牌」。  
2. **流局查花猪**：仅看 tehai，副露全是定缺花色时不会被判花猪，少罚分。

**影响**：  
- 和牌：可能出现规则上不应成立的和牌被接受。  
- 流局：花猪判定过松，查花猪罚分偏少。

**建议**：  
- 增加「整手」定缺检查：如 `has_ding_que_tiles_in_hand(tehai, pons, minkans, ankans, ding_que)`，对 tehai 与 pons/minkans/ankans 中的牌按花色检查。  
- `can_agari` 在 AgariCalculator 调用处传入副露，使用上述整手检查。  
- `check_ding_que_complete()` 改为：定缺花色在 tehai 与所有副露中均无残留。

---

### Bug #3：金钩钓番数注释错误（文档/注释）

**位置**：`libblood/src/algo/agari.rs` 约 205 行。

**问题**：注释写「5. 金钩钓（JinGouDiao）：+2番」，实际为 `div_fan += 1`，与 rules.md 的 +1 番一致。

**建议**：将注释改为「+1番」。

---

### Bug #4：Hora 未 broadcast 导致 PlayerState.players_agari 滞后（已修复）

**位置**：`libblood/src/arena/board.rs`，`handle_hora()`。

**问题**：`handle_hora()` 只把 Hora 写入 log（`add_log_no_meta(hora)`），未对 4 个 `player_states` 做 `broadcast(&hora)`。  
- 对局层 `self.players_agari` 在 handle_hora 里已正确更新。  
- 各 `PlayerState` 的 `players_agari` 只在 `update(ev)` 时更新，而 Hora 从未被 broadcast，所以各 state 的「谁已和牌」会滞后。  
- `agent_context()` 把 `player_states` 传给 agent 时，agent 拿到的 state 中 `players_agari` 可能缺少本局刚发生的和牌，obs/决策可能错误。

**修复**（已做）：在 `add_log_no_meta(hora)` 前调用 `self.broadcast(&hora)`，使各 `PlayerState` 执行 `hora()` 并更新 `players_agari`。

---

### 其他已处理/无问题项

- **抢杠**：先应用 Kakan 再在 Hora 时回滚 MinKan→Pon、刮风退款、根不计，逻辑与 CHANKAN_STATE_FIX_ANALYSIS.md 一致，无误。  
- **天胡/地胡**：tiles_left==55 且 kans==0 判定第一巡，正确。  
- **多响**：多 Ron 时逐个 handle_hora、杠收入转移/退款、下一轮 tsumo_actor 跳过已和牌者，逻辑正确。  
- **Point/庄家**：无庄家倍数，tsumo_total/ron 与规则一致。  
- **杠支付**：Ankan/Kakan 仅对「未和牌」玩家收/付（`!self.players_agari[i]`）；Daiminkan 仅 discarder 付 2000，逻辑正确。

---

## 四、修复优先级建议

| 优先级 | Bug | 影响 | 建议 |
|--------|-----|------|------|
| 高 | #1 SP 自摸计分逻辑 | 自摸被算成「和牌者得 base_points、三家共付 base_points/3」，违反规则（自摸最低 6000） | 已修复：和牌者得 base_points×3，三家各付 base_points |
| 中 | #2 定缺/花猪不含副露 | 花猪和牌、查花猪漏判 | 增加整手定缺检查并接入和牌与流局 |
| 中 | #4 Hora 未 broadcast | 各 PlayerState.players_agari 滞后，agent 看到的「谁已和牌」错误 | 已修复：handle_hora 中 broadcast(Hora) 再 add_log |
| 低 | #3 金钩钓注释 | 仅文档 | 注释改为 +1 番 |

---

## 五、规则与代码索引

- 规则总览：`rules.md`  
- 番数/和牌：`libblood/src/algo/agari.rs`  
- 点数：`libblood/src/algo/point.rs`  
- 局内计分/流局/抢杠：`libblood/src/arena/board.rs`  
- 定缺：`libblood/src/ding_que.rs`、`player_state.rs`（check_ding_que_complete）、`update.rs`（DingQue/discard）  
- 抢杠状态与回滚：`docs/CHANKAN_STATE_FIX_ANALYSIS.md`  
- 训练奖励：`mortal/reward_calculator.py`、`libblood/src/dataset/gameplay.rs`（分数来源）
