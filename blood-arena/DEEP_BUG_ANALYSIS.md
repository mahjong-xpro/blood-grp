# 人机对战深度 Bug 分析

基于当前代码与数据流做的系统排查，按模块列出已发现的问题与修复建议。

---

## 一、数据流与模块边界

```
前端 (arena.js / index.html)
  ↔ WebSocket (state_update / action_request / game_over)
后端 (main.py → game_manager.py)
  ↔ action_queue / state_queue
GameManager._run_libblood
  → HumanEngine (player 0) + MortalEngine (players 1,2,3)
  → libblood OneVsThree → BatchGame → set_scene / get_reaction
  → MjaiLogBatchAgent.react_batch → Python HumanEngine.react_batch
  → HumanEngine.end_game(index, scores)  [仅 scores，无 tehais]
```

---

## 二、已确认的 Bug

### BUG-1：牌局结束时未下发四家手牌，导致「显示AI手牌」无效（中）

**现象**：勾选「显示AI手牌（结束时）」且牌局结束后，三家 AI 手牌区仍只显示背牌，无法看到真实牌面。

**原因**：
- 前端已实现：`state.finalTehais`、`handTiles(offset)`，且 `game_over` 分支会写 `state.finalTehais = msg.tehais`。
- 后端 `HumanEngine.end_game(index, scores)` 只收到 `scores`，发出的 `game_over` 仅含 `scores`，**不含 `tehais`**。
- libblood 侧：`GameResult` 无 `final_tehais` 字段；`mjai_log.rs` 的 `end_game` 只调用  
  `engine.call_method1("end_game", (index, game_result.scores))`，未传手牌。

**结论**：整条链路上「结束局四家手牌」未从 Rust 传到 Python 再传到前端，因此功能无法生效。

**修复建议**：
1. 在 libblood `GameResult` 中增加 `final_tehais: Option<Vec<[String; 14]>>`（或与现有手牌表示一致的结构），在游戏结束时由 Board/Game 写入。
2. `mjai_log.rs` 的 `end_game` 将 `game_result.final_tehais` 一并传给 Python：  
   `call_method1("end_game", (index, game_result.scores, game_result.final_tehais))`。
3. Python `HumanEngine.end_game(index, scores, final_tehais=None)` 在 `game_over` 消息中加入 `tehais`（若存在）。
4. 前端已就绪，无需改逻辑，只需后端开始下发 `tehais`。

---

### BUG-2：动作队列残留导致下一回合误用旧操作（中）

**现象**：用户在同一回合内连续点击（如连续两次出牌），或网络延迟导致多条动作先进入队列，第一次 `get()` 用掉合法动作后，下一次人类回合的 `get()` 会拿到**上回合留下的动作**。若该动作对新状态不合法（例如牌已打出、或已非出牌阶段），会触发「Ignored invalid action」并重发 state/request；若恰好仍合法（例如又摸到同张牌），会误把「上一次的意图」当作本回合操作执行。

**位置**：`blood-arena/backend/game_manager.py`，`HumanEngine.react_batch` 中在 `action_queue.get()` 返回合法动作并 `return` 后，未清空队列中剩余动作。

**修复建议**：在 `return [json.dumps(mjai_action)]` 之前，将当前队列中剩余动作全部取出并丢弃，保证下一回合只处理本回合新产生的操作。例如：

```python
# 在 return 前
try:
    while True:
        self.action_queue.get_nowait()
except queue.Empty:
    pass
return [json.dumps(mjai_action)]
```

---

### BUG-3：AI 手牌张数按「每副露 3 张」计算，杠被算错（低）

**现象**：有暗杠（ankan）或明杠（daiminkan/kakan）时，他家手牌区的背牌张数会多 1 张。因为每副露被统一按 3 张算，而杠是 4 张。

**位置**：`blood-arena/frontend/js/arena.js`，`getHandTileCount(p)`：

```js
const meldCount = state.fuuro[p] ? state.fuuro[p].length : 0;
let count = 13 - (meldCount * 3);
```

**修复建议**：按每副露实际张数扣减。例如：

```js
const fuuro = state.fuuro[p] || [];
const meldTiles = fuuro.reduce((sum, m) => sum + (m.tiles ? m.tiles.length : 3), 0);
let count = 13 - meldTiles;
```

再在「当前为该玩家回合且对局未结束」时 +1（摸牌）。这样 pon=3、kan=4 都正确。

---

### BUG-4：AI 分析中 ding_que 推荐的 tile 为 None（低）

**现象**：当推荐动作为定缺（action index 31–33）时，`idx_to_tile(best_idx)` 对 31–33 返回 `None`（因只处理 0–26 的牌型），前端若用 `best_action.pai` 显示会得到 undefined。

**位置**：`blood-arena/backend/game_manager.py`，`_get_ai_analysis` 中 `best_tile = idx_to_tile(best_idx)`，而 `idx_to_tile` 仅在 `0 <= idx < 27` 时返回牌字符串。

**修复建议**：对 `best_idx >= 31` 的定缺情况，不设 `pai` 或设为一个占位（如 `"ding_que"`），避免前端依赖 `pai` 显示推荐牌。

---

## 三、与历史文档的对应关系

| 历史文档中的 Bug | 当前状态 |
|------------------|----------|
| Kakan 未实现 | **已修复**：`_translate_to_mjai` 已按 `can_kakan`/`kakan_candidates`/`self.peng` 实现加杠。 |
| state.gaming 从未为 true | **已修复**：`replayEvents` 中 `start_kyoku` 设 `gaming=true`，`handleActionRequest` 也设 `gaming=true`。 |
| 初始分数 25000 | **已修复**：已改为 60000。 |
| game_manager 重复/死代码 | 需再确认：若仍存在 358–388 行重复定义，建议删除。 |
| game_over 的 scores 类型 | 无问题，前端正常使用。 |

---

## 四、建议修复优先级

| 优先级 | Bug | 影响 |
|--------|-----|------|
| P1 | BUG-1 结束局未传 tehais | 「显示AI手牌（结束时）」永远不生效，需改 Rust + 后端。 |
| P1 | BUG-2 动作队列残留 | 双点/延迟可能导致下一回合误用旧操作，逻辑错误。 |
| P2 | BUG-3 手牌张数杠算错 | 仅影响他家背牌显示张数，体验问题。 |
| P2 | BUG-4 定缺推荐 pai 为 None | 仅影响推荐展示，可容错处理。 |

---

## 五、已核对无问题的部分

- **定缺**：前端 `canDiscardTile` / `getDingqueSuitChar` / `hasDingqueTilesInHand` 与后端事件一致；后端 shadow state 与 `ding_que` 事件正确。
- **碰/杠/和**：`last_kawa`、`last_tsumo_tile`、`last_cans`（含 can_daiminkan/can_kakan/can_ankan）在 `_translate_to_mjai` 中用法正确；kakan 的 mjai 格式 `pai` + `consumed: [p,p,p]` 与 libblood 一致。
- **观测编码**：AI 输入包含河牌、副露、定缺、分数等（见 `libblood/src/state/obs_repr.rs`），无缺漏。
- **事件重放**：`replayEvents` 对 start_kyoku、ding_que、tsumo、dahai、pon/kan/ankan/daiminkan/kakan、agari/hora 的处理与后端事件一致；`tilesLeft` 在 start_kyoku 置 56、tsumo 时减 1，正确。

---

## 六、可选优化（非 Bug）

- **乐观更新**：用户出牌后，可在前端先从 `state.tehai` 移除该牌并更新 `currentActor`，等下次 `state_update` 再以事件为准；可减少「已出牌仍短暂显示」的错觉。
- **game_over 后清空 action_queue**：当前仅在 `start_game_thread` 时清空；若在 `end_game` 后也清空一次，可避免重开局时误用上局残留动作（若存在重连或快速「再来一局」）。

上述为本次深度分析结论；建议优先修复 BUG-1（需动 Rust）与 BUG-2（仅后端 Python）。
