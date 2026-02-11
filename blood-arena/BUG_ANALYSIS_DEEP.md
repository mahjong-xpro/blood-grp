# 血战 Arena 深度 Bug 分析

基于代码全链路追踪的系统化排查，按严重程度与模块分类。

---

## 一、数据流速览

```
前端 (arena.js) ←WebSocket→ main.py (broadcast_loop) ←state_queue→ GameManager 线程
                                 ↑
                          action_queue ← 前端 action
                                 ↓
                    libblood OneVsThree → HumanEngine.react_batch
                         └→ set_scene → update_state (AI 回合转发)
```

---

## 二、已确认 Bug（高优先级）

### BUG-1：牌局结束时未下发四家手牌 ✅ 已修复

**现象**：勾选「显示AI手牌（结束时）」后，牌局结束仍只显示背牌。

**修复**：
- libblood: `GameResult` 增加 `final_tehais: Option<Vec<Vec<String>>>`，游戏结束时从 `player_states` 的 `tehai` 导出
- `hand.rs`: 新增 `tehai_to_strings()` 将 `[u8; 27]` 转为 mjai 牌面字符串列表
- `mjai_log.rs`: `end_game` 将 `final_tehais` 一并传给 Python
- `game_manager.py`: `end_game(index, scores, final_tehais=None)` 在 `game_over` 中加入 `tehais`（若存在）

---

### BUG-2：action_request 与 state_update 顺序导致动作栏时序异常

**现象**：后端先发 `action_request`，后发 `state_update`。若用户快速点击，可能在 `state_update` 未到达前就出牌。

**分析**：
- 设计意图：先发 `action_request` 以便用户尽早操作
- 若用户先点牌再看到对手出牌动画，属于 UX 问题，但逻辑上后端会校验
- 更严重的是：`state_update` 晚到时，`validActionsShown` 会由 replay 的 dahai 分支设为 true。若 `action_request` 先到且 `validActionsShown = false`，动作栏会一直隐藏直到 dahai 被 replay

**结论**：当前逻辑在 dahai replay 时设置 `validActionsShown = true`，能弥补顺序问题。但若网络延迟大，用户可能先看到空状态再看到动作栏，体验一般。

---

### BUG-3：重连时若 latest 为 game_over 可能未正确处理

**现象**：断线重连时，`connect` 会发送 `shared_state['latest']`。若 latest 为 `game_over`，前端会收到并进入 `handleMessage`。`handleMessage` 已处理 `game_over`，会设置 `state.gameEnded`、`phase` 等。

**结论**：已正确处理，无 bug。

---

## 三、潜在 Bug（中优先级）

### BUG-4：game_states 索引可能越界

**位置**：`game_manager.py` 第 206 行  
```python
wrapper = game_states[self.player_id]
```

**分析**：
- Human 固定为 player_id=0
- `game_states` 由 MjaiLogBatchAgent 在 `set_scene` 时逐玩家 push
- 当仅人类需要行动时，`game_states` 可能只有 1 个元素，`game_states[0]` 正确
- 当多人需行动（如多人和牌）时，会有多个元素，索引按 player_id 对应

**结论**：OneVsThree 中 Human 为 0，正常仅人类需行动时只有 1 个元素，当前无问题。若将来扩展多人决策，需确认索引语义。

---

### BUG-5：replay 中 lastReplayedEventCount 与 events.length 不一致

**现象**：若 `state_update` 两次推送的 `events` 长度相同，replay 会因 `events.length === last` 直接 return，不重放。

**分析**：此为去重逻辑，避免重复 replayed。若两次 `events` 内容相同，不重放是预期行为。

**潜在问题**：若第一次 `state_update` 的 replay 因异步未完成，第二次同样长度的 `state_update` 到达时会直接 return，导致第一次 replay 的结果可能不完整。但因 `replayId` 机制，新 `state_update` 会触发新的 `replayEvents`，旧 replay 会在 `delay` 后检查 `myReplayId !== replayId` 并提前退出，不会覆盖新状态。需注意 `lastReplayedEventCount` 的更新时机：仅在 loop 结束时 `state.lastReplayedEventCount = events.length`，若中途 return 则不会更新。

**结论**：设计合理，暂无发现越界或状态错乱。

---

### BUG-6：HumanEngine Shadow State 与 libblood 不同步

**场景**：`update_state` 更新 shadow state，但 `react_batch` 使用 `game_states` 中的 `player_state`（来自 libblood），两者可能短暂不一致。

**分析**：
- `update_state` 在 AI 回合由 `_ChampionWithHumanObserver` 调用
- `react_batch` 仅在人类回合调用，此时 `game_states` 来自 libblood 最新状态
- shadow state 主要用于 `_translate_to_mjai`（如 `last_kawa`、`last_tsumo_tile`、`peng`），与 `game_states` 的 `last_cans` 互补

**结论**：若 `update_state` 与 `react_batch` 的 events 一致，shadow state 应正确。唯一风险是 `update_state` 的 events 与 `react_batch` 的 `events_json` 来自不同批次，但两者都来自同一 `ctx.log`，理论上一致。

---

## 四、边界与健壮性

### 4.1 异常处理

- `update_state` 中 `json.loads(events_json)` 异常已被 try/except 捕获
- `_translate_to_mjai` 中 `last_cans`、`kakan_candidates` 等缺失时使用 `getattr(..., False)` 或空列表
- 前端 `playDiscardSound`、`playActionSound` 使用 `try/catch` 和 `.catch(() => {})` 静默失败

### 4.2 音效

- 碰/杠/胡音效已接入，使用 `pon.m4a`、`kan.m4a`、`tsumo.m4a`、`ron.m4a`
- 杠后摸牌显示逻辑已修正（去掉错误的 `expectedWithTsumo` 判断）

### 4.3 前端 v-cloak

- `arena.css` 已定义 `[v-cloak] { display: none }`，挂载前隐藏占位，正常。

---

## 五、建议修复优先级

| 优先级 | Bug | 影响 |
|--------|-----|------|
| P0 | BUG-1 结束局未下发 tehais | 「显示AI手牌」功能始终无效 |
| P1 | 验证 action_request/state_update 顺序对 UX 的影响 | 快速操作下是否会有异常 |

---

## 六、附录：关键代码路径

- **前端 replay**：`arena.js` `replayEvents()`，增量重放，`lastReplayedEventCount` 去重
- **后端 react_batch**：`game_manager.py` 第 198 行，先 `action_request` 再 `state_update`
- **AI 回合转发**：`_ChampionWithHumanObserver.update_state` 调用 `HumanEngine.update_state`
- **game_over**：`HumanEngine.end_game` 构造 msg，不含 `tehais`
