# 继续分析 - 更多 Bug 清单

基于对前端、后端、libblood 数据流的追溯，补充发现以下潜在问题。按模块与严重性分类。

---

## 一、前端 arena.js

### BUG-F：start_kyoku / start_game 未重置 currentActor（低）

**现象**：新局开始时，`currentActor` 仍为上一局末值（如 0、1、2、3），回合指示可能短暂显示错误（如「等待 下家」而实际在定缺阶段）。

**位置**：`replayEvents` 中 `start_kyoku`、`start_game` 分支。

**原因**：仅重置了 `canDiscard`、`validActions` 等，未重置 `currentActor`。定缺阶段尚未有 tsumo，`currentActor` 一直为旧值。

**建议**：在 `start_kyoku` 和 `start_game` 中增加 `state.currentActor = -1`。

---

### BUG-G：handleActionRequest 定缺时未设 currentActor（低）

**现象**：收到定缺 `action_request` 时未设置 `currentActor`，回合指示沿用上一状态。

**位置**：`handleActionRequest`，`isDingQue === true` 分支。

**建议**：定缺时设置 `state.currentActor = -1`，与「等待中...」语义一致。

---

### BUG-H：replay 多个并发时 lastReplayedEventCount 竞态（低）

**现象**：`state_update` 可能连续触发两次 replay（set_scene + react_batch），第二个 replay 依赖 `events.length === last` 提前返回。若两次 `events` 长度相同，逻辑正确；若网络/时序异常导致顺序错乱，`lastReplayedEventCount` 可能与实际不符。

**现状**：当前 `events.length === last` 正确去重，风险较低。

**建议**：保持现状，必要时可增加 replay 版本号或序列号做更强一致性校验。

---

### BUG-I：action_request 与 dahai 的校验不完整（中）

**现象**：后端校验 `atype === "dahai"` 时，未校验 `pai` 是否在合法打牌集合内。若前端因 bug 或篡改发送非法 `pai`，libblood 可能报错或导致内部状态异常。

**位置**：`game_manager.py` 的 `react_batch` 校验逻辑。

**建议**：在 Python 层用 `player_state` 或 shadow state 校验 `pai` 合法性，非法则重发 state/request，不交给 libblood。

---

## 二、后端 game_manager.py

### BUG-2：动作队列残留（已记录，中）

**现象**：同一回合内多次点击或网络延迟时，`action_queue` 可能残留旧动作，下一回合误用。

**现状**：`DEEP_BUG_ANALYSIS.md` 已记录，且已实现 `get_nowait` 清空。

**确认**：当前代码在 `return` 前已有清空逻辑，应已修复。

---

### BUG-4：定缺推荐 pai 为 None（已记录，低）

**现象**：`best_idx >= 31` 时 `idx_to_tile` 返回 `None`，`best_action.pai` 为 undefined。

**建议**：对定缺情况设置 `pai: null` 或占位值，避免前端依赖 `pai` 显示。

---

### BUG-J：update_state 与 react_batch 的 events 可能不一致（低）

**现象**：`update_state`（set_scene）与 `react_batch` 的 `events` 来自同一 `ctx.log`，理论上一致。若 libblood 在 `set_scene` 与 `get_reaction` 之间追加事件，两者可能不同。

**现状**：从 `game.rs` 看，commit 前不会修改 log，一致性应可保证。

**建议**：保持观察，如有异常再加强校验。

---

## 三、数据流与消息顺序

### BUG-K：重连时仅拿到 latest 一条消息（中）

**现象**：断线重连时，`connect` 只发送 `shared_state['latest']`。若最新为 `action_request`，前端会设置 `canDiscard=true`，但可能缺少之前的 `state_update`，导致 `tehai`、`discards` 等与后端不一致。

**位置**：`game_manager.connect`、前端 `onopen`。

**建议**：重连时如需完整恢复，可考虑发送最近若干条消息，或让前端主动请求完整状态。

---

### BUG-L：game_over 单局结束时 phase 未更新（低）

**现象**：`is_match_over=false` 时，仅清 `canDiscard`/`validActions`，未设 `state.phase`。界面仍为 `playing`，可能短暂显示「等待 下家」等，直到下一局 `start_kyoku` 到达。

**建议**：单局结束时可选设置 `state.phase = 'between_games'` 或类似，用于区分「局中」与「局间」。非必须，可根据 UX 需求决定。

---

## 四、与现有文档的对应

| 文档中的 Bug | 状态 |
|--------------|------|
| BUG-1 结束局未传 tehais | 未修复，需改 Rust + 后端 |
| BUG-2 动作队列残留 | 已实现清空逻辑 |
| BUG-3 手牌张数杠算错 | 已修复（按 m.tiles.length） |
| BUG-4 定缺推荐 pai 为 None | 未修复 |
| 第二轮无法打牌 | 已修复（replay 不再修改 canDiscard） |

---

## 五、总结优先级

| 优先级 | Bug | 影响 |
|--------|-----|------|
| P1 | BUG-1 结束局未传 tehais | 显示 AI 手牌功能无效 |
| P2 | BUG-F start_kyoku 未重置 currentActor | 回合指示短暂错误 |
| P2 | BUG-G 定缺时未设 currentActor | 回合指示错误 |
| P2 | BUG-K 重连时状态不完整 | 断线重连后可能 desync |
| P3 | BUG-4 定缺推荐 pai 为 None | 推荐展示异常 |
| P3 | BUG-L game_over 单局 phase | 局间状态展示 |

---

## 六、已核对无问题的部分

- **getHandTileCount**：offset 0–3 对应 player id（myPlayerId=0 时一致），杠牌张数正确。
- **validActionsShown**：与对手 dahai 同步后再显示碰/杠/胡，逻辑正确。
- **optimisticDahai**：打牌乐观更新与 replay 的匹配逻辑正确。
- **replay 去重**：`events.length === last` 正确跳过重复 state_update。
- **第二轮打牌**：canDiscard 由 action_request 独占控制，replay 不再修改。
