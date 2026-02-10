# 人机对战系统深度分析与 Bug 报告

## 一、系统架构概览

### 1.1 数据流

```
前端 (Vue + WebSocket)
    ↓ start_game / action (dahai, ding_que, action bar)
后端 main.py (FastAPI + 单 consumer broadcast_loop)
    ↓ action_queue / state_queue
GameManager._run_libblood (工作线程)
    ↓ HumanEngine + MortalEngine
libblood OneVsThree (BatchGame)
    → set_scene(index, log, state) → update_state() + 后续 get_reaction() → react_batch(game_states)
    → HumanEngine 阻塞在 action_queue.get()，收到合法动作用 _translate_to_mjai 转为 MJAI 事件返回
```

- **Poll 阶段**：对每个需要行动的玩家调用 `set_scene`；Human 的 `update_state` 会向 state_queue 推状态，前端通过 WebSocket 收到 `state_update`。
- **Commit 阶段**：对同一批玩家调用 `get_reaction`；MjaiLogBatchAgent 先 `evaluate()` 即调用 Python 的 `react_batch(game_states)`，Human 在内部等待前端动作后返回一个 MJAI 事件的 JSON 字符串。

### 1.2 关键约定

- 1v3 时 Challenger（Human）只有 player_id=0，故 `game_states` 仅 1 个元素，`game_states[self.player_id]` 即 `game_states[0]` 正确。
- 前端 `state.myPlayerId === 0`，手牌/定缺/弃牌等均按 player 0 为“自己”。
- 事件流来自 libblood Board 的 `ctx.log`，**不包含 `start_game`**，只包含 `start_kyoku` 及后续对局事件。

---

## 二、已发现 Bug 与修复建议

### Bug 1：加杠（Kakan）未实现（严重）

**位置**：`blood-arena/backend/game_manager.py`，`HumanEngine._translate_to_mjai`，`act_type == "kan"` 的 else 分支。

**现象**：玩家已有碰（Pon），摸到第四张后点“杠”时，后端只检查了“手牌四张相同”（暗杠 Ankan），没有检查“碰 + 手牌一张”（加杠 Kakan），因此直接 `return {"type": "none"}`，导致加杠无法执行。

**原因**：注释里写了 “Check Kakan (1 in hand + 3 in Pon)” 和 “We don't track 'peng' in shadow state yet”，但 `update_state` 里已经维护了 `self.peng`（pon 时 append，kakan 时 remove），只是 `_translate_to_mjai` 的完整实现没有使用 `self.peng`。

**修复**：在 Ankan 检查之前，先根据 `self.peng` 检查是否有“碰的牌 + 手牌还有一张”，有则返回 `{"type": "kakan", "actor": actor_id, "pai": p, ...}`；否则再检查四张相同并返回 ankan。

---

### Bug 2：前端 `state.gaming` 从未为 true（严重）

**位置**：`blood-arena/frontend/js/arena.js`，`replayEvents` 仅在 `ev.type === 'start_game'` 时设置 `state.gaming = true`。

**现象**：点击“开始对局”后，后端正常开局并推送 `state_update`（事件多为 `start_kyoku`、定缺等），但事件流中**没有** `start_game`（log 来自 Board，不包含 dump 时的 start_game）。因此 `state.gaming` 一直为 false，界面表现为“未在游戏中”（例如回合/状态依赖 gaming 时异常）。

**修复**：
- 在 `replayEvents` 中，当遇到 `start_kyoku` 时也设置 `state.gaming = true`；和/或
- 在 `handleActionRequest` 中，收到任意 `action_request` 时设置 `state.gaming = true`（表示已进入对局流程）。

---

### Bug 3：初始分数与规则不一致

**位置**：`blood-arena/frontend/js/arena.js`，`replayEvents` 里 `start_game` / `start_kyoku` 分支的 `state.scores` 初始化为 `[25000, 25000, 25000, 25000]`。

**规则**：`rules.md` 与 `libblood` 的 `INITIAL_SCORE` 均为 60000（4×60000 零和）。

**修复**：将前端初始分数改为 `[60000, 60000, 60000, 60000]`，与规则和后端一致。

---

### Bug 4：game_manager 中重复/死代码

**位置**：`blood-arena/backend/game_manager.py`，约 358–388 行。

**现象**：`start_game_thread` 和 `_run_libblood` 各出现两段；第一段 `_run_libblood` 仅为 `pass`，且第一段 `start_game_thread` 之后被第二段覆盖，形成无用代码，增加阅读和维护成本。

**修复**：删除 358–388 行之间的重复定义（保留后面完整的 `start_game_thread` 与 `_run_libblood`）。

---

### Bug 5：game_over 的 scores 类型

**位置**：前端 `handleMessage` 中 `msg.type === 'game_over'` 时 `alert(\`... ${msg.scores.join(', ')}\`)`。

**说明**：后端传的 `scores` 来自 Rust `game_result.scores`（i32 数组），JSON 序列化后为数字数组，`join` 正常。无类型错误，仅作记录。

---

### Bug 6（可选）：打牌后手牌/回合的乐观更新

**现象**：用户点击出牌后，要等下一轮 `state_update`（多手之后）才会从 `state.tehai` 移除该牌并更新 `state.currentActor`，中间可能有多手 AI 操作，界面会短暂仍显示“你的回合”和未减少的手牌。

**建议**：可在发送 `dahai` 时做乐观更新：从 `state.tehai` 移除该牌、将 `state.currentActor` 设为下家；若后端之后因非法动作重发状态，再以事件重放为准。当前实现为“保守不乐观”，逻辑正确，仅体验可优化。

---

## 三、与规则/实现的交叉核对

- **定缺**：Human 的 `update_state` 处理 `start_kyoku` 并维护 `tehai`；定缺选择通过 `ding_que` 事件与 `_translate_to_mjai` 正确传递；前端定缺面板与 `doDingQue` 一致。
- **碰/杠/和**：`last_kawa` / `last_tsumo_tile` 用于区分荣和/自摸、大明杠等，逻辑正确；仅 Kakan 缺失（见 Bug 1）。
- **番数/计分**：由 libblood 内部完成，前端只展示后端下发的 scores，无额外计算，无发现错误。

---

## 四、修复优先级

| 优先级 | Bug | 影响 |
|--------|-----|------|
| P0 | Kakan 未实现 | 加杠无法执行，对局逻辑错误 |
| P0 | state.gaming 未设 | 界面状态错误，可能影响回合/按钮等 |
| P1 | 初始分数 25000 | 与规则不一致，显示误导 |
| P2 | 重复/死代码 | 可维护性 |

上述 P0/P1 建议尽快修复；P2 在整理代码时一并删除。
