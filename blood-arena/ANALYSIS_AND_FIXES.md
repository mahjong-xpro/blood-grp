# 人机对战深度分析与修复

## 一、架构与数据流

```
[浏览器 Vue] ←WebSocket→ [FastAPI main.py] ←Queue→ [GameManager 线程]
     ↑                          ↑                        ↑
  state_update              state_queue              libblood
  (events 重放)             action_queue             HumanEngine.react_batch
```

- **后端**：`main.py` 接受 WebSocket，启动 `GameManager` 线程；线程内运行 `arena.OneVsThree().py_vs_py(human, ai_engine, seed, 1)`。
- **人类决策**：Rust 调用 `HumanEngine.react_batch(game_states)` → 后端把 `state_update` 放入 `state_queue` → `main.py` 的 sender 发给前端 → 后端再 `action_queue.get()` 阻塞等待 → 前端发动作 → 放入 `action_queue` → `react_batch` 返回该动作。

## 二、发现的 Bug 与问题

### 1. 重连后不处理 `game_over`（严重）

- **现象**：重连时若已结束，后端会发 `shared_state['latest']`，可能是 `{ type: "game_over", scores: [...] }`。
- **问题**：前端 `onmessage` 只处理 `msg.type === "state_update"`，不处理 `game_over`，导致结束页/最终分数不显示。
- **修复**：在 `onmessage` 中增加对 `game_over` 的处理，展示最终分数并标记对局结束。

### 2. 碰（Pon）的 `consumed` 错误（严重）

- **现象**：前端发送 `{ type: "pon", actor, target, pai, consumed: [] }`。
- **问题**：libblood 的 `validate_reaction` 要求 `consumed` 的两张牌必须在手牌中（`ensure_tiles_in_hand`），空数组会导致校验失败或异常。
- **修复**：在“碰”时根据当前手牌与对方打出的 `pai` 计算两张同牌，将 `consumed` 设为这两张牌（例如从手牌中取两个与 `pai` 相同的牌）。

### 3. 游戏在连接时即开始，与“开始对局”按钮语义不符

- **现象**：WebSocket 一连接就调用 `game_manager.start_game_thread()`，对局立即开始。
- **问题**：用户以为要点“开始对局”才开始，实际上连接即开始，按钮语义误导。
- **修复**：仅在收到前端发来的 `start_game` 时再调用 `start_game_thread()`，连接时仅做就绪，不启动对局。

### 4. 前端未处理 `ankan` / `kakan` 事件

- **现象**：服务器会广播 `ankan`、`kakan`，前端只处理了 `pon`、`daiminkan`、`chi`。
- **问题**：收到暗杠/加杠后，副露与手牌不同步，界面显示错误。
- **修复**：在 `handleEvent` 中增加对 `ankan`、`kakan` 的处理（更新副露与手牌），与现有 `pon`/`daiminkan` 一致。

### 5. `consumed` 可能为 undefined 导致报错

- **现象**：`handleEvent` 里 `consumed.forEach(...)`，若事件缺少 `consumed` 会抛错。
- **修复**：使用 `(consumed || []).forEach(...)` 或可选链，避免未定义。

### 6. 后端默认分数与规则不一致

- **现象**：`_reconstruct_state` 里 `scores = [25000] * 4`，血战到底规则为 60000 起。
- **修复**：默认分数改为 60000。

### 7. 多标签页共用队列导致动作错乱

- **现象**：多个标签页连接同一后端，共用一个 `action_queue`，多个动作会排队，可能被同一局误用。
- **建议**：文档说明“仅支持单标签页”，或后端按连接/会话隔离队列（后续优化）。

### 8. 对局结束后无法再开一局

- **现象**：`_run_libblood` 跑完一局后线程结束，`running = False`，用户无法在同一会话再开一局。
- **建议**：在“对局结束”后允许用户再次发送 `start_game`，重新 `start_game_thread`（当前实现会在 `running` 为 False 时重新开线程，可保留并在 UI 上提供“再来一局”）。

## 三、已实施的修复

- **前端**
  - `onmessage` 增加对 `game_over` 的处理：写入 `scores`、设置 `gameEnded`，重连后可看到结束与分数。
  - 碰/杠：用 `getPonConsumed(tehai, pai)` / `getMinkanConsumed(tehai, pai)` 从手牌算出正确 `consumed` 再发送；仅在 `consumed.length >= 2`（碰）或 `>= 3`（杠）时显示对应按钮。
  - `handleEvent`：支持 `ankan`、`kakan`；`consumed` 使用 `(consumed || []).forEach` 避免 undefined。
  - 定缺/出牌等状态文案改为中文（如「请选择定缺花色」「轮到你出牌」）。
- **后端**
  - 对局线程改为仅在收到前端 `start_game` 时启动（`receiver` 内调用 `start_game_thread`），连接时不再自动开局。
  - `_reconstruct_state` 默认分数已为 60000（血战到底规则）。

## 四、后续可优化

- 加杠/暗杠：前端提供 加杠、暗杠 按钮并与后端协议一致。
- 剩余牌数：若后端在事件中提供剩余张数，前端在中央显示。
- 断线重连：重连时若对局未结束，用 `latest` 的 `state_update` 重放并恢复 UI。
- 音效：和牌、出牌等简单音效。
- 单会话多局：对局结束后清空/重置队列与状态，允许立即再开一局而不刷新页面。
