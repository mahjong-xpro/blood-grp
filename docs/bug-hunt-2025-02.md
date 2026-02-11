# Bug 排查报告 (2025-02 续)

## 一、已修复（上一轮）

- **validActionsShown**：碰/杠/胡按钮被 `!hasPonKanHu` 覆盖为 false，永不显示 → 已改为 `validActionsShown = true`
- **hora/ryukyoku/杠 deltas**：replay 未更新 scores → 已在前端 replay 中应用 deltas

## 二、已按推荐修复（本轮）

- **BUG-A**：`handleActionRequest` 增加 `actions || []` 空值防护
- **BUG-B**：`broadcast` 保存 `last_state_update`；`connect` 时若 latest 为 action_request 先发 state_update
- **BUG-C**：game_over 单局结束设 `phase = 'between_games'`
- **BUG-D**：update_state 异常时发送降级 state_update
- **BUG-E**：tilesLeft 初始值改为 56
- **BUG-F**：start_kyoku 显式重置 `currentActor = -1`

---

## 三、需核实/非 Bug

| 项目 | 说明 |
|------|------|
| **broadcast_loop _thread_finished** | 收到后 `continue`，不向客户端发送，逻辑正确 |
| **hand(0) vs handTiles(0)** | 自家用 `hand(0)`（仅 tehai）+ 单独 tsumoTile，他家用 `handTiles(offset)`，逻辑正确 |
| **定缺 panel** | `phase === 'dingque'` 时显示，不依赖 currentActor |
| **validActions :key** | 后端最多各一种 pon/kan/hu/pass，key 唯一 |
| **hasAuthoritativeHand** | `data.tehais[myPlayerId]` 为 [] 时 truthy 仍为 true，会应用空手牌，符合预期 |

---

## 四、建议修复优先级（均已修复）

| 优先级 | Bug | 状态 |
|--------|-----|------|
| P1 | BUG-A actions 空值 | ✅ |
| P2 | BUG-B 重连状态不完整 | ✅ |
| P3 | BUG-C 单局结束 phase | ✅ |
| P4 | BUG-D update_state 异常 | ✅ |
| P5 | BUG-E/F | ✅ |
