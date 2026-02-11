# 手牌显示与状态更新逻辑梳理

## 1. 数据流总览

```
libblood (poll)  → set_scene → update_state → state_queue.put(state_update)
libblood (commit) → get_reaction → react_batch → state_queue.put(action_request)
                                         → state_queue.put(state_update)
broadcast_loop   → 按 FIFO 取 state_queue → WebSocket 发送
前端 handleMessage → 按顺序处理 (state_update 会 await updateFullState)
```

## 2. 消息顺序（人类回合）

1. **Poll 阶段**：set_scene(human) → update_state → `state_update`
2. **Commit 阶段**：react_batch → `state_update`（先）→ `action_request`（后）

队列顺序：`state_update`（来自 update_state）→ `state_update`（来自 react_batch）→ `action_request`

**修复**：react_batch 改为先发 state_update 再发 action_request，保证前端先完成 replay 再显示可出牌。

## 3. 前端处理流程

### 3.1 handleMessage

- `state_update` → `await updateFullState(msg.data)`
- `action_request` → `handleActionRequest(actions)`（同步）
- `game_over` → 更新 scores、finalTehais 等

### 3.2 updateFullState

```javascript
await replayEvents(events, replayId, hasAuthoritativeHand, authData);  // 串行等待
// 等待 replay 完成并应用 authData 后再处理下一条消息
```

### 3.3 replayEvents（异步）

- 处理 events（start_kyoku、tsumo、dahai、pon/kan 等）
- 在 **他家 dahai** 前有 1500ms delay（首个 action 不 delay）
- 若 `hasAuthoritativeHand`，**不**在 replay 中修改 tehai/tsumoTile
- **结束时**应用 authData：覆盖 tehai、tsumoTile

### 3.4 已修复问题

**A. fire-and-forget 导致竞态** → 改为 `await replayEvents`，消息串行处理

**B. action_request 先于 state_update** → 后端改为先发 state_update，再发 action_request

**C. 权威手牌在 replay 结束才应用** → 保持；replay 结束时应用 authData，保证与 events 顺序一致

## 4. 后端手牌格式

- **tehais**：13 张基础手牌（不含刚摸的牌）
- **my_tsumo**：刚摸的 1 张，可为 null
- 显示总牌数 = `tehais.length + (my_tsumo ? 1 : 0)` = 14

## 5. 已实施修复

1. **await replay**：updateFullState 等待 replay 完成后再返回，保证下一条消息处理时状态已更新
2. **消息顺序**：react_batch 先发 state_update，再发 action_request
3. **authData 在 replay 结束时应用**：保证 AI 出牌动画 → 人类摸牌 的视觉顺序正确
