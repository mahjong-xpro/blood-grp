# 河牌重放问题深度分析

## 一、数据流

```
libblood Board.log (当前局完整 log)
    ↓
set_scene → HumanEngine.update_state(events_json) → state_queue.put(state_update)
    ↓
poll 完成 → get_reaction → HumanEngine.react_batch
    → state_queue.put(state_update)  // 再次发送 events + analysis
    → state_queue.put(action_request)
    ↓
broadcast_loop 依次推送给前端
```

**结论**：同一人类回合内，前端可能收到 **多次** `state_update`，且 events 内容相同或递增。

## 二、根本原因

### 1. 完整重放导致视觉“回放”

每次 `replayEvents` 都从 `start_kyoku` 开始，会执行：

```js
state.discards = [[], [], [], []];  // 河牌清空
// 然后 for 每个 dahai: state.discards[ev.actor].push(ev.pai)
```

所以每次都会：**河牌先清空，再按顺序重新添加**。即使用户只打了一张牌，也会看到整条河从头“重播”一遍。

### 2. 同一次轮次可能收到多次 state_update

- `update_state`（set_scene）会发一次
- `react_batch` 也会发一次（带 analysis）

若 events 数量相同，应视为**重复**，不应再重放。

### 3. 增量处理的条件

- `events.length > lastReplayedEventCount`：新增了事件，只处理 `[last, length)` 段
- `events.length === lastReplayedEventCount`：重复推送，**跳过**
- `events.length < lastReplayedEventCount`：新一局，需从 0 开始完整处理

## 三、修复方案

1. **增量重放**：只处理 `[lastReplayedEventCount, events.length)` 的新事件，不碰历史。
2. **去重**：当 `events.length === lastReplayedEventCount` 时直接 return，不重放。
3. **新局识别**：`events.length < lastReplayedEventCount` 时，从 0 开始处理（新一局）。
4. **保留延迟**：仅对新增动作在之间加 1.5 秒延迟。
