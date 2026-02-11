# 第二轮无法打牌 - 根因分析

## 一、现象

第一轮（第一次出牌）正常，第二轮（再次轮到玩家出牌）时无法点击打牌。

## 二、消息流（libblood → 前端）

```
libblood game.rs:
  poll()     → 对需行动的玩家调用 set_scene
            → MjaiLogBatchAgent.set_scene → update_state(Python) → state_queue.put(state_update)
  commit()   → 对需行动的玩家调用 get_reaction
            → evaluate() → react_batch(Python)
            → state_queue.put(action_request)
            → state_queue.put(state_update)
```

因此前端收到的顺序为：

1. **state_update**（来自 set_scene / update_state）
2. **action_request**（来自 react_batch）
3. **state_update**（来自 react_batch，带 analysis）

## 三、根因

### 问题：replay 与 action_request 对 canDiscard 的冲突

| 步骤 | 消息 | 处理 |
|-----|------|------|
| 1 | state_update | replayEvents（异步）→ 处理完整事件序列 |
| 2 | action_request | handleActionRequest → **canDiscard = true** |
| 3 | state_update | replayEvents（异步）→ 再次处理完整事件序列 |

第二轮时，events 中始终包含**第一轮的 dahai(0)**。

replay 中 dahai 分支逻辑：

```js
if (ev.actor === state.myPlayerId) {
    state.canDiscard = false;  // ← 问题所在
    // ...
}
```

- 第一次 replay：处理 dahai(0) → canDiscard = false
- action_request：canDiscard = true
- 第二次 replay：**再次**处理 dahai(0) → canDiscard = false

第三次处理（第二次 replay）在 action_request 之后执行，因此覆盖了正确的 canDiscard，导致第二轮无法打牌。

### 时序示意

```
t0: state_update 1 到达 → replay 1 启动（异步）
t1: action_request 到达 → canDiscard = true ✓
t2: state_update 2 到达 → replay 2 启动（异步）
t3: replay 2 执行到 dahai(0) → canDiscard = false ✗ 覆盖 t1
t4: 用户看到 canDiscard = false，无法点击
```

## 四、设计问题

`canDiscard` 被两个来源共同修改：

1. **action_request**：唯一应表示“当前能否出牌”的来源
2. **replay**：处理历史事件时，在看到自家 dahai 时也把 canDiscard 设为 false

replay 负责的是**视觉状态**（手牌、河牌、副露、currentActor），不应决定**交互状态**（canDiscard）。  
历史中的 dahai(0) 与当前能否出牌无关，replay 不应在其上修改 canDiscard。

## 五、修复方案（非打补丁）

原则：**canDiscard 仅由 action_request 和用户交互驱动，replay 不再修改 canDiscard**。

### 1. 从 replay 中移除对 canDiscard 的修改

- dahai 分支：去掉 `state.canDiscard = false`
- pon/kan 分支：去掉 `state.canDiscard = false`

canDiscard 的更新只保留在：

- **handleActionRequest**：根据是否有 dahai 设置 canDiscard
- **onTileClick**：出牌后置为 false（乐观更新）
- **doAction**：选择 pon/kan/hu/pass 后置为 false
- **game_over / start_kyoku / start_game**：重置为 false

### 2. 安全性

- 用户出牌后：onTileClick 已置 canDiscard = false
- 下一家收到 action_request 时：若为 pon/kan/hu，则 canDiscard = false；若为 dahai，则 canDiscard = true
- replay 只负责展示，不再覆盖 canDiscard

## 六、结论

根因是 **replay 与 action_request 对 canDiscard 的职责边界不清**：replay 在重放历史 dahai 时错误地覆盖了 action_request 设置的正确值。  
修复方式是：**明确 canDiscard 的权威来源为 action_request，replay 不再修改 canDiscard**。
