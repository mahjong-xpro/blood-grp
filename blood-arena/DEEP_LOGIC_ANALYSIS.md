# 手牌逻辑深度分析

## 〇、业务逻辑 Bug 清单

| Bug | 描述 | 严重性 |
|-----|------|--------|
| BUG-A | game_over 未清 canDiscard/validActions，局终仍可点击出牌 | 高 ✓ |
| BUG-B | replay 处理自家 dahai 后未清 canDiscard，currentActor 已变下一家 | 高 ✓ |
| BUG-C | start_kyoku 未清 canDiscard/validActions，新局可能残留 | 中 ✓ |
| BUG-D | dahai 后 currentActor 未跳过已和牌玩家，(actor+1)%4 错误 | 高 ✓ |

## 一、数据流

```
用户点击 → onTileClick → send(dahai) + 乐观更新
                        ↓
后端处理 → state_queue.put(state_update)
                        ↓
前端 handleMessage → updateFullState → replayEvents (fire-and-forget)
                        ↓
replay 处理 dahai → 若 optimisticDahai 匹配则跳过 remove/add
```

## 二、乐观更新 vs Replay 的竞态

| 场景 | 乐观更新 | Replay dahai | 结果 |
|-----|---------|--------------|------|
| 正常 | 移除1、加tsumo → 13 | 匹配 optimisticDahai，跳过 | 13 ✓ |
| 一对牌 | 移除1、加tsumo → 13 | 匹配，跳过（否则 indexOf 会找到另一张再移除） | 13 ✓ |
| 打摸牌 | 只清 tsumoTile | state.tsumoTile===ev.pai 分支 | 13 ✓ |
| 对手碰/杠 | 同上 | action_request 先到但不清 optimisticDahai | 13 ✓ |

## 三、潜在 Bug

### BUG-1: else 分支未清 optimisticDahai（防御性）

当走 else 分支（idx>-1 或 idx=-1）时，若存在陈旧的 optimisticDahai，应在处理完后清空，避免影响后续回合。

### BUG-2: start_kyoku 在增量 replay 时

若 `events.length < last`（新局），startIdx=0，会执行 start_kyoku 并清 optimisticDahai。此时 tehai 来自 ev.tehais，状态正确。但若上次乐观更新尚未被 replay 消费，optimisticDahai 会残留；在新局中清空是正确行为。

### BUG-3: 消息顺序 action_request → state_update

后端 react_batch 先发 action_request，再发 state_update。若用户快速点击，可能在 state_update 到达前完成乐观更新。此时 optimisticDahai 已设置，等 state_update 到达后 replay 会正确匹配。

### BUG-4: 打摸牌时 tsumoTile 已被清空

打摸牌：idx==='tsumo'，只清 tsumoTile。Replay 时 `state.tsumoTile === ev.pai` 为 false（已清空），会走 else。`indexOf(ev.pai)` 在 tehai 中找不到（摸牌不在 tehai），idx=-1，仅清 tsumoTile。手牌保持 13 ✓。

## 四、加固建议

1. **else 分支清空 optimisticDahai**：任一处理完自家 dahai 的分支都应清空 optimisticDahai
2. **保持当前 start_kyoku / start_game 中的清空逻辑**

---

## 五、继续分析（回合指示与状态）

### BUG-E: currentActor === -1 时回合指示错误（已修复 ✓）

**现象**：定缺阶段、或尚未收到 tsumo 等事件时，`currentActor` 为 -1，原逻辑 `['下家','对家','上家'][state.currentActor > 0 ? state.currentActor-1 : 0]` 会退化为索引 0，显示「等待 下家...」，但此时可能是等待用户选定缺或 AI 思考，并非特定玩家。

**修复**：新增 `turnIndicatorText` 计算属性，当 `currentActor === -1` 时显示「等待中...」。

### game_over 单局 vs 整场

- **单局结束**（`is_match_over=false`）：不设 `gameEnded=true`，仅清 `canDiscard`/`validActions`。随后 `start_kyoku` 会设 `gameEnded=false`，避免中途短暂显示「再来一局」。
- **整场结束**（`is_match_over=true`）：设 `gameEnded=true`、`matchOver=true`，显示「再来一局」按钮。

### getHandTileCount 杠牌逻辑（已正确）

`getHandTileCount` 已按副露实际张数计算：`m.tiles ? m.tiles.length : 3`，pon=3、kan=4 均正确。
