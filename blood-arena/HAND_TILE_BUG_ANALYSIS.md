# 手牌数量异常 Bug 深度分析

## 现象

打着打着，牌变少了（手牌逐渐减少）。

## 数据流

```
1. 后端 react_batch: 发送 action_request → state_update
2. 前端 handleMessage: 先处理 action_request，再处理 state_update
3. state_update 触发 replayEvents（fire-and-forget，异步）
4. 用户点击出牌 → send dahai → state.tsumoTile = null
5. 后端收到 action，处理，下一轮发 state_update
6. 前端 replayEvents 处理 tsumo、dahai 等事件
```

## 关键时序

| 时刻 | 事件 | state.tehai | state.tsumoTile |
|-----|------|-------------|-----------------|
| T0 | start_kyoku | 13 | null |
| T1 | tsumo (我) | 13 | 摸牌 |
| T2 | 用户点击打手牌 | 13 | **null（被清空）** |
| T3 | replay 处理 dahai | 12 | null |

**Bug 根因**：T2 时 `onTileClick` 无条件执行 `state.tsumoTile = null`，但未从 tehai 移除打出的牌、也未把摸牌并入 tehai。等 T3 replay 处理 dahai 时：

- 会执行 `tehai.splice(idx, 1)` 移除打出的牌 ✓
- 会检查 `if (state.tsumoTile)` 决定是否加入摸牌 ✗ **此时 tsumoTile 已被清空，不会加入**

结果：**移除 1 张，未补 1 张，净少 1 张牌**。每次从手牌打牌都会少一张，牌数持续减少。

## 修复方案

**乐观更新**：在用户点击打手牌时，立即更新本地状态：

1. 从 tehai 移除打出的牌
2. 将 tsumoTile 加入 tehai（若存在）
3. 清空 tsumoTile

这样 replay 处理 dahai 时，ev.pai 已不在 tehai（idx=-1），会走「已并入」分支，只清空 tsumoTile，不再重复操作，牌数保持正确。

## 其他相关修复（防多牌）

- **tsumo**：仅当 `!tehai.includes(ev.pai)` 时设置 tsumoTile，避免 fire-and-forget 重复设置导致多牌
- **dahai**：当 idx=-1 时执行 `tsumoTile = null`，避免摸牌既在 tehai 又单独显示
