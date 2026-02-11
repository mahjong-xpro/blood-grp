# 手牌 15 张 Bug 根因深度分析

## 1. 问题描述

- **现象**：开局或摸牌后手牌显示 15 张（应为 14 张）
- **当前补丁**：在 `tsumo` 分支加 `if (state.tehai.length < 14)` 再设置 `tsumoTile`，防止 15 张

## 2. 数据模型与显示逻辑

### 2.1 手牌表示

```
总牌数 = state.tehai.length + (state.tsumoTile ? 1 : 0)
```

- **tehai**：13 张基础手牌（或碰/杠后的 10/7/4 张）
- **tsumoTile**：刚摸的牌，单独显示（庄家首摸、普通摸牌、杠后补牌）

### 2.2 显示代码 (index.html)

```html
<div class="tile" v-for="(t, i) in hand(0)" />   <!-- hand(0) = state.tehai -->
<div class="tile tsumo" v-if="state.tsumoTile" />  <!-- 自摸牌单独显示 -->
```

15 张出现等价于：`tehai.length + (tsumoTile?1:0) = 15`，即：
- 情况 A：`tehai.length = 14` 且 `tsumoTile` 有值
- 情况 B：`tehai.length = 15`

## 3. 数据流与事件来源

### 3.1 事件来源

| 来源 | 时机 | 内容 |
|------|------|------|
| `update_state` | libblood `set_scene`（poll 阶段） | `ctx.log` 序列化为 `events_json` |
| `react_batch` | libblood `get_reaction`（commit 阶段） | `game_states[i].events_json`（与 set_scene 同源） |

两处都来自同一 `log`，事件内容一致。

### 3.2 消息顺序

```
poll: set_scene(human) → update_state → state_queue.put(state_update)
commit: get_reaction → react_batch → state_queue.put(action_request) → state_queue.put(state_update)
```

因此前端可能收到：
1. `state_update`（来自 update_state）
2. `action_request`
3. `state_update`（来自 react_batch）

### 3.3 libblood 的牌局流程

```
haipai(): StartKyoku(tehais = haipai, 13x4) → Tsumo(oya, pai)
```

- `StartKyoku.tehais` 固定为 13 张/人
- 庄家首摸来自单独 `Tsumo` 事件

## 4. 根因分析

### 4.1 tehai 何时会变成 14？

tehai 的修改只在 replay 中发生：

| 事件 | 修改 |
|------|------|
| `start_kyoku` | `tehai = ev.tehais[myPlayerId]`（替换） |
| `dahai`（自家） | 从 tehai 移除 1，有 tsumoTile 则 push 到 tehai |
| `pon/kan/...` | 从 tehai 移除 consumed，有 tsumoTile 则 push |

- 若 `ev.tehais` 是 13 张：`start_kyoku` 后 tehai 恒为 13
- `dahai`/`pon` 后：remove 1 + push tsumoTile → 总数仍为 13
- **唯一能让 tehai 变成 14 的路径**：`start_kyoku` 的 `ev.tehais[myPlayerId]` 本身是 14 张

### 4.2 15 张的形成路径

```
tehai.length = 14 且 tsumoTile 有值 → 显示 15 张
```

因此需要同时满足：
1. `start_kyoku` 中的 `ev.tehais[myPlayerId]` 为 14 张
2. 随后 `tsumo` 事件又设置了 `tsumoTile`

按 libblood 设计，`StartKyoku` 的 `tehais` 固定为 13，不会把首摸合并进去，所以正常流程不应出现 14。

### 4.3 可能的异常来源

#### 假设 1：后端发出 14 张 tehais

- **检查点**：libblood `Event::StartKyoku`、`haipai`、JSON 序列化
- **结论**：`haipai` 为 `[[Tile; 13]; 4]`，且 `StartKyoku` 直接用 `haipai`，无合并首摸逻辑
- **可能性**：低，除非存在未发现的序列化或透传错误

#### 假设 2：replay 时序/竞态

- `replayEvents` 为 fire-and-forget，不 await
- 多个 `state_update` 可导致多个 replay 并发：
  - 旧 replay 可能被 `myReplayId !== replayId` 中止
  - 新 replay 从 `startIdx` 开始，可能基于错误初始状态

关键代码：

```javascript
const startIdx = (events.length < last) ? 0 : last;
if (startIdx === 0) state.tsumoTile = null;
```

- `events.length < last`：新局或事件回退，从头重放，会清 `tsumoTile`
- `startIdx > 0`：增量重放，不会处理 `start_kyoku`，tehai 来自前一状态

若存在竞态：
- Replay A 处理到 `start_kyoku`（tehai=13）
- Replay B 开始，`last` 尚未更新，`startIdx` 计算错误
- 可能出现“部分重放 + 部分重置”的混乱状态

#### 假设 3：重复或乱序重放

- 多次 `state_update` 导致同一批事件被重放多次
- `if (events.length === last) return` 可避免重复，但需 `lastReplayedEventCount` 已正确更新
- 若 `last` 更新滞后或与另一 replay 交错，仍可能重复处理

#### 假设 4：dahai 分支未正确移除牌

dahai 分支逻辑：

```javascript
if (state.tsumoTile === ev.pai) {
    state.tsumoTile = null;  // 摸切
} else {
    const idx = state.tehai.indexOf(ev.pai);
    if (idx > -1) {
        state.tehai.splice(idx, 1);
        if (state.tsumoTile) state.tehai.push(state.tsumoTile);
        state.tsumoTile = null;
    }
}
```

- 若 `ev.pai` 与 `state.tehai` 中牌格式不一致（如 `"1m"` vs `"1mr"`），`indexOf` 失败，`idx === -1`
- 此时只清 `tsumoTile`，不执行 splice，tehai 不减少
- 下一轮 `tsumo` 再设 `tsumoTile` → 可能出现 14 + 1 = 15
- 注：血战到底无赤牌，牌格式通常为 `"1m"`–`"9s"`，该情形概率较低

## 5. 结论与建议

### 5.1 最可能根因

1. **dahai 匹配失败**：`ev.pai` 与 tehai 中牌表示不一致，导致 `indexOf` 失败，手牌未被正确移除，后续 tsumo 叠加成 15 张。
2. **状态竞态**：多 replay 并发 + `lastReplayedEventCount` 更新时机，导致部分重放、部分重置，出现异常中间状态。

### 5.2 非补丁式修复思路

1. **统一牌表示**  
   - 在 libblood、Python、前端之间约定统一格式（如 `"1m"` / `"1mr"` 规则）  
   - 在 dahai 分支做格式兼容或规范化后再 `indexOf`

2. **replay 串行化**  
   - 对 `replayEvents` 使用队列或锁，确保同一时刻只执行一个 replay  
   - 或改为 `await replayEvents`，与 `handleMessage` 的流程对齐

3. **明确状态来源**  
   - 将“权威状态”放在后端，前端只做展示  
   - 或将 `state_update` 设计为“完整状态快照 + 增量事件”的混合模式，减少依赖多次重放推导状态

4. **增加调试**  
   - 在 tsumo/dahai 分支打点：`tehai.length`、`tsumoTile`、`ev.pai`、`indexOf` 结果  
   - 记录 `state_update` 顺序与 `lastReplayedEventCount` 变化，定位竞态

### 5.3 验证步骤

1. 在 dahai 分支加日志：当 `idx === -1` 且 `ev.actor === myPlayerId` 时，打印 `ev.pai`、`state.tehai`、`state.tsumoTile`
2. 在 `replayEvents` 开头/结尾记录 `replayId`、`last`、`startIdx`、`events.length`
3. 复现 15 张时，确认是否伴随 `indexOf` 失败或 replay 并发
