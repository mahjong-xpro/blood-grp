# 打牌中途少一张牌 Bug 根因深度分析与重构方案

## 1. 问题描述

- **现象**：打牌过程中，手牌突然少一张（应为 13/14 张，实际少 1）
- **用户要求**：深度分析、非补丁、重构业务流程

## 2. 当前架构的根本问题

### 2.1 双源状态推导

前端手牌由**两套逻辑**共同推导：

| 来源 | 时机 | 修改 tehai/tsumoTile 的方式 |
|------|------|---------------------------|
| **乐观更新** | 用户点击出牌 | 立即 splice/push，设 optimisticDahai |
| **事件重放** | 收到 state_update | 按 events 顺序 replay，分支处理 dahai/pon/kan |

两套逻辑必须在时序、逻辑上完全一致，否则会出现：
- 多删（少牌）
- 少删（15 张）
- 重复处理同一条 dahai

### 2.2 关键代码路径

**dahai 分支（arena.js：290-312）**：

```javascript
if (state.tsumoTile === ev.pai) {
    state.tsumoTile = null;           // 摸切
} else if (state.optimisticDahai === ev.pai) {
    state.tsumoTile = null;           // 已乐观更新，跳过 splice 避免重复删
    // 不在此清 optimisticDahai：若同一 dahai 被重复 replay，else 分支会 indexOf 找到另一张同牌并误删
} else {
    const idx = state.tehai.indexOf(ev.pai);
    if (idx > -1) {
        state.tehai.splice(idx, 1);
        if (state.tsumoTile) state.tehai.push(state.tsumoTile);
        state.tsumoTile = null;
    } else {
        state.tsumoTile = null;       // 找不到牌，只清 tsumoTile
    }
}
```

**设计意图**：
- 同牌多张时，用 optimisticDahai 避免 replay 再次 splice 导致误删另一张
- 因此 optimisticDahai 不能随便清，否则重复 replay 会走到 else 分支，indexOf 会删错牌

**脆弱点**：
1. optimisticDahai 的存活时机依赖消息顺序和 replay 是否重复
2. start_kyoku / start_game 会清 optimisticDahai，若在「已出牌未确认」的窗口期内收到，会导致下一次 replay 走 else 分支
3. replay 为 fire-and-forget，多个 state_update 可并发，lastReplayedEventCount 与事件边界可能错乱

## 3. 少一张牌的形成路径

### 3.1 路径 A：dahai 被 splice 两次

1. 用户打出 5m（手牌有两张 5m）
2. 乐观更新：splice 一张 5m，push tsumoTile，设 optimisticDahai = "5m"
3. 正常 replay：命中 optimisticDahai 分支，不 splice ✓
4. 若 optimisticDahai 被提前清（如 start_kyoku）、或 replay 异常走 else 分支：
   - else 分支 indexOf("5m") 找到**剩余那张** 5m
   - splice 删除 → 多删一张 → **少一张**

### 3.2 路径 B：tsumo 未正确设置

1. `state.tehai.length < 14` 的补丁：tehai 已达 14 时不设 tsumoTile
2. 若因 replay 竞态或状态错误，tehai 被错误置为 14，下一次 tsumo 就不会设 tsumoTile
3. 显示 14 张（实际应为 13+tsumo=14），用户感觉「少摸了一张」

### 3.3 路径 C：增量 replay 的边界错误

```javascript
const startIdx = (events.length < last) ? 0 : last;
```

- `events.length < last`：新局或事件回退，startIdx=0，会清 tsumoTile
- 多个 replay 并发时，若 last 更新滞后或交错，startIdx 可能算错
- 导致部分事件被跳过或重复处理，tehai 与真实状态分叉

### 3.4 路径 D：pon/kan 的 consumed 与 tsumoTile 重叠

pon 时 consumed 来自手牌，tsumoTile 可能是刚摸的同一张牌。若遍历 consumed 时先清 tsumoTile、再 indexOf 找牌，逻辑需严格一致；否则可能多删或少删。

## 4. 根因总结

| 根因 | 说明 |
|------|------|
| **双源推导** | 乐观更新 + 事件重放共同维护手牌，易分叉 |
| **optimisticDahai 生命周期脆弱** | 依赖消息顺序与 replay 次数，易被误清或误用 |
| **replay 并发** | fire-and-forget，多 replay 交织，last 与 startIdx 不可靠 |
| **无权威状态** | 后端有完整 player_state，但未下发给前端，前端只能从 events 推导 |

## 5. 重构方案：后端为唯一权威状态

### 5.1 核心思路

**手牌不再由前端 replay 推导，改为由后端直接下发。**

- libblood 已有 `PlayerState.tehai`、`last_self_tsumo`
- `GameState` 已包含 `state: PlayerState`
- 只需在 `state_update` 中附带 `tehais`、`my_tsumo`，前端直接使用

### 5.2 协议变更

**state_update 扩展**：

```json
{
  "type": "state_update",
  "data": {
    "events": [...],
    "tehais": [["1m","2m",...], [], [], []],   // 四家手牌（自家可见，他家可为空）
    "my_tsumo": "5m" | null,                    // 自家刚摸的牌
    "analysis": {}
  }
}
```

- `tehais[myPlayerId]`：自家手牌，由后端 PlayerState 算出
- `my_tsumo`：后端 `last_self_tsumo` 转字符串

### 5.3 实现要点

1. **libblood → Python**  
   - `set_scene` 时已有 `state: PlayerState`  
   - 在 `update_state` 调用处增加参数：`tehais`、`my_tsumo`（或直接传 state 由 Python 抽取）

2. **Python → 前端**  
   - HumanEngine.update_state 构造 `tehais`、`my_tsumo` 放入 `state_update.data`

3. **前端**  
   - 收到 `state_update` 若带 `tehais`、`my_tsumo`：  
     - `state.tehai = data.tehais[myPlayerId] ?? state.tehai`  
     - `state.tsumoTile = data.my_tsumo ?? null`  
   - replay 仅用于：碰/杠/胡动画、他家打牌、流局等**不涉及自家手牌**的逻辑  
   - 或：replay 保留，但**不再修改 tehai/tsumoTile**（仅更新 discards、fuuro、currentActor 等）

### 5.4 乐观更新策略

- **方案 A（推荐）**：保留乐观更新，但收到带 `tehais` 的 state_update 时，**用服务器状态覆盖**，不再用 replay 改动 tehai  
- **方案 B**：取消乐观更新，等服务器确认后再更新界面（延迟略高，但逻辑最简单）

### 5.5 迁移步骤

1. 后端：在 `update_state` 和 `react_batch` 的 state_update 中增加 `tehais`、`my_tsumo`
2. 前端：优先使用 `data.tehais`、`data.my_tsumo`，无则回退到 replay
3. 实测无问题后，从 replay 中移除对 tehai/tsumoTile 的修改
4. 逐步删除 optimisticDahai 及相关分支

## 6. 小结

- **少一张牌** 来自：双源推导 + optimisticDahai 时机 + replay 并发的组合效应  
- **根本解决**：后端下发权威手牌，前端只展示，不再用 replay 推导自家 tehai  
- **重构收益**：消除 15 张、少一张、重复 replay 等手牌相关 bug，逻辑更清晰、更易维护

---

## 7. 已实施重构（2025-02）

**约束**：不修改 libblood，以免影响训练相关代码。

### 7.1 后端（blood-arena）

- **update_state**：在 `state_update` 中附带 `tehais`、`my_tsumo`（来自 shadow state）
- **react_batch**：从 libblood `PlayerState` 取权威手牌，同步到 shadow state，并在 `state_update` 中附带 `tehais`、`my_tsumo`
- **新增**：`_tehai_from_counts()`，将 `[u8;27]` 转为 mjai 牌面字符串列表

### 7.2 前端（arena.js）

- **updateFullState**：收到 `tehais` 时，直接覆盖 `state.tehai`、`state.tsumoTile`，并清空 `optimisticDahai`
- **replayEvents**：新增 `hasAuthoritativeHand`，为 true 时不再修改 tehai/tsumoTile（start_kyoku、tsumo、dahai、pon/kan 分支均跳过）
