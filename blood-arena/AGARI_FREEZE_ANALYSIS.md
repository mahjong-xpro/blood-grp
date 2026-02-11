# 和牌后卡死与剩余牌数 Bug 分析

## 一、现象

1. **卡死**：下家胡牌后，用户再胡牌，牌局卡死。
2. **剩余牌数不正确**：tilesLeft 显示错误。

## 二、根因

### 2.1 卡死根因

libblood 的 `set_scene` 只对**需要行动的玩家**调用。人胡牌后 `agari[0]=true`，不再需要行动，`set_scene` 不会再调人类引擎，只会调 AI 引擎。

因此：
- 人胡牌后，后续是 AI 对局（玩家 2、3）
- `set_scene` 只调用 AI 的 `MjaiLogBatchAgent`
- AI 的 `update_state` 要么不存在，要么没有把状态推给前端
- 前端永远收不到包含 hora 及后续事件的 `state_update`
- 界面停留在「等待」状态，表现为卡死

### 2.2 剩余牌数根因

`tilesLeft` 在每次 `tsumo` 事件时减 1。若因卡死收不到 AI 回合的 `state_update`，就收不到后续的 `tsumo` 事件，`tilesLeft` 不会继续减少，数值会偏高。

本质与卡死是同一问题：**AI 回合没有向前端推送 state_update**。

## 三、修复方案

用 `_ChampionWithHumanObserver` 包装 AI 引擎：
- 保留 AI 的 `react_batch`、`start_game` 等行为
- 在 `update_state` 中转发给 `HumanEngine.update_state`
- libblood 在 AI 回合调用 `set_scene` 时，会尝试调用 `update_state`，包装器将其转发给 `HumanEngine`，从而向前端推送 `state_update`

这样，AI 回合也会向前端推送事件，卡死和剩余牌数问题同时解决。

## 四、修改位置

`blood-arena/backend/game_manager.py`：

1. 新增 `_ChampionWithHumanObserver` 类
2. 将 `env.py_vs_py(human, ai_engine, ...)` 改为 `env.py_vs_py(human, champion, ...)`，其中 `champion = _ChampionWithHumanObserver(ai_engine, human)`
