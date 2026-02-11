# 出牌与动作栏逻辑深度分析

## 一、不该出牌时不能出牌

**现状**：`canDiscard` 由后端 `action_request` 控制，后端仅在轮到玩家时发送 `dahai` 动作。理论上不会出现「不该出牌却能出」的情况。

**风险**：消息乱序或 replay 延迟可能导致 `currentActor` 与 `canDiscard` 短暂不一致。

**加固**：在 `onTileClick` 中增加 `isMyTurn` 校验，形成双重防护。

## 二、对手未出牌时不显示碰/杠

**现状**：后端先发 `action_request`（含 pon/kan），再发 `state_update`（含对手 dahai）。前端会先展示动作栏，再通过 replay 显示对手出牌，导致「碰/杠先于对手出牌出现」。

**修复**：引入 `validActionsShown`：
- 收到含 pon/kan/hu 的 `action_request` 时设为 `false`
- 在 replay 中处理**对手**的 `dahai` 时设为 `true`
- 动作栏仅在 `validActionsShown` 为 true 时显示
- 仅有 `pass` 时立即显示（无需等待对手出牌）

## 三、出牌音效

**资源**：`/static/audio/` 已有各牌面音效（如 `5m.m4a`）。

**实现**：打牌时播放对应牌面音效；`?` 或未知牌使用默认音效。
