# 界面只显示一半（下一半看不到）— 深度分析与修复

## 原因分析

1. **body** 使用 `height: 100vh` + `overflow: hidden`，视口外的内容被裁掉，所以“下半”并不是没渲染，而是被裁掉看不见。

2. **总高度大于 100vh**：
   - **#app** 使用 `min-height: 100vh`，没有上限，会随内容增高。
   - **.app-root** 使用 `min-height: 100vh` + `flex: 1`，即“至少 100vh”，再叠上顶栏(48px)和操作栏，总高 = 48px + 100vh + 操作栏 > 100vh，所以必然溢出。

3. **Flex 子项不收缩**：
   - Flex 子项默认 `min-height: auto`，不会比内容更小，所以 `.main-content`、`.game-board-container` 会按内容高度撑开，不会为底部的操作栏“让位”。
   - 结果：牌桌区域把空间占满，操作栏被挤到视口下方，被 body 的 overflow: hidden 裁掉。

## 修复思路

- 整页**严格限制在 100vh**，不溢出。
- **顶栏**、**操作栏** 固定高度且不收缩（flex-shrink: 0）。
- **中间主区域** 使用 `flex: 1` 且 **min-height: 0**，允许被压缩，把剩余空间留给牌桌；牌桌再在剩余空间内做正方形（aspect-ratio + max-height: 100%）。

## 已做修改

| 选择器 | 修改 |
|--------|------|
| html | `height: 100%`，方便子级用百分比高度。 |
| body | 保留 `overflow: hidden`，增加 `height: 100%`、`min-height: 100vh`。 |
| #app | `min-height` 改为 **height: 100vh**，并 **overflow: hidden**，整页高度锁定为视口。 |
| .app-root | 去掉 `min-height: 100vh`，改为 **min-height: 0**，**overflow: hidden**，让中间可收缩。 |
| .main-content | 增加 **min-height: 0**，保证在 flex 中可被压缩。 |
| .game-board-container | 增加 **min-height: 0**，padding 略减为 12px。 |
| .game-board | 使用 **width: auto; height: auto; max-width: min(100%, 900px); max-height: 100%; aspect-ratio: 1/1**，在剩余空间内成正方形且不溢出。 |

效果：顶栏、牌桌、操作栏都在同一视口内，下半（操作栏及牌桌底部）不再被裁掉。
