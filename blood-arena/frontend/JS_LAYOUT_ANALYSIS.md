# JS 与左右样式不一致 — 深度分析

## 1. 数据流概览

- **playerZones** = `PLAYER_ZONES`（常量），顺序：`top(seat 2)` → `left(seat 3)` → `right(seat 1)` → `bottom(seat 0)`。
- 模板用 `v-for="z in playerZones"`，`:key="z.zone"`，所以 **DOM 顺序** 固定为：上 → 左 → 右 → 下（对应 class `player-2` → `player-3` → `player-1` → `player-0`）。
- 每个区域用 **z.seat** 取数据：`state.players[z.seat]`，用 **z.pose** 取图：`getPaiImage(tile, z.pose)`。
- 布局完全由 CSS 的 `.player-0` ~ `.player-3` 控制（绝对定位 + 旋转），与 DOM 顺序无关；**样式** 只和 class、以及图片资源有关。

## 2. 根本原因：左右用了不同的 pose → 不同的图片

```js
// app.js
const PLAYER_ZONES = [
    { zone: 'top',  seat: 2, pose: 2 },
    { zone: 'left', seat: 3, pose: 3 },  // 左家 pose = 3
    { zone: 'right', seat: 1, pose: 1 }, // 右家 pose = 1
    { zone: 'bottom', seat: 0, pose: 0 }
];
```

- 牌背：`getPaiImage('?', pose)` → `/static/images/p_bk_0|1|2|3.gif`
- 牌面：`getPaiImage(tile, pose)` → `/static/images/p_${name}_0|1|2|3.(gif|png)`

所以：
- **右家** 始终用 **pose 1** 的图（`p_bk_1.gif`、`p_*_1.*`）。
- **左家** 始终用 **pose 3** 的图（`p_bk_3.gif`、`p_*_3.*`）。

只要资源里 `pose 1` 和 `pose 3` 不是同一张图（或不是左右对称的图），左右两边的**牌面/牌背**就会不一样，看起来就像“样式不一致”。  
这不是 CSS 写错，而是 **JS 传的 pose 不同 → 左右加载的图片不同**。

## 3. 其他 JS 点（与“样式一致”的关系）

| 项目 | 结论 |
|------|------|
| `handleTileClick(tile, i)` 定义只用了 `tile` | 仅少用了一个参数，不影响样式。 |
| `state.players[z.seat]` | 数据按座位取，正确，不影响左右对称。 |
| `:key="z.zone"` | 稳定、按区域，无副作用。 |
| DOM 顺序 2→3→1→0 | 四块都是 `position: absolute`，视觉顺序由 CSS 的 transform 决定，与 DOM 顺序无关。 |

结论：**和“样式一直不一样”直接相关的，主要是 pose 导致左右图片不同。**

## 4. 推荐改法：左右共用同一套图 + CSS 镜像

- **JS**：让左家（seat 3）在**取图时**也用和右家相同的 pose（例如都用 pose 1），这样左右用同一套资源。
- **CSS**：对左家（`.player-3`）的牌图做水平镜像（如 `transform: scaleX(-1)`），视觉上仍是“从左边看”，但素材一致。

这样“样式”在数据层（同一张图）和视觉层（左右对称）都一致，后续要统一改样式也只需改一套图 + 一处 CSS。
