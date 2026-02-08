# 左右两家布局深度分析

## 1. 当前 DOM 与 Flex 结构

- 四家共用同一 HTML 结构（v-for），DOM 完全一致。
- `.player-area`：`flex-direction: column`，`justify-content: flex-end`，`align-items: center`，`gap: 8px`。
- 子元素顺序：`kawa-area` (order 1) → `player-info-and-hand` (order 2)，内部为 `player-info-card` → `hand-row`（tehai + fuuro）。

在**未旋转**的坐标系下：
- 列的方向：上 = 靠近牌桌“上方”，下 = 靠近牌桌“下方”。
- 因为 flex-end，整块内容在列的**底部**：从上到下 = 空白 → kawa → info+hand。
- 所以**列的顶部**是 kawa，**列的底部**是 hand。

## 2. 旋转与视口方向的对应关系

- `rotate(-90deg)`（逆时针）：元素“上”→ 视口“右”，“下”→ 视口“左”。
- `rotate(90deg)`（顺时针）：元素“上”→ 视口“左”，“下”→ 视口“右”。

因此：
- **当前 player-1 (右家) 用 `-90deg`**：列“下”(hand) → 视口“左”，列“上”(kawa) → 视口“右”。  
  → 手牌在**左侧**（靠中心），河牌在**右侧**（靠边）。和“右家手牌在右边缘”的预期**相反**。
- **当前 player-3 (左家) 用 `90deg`**：列“下”(hand) → 视口“右”，列“上”(kawa) → 视口“左”。  
  → 手牌在**右侧**（靠中心），河牌在**左侧**（靠边）。和“左家手牌在左边缘”的预期**相反**。

结论：**左右两家的旋转方向目前是反的**，导致两边“谁靠中心、谁靠边”都错，而且错法不同，所以看起来“完全不一样”。

## 3. 正确对应关系

- **右家 (player-1)**：手牌应在视口**右侧** → 需要列的“下”→ 视口“右” → 用 **`rotate(90deg)`**。
- **左家 (player-3)**：手牌应在视口**左侧** → 需要列的“下”→ 视口“左” → 用 **`rotate(-90deg)`**。

因此应**对调**当前左右两家的旋转角度。

## 4. 左家独有规则带来的不对称

当前仅对 player-3 有：
- `.player-area.player-3 .hand-row { flex-direction: row-reverse; }`
- `.player-area.player-3 .tehai-area { flex-direction: row-reverse; }`
- `.player-area.player-3 .fuuro-area { margin-left: 0; margin-right: 20px; }`

右家没有对应设置，导致：
- 左右两家的 flex 子顺序、margin 不一致；
- 旋转后，信息卡与牌列的相对位置、空白分布会不同。

要“样式大小完全一样”，应让左右**共用同一套 flex 和尺寸规则**，**唯一**区别只有旋转角度（90deg / -90deg），这样才是严格镜像。

## 5. 修改方案

1. **对调旋转**：player-1 改为 `rotate(90deg)`，player-3 改为 `rotate(-90deg)`，使两边都是“kawa 靠中心、hand 靠边缘”。
2. **去掉左家独有规则**：删除 player-3 的 row-reverse 和 fuuro 的 margin-right，左右使用相同的 `.hand-row` / `.tehai-area` / `.fuuro-area` 样式。
3. **保持现有左右共用样式**：gap、min-width、info 卡宽、hand-row margin、kawa max-width 等继续用同一组选择器 `.player-1, .player-3`，保证尺寸与间距一致。

这样左右两家在布局和样式上完全一致，仅通过旋转方向形成镜像。
