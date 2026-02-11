# 左右玩家河牌间距深度分析

## 1. 影响间距的因素

### 1.1 直接因素

| 因素 | 位置 | 当前值 | 作用 |
|------|------|--------|------|
| `gap` | `.tile-group.river` | 2px (ME/对家) | 牌与牌之间的空隙 |
| `gap` | `.zone-left/right .tile-group.river` | 0 (已覆盖) | 左右河牌间距 |
| 牌尺寸 | `.river .tile` | 26×36px | 布局中的占位 |

### 1.2 间接因素

| 因素 | 位置 | 当前值 | 可能影响 |
|------|------|--------|----------|
| `.tile-group` 基类 | 行 279 | gap: 2px | 被 .tile-group.river 继承/覆盖 |
| `.tile` | 行 474 | border-radius: 4px | 不增加布局空间 |
| `.tile img` | 行 484 | max-width/height: 100% | 不增加布局空间 |
| `transform: rotate` | 行 312-314 | 90deg | transform 不改变布局盒，布局仍为 26×36 |

### 1.3 根本原因

**左右河牌为 column 布局，主轴为竖直方向。**
- 每张牌在主轴上的尺寸 = **height = 36px**
- 中心距 = 36px + gap
- 即使 gap=0，中心距仍为 **36px**

**ME 河牌为 row 布局，主轴为水平方向。**
- 每张牌在主轴上的尺寸 = **width = 26px**
- 中心距 = 26px + 2px = **28px**

**结论**：左右河牌间距大的根本原因是**牌在主轴上的尺寸为 36px**，而 ME 为 26px。gap 已设为 0，无法再减小。要减小间距，必须减小牌在主轴（竖直）上的尺寸。

## 2. 可行方案

### 方案 A：交换左右河牌宽高（推荐）

```css
.zone-left .river .tile,
.zone-right .river .tile {
    width: var(--tile-h-sm);   /* 36px 跨轴 */
    height: var(--tile-w-sm);  /* 26px 主轴 */
}
```

- 主轴尺寸变为 26px，中心距 = 26px（与 ME 一致）
- 需配合 `object-fit: contain` 避免图片被拉伸变形

### 方案 B：负 margin 重叠

```css
.zone-left .river .tile:not(:first-child),
.zone-right .river .tile:not(:first-child) {
    margin-top: -8px;
}
```

- 牌会重叠约 8px，中心距约 28px
- 可能产生叠影，观感一般
