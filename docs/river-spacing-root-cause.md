# 左右河牌间距问题根因分析

## 现象
1. 河牌之间间距太大
2. **第一张和第二张的间距比其它河牌之间的间距更大**

## 关键发现

### 1. :first-child 导致的不对称

当前使用 `margin-top: -6px` 仅作用于 `:not(:first-child)`，即第 2、3、4… 张牌。

在 **column-reverse** 下：
- DOM 第 1 项 = 视觉上最底部（最早出的牌）
- DOM 最后一项 = 视觉上最顶部（刚出的牌）

若用户说的「第一张」= 视觉最顶（最新），则第一张 = **:last-child**，第二张 = **:nth-last-child(2)**。  
第二张有 `margin-top: -6px`，第一张没有，所以第一、二张之间的间距确实会与其他相邻牌不同。

### 2. 统一用 margin-bottom 替代 margin-top

改用 `margin-bottom: -8px` 作用于 `:not(:last-child)`：
- 除最顶那张外，每张牌都有负 margin
- 每对相邻牌之间的间距统一为 36 - 8 = 28px
- 第一张和第二张会与其他相邻牌一致

### 3. 显式覆盖 gap

确保左右河牌不受父级 `.tile-group` 的 `gap: 2px` 影响，显式设置 `row-gap: 0`。
