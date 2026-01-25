# 日本麻将相关函数删除计划

## 需要删除的函数

### tile.rs
1. `deaka()` - 75处使用，实现只是返回self，可以内联替换
2. `akaize()` - 未找到使用，直接删除
3. `is_aka()` - 2处使用（obs_repr.rs），总是返回false，删除函数和检查
4. `is_jihai()` - 未找到使用，直接删除

### hand.rs
1. `hand_with_aka()` - 只在测试中使用，可以删除或重命名

## 替换策略

### deaka() 替换
- `tile.deaka()` → `tile`（直接使用）
- `self.deaka()` → `self`（直接使用）

### is_aka() 替换
- `if tile.is_aka() { ... }` → 删除整个if块（因为总是false）

### 其他函数
- `akaize()` → 删除
- `is_jihai()` → 删除
- `hand_with_aka()` → 删除或重命名

## 执行顺序

1. 先替换所有 `deaka()` 调用
2. 删除所有 `is_aka()` 检查
3. 删除未使用的函数
4. 验证编译
