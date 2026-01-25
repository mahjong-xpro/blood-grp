# 日本麻将代码删除计划

## 目标
删除所有日本麻将特有的代码和业务逻辑，不仅仅是注释。

## 需要删除的功能模块

### 1. Riichi（立直）相关
- [ ] `ActionCandidate.can_riichi` 字段
- [ ] `PlayerState` 中 `can_riichi` 的检查逻辑（`tsumo()` 方法中）
- [ ] `stat.rs` 中所有 riichi 相关字段（虽然已 deprecated，应完全删除）
- [ ] 所有 riichi 相关的统计和计算

### 2. Chi（吃）相关
- [ ] `ActionCandidate.can_chi_low`, `can_chi_mid`, `can_chi_high` 字段
- [ ] `ActionCandidate.can_chi()` 方法
- [ ] `PlayerState.set_can_chi_from_tile()` 方法
- [ ] 所有调用 `set_can_chi_from_tile()` 的地方
- [ ] `KawaItem.chi_pon` 字段（如果只用于 chi）

### 3. Dora（宝牌）相关
- [ ] `dora_indicators` 字段
- [ ] `dora_marker` 字段
- [ ] `uradora` / `ura_markers` 相关
- [ ] 所有 dora 计算逻辑

### 4. 其他日本麻将特有字段
- [ ] `honba`, `kyotaku` 字段
- [ ] `bakaze`, `jikaze` 字段（如果只用于日本麻将）
- [ ] `furiten` 相关逻辑（需要确认是否血战到底也需要）
- [ ] `tedashi` / `is_tedashi`（需要确认是否血战到底也需要）
- [ ] `yakuman`, `nagashi_mangan` 在 stat.rs 中的字段

## 执行顺序

1. 先删除 chi 相关（最简单，血战到底没有吃）
2. 删除 riichi 相关
3. 删除 dora 相关
4. 删除其他日本麻将特有字段
5. 验证编译和测试
