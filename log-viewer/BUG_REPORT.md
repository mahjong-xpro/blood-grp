# Bug 分析报告

## 已修复的 Bug

### 1. **手牌重复牌处理错误** ✅ 已修复
**位置**: `replay.js:185-189`
**问题**: `dahai` 事件处理中，使用 `indexOf` 查找牌，如果手牌中有重复的牌（例如两个"1m"），`indexOf` 只会找到第一个，可能导致移除错误的牌。

```javascript
// 当前代码
const tileIndex = dahaiPlayer.tehai.indexOf(event.pai);
if (tileIndex >= 0) {
    dahaiPlayer.tehai.splice(tileIndex, 1);
    dahaiPlayer.kawa.push(event.pai);
}
```

**修复建议**: 应该从后往前查找，或者使用更精确的匹配逻辑。

### 2. **Kakan 副露查找逻辑错误** ✅ 已修复
**位置**: `replay.js:219-231`
**问题**: `kakan` 事件中，查找pon的逻辑有误。pon的meld结构是 `[event.pai, ...event.consumed]`，但查找时只检查 `meld[0] === event.pai`，这不够准确。应该查找包含该牌的3张牌的meld。

```javascript
// 当前代码
for (let meld of kakanPlayer.fuuro) {
    if (meld.length === 3 && meld[0] === event.pai) {
        meld.push(event.pai);
        break;
    }
}
```

**修复建议**: 应该检查meld中是否包含该牌，而不是只检查第一个元素。

### 3. **Vue 响应式更新问题** ✅ 已修复
**位置**: `replay.js:57-70`
**问题**: 直接修改 Vue 对象的属性可能不会触发响应式更新。Vue 3 需要确保使用响应式的方式更新数据。

```javascript
// 当前代码
gameState.players.forEach((p, i) => {
    if (window.vueApp.players[i]) {
        window.vueApp.players[i].name = p.name;
        window.vueApp.players[i].score = p.score;
        window.vueApp.players[i].dingque = p.dingque;
    }
});
```

**修复建议**: 应该直接替换整个数组或使用 Vue 的响应式方法。

### 4. **剩余牌数计算不准确** (中等)
**位置**: `replay.js:161, 180`
**问题**: 
- `start_kyoku` 时固定设置为56，但实际应该根据游戏规则计算
- `tsumo` 时减1，但没有考虑杠牌（kan）会减少牌数
- 没有处理 `daiminkan`、`ankan`、`kakan` 对剩余牌数的影响

**修复建议**: 需要正确处理所有影响剩余牌数的事件。
**状态**: 已添加基本检查，但完整计算需要根据实际游戏规则实现。

### 5. **事件类型大小写问题** ✅ 已确认
**位置**: `replay.js:171`
**问题**: 使用 `'ding_que'`，但需要确认JSON序列化后的实际格式。根据Rust代码，应该是 `snake_case`，但需要验证。
**状态**: 已确认示例日志使用snake_case格式，代码正确。

### 6. **Pon/副露处理中手牌移除可能失败** ✅ 已修复
**位置**: `replay.js:192-198, 201-207, 210-216`
**问题**: 在 `pon`、`daiminkan`、`ankan` 中，使用 `indexOf` 查找并移除手牌，如果手牌中没有对应的牌（数据不一致），会静默失败。

**修复建议**: 应该添加错误检查或日志。
**状态**: 已添加从后往前查找逻辑和警告日志。

### 7. **手牌排序问题** ✅ 已修复
**位置**: `replay.js:94-100`
**问题**: 手牌显示时没有排序，可能看起来混乱。麻将手牌通常应该按花色和数字排序。

### 8. **Tsumo 牌显示问题** ✅ 已修复
**位置**: `replay.js:177-180`
**问题**: `tsumo` 后，新摸的牌应该在手牌中显示，但可能需要特殊标记（已经有 `tile-tsumo` 类）。问题是如果手牌中有多张相同的牌，无法区分哪张是新摸的。

### 9. **事件处理顺序问题** (轻微)
**位置**: `replay.js:328-341`
**问题**: `goToEvent` 中，每次都要重置并重新处理所有事件，对于大量事件可能性能较差。

### 10. **缓存路径匹配可能失败** (中等)
**位置**: `app.py:283-310`
**问题**: 路径匹配策略可能过于复杂，某些情况下可能匹配到错误的日志。

### 11. **缺少错误处理** ✅ 已修复
**位置**: 多处
**问题**: 很多地方缺少错误处理，例如 `processEvent` 中如果事件格式不正确，会静默失败。
**状态**: 已添加事件验证和默认case处理。

### 12. **手牌数组引用问题** ✅ 已修复

## 待优化的问题

### 4. **剩余牌数计算不准确** (中等)
**位置**: `replay.js:164`
**问题**: `p.tehai = event.tehais[i] || []` 直接赋值数组引用，如果后续修改，可能影响原始数据。

**修复建议**: 应该创建数组副本。
