# 抢杠（Chankan）逻辑修复

## 问题描述

抢杠（枪杠）是指：当其他玩家进行加杠（kakan）时，如果这张牌正好是你听的牌，你可以抢杠和牌。

原来的代码中，抢杠的番数计算可能不正确，因为：
1. 抢杠应该算作"杠上炮"（`is_kan_discard = true`）
2. 但在 `agari_points()` 中，`is_kan_discard` 只检查了 `dahai()` 中的情况，没有检查抢杠的情况

## 修复内容

### 1. 添加 `last_discard_was_after_kan` 字段

在 `PlayerState` 中添加了 `last_discard_was_after_kan` 字段，用于跟踪最后打出的牌是否来自杠（用于杠上炮）。

### 2. 修复 `dahai()` 函数

在 `dahai()` 中，当检测到 `intermediate_kan` 不为空时（表示刚有人杠），设置 `last_discard_was_after_kan = true`。

### 3. 修复 `agari_points()` 函数

在 `agari_points()` 中，正确检查：
- **抢杠（chankan）**：通过 `chankan_chance.is_some()` 判断
- **杠上炮（kan discard）**：通过 `last_discard_was_after_kan` 判断

两者都设置 `is_kan_discard = true`，从而正确计算番数。

## 番数计算

### 抢杠（Chankan）
- **条件**：在别人加杠时，如果听的牌正好是加杠的牌，可以抢杠和牌
- **番数**：平胡1番 + 杠上炮1番 = **2番**
- **计分**：2000点（荣和）

### 杠上炮（GangShangPao）
- **条件**：其他玩家杠牌后打出的牌和牌（荣和）
- **番数**：平胡1番 + 杠上炮1番 = **2番**
- **计分**：2000点（荣和）

## 测试用例

在 `agari.rs` 的测试中添加了 Test 13，验证抢杠的番数计算：
- 抢杠应该算作2番（平胡1番 + 杠上炮1番）

## 相关文件

- `libblood/src/state/player_state.rs` - 添加 `last_discard_was_after_kan` 字段
- `libblood/src/state/update.rs` - 在 `dahai()` 中设置 `last_discard_was_after_kan`
- `libblood/src/state/agent_helper.rs` - 在 `agari_points()` 中正确检查抢杠和杠上炮
- `libblood/src/algo/agari.rs` - 添加抢杠测试用例
