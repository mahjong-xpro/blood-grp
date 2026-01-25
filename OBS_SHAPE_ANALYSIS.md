# Version 4 Observation Shape 分析

## 问题描述

实际运行结果显示 `idx` 一直在增加：
- 1024 → 1026 → 1038 → 1040/1048/1050 → 1052

最终确定需要 **1052 行**。

## 理论计算

### SP table 编码之前的行数（875 行）

1. **基础编码（27 行）**
   - tehai: 4 行
   - akas_in_hand (兼容): 3 行
   - scores (version 4，每个玩家 2 行): 4 * 2 = 8 行
   - rank: 4 行
   - kyoku: 4 行
   - honba (version 4: rescale only): 1 行
   - kyotaku (version 4: rescale only): 1 行
   - bakaze/jikaze (兼容): 2 行

2. **ding_que 编码（14 行）**
   - ding_que suit: 3 行
   - ding_que complete: 1 行
   - ding_que remaining (version 4: rescale only): 1 行
   - other players ding_que (3个玩家): 3 * 3 = 9 行

3. **其他编码（834 行）**
   - kyoku (version 2|3|4: rescale only): 1 行
   - dora_indicators (encode_tile_set = 7): 7 行
   - self_kawa: 6 * 4 + 18 * 4 = 96 行
   - self_kawa overview (version 3|4): 1 行
   - other_kawa (3个玩家，version 4): 3 * (6 * 8 + 18 * 8 + 3) = 585 行
   - tiles_left: 1 行
   - dora (4个，每个 rescale only): 4 * 1 = 4 行
   - doras_unseen (rescale only): 1 行
   - kawa_overview (3个玩家，每个 encode_tile_set = 7): 3 * 7 = 21 行
   - fuuro_overview (3个玩家): 3 * 4 * 5 = 60 行
   - ankan_overview (3个玩家): 3 * 1 = 3 行
   - tiles_seen (version 2|3|4): 1 行
   - last_tedashis (3个玩家): 3 * 3 = 9 行
   - riichi sutehais (兼容): 3 * 3 = 9 行
   - riichi declared (兼容): 3 行
   - riichi accepted (兼容): 3 行
   - waits: 1 行
   - furiten (兼容): 1 行
   - shanten (one_hot: cap + 1 = 7): 7 行
   - riichi_accepted (兼容): 1 行
   - at_kan_select: 1 行
   - can_pass: 3 行
   - discard_candidates: 5 行
   - riichi (兼容): 1 行
   - chi (兼容): 3 行
   - pon: 1 行
   - kan: 1 行
   - ankan: 1 行
   - kakan: 1 行
   - agari: 1 行
   - ryukyoku: 1 行

**总计：27 + 14 + 834 = 875 行**

### SP table 编码（最大路径：111 行）

**路径 1: empty table + can_discard=True（最大）**
- idx += 2 + 2*27 + 2 = 58 行
- encode_sp_table: +51 行
- idx += 2 (best ev/win prob discard): +2 行
- **小计: 58 + 51 + 2 = 111 行**

**路径 2: empty table + can_discard=False**
- idx += 2 + 2*27 + 1 + 1 = 58 行
- encode_sp_table: +51 行
- **小计: 58 + 51 = 109 行**

**路径 3: non-empty table + can_discard=True**
- encode_ev: +2 行
- required tiles: +54 行 (2*27)
- max required tiles: +2 行
- encode_sp_table: +51 行
- idx += 2 (best ev/win prob discard): +2 行
- **小计: 2 + 54 + 2 + 51 + 2 = 111 行**

**路径 4: non-empty table + can_discard=False**
- encode_ev: +2 行
- required tiles: +55 行 (2*27 + 1)
- first required: +1 行
- encode_sp_table: +51 行
- **小计: 2 + 55 + 1 + 51 = 109 行**

**路径 5: single_player_tables() 失败**
- encode_ev: +2 行
- idx += 2*27 + 2 + 3*MAX_NUM_TURNS = 2 + 54 + 2 + 51 = 109 行
- **小计: 2 + 54 + 2 + 51 = 109 行**

**最大路径：111 行**

### 理论最大总计

**875 + 111 = 986 行**

## 实际需求

**1052 行**（基于运行时错误 `idx=1052`）

## 差异分析

**差异：1052 - 986 = 66 行**

### 可能的原因

1. **某些编码步骤被遗漏**
   - 可能某些条件分支导致额外的编码步骤
   - 可能某些编码步骤在某些情况下被跳过，但在其他情况下不会

2. **某些编码步骤的计算不准确**
   - 可能某些编码步骤实际使用的行数比计算的多
   - 可能某些条件分支导致额外的行数

3. **某些编码路径需要额外的空间**
   - 可能某些编码路径需要额外的缓冲空间
   - 可能某些编码路径需要对齐或填充

### 关键发现

1. **`encode_tile_set` 使用 7 行，不是 1 行**
   - 这影响了 `dora_indicators` 和 `kawa_overview` 的计算

2. **Version 4 的 `IntegerEncoder` 不支持 `rbf_intervals`**
   - 只支持 `one_hot` 和 `rescale`
   - 这影响了 `honba`, `kyotaku`, `ding_que remaining`, `dora`, `doras_unseen` 的计算

3. **`can_discard=False` 情况下的编码逻辑**
   - empty table: `self.idx += 2 + 2 * 27 + 1 + 1 = 58 行`
   - non-empty table: `encode_ev (+2) + self.idx += 2 * 27 + 1 (55) + self.idx += 1 (1) = 58 行`
   - 两种情况都是 58 行，这是对的

4. **`can_discard=True` 情况下的编码逻辑**
   - 在 `non-empty table` 的情况下，`encode_sp_table` 后需要额外的 2 行（best ev/win prob discard）
   - 这与 `empty table` 的情况保持一致

## 解决方案

1. **临时解决方案**：将 `obs_shape(4)` 设置为 **1052**，基于实际运行结果
2. **长期解决方案**：深入分析编码逻辑，找出差异 66 行的来源，并优化以减少空间使用

## TODO

- [ ] 找出差异 66 行的具体来源
- [ ] 优化编码逻辑以减少空间使用
- [ ] 验证所有编码路径的行数计算
- [ ] 确保编码逻辑的一致性
