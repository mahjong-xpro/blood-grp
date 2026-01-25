# Observation Encoding 深度分析计划

## 目标
找出 observation encoding 中所有可能的 bug，特别是导致实际需要 1066 行而理论计算只有 986 行的 80 行差异。

## 分析计划

### 阶段1：基础编码部分验证

#### 1.1 tehai 编码
- **代码位置**: `obs_repr.rs:130-139`
- **预期行数**: 4 行
- **检查点**:
  - [ ] `idx += 4` 是否正确
  - [ ] 是否所有手牌都被正确编码

#### 1.2 akas_in_hand (兼容)
- **代码位置**: `obs_repr.rs:141-143`
- **预期行数**: 3 行
- **检查点**:
  - [ ] `idx += 3` 是否正确

#### 1.3 scores 编码
- **代码位置**: `obs_repr.rs:145-161`
- **预期行数**: 4 个玩家 × 2 行 = 8 行
- **检查点**:
  - [ ] version 4: 每个玩家 `idx += 1` (rescale only)
  - [ ] 总共 4 次 `idx += 1` = 4 行
  - [ ] 但是还有第一次的 `idx += 1` (line 148)
  - [ ] **潜在问题**: 每个玩家实际是 `idx += 1` (line 148) + `idx += 1` (line 157) = 2 行
  - [ ] 4 个玩家 × 2 行 = 8 行 ✓

#### 1.4 rank, kyoku
- **代码位置**: `obs_repr.rs:163-174`
- **预期行数**: 4 + 4 = 8 行
- **检查点**:
  - [ ] rank: `idx += 4`
  - [ ] kyoku: `idx += 4`

#### 1.5 honba, kyotaku
- **代码位置**: `obs_repr.rs:176-190`
- **预期行数**: version 4 应该是 1 + 1 = 2 行 (rescale only)
- **检查点**:
  - [ ] **关键**: `IntegerEncoder` 在 version 4 中，如果同时设置了 `rescale(true)` 和 `rbf_intervals(3)`，会如何处理？
  - [ ] 查看 `IntegerEncoder.encode()` 的 version 4 分支
  - [ ] **潜在 BUG**: line 185 和 189 都调用了 `rbf_intervals(3)`，但 version 4 不支持 rbf_intervals！
  - [ ] 需要检查：version 4 时，`rbf_intervals` 是否会被忽略

#### 1.6 bakaze/jikaze (兼容)
- **代码位置**: `obs_repr.rs:192-194`
- **预期行数**: 2 行
- **检查点**:
  - [ ] `idx += 2` 是否正确

#### 1.7 ding_que 编码
- **代码位置**: `obs_repr.rs:196-230`
- **预期行数**: 3 + 1 + 1 + 9 = 14 行
- **检查点**:
  - [ ] ding_que suit: `idx += 3`
  - [ ] ding_que complete: `idx += 1`
  - [ ] ding_que remaining: **关键检查**
    - [ ] line 215-218: `IntegerEncoder` 设置了 `rescale(true)` 和 `rbf_intervals(3)`
    - [ ] version 4 时，`rbf_intervals` 会被忽略，只使用 `rescale` = 1 行
  - [ ] other players: 3 × 3 = 9 行

### 阶段2：手牌和牌河编码验证

#### 2.1 kyoku (version 2|3|4)
- **代码位置**: `obs_repr.rs:232-238`
- **预期行数**: 1 行 (rescale only)
- **检查点**:
  - [ ] `IntegerEncoder` 只设置了 `rescale(true)`，没有 `rbf_intervals`
  - [ ] version 4: 1 行 ✓

#### 2.2 dora_indicators
- **代码位置**: `obs_repr.rs:240-242`
- **预期行数**: 7 行 (encode_tile_set)
- **检查点**:
  - [ ] `encode_tile_set()` 确实使用 7 行 (line 755)

#### 2.3 self_kawa
- **代码位置**: `obs_repr.rs:244-268`
- **预期行数**: 6 × 4 + 18 × 4 + 1 = 97 行
- **检查点**:
  - [ ] 前 6 个: 6 × 4 = 24 行
  - [ ] 后 18 个: 18 × 4 = 72 行
  - [ ] overview (version 3|4): 1 行
  - [ ] **潜在问题**: line 248 和 255 的 `idx +=` 计算是否正确？

#### 2.4 other_kawa
- **代码位置**: `obs_repr.rs:270-313`
- **预期行数**: 3 个玩家 × (6 × 8 + 18 × 8 + 3) = 3 × 195 = 585 行
- **检查点**:
  - [ ] 每个玩家前 6 个: 6 × 8 = 48 行
  - [ ] 每个玩家后 18 个: 18 × 8 = 144 行
  - [ ] version 3|4: +3 行 (line 309)
  - [ ] **潜在问题**: line 275 和 282 的 `idx +=` 计算是否正确？

### 阶段3：其他编码验证

#### 3.1 tiles_left, dora, doras_unseen
- **代码位置**: `obs_repr.rs:315-332`
- **预期行数**: 1 + 4 + 1 = 6 行
- **检查点**:
  - [ ] tiles_left: 1 行
  - [ ] dora (4个): **关键检查**
    - [ ] line 322-325: `IntegerEncoder` 设置了 `rescale(true)` 和 `rbf_intervals(3)`
    - [ ] version 4 时，`rbf_intervals` 会被忽略，只使用 `rescale` = 1 行
    - [ ] 4 个 dora × 1 行 = 4 行
  - [ ] doras_unseen: **关键检查**
    - [ ] line 329-332: `IntegerEncoder` 设置了 `rescale(true)` 和 `rbf_intervals(4)`
    - [ ] version 4 时，`rbf_intervals` 会被忽略，只使用 `rescale` = 1 行

#### 3.2 kawa_overview
- **代码位置**: `obs_repr.rs:334-336`
- **预期行数**: 3 × 7 = 21 行
- **检查点**:
  - [ ] 每个玩家使用 `encode_tile_set` = 7 行

#### 3.3 fuuro_overview, ankan_overview
- **代码位置**: `obs_repr.rs:338-362`
- **预期行数**: 60 + 3 = 63 行
- **检查点**:
  - [ ] fuuro_overview: 3 × 4 × 5 = 60 行
  - [ ] ankan_overview: 3 × 1 = 3 行

#### 3.4 tiles_seen, last_tedashis, riichi相关
- **代码位置**: `obs_repr.rs:364-398`
- **预期行数**: 1 + 9 + 9 + 3 + 3 = 25 行
- **检查点**:
  - [ ] tiles_seen: 1 行 (version 2|3|4)
  - [ ] last_tedashis: 3 × 3 = 9 行
  - [ ] riichi sutehais (兼容): 3 × 3 = 9 行
  - [ ] riichi declared (兼容): 3 行
  - [ ] riichi accepted (兼容): 3 行

#### 3.5 waits, furiten, shanten
- **代码位置**: `obs_repr.rs:400-417`
- **预期行数**: 1 + 1 + 7 + 1 = 10 行
- **检查点**:
  - [ ] waits: 1 行
  - [ ] furiten (兼容): 1 行
  - [ ] shanten: **关键检查**
    - [ ] line 413: `IntegerEncoder::new(n, 6).one_hot(true)`
    - [ ] version 4: one_hot 使用 `cap + 1 = 6 + 1 = 7` 行
  - [ ] riichi_accepted (兼容): 1 行

#### 3.6 动作编码
- **代码位置**: `obs_repr.rs:419-551`
- **预期行数**: 1 + 3 + 5 + 1 + 3 + 1 + 1 + 1 + 1 + 1 = 18 行
- **检查点**:
  - [ ] at_kan_select: 1 行
  - [ ] can_pass: 3 行
  - [ ] discard_candidates: 5 行
  - [ ] riichi (兼容): 1 行
  - [ ] chi (兼容): 3 行
  - [ ] pon: 1 行
  - [ ] kan: 1 行
  - [ ] ankan: 1 行
  - [ ] kakan: 1 行
  - [ ] agari: 1 行
  - [ ] ryukyoku: 1 行

### 阶段4：SP table 编码（最关键）

#### 4.1 empty table + can_discard=True
- **代码位置**: `obs_repr.rs:563-569`
- **预期行数**: 58 + 51 + 2 = 111 行
- **检查点**:
  - [ ] line 565: `idx += 2 + 2 * 27 + 2` = 58 行
  - [ ] line 567: `encode_sp_table()` = 51 行
  - [ ] line 569: `idx += 2` = 2 行

#### 4.2 empty table + can_discard=False
- **代码位置**: `obs_repr.rs:570-574`
- **预期行数**: 58 + 51 = 109 行
- **检查点**:
  - [ ] line 572: `idx += 2 + 2 * 27 + 1 + 1` = 58 行
  - [ ] line 574: `encode_sp_table()` = 51 行

#### 4.3 non-empty table + can_discard=True
- **代码位置**: `obs_repr.rs:585-636`
- **预期行数**: 2 + 54 + 2 + 51 + 2 = 111 行
- **检查点**:
  - [ ] line 585: `encode_ev()` = 2 行
  - [ ] line 603: `idx += 2 * 27` = 54 行
  - [ ] line 616: `idx += 2` = 2 行
  - [ ] line 630: `encode_sp_table()` = 51 行
  - [ ] line 635: `idx += 2` = 2 行

#### 4.4 non-empty table + can_discard=False
- **代码位置**: `obs_repr.rs:617-627`
- **预期行数**: 2 + 55 + 1 + 51 = 109 行
- **检查点**:
  - [ ] line 585: `encode_ev()` = 2 行
  - [ ] line 619: `idx += 2 * 27 + 1` = 55 行
  - [ ] line 626: `idx += 1` = 1 行
  - [ ] line 630: `encode_sp_table()` = 51 行
  - [ ] **关键检查**: line 621-624 使用 `self.idx` 赋值，但这是在 `idx += 2 * 27 + 1` 之后
  - [ ] 这意味着 `self.idx` 指向的是第 55 行的位置
  - [ ] 然后 `idx += 1` 又增加了 1 行
  - [ ] 所以实际是：55 行（预留）+ 1 行（使用后增加）= 56 行？
  - [ ] **潜在 BUG**: 这里可能有计算错误

#### 4.5 single_player_tables() 失败
- **代码位置**: `obs_repr.rs:638-655`
- **预期行数**: 2 + 107 + 2 = 111 行 (can_discard=True) 或 2 + 107 = 109 行 (False)
- **检查点**:
  - [ ] line 645: `encode_ev()` = 2 行
  - [ ] line 649: `idx += 2 * 27 + 2 + 3 * MAX_NUM_TURNS` = 54 + 2 + 51 = 107 行
  - [ ] line 654: `idx += 2` (can_discard=True) = 2 行

### 阶段5：关键函数深度检查

#### 5.1 IntegerEncoder.encode() - version 4
- **代码位置**: `obs_repr.rs:92-104`
- **检查点**:
  - [ ] line 93: `debug_assert!(self.one_hot || self.rescale)`
  - [ ] line 95-98: one_hot 分支，使用 `cap + 1` 行
  - [ ] line 99-103: rescale 分支，使用 1 行
  - [ ] **关键**: `rbf_intervals` 在 version 4 中**完全被忽略**
  - [ ] **潜在 BUG**: 如果代码中设置了 `rbf_intervals`，但 version 4 不支持，可能导致行数计算错误

#### 5.2 encode_tile_set()
- **代码位置**: `obs_repr.rs:737-756`
- **检查点**:
  - [ ] line 755: `self.idx += 7` ✓
  - [ ] 确认使用 7 行

#### 5.3 encode_sp_table()
- **代码位置**: `obs_repr.rs:687-735`
- **检查点**:
  - [ ] line 694: empty case: `idx += 3 * MAX_NUM_TURNS` = 51 行
  - [ ] line 734: normal case: `idx += 3 * MAX_NUM_TURNS` = 51 行
  - [ ] **关键**: `encode_sp_table` 不处理 `best ev/win prob discard` 的 2 行
  - [ ] 这 2 行需要在调用后手动添加（已经在代码中处理）

#### 5.4 encode_self_kawa() 和 encode_kawa()
- **代码位置**: `obs_repr.rs:758-797`
- **检查点**:
  - [ ] `encode_self_kawa`: 如果有 item，先 `idx += 2`，然后 `idx += SELF_KAWA_ITEM_CHANNELS (4)`
  - [ ] 如果没有 item，直接 `idx += SELF_KAWA_ITEM_CHANNELS (4)`
  - [ ] `encode_kawa`: 如果有 item，先 `idx += 2`，然后 `idx += 4`，最后 `idx += KAWA_ITEM_CHANNELS (8)`
  - [ ] 如果没有 item，直接 `idx += KAWA_ITEM_CHANNELS (8)`

### 阶段6：潜在问题检查清单

#### 6.1 IntegerEncoder 的 rbf_intervals 问题
- [ ] **关键 BUG**: 在多个地方，`IntegerEncoder` 同时设置了 `rescale(true)` 和 `rbf_intervals(N)`
- [ ] 但在 version 4 中，`rbf_intervals` 会被忽略
- [ ] 需要检查所有使用 `IntegerEncoder` 的地方，确认 version 4 时的行数计算
- [ ] **位置**:
  - [ ] honba/kyotaku (line 183-190)
  - [ ] ding_que remaining (line 215-218)
  - [ ] dora (line 322-325)
  - [ ] doras_unseen (line 329-332)

#### 6.2 non-empty table + can_discard=False 的行数计算
- [ ] **关键检查**: line 619-626
- [ ] `idx += 2 * 27 + 1` = 55 行
- [ ] 然后使用 `self.idx` 赋值 `first_candidate.required_tiles`
- [ ] 然后 `idx += 1` = 1 行
- [ ] **问题**: 这 55 行是预留空间，还是实际使用的空间？
- [ ] 如果 `first_candidate.required_tiles` 为空，这 55 行是否仍然被使用？

#### 6.3 是否有重复编码
- [ ] 检查是否有某些编码步骤被重复执行
- [ ] 检查是否有条件分支导致额外编码

#### 6.4 条件分支的完整性
- [ ] 检查所有 `if/else` 分支
- [ ] 检查所有 `match self.version` 分支
- [ ] 确保每个分支的行数都被正确计算

## 下一步行动

1. 系统检查每个 `IntegerEncoder` 的使用，确认 version 4 时的行数
2. 仔细检查 `non-empty table + can_discard=False` 的行数计算
3. 检查是否有遗漏的编码步骤
4. 验证所有条件分支的行数计算
