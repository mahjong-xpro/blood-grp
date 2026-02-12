# 点炮 Ron 检测修改深度分析

## 修改概要

1. **Ron 优先级**（game.rs）：有人能胡时，仅向能胡的玩家请求反应，能碰/杠者视为过
2. **点炮 Ron**（update.rs dahai）：以 has_yaku 为准，不再依赖 waits[]
3. **抢杠 Chankan**（update.rs kakan）：同上，以 has_yaku 为准

---

## 一、修改影响范围

| 模块 | 影响 | 说明 |
|------|------|------|
| libblood/arena/game.rs | poll / commit | Ron 优先级过滤 |
| libblood/state/update.rs | dahai / kakan | Ron 检测逻辑 |
| blood-arena 后端 | 间接 | 依赖 last_cans |
| 训练 pipeline | 间接 | 更正确的对局日志 |

---

## 二、潜在 Bug 与风险

### 1. ✅ 已排除：hand_total 公式

**公式**：`hand_total = tehai_sum + 3*(pons + minkans + ankans)`

- 无副露：13 张，hand_total = 13 ✓
- 1 碰：10 张，hand_total = 10+3 = 13 ✓
- 1 明杠/暗杠：10 张，hand_total = 10+3 = 13 ✓
- 2 碰：7 张，hand_total = 7+6 = 13 ✓

与 `hand14_for_division` 的「每种副露 +3」一致。

### 2. ✅ 已排除：虚假正例（误判可胡）

`has_yaku` 会校验 AGARI_TABLE 和定缺，无效结构返回 false。误判可胡需 AGARI_TABLE 或定缺逻辑出错，属既有问题。

### 3. ✅ 已排除：tiles_seen 时序

点炮时先 `witness_tile` 再检查，`tiles_seen` 已包含本次打出的牌。

### 4. ✅ 已排除：temporary_furiten

过水不胡时仍会检查，不会误判可再次 Ron。

### 5. ✅ 已修复：抢杠时 HumanEngine.last_kawa

**位置**：`game_manager.py` 中 `update_state()`

**问题**：抢杠 Ron 时，`_translate_to_mjai` 需要 `last_kawa = (target, pai)` 来构造 hora。但 `kakan` 分支未设置 `last_kawa`，只依赖上次 `dahai` 的 `last_kawa`。抢杠时上次不是 `dahai` 而是 `kakan`，导致 target/pai 错误。

**修复**：在 `update_state` 的 `kakan` 分支，当 `actor != self.player_id` 时，设置 `last_kawa = (actor, pai)`。

### 6. ✅ 抢杠与点炮的差异

- **点炮**：需检查 `tiles_seen < 4`
- **抢杠**：不检查 `tiles_seen`，加杠的牌必定可用

---

## 三、Ron 优先级对 Agent 的影响

| 场景 | 行为 | 正确性 |
|------|------|--------|
| 人类能 Ron，AI 能碰 | 仅询问人类 | ✓ |
| AI 能 Ron，人类能碰 | 仅询问 AI，人类不弹出 | ✓ |
| 无人能 Ron | 正常询问所有能碰/杠者 | ✓ |

被跳过时：`last_reactions[player] = EventExt::default()`（相当于 `None`），不会调用 `get_reaction`，与「过」一致。

---

## 四、性能与兼容性

- **性能**：每局对局约多 50–70 次 `has_yaku`（hash + 定缺检查），开销可忽略
- **旧日志**：点炮时本可 Pon 却选了 Pon 的旧日志仍合法，新逻辑不会使其失效
- **训练**：新对局更符合规则，训练数据质量提升

---

## 五、结论

- 点炮 Ron、抢杠逻辑正确
- HumanEngine 已为抢杠设置 `last_kawa`，抢杠时点击胡可正确构造 hora
