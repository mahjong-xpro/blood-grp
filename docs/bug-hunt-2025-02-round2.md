# Bug 排查报告 (2025-02 第二轮)

## 一、已修复（本轮）

- **BUG-G**：JSON.parse 增加 try/catch，解析失败时 log 并跳过
- **BUG-J**：未知消息类型时 console.warn
- **BUG-K**：state_update 无 data 时 console.warn
- **BUG-L**：startGame 重置 phase = 'idle'

**BUG-H（杠类型歧义）**：用户确认 kakan 与 ankan 互斥，不成立。

## 二、第三轮修复

- **drain 循环**：handleMessage 抛错时 processing 永为 true → 增加 try/catch/finally
- **replay dahai**：ev.actor 为 undefined 时 nextActor 为 NaN → 用 `actor = ev.actor != null ? ev.actor : 0`
- **replay tsumo**：ev.actor 为 undefined 时 currentActor 被设为 undefined → 保留原值
- **backend update_state**：ev["tehais"]、ev["pai"]、ev["consumed"] 缺失时 KeyError → 用 ev.get
- **player(offset)**：scores[id] 为 undefined 时显示 undefined → 用 `?? 0`

---

## 三、新发现 Bug（按优先级）

### BUG-G：WebSocket onmessage 中 JSON.parse 无 try/catch（高）✅ 已修复

**修复**：try/catch 包裹，解析失败时 console.error 并 return，不推入队列。

---

### BUG-H：杠类型歧义（中）❌ 不适用

**说明**：kakan 与 ankan 互斥，同一回合不会同时出现。

---

### BUG-I：kakan 多候选时取第一个（中）

**位置**：`game_manager._translate_to_mjai` kakan 分支

**现象**：存在多个可加杠的牌（如多个 pon）时，后端遍历 `kakan_candidates` 或 `peng`，返回第一个找到的牌。用户无法选择要加杠的牌。

**影响**：多碰时可能杠错牌。

---

### BUG-J：handleMessage 对未知 type 静默忽略（低）✅ 已修复

**修复**：增加 `else if (msg && msg.type) { console.warn('Unknown message type:', msg.type); }`。

---

### BUG-K：state_update 无 data 时静默跳过（低）✅ 已修复

**修复**：`!data` 时 `console.warn('state_update: missing data')` 并 return。

---

### BUG-L：startGame 未重置 phase（极低）✅ 已修复

**修复**：startGame 中增加 `state.phase = 'idle'`。

---

## 四、需核实

| 项目 | 说明 |
|------|------|
| **can_ankan 与 can_kakan 互斥** | 已确认，BUG-H 不适用 |
| **last_kawa 与 hora** | ron 后 last_kawa 未清，但该局 human 不再行动，无影响 |
| **backend receive_json** | 异常时 catch 并 disconnect，逻辑合理 |

---

## 五、建议修复优先级

| 优先级 | Bug | 状态 |
|--------|-----|------|
| P1 | BUG-G JSON.parse | ✅ |
| P2 | BUG-H 杠类型歧义 | ❌ 互斥不适用 |
| P2 | BUG-I kakan 多候选 | 未修复（需 UI 改动） |
| P3 | BUG-J 未知消息 | ✅ |
| P3 | BUG-K 无 data | ✅ |
| P4 | BUG-L startGame phase | ✅ |
