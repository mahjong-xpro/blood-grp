# 定缺辅助学习 — 总结

## 背景

定缺难学：每局只做一次三选一、回报滞后、主奖励稀疏，策略在定缺上收敛慢。为此增加**辅助学习**与**启发式改进**，在不改网络结构的前提下给定缺步更强的学习信号。

---

## 已实现内容

### 1. 方案 A：定缺步奖励塑形

| 位置 | 改动 |
|------|------|
| **mortal/config.toml** | `[aux]` 增加 `ding_que_aux_enabled`、`ding_que_aux_scale`（默认 0.02） |
| **mortal/dataloader.py** | 从 `take_ding_que_quality()` 取质量；仅当 `actions[i] in {31,32,33}` 时设 `ding_que_bonus[i]`，否则 0；每条样本多一维 `dq_bonus` |
| **mortal/train.py** | 解包 `ding_que_bonus`；Q 目标改为 `gamma^steps_to_done * kyoku_rewards + ding_que_bonus` |

- **作用**：在定缺步给即时奖励（启发式质量 × scale），拉高「选好缺」的长期回报，不改变其他步的 Q。
- **关闭**：`ding_que_aux_enabled = false`。

---

### 2. 方案 B：定缺步 CE 监督

| 位置 | 改动 |
|------|------|
| **libblood/src/dataset/grp.rs** | 新增 `ding_que_best_suit: Vec<[u8; 4]>`、`best_ding_que_suit_index(tehai)`、`take_ding_que_best_suit()`；replay 时每局每人写入启发式最佳花色 0/1/2 |
| **mortal/config.toml** | `[aux]` 增加 `ding_que_ce_weight`（默认 0.1） |
| **mortal/dataloader.py** | 取 `take_ding_que_best_suit()`；定缺步写入 0/1/2，非定缺步 -1；每条样本多一维 `dq_best_suit` |
| **mortal/train.py** | 解包 `ding_que_best_suit`；仅对 `ding_que_best_suit >= 0` 的样本在 Q 头动作 31/32/33 上做 CE 损失，乘以 `ding_que_ce_weight` 加入总 loss；记录 `loss/ding_que_ce_loss` |

- **作用**：用启发式「最佳花色」做监督，直接拉高策略选该花色的概率，与方案 A 叠加。
- **关闭**：`ding_que_ce_weight = 0`。

---

### 3. 启发式改进：刻子/顺子质量 + 进张种类

| 位置 | 改动 |
|------|------|
| **libblood/src/dataset/grp.rs** | **刻子**：去掉该门时每失去一个 刻子 +0.8。**顺子**：新增 `count_sequences_in_suit` 贪心数该门完整 顺子 个数，每失去一个 +0.7。**进张种类**：`count_improvement_kinds`，减去 improvement_bonus（上限 2.0）。 |

- **作用**：优先定缺「刻子/顺子少」的门；去掉后进张宽则 cost 更低。不依赖牌山/`tiles_seen`。
- **单测**：`ding_que_count_sequences_in_suit`、`ding_que_cost_penalizes_triplet_and_sequence`、`ding_que_cost_prefers_suit_with_no_groups`（需在能链接 Python 的环境下跑 `cargo test`，如 maturin/pytest）。

---

## 配置速查（mortal/config.toml）

```toml
[aux]
next_rank_weight = 0.2
ding_que_aux_enabled = true
ding_que_aux_scale = 0.02
ding_que_ce_weight = 0.1
```

---

## 涉及文件一览

| 文件 | 变更类型 |
|------|----------|
| `docs/DING_QUE_AUXILIARY_LEARNING.md` | 方案说明与实现细节 |
| `docs/DING_QUE_AUXILIARY_LEARNING_SUMMARY.md` | 本总结 |
| `libblood/src/dataset/grp.rs` | 定缺 cost 启发式、`ding_que_best_suit`、进张种类 |
| `mortal/config.toml` | 定缺辅助与 CE 权重 |
| `mortal/dataloader.py` | 每步 `ding_que_bonus`、`ding_que_best_suit` |
| `mortal/train.py` | Q 目标含 bonus、定缺 CE 损失与统计 |

---

## 可选后续

- 若 replay 能拿到近似 `tiles_seen`，可用 SP 的 `get_required_tiles` 进张数加权替代「种类数」，使 cost 更贴近真实期望。
