# 定缺辅助学习 — 回顾

## 一、动机

定缺在血战到底里是**每局只做一次**的三选一（万/筒/条），但：

- **回报滞后**：局末得分要很多步后才出现，信用分配难。
- **信号稀疏**：主奖励是得分/排名，定缺混在大量后续决策里，梯度对「选哪门缺」区分度弱。
- **探索有限**：只有 3 个动作，无额外信号时策略在定缺上易收敛慢或次优。

因此在不改网络结构的前提下，给定缺步增加**辅助学习信号**和更合理的**启发式**，让「选好缺」更容易被学到。

---

## 二、已做内容总览

| 层次 | 内容 | 作用 |
|------|------|------|
| **训练信号** | 方案 A：定缺步奖励塑形 | 定缺步即时 bonus 并入 Q 目标，拉高「选好缺」的长期回报 |
| **训练信号** | 方案 B：定缺步 CE 监督 | 用启发式最佳花色做交叉熵，直接拉高选该花色的概率 |
| **启发式** | 刻子/顺子质量 | 去掉该门时，每失去一个刻子 +0.8、每失去一个顺子 +0.7，优先定缺「无成组」的门 |
| **启发式** | 进张种类 | 去掉该门后，有多少种牌加一张能减向听 → 种类越多 cost 减越多（上限 2.0） |
| **文档与测试** | 方案说明、总结、单测 | 行为可复现、可回归、可关闭做消融 |

---

## 三、数据与训练流程

1. **Replay**（`libblood/grp.rs`）  
   每局每人：算三种缺门的 cost（刻子/顺子/向听/进张种类）→ 得到 `ding_que_quality`（+1/0/-1）和 `ding_que_best_suit`（0/1/2）。

2. **Dataloader**（`mortal/dataloader.py`）  
   - 定缺步（action ∈ {31,32,33}）：`ding_que_bonus = quality × scale`，`ding_que_best_suit = 最佳花色索引`。  
   - 非定缺步：bonus=0，best_suit=-1。  
   每条样本多两维：`dq_bonus`、`dq_best_suit`。

3. **Train**（`mortal/train.py`）  
   - Q 目标：`gamma^steps_to_done * kyoku_rewards + ding_que_bonus`。  
   - 定缺步上对动作 31/32/33 做 CE(策略, ding_que_best_suit)，乘以 `ding_que_ce_weight` 加入总 loss。

4. **配置**（`mortal/config.toml`）  
   - `ding_que_aux_enabled` / `ding_que_aux_scale`：控制方案 A。  
   - `ding_que_ce_weight`：控制方案 B。  
   关闭任一即可做消融。

---

## 四、启发式 cost 公式（grp.rs）

对「选某门为缺」的 cost（越低越适合定缺该门）：

- **base**：去掉该门后的向听数。
- **+ 刻子惩罚**：该门每有一个刻子（同种≥3 张）+0.8。
- **+ 顺子惩罚**：该门贪心数出的完整顺子个数 × 0.7。
- **− 对对潜力**：剩余手牌中 pair/triplet 种类≥4 时 −0.5。
- **− 进张种类 bonus**：去掉该门后「加一张能减向听」的牌种类数 × 0.12，上限 2.0。

由此得到每局每人的 `ding_que_quality`（相对三选一的好坏）和 `ding_que_best_suit`（cost 最低的花色索引）。

---

## 五、涉及文件

| 文件 | 角色 |
|------|------|
| `docs/DING_QUE_AUXILIARY_LEARNING.md` | 方案与实现细节 |
| `docs/DING_QUE_AUXILIARY_LEARNING_SUMMARY.md` | 速查总结 |
| `docs/DING_QUE_AUXILIARY_LEARNING_REVIEW.md` | 本回顾 |
| `libblood/src/dataset/grp.rs` | 定缺 cost、刻子/顺子/进张种类、ding_que_best_suit、单测 |
| `mortal/config.toml` | 定缺辅助与 CE 权重 |
| `mortal/dataloader.py` | 每步 dq_bonus、dq_best_suit |
| `mortal/train.py` | Q 目标含 bonus、定缺 CE loss |

---

## 六、可选后续

- 若 replay 能拿到近似 `tiles_seen`，可用 SP 的 `get_required_tiles` 进张数加权替代「种类数」，使 cost 更贴近真实期望。
- 单测需在能链接 Python 的环境（如 maturin/pytest）下跑 `cargo test`。
