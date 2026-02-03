# TensorBoard 快照分析与配置调整

> 基于 `curl http://localhost:6007/` 获取的 TensorBoard 数据（step ~87k），对当前指标进行分析并调整 `mortal/config.toml`。

---

## 一、数据快照（step ~87k）

| 指标 | 最新值 | 趋势 |
|------|--------|------|
| **loss/dqn_loss** | 0.410 | 50k 后由 ~0.37 缓慢上升，进入平台期 |
| **loss/next_rank_loss** | 0.662 | 稳定 |
| **loss/ding_que_ce_loss** | 0.062 | 由 ~0.12 持续下降 |
| **ding_que/match_rate** | 71.3% | 稳定 |
| **test_play/avg_ranking** | 2.54 | 波动，略差于随机基线 2.5 |
| **test_play/avg_pt** | 2.93 | 略低于随机基线 3.0 |
| **hparam/lr** | 1e-4 | 恒定 |

---

## 二、分析结论

1. **dqn_loss 缓慢上升**：50k→87k 从 ~0.37 升至 ~0.41，需警惕过拟合或数据分布漂移；启用 lr decay 有助于稳定。
2. **test_play 略弱于随机**：avg_pt ~2.93 < 3.0，avg_rank ~2.54 > 2.5；可加强排名辅助、降低探索以提高 exploit。
3. **定缺辅助正常**：match_rate ~71%，ce_loss 下降，无需调整。
4. **探索过高**：boltzmann_epsilon=0.5 适合初期，当前 step~87k 宜进入中期，降低探索以提升 exploit。

---

## 三、已实施的配置调整

| 配置项 | 原值 | 新值 | 理由 |
|--------|------|------|------|
| **boltzmann_epsilon** | 0.5 | 0.15 | 进入中期，降低探索，提高 exploit |
| **boltzmann_temp** | 0.5 | 0.2 | 配合 epsilon 降探索 |
| **optim.scheduler.final** | 1e-4 | 5e-5 | 启用 lr decay，缓解 dqn_loss 缓慢上升 |
| **next_rank_weight** | 0.2 | 0.25 | 略强化排名预测，期望改善 avg_pt/avg_rank |

---

## 四、后续监控建议

- 关注 **dqn_loss** 是否在 lr decay 后趋于平稳或略降
- 关注 **test_play/avg_pt**、**avg_ranking** 是否随探索降低而改善
- 若 test_play 稳定优于随机后，可进一步将 boltzmann_epsilon 降至 0.005（后期）
