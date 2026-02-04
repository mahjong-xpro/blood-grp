# TensorBoard 指标分析（换对手后）

> 在浏览器打开 `http://localhost:6007/` 对照本文查看。**刚换了对手**时，test_play 曲线会相对新对手重新建立，需要重新建立基线预期。

---

## 一、指标清单与含义

### 1. Loss（训练损失）

| 指标 | 含义 | 换对手后注意 |
|------|------|----------------|
| **loss/dqn_loss** | Q 网络 TD 损失 | 换对手后数据分布可能变，短期波动正常；若持续明显上升考虑 lr decay 或过拟合。 |
| **loss/cql_loss** | CQL 保守项（offline 时有） | online 时不写。 |
| **loss/next_rank_loss** | 排名预测辅助损失 | 稳定即可。 |
| **loss/ding_que_ce_loss** | 定缺与启发式一致性的 CE 损失 | 下降或稳定即正常。 |
| **ding_que/match_rate** | 定缺步与启发式最佳花色一致率 | 70%+ 可接受，换对手不影响此项。 |
| **hparam/lr** | 当前学习率 | 随 scheduler 从 peak 向 final 衰减。 |

### 2. Test Play（对新对手 1v3）

**对手来源**：`config.toml` → `[baseline.test]` → `state_file`（你刚换的就是这个 baseline）。

| 指标 | 含义 | 参考基线 |
|------|------|----------|
| **test_play/avg_ranking** | 挑战者（当前模型）平均名次 | 随机 ≈ 2.5；**&lt; 2.5** 表示比随机强。 |
| **test_play/avg_pt** | 平均排名分（1st=6, 2nd=4, 3rd=2, 4th=0） | 随机 ≈ 3.0；**&gt; 3.0** 表示比随机强。 |
| **test_play/ranking** (1st/2nd/3rd/4th) | 名次分布 | 换强对手后 1st↓、4th↑ 正常；看趋势是否随 step 改善。 |
| **test_play/behavior** (agari/houjuu/fuuro) | 和牌率、流局率、副露率 | 用于判断风格是否合理。 |
| **test_play/agari_point** (overall/fuuro) | 和牌时平均得点、副露和平均得点 | 与对手强度相关。 |
| **test_play/fuuro_num**, **fuuro_point** | 副露次数与副露相关得点 | 辅助判断进攻/防守平衡。 |

### 3. 直方图（Distributions）

- **q_predicted** / **q_target**：Q 值分布，避免严重偏移或塌缩。

---

## 二、换对手后怎么读曲线

1. **test_play 会“重置”**  
   新对手强度不同，**avg_ranking**、**avg_pt**、**ranking** 会跳变，这是预期现象。不要和换对手前的绝对值直接比，而是看**同一对手下随 step 的走势**。

2. **建议看的时间窗口**  
   - 换对手后至少跑完 **1～2 个 test_every**（例如 5k～10k step），再判断趋势。  
   - 关注：**avg_pt** 是否逐步 &gt; 3.0、**avg_ranking** 是否逐步 &lt; 2.5。

3. **对手更强时**  
   - 初始 avg_pt 可能 &lt; 3.0、avg_ranking &gt; 2.5。  
   - 若曲线随 step 缓慢改善，说明在适应新对手；若长期不改善，再考虑调 lr、探索或数据。

4. **对手更弱时**  
   - 初始可能 avg_pt &gt; 3.0。  
   - 重点看是否稳定或继续升，避免过拟合到弱对手。

---

## 三、在 TensorBoard 里怎么操作

1. 左侧选 **SCALARS**，勾选或取消勾选上述指标，对比多条 run（若有多次实验）。  
2. 横轴用 **Step**；换对手后可在 **Smoothing** 调成 0.3～0.6 看趋势。  
3. 若有多条曲线（例如不同 baseline），用左侧 **Runs** 勾选要对比的 run。  
4. 记录当前 **step 范围**（例如 0～100k），换对手后的第一个 test_play 点通常会有明显跳变。

---

## 四、快速检查清单（换对手后）

在 `http://localhost:6007/` 依次确认：

- [ ] **loss/dqn_loss** 无长期单边上升（可接受小幅波动）。  
- [ ] **test_play/avg_pt** 在换对手后 1～2 个 test 周期内，趋势是否向 &gt; 3.0 发展。  
- [ ] **test_play/avg_ranking** 趋势是否向 &lt; 2.5 发展。  
- [ ] **ding_que/match_rate** 是否在 70% 左右或以上。  
- [ ] **hparam/lr** 是否按配置从 peak 衰减到 final。

若你把当前 TensorBoard 的 step 范围、以及某几个指标的大致数值（如最新 avg_pt、avg_ranking、dqn_loss）贴出来，可以据此给更具体的调参建议（如是否调 lr、探索、test_every 等）。
