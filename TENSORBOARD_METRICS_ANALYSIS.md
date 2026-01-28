# TensorBoard 指标分析报告

> 数据来源: `curl` 请求 `http://192.168.5.12:6006/data/plugin/scalars/...`  
> 配置依据: `mortal/config.toml`  
> 业务: 血战到底麻将 Mortal 自博弈训练（DQN + 排名预测辅助）

---

## 一、数据获取方式

TensorBoard 未提供官方 REST 文档，但 scalars 插件支持以下接口（run 为日志目录名，当前为 `.`）：

```bash
# 列出所有 tag
curl -s "http://192.168.5.12:6006/data/plugin/scalars/tags"

# 获取某 tag 的时序数据（返回 JSON 数组: [[wall_time, step, value], ...]）
curl -s "http://192.168.5.12:6006/data/plugin/scalars/scalars?run=.&tag=<TAG_URI>"
```

示例：`tag=loss%2Fdqn_loss` 对应 `loss/dqn_loss`。

---

## 二、当前训练进度与配置对应

| 配置项 (config.toml) | 值 | 说明 |
|----------------------|----|------|
| `save_every` | 1000 | 每 1000 step 存盘 + 写 TensorBoard |
| `test_every` | 5000 | 每 5000 step 跑一次 test_play 评估 |
| `[env] pts` | [6, 4, 2, 0] | 1位=6 分、2位=4、3位=2、4位=0（线性权重） |
| `[cql] min_q_weight` | 0.0 | 当前未启用 CQL 正则 |
| `[optim.scheduler] peak/final` | 1e-4 | 学习率固定 1e-4 |
| `[train_play.default] boltzmann_epsilon` | 0.5 | 自对局探索率 0.5（偏探索） |

**拉取到的数据范围**  
- 最新 step: **144,000**（约 144k × 2k ≈ 2.88亿 sample 量级的训练步数，与 batch_size=2048 及每 step 的 sample 数有关，此处按“步数”理解即可）。  
- loss 系列: 每 1000 step 一个点（144 个点）。  
- test_play 系列: 每 5000 step 一个点（28 个点：5k～140k）。

---

## 三、核心指标与业务含义

### 3.1 业务目标（与 config 一致）

- **排名**: 平均排名 `avg_ranking` 越接近 1 越好（1=一位率最高）。  
- **得分**: 使用线性权重 `pts = [6,4,2,0]`，平均分 `avg_pt` 的理论平均为 3.0；**avg_pt > 3.0** 表示优于随机，越高越好（train.py 里用 `avg_pt` 与 `avg_rank` 共同决定是否更新 best 模型）。

### 3.2 test_play/avg_ranking（平均排名）

| 阶段 (step) | avg_ranking | 简要说明 |
|-------------|-------------|----------|
| 5k  | 2.56 | 起步略优于随机（2.5 为随机期望） |
| 30k | 2.31 | 有提升 |
| 85k | 2.22 | 明显优于随机 |
| 125k| 2.03 | 接近 2.0 |
| **130k** | **1.92** | 已明显优于随机 |
| **140k** | **1.91** | 当前最佳，趋势平稳略优 |

**结论**  
- 从约 2.5+ 降到 1.91，排名能力持续变好，且近期（约 85k～140k）稳定在 2.0～2.2 并继续微幅改善。  
- 与配置中的“恢复/探索阶段”（boltzmann_epsilon=0.5）一致：先靠探索把策略拉起来，再逐步收敛。

### 3.3 test_play/avg_pt（加权平均得分）

| 阶段 (step) | avg_pt | 说明 |
|-------------|--------|------|
| 5k   | -9.0  | 初期远低于随机 |
| 15k  | 7.4   | 越过随机线 3.0 |
| 30k  | 11.8  | 明显优于随机 |
| 85k  | 17.0  | 稳定优于随机 |
| 125k | 27.6  | 已很强 |
| **140k** | **36.4** | 当前最佳 |

**结论**  
- 配置中“优于随机”的阈值为 **avg_pt >= 3.0**；当前 36.4 远高于该阈值，说明在 test_play 上已明显强于随机。  
- 与 `avg_ranking` 改善一致：排名变好、得分同步升高，无矛盾。

### 3.4 loss/dqn_loss（DQN TD 损失）

- **前期**: step 1k～5k 约 0.15～0.20，有轻微上升（0.17～0.20）。  
- **中期**: 约 0.16～0.19 波动。  
- **近期**: 140k～144k 约 **0.21～0.23**，略有抬升但仍属正常波动。

**结论**  
- 未出现持续爆炸或 NaN；小幅上升可能来自探索（boltzmann_epsilon=0.5）或数据分布变化，在自博弈训练中常见。  
- 若后续持续单边上升且伴随 avg_ranking/avg_pt 变差，再考虑降学习率或调 CQL。

### 3.5 loss/next_rank_loss（排名预测辅助损失）

- **前期**: 1k 约 1.29，5k 约 0.87，下降很快。  
- **后期**: 140k～144k 约 **0.50～0.71**，有波动但整体稳定在 0.5 左右。

**结论**  
- 排名预测任务收敛良好（config 中 `[aux] next_rank_weight = 0.2`），对主任务有辅助且未抢主 loss 风头。

### 3.6 loss/cql_loss（CQL 项）

- 当前配置 **min_q_weight = 0.0**，CQL 项未启用，曲线约在 1.16 附近平稳，仅作占位，**无需作为优化依据**。  
- 若之后开启 CQL（例如设为 1.0～5.0），再关注该曲线是否稳定、与 dqn_loss 的平衡。

### 3.7 hparam/lr（学习率）

- 全程 **1e-4** 恒定，与 `config.toml` 中 `peak = final = 1e-4`、`warm_up_steps = 0`、`max_steps = 0` 一致。

---

## 四、与配置、业务的对应关系

1. **test_every=5000**  
   - TensorBoard 中 test_play 指标为每 5k step 一个点，与配置一致；最新评估在 140k，下一次在 145k。

2. **save_every=1000**  
   - loss / lr 等为每 1k step 一个点，与配置一致。

3. **env.pts = [6,4,2,0]**  
   - avg_pt 使用相同线性权重；>3.0 即优于随机，当前 36+ 说明策略已远优于随机，与 avg_ranking≈1.91 相符。

4. **自对局与探索**  
   - boltzmann_epsilon=0.5、boltzmann_temp=1.0 表示当前仍偏“探索/恢复”阶段；指标上排名和得分都在稳步变好，说明数据质量和策略学习都在正向发展，可考虑在后续阶段按配置注释逐步把 epsilon/temp 降到 0.1～0.005 以加强利用。

5. **best 模型更新逻辑（train.py）**  
   - 以 `avg_pt` 提升且 `avg_rank` 不差为准更新 best；当前 140k 时 avg_pt=36.4、avg_ranking=1.91，若为历史最佳则会写入 `best_state_file`。

---

## 五、简要结论与建议

- **整体**  
  - 训练步数 144k，test_play 每 5k 评估一次；avg_ranking 从约 2.5 降到 1.91，avg_pt 从负值升到 36+，DQN 与排名辅助 loss 均正常，无异常发散。  
  - 与当前 config（探索为主、CQL 关闭、固定 lr）和业务目标（排名与加权得分）一致。

- **建议**  
  1. 继续跑到至少 150k～200k，观察 avg_ranking 是否稳定在 1.9 以下或继续微降、avg_pt 是否稳定或缓升。  
  2. 若进入“中期/后期”，按 config 注释逐步下调 `boltzmann_epsilon`（如 0.5→0.2→0.05）和 `boltzmann_temp`，观察 test_play 是否再上一档。  
  3. 若后续要启用 CQL，将 `min_q_weight` 从 0 调到 1.0～2.0 再观察 dqn_loss/cql_loss 与 test_play 的平衡。  
  4. 定期用 1v3 或 baseline 对局（config 中 `[1v3]`）做胜率/排名校验，与 TensorBoard 的 test_play 指标交叉验证。

---

## 六、附录：本次使用的 curl 示例

```bash
# 列出所有 scalar tags
curl -s "http://192.168.5.12:6006/data/plugin/scalars/tags"

# 获取 loss/dqn_loss 时序
curl -s "http://192.168.5.12:6006/data/plugin/scalars/scalars?run=.&tag=loss%2Fdqn_loss"

# 获取 test_play/avg_ranking 时序
curl -s "http://192.168.5.12:6006/data/plugin/scalars/scalars?run=.&tag=test_play%2Favg_ranking"

# 获取 test_play/avg_pt 时序
curl -s "http://192.168.5.12:6006/data/plugin/scalars/scalars?run=.&tag=test_play%2Favg_pt"
```

Tag 中的 `/` 需编码为 `%2F`（如 `test_play%2Favg_pt`）。

---

**文档生成时间**: 2026-01-28  
**TensorBoard 地址**: http://192.168.5.12:6006/
