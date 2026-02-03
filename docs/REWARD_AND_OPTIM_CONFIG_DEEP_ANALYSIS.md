# 奖励系统与优化配置深度分析

> 基于 `mortal/reward_calculator.py`、`mortal/dataloader.py`、`mortal/train.py`、`mortal/config.toml`、`rules.md` 及既有文档的逐项核对与**设计级**分析。

---

## 一、奖励系统深度分析

### 1.1 主奖励架构

```
数据源(libblood/grp) → scores_history + final_scores
                    → RewardCalculator.calc_delta_points()
                    → kyoku_rewards (每局得分变化 / 10000)
                    → train: q_target_mc = γ^steps_to_done * kyoku_rewards + ding_que_bonus
```

**核心公式**（`reward_calculator.py`）：
```python
seq = [scores_history[:, player_id], final_scores[player_id]]
delta_points = seq[1:] - seq[:-1]  # 每局得失分
kyoku_rewards = delta_points / 10000.0  # 1.0 reward ≈ 10000 点
```

**语义**：
- `kyoku_rewards[k]` = 玩家在第 k 局的**得分变化**（局初 → 下一局初，最后一局为局初 → 终局）
- 每条样本取 `kyoku_rewards[at_kyoku[i]]`，即**该步所在局**的局得失分
- 同一局内**所有 step 共用同一个 kyoku_rewards[k]**，不做局内信用分配

### 1.2 设计特性与权衡

| 维度 | 现状 | 影响 |
|------|------|------|
| **零和** | 每局四人 delta 之和 = 0 | 与血战到底规则一致，不引入虚假激励 |
| **信用分配** | 局内共享 delta | 无法区分「哪一步」贡献了本局得分；长轨迹时折扣减弱 |
| **量级** | 约 ±1.0（±10000 点） | 与 Q 值量级匹配，便于 DQN 学习 |
| **时间折扣** | γ^steps_to_done | 越接近局末的 step 越重视；steps_to_done 由 dataloader 逆序累加 |

### 1.3 潜在问题与改进方向

#### 问题 1：局内信用分配缺失

- **现状**：同局内打牌、碰、杠、和牌等所有 step 共享同一 delta
- **影响**：放铳者的「错误打牌」与和牌者的「正确和牌」在局内被同等对待（都得到该局 delta）
- **改进方向**（可选）：
  - 保持当前设计，依赖**长期多局**平均效应；或
  - 引入 per-step 奖励塑形（如：和牌步 +ε、放铳步 -ε），需谨慎避免 reward hacking

#### 问题 2：gamma 与长局

- **现状**：`gamma = 0.995`，局内 step 数可达 50+，γ^50 ≈ 0.78
- **影响**：早期 step 的回报衰减明显，模型可能偏向「局末几步」
- **调参建议**：
  - 若希望更重视眼前（如定缺、早期打牌）：可试 `gamma = 0.95`～`0.98`
  - 若希望更重视长期（多局排名）：保持 0.995 或略高

#### 问题 3：reward 缩放与 Q 量级

- **现状**：1.0 reward = 10000 点；Q 预测与 q_target_mc 量级约 ±1～2
- **观察**：`loss/dqn_loss` 稳定在 0.35～0.40，说明 Q 与 target 的 MSE 已进入平台
- **建议**：当前缩放合理；若 Q 长期发散，可检查 gamma、学习率或引入 reward clipping

### 1.4 定缺辅助系统

| 组件 | 配置项 | 作用 | 量级 |
|------|--------|------|------|
| **塑形 (A)** | `ding_que_aux_scale = 0.02` | 定缺步 Q 目标 += quality × 0.02 | 约 ±0.02 |
| **CE 监督 (B)** | `ding_que_ce_weight = 0.1` | 定缺步 CE(q_out[31:34], best_suit) | 梯度项，不改 Q 目标 |

**与主奖励关系**：
- 主奖励仍为主导（约 ±1.0），定缺项为小幅修正
- 不破坏零和、不改变非定缺步目标
- `ding_que/match_rate` 当前 ~71%，CE loss 持续下降，说明辅助有效

**调参建议**：
- `ding_que_aux_scale`：0.01～0.05 为宜，不宜超过 0.05
- `ding_que_ce_weight`：0.1 可维持；若 match_rate 已达 90%+ 可微降，避免过拟合启发式

---

## 二、优化配置深度分析

### 2.1 当前配置总览

| 类别 | 配置项 | 当前值 | 说明 |
|------|--------|--------|------|
| **环境** | gamma | 0.995 | 折扣因子 |
| | pts | [6,4,2,0] | 线性排名权重，与 rules 一致 |
| **优化器** | peak / final | 1e-4 | 学习率（恒定） |
| | warm_up_steps | 1000 | 预热步数 |
| | max_steps | 1000000 | 总步数上限 |
| | betas | [0.9, 0.999] | AdamW 动量 |
| | weight_decay | 0.1 | 权重衰减 |
| | max_grad_norm | 1.0 | 梯度裁剪 |
| **Loss 权重** | min_q_weight | 0.0 | CQL 关闭 |
| | next_rank_weight | 0.2 | 排名预测辅助 |
| | ding_que_ce_weight | 0.1 | 定缺 CE 辅助 |
| **训练节奏** | batch_size | 4096 | 每 step 样本数 |
| | save_every | 1000 | 存盘/写 TensorBoard 间隔 |
| | test_every | 5000 | test_play 间隔 |
| | opt_step_every | 1 | 每 step 做一次 optimizer.step |

### 2.2 学习率调度

**实现**（`lr_scheduler.py`）：LinearWarmUpCosineAnnealingLR
- 前 `warm_up_steps` 步：线性从 init(1e-8) 升到 peak(1e-4)
- `warm_up_steps` 到 `max_steps`：余弦退火到 final(1e-4)
- **当前**：peak = final = 1e-4 → **实际为恒定 lr = 1e-4**（仅 warmup 有效）

**优化建议**：
- 若希望长期收敛更稳：设 `final = 5e-5` 或 `2e-5`，让 lr 随步数衰减
- 若训练步数较短（<100k）：保持恒定 1e-4 亦可

### 2.3 梯度与正则

| 配置 | 作用 | 建议 |
|------|------|------|
| max_grad_norm = 1.0 | 梯度裁剪，防爆炸 | 当前合理；无 CQL 时梯度主要来自 DQN，1.0 可防异常 batch |
| weight_decay = 0.1 | L2 正则 | 常见 0.01～0.1；当前偏大，若过拟合可略降 |
| min_q_weight = 0 | CQL 关闭 | Zero Start / online 探索阶段合理；若改用离线数据可逐步开启 0.3～0.5 |

### 2.4 辅助任务权重

```
loss = dqn_loss + next_rank_loss * 0.2 + ding_que_ce_loss * 0.1
```

| 任务 | 权重 | 当前表现 | 建议 |
|------|------|----------|------|
| DQN | 1.0 | 主任务，dqn_loss ~0.36 | 保持 |
| next_rank | 0.2 | next_rank_loss ~0.66，优于随机 | 可微调 0.15～0.25 |
| ding_que_ce | 0.1 | ce_loss 下降，match_rate ~71% | 维持 0.1 |

### 2.5 探索与利用（train_play）

| 配置 | 当前值 | 阶段建议 |
|------|--------|----------|
| boltzmann_epsilon | 0.5 | 初期强探索；中期 0.15；后期 0.005 |
| boltzmann_temp | 0.5 | 同上，随阶段降至 0.2、0.05 |

**当前**：config 注释已标明分阶段建议；若 test_play 已稳定优于随机，可逐步降低探索，提高 exploit。

---

## 三、分阶段优化配置建议

### 3.1 初期（Zero Start，0～50k step）

| 维度 | 建议 |
|------|------|
| gamma | 0.995 |
| lr | 1e-4 恒定 |
| CQL | 0（保持） |
| 探索 | boltzmann_epsilon=0.5, temp=0.5 |
| 定缺 | ding_que_aux_scale=0.02, ce_weight=0.1 |
| max_grad_norm | 1.0 |

### 3.2 中期（50k～200k step）

| 维度 | 建议 |
|------|------|
| gamma | 0.995 或略降至 0.98（若希望更重视眼前） |
| lr | 可试 5e-5～1e-4，或启用 decay（final=5e-5） |
| CQL | 仍 0（online）；若改离线可试 0.3 |
| 探索 | boltzmann_epsilon=0.15, temp=0.2 |
| next_rank_weight | 可微调 0.2～0.25 |
| 定缺 | 维持 |

### 3.3 后期（200k+ step）

| 维度 | 建议 |
|------|------|
| lr | 2e-5～5e-5（衰减或手动降） |
| 探索 | boltzmann_epsilon=0.005, temp=0.05 |
| CQL | 若离线数据占比高，可试 0.5 |
| 定缺 | match_rate 高时可略降 ding_que_ce_weight |

---

## 四、奖励系统可配置化建议（可选）

当前 reward 缩放（/10000）与 gamma 硬编码在 dataloader/train 中。若需灵活调参，可考虑：

1. **config 新增 `[env]` 扩展**：
   ```toml
   [env]
   gamma = 0.995
   reward_scale = 10000.0   # 1.0 reward = reward_scale 点
   ```
2. **dataloader**：`kyoku_rewards = delta_points / config['env']['reward_scale']`
3. **train**：`gamma = config['env']['gamma']`（已实现）

这样可在不改代码的情况下做 reward scale / gamma 消融实验。

---

## 五、总结表

| 类别 | 当前状态 | 优先优化点 |
|------|----------|------------|
| **主奖励** | 局得分变化/10000，零和，局内共享 | 保持；可选 reward_scale 配置化 |
| **定缺辅助** | 塑形+CE，量级小，有效 | 维持；match_rate 高时可微调 |
| **gamma** | 0.995 | 按需求试 0.95～0.995 |
| **lr** | 1e-4 恒定 | 长期训练可启用 decay |
| **CQL** | 0 | online 保持；离线可逐步开启 |
| **探索** | 0.5/0.5 | 按阶段降至 0.15→0.005 |
| **辅助权重** | next_rank=0.2, ding_que_ce=0.1 | 可微调，非瓶颈 |

整体结论：**奖励系统设计正确、与规则一致**；**优化配置可随训练阶段做渐进调参**，优先关注 gamma、探索参数与 lr decay。
