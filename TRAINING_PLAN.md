# 血战到底 AI 超人类训练计划 V2

> 基于全部代码审计（ISSUES.md 77 文件）和 70k+ 步纯自博弈实战经验制定。
> 目标：从零知识自博弈达到超人类水平。
> 配套配置文件: `mortal/config.fresh_start.toml`

---

## 1. 系统架构总览

### 1.1 模型结构

```
输入 (obs: 423×27)
  │
  ▼
Brain (ResNet: SuitAwareConv1d × 30 blocks, 192 ch)
  │  - 花色隔离卷积: 万/筒/条独立卷积，无跨花色边界污染 (MODEL-04)
  │  - BatchNorm + Mish 激活 + ChannelAttention (SE-Net)
  │  - 最终 Conv 64ch → Flatten → Linear(1728→1024)
  │
  ├─→ DQN (Dueling: V-stream + A-stream, 各 512 隐藏层)  →  Q(s,a) [35 actions]
  │     - 动作空间: 弃牌(27) + 碰(1) + 杠(1) + 和(1) + pass(1) + 定缺(3) + 流局(1)
  │     - V/A 独立流 (MODEL-01), A 输出层零初始化
  │
  └─→ AuxNet (1024→512→88)  →  辅助任务
        - next_rank: 排名预测 (4 分类)
        - opp_wait: 对手听牌预测 (3×27 = 81, BCE)
        - ding_que: 定缺分类 (3 分类, CE, MODEL-03 独立分支)
```

### 1.2 参数规模

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Brain (ResNet) | ~7.5M | 192ch × 30 blocks, SuitAwareConv1d |
| DQN | ~1.07M | Dueling V/A 各 512 hidden |
| AuxNet | ~568K | 512 hidden → (4+81+3) |
| **合计** | **~9.1M** | |

### 1.3 训练框架

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  Client ×7  │────▶│  Server (buffer/drain)│────▶│  Trainer    │
│  (self-play)│     │  port 5000            │     │  (GPU cuda:0)│
│  每卡 1 个  │◀────│                       │◀────│             │
└─────────────┘     └──────────────────────┘     └─────────────┘
  生成对局数据       存储/分发数据                  DQN + 辅助损失
  6400 games/batch   capacity: 40000               batch: 4096
```

- **Online 自博弈**: Client 用当前模型对局 → Server 存储 → Trainer 读取训练
- **TD(λ=1.0)**: 等价于 Monte Carlo 回报（BUG-06: 无 V(s') bootstrap 时必须 λ=1）
- **Baseline**: 定期手动更新 (`cp mortal.pth baseline.pth`)，test_play 对比评估

---

## 2. 硬件要求

| 资源 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 1× RTX 4090 (24GB) | 1× RTX 4090 训练 + 7× 4090 self-play |
| CPU | 16 核 | 64 核（self-play 瓶颈在 CPU） |
| 内存 | 32GB | 128GB |
| 存储 | 100GB SSD | 500GB+ NVMe（对局日志累积快） |
| 训练精度 | FP32 | FP32（AMP/FP16 已确认不稳定，PERF-02）|

---

## 3. 启动前检查清单

```bash
# 1. 目录准备
mkdir -p /data/mortal/{train_play,test_play,buffer,drain,logs}

# 2. 编译 Rust 库（需要 Python 环境 + PyO3）
cd /path/to/blood
pip install maturin
maturin develop --release -m libblood/Cargo.toml

# 3. Python 依赖
pip install torch numpy numba tensorboard tqdm indicatif

# 4. 激活 fresh start 配置
cp mortal/config.fresh_start.toml mortal/config.toml

# 5. 验证可启动
cd mortal && python -c "from config import config; print('OK:', config['control']['version'])"
```

---

## 4. 深度现状分析

### 4.1 训练实况总结（Step 0-70k）

纯自博弈 70k 步，3 次 baseline 更新（20k, 38k, 55k），关键发现：

- **已掌握**: 定缺 98.5%、基本出牌、碰杠判断（副露率从 95% 降至 77%）
- **未掌握**: 防守（放铳率 33.8%）、番数追求（和牌平均 5,500 点 ≈ 1-2 番）、对手建模
- **核心瓶颈**: 高探索率（30%）产生嘈杂数据 + 无排名奖励 → 模型无法学习防守

### 4.2 奖励信号量级分析

代码实际奖励计算（`dataloader.py:232`）：

```python
kyoku_rewards = calc_delta_points(...) / 10000.0  # 1.0 reward = 10,000 点
```

Phase 1 各奖励信号量级（每局 kyoku）：

- 分数差: 和牌 ±0.1~1.6（1番=0.1, 2番=0.2, 3番=0.4, 4番=0.8, 5番封顶=1.6）
- action_bonus: 和牌 +0.1 / 放铳 -0.1
- rank_bonus: **关闭** ← 这是防守学不会的关键原因
- ding_que_bonus: ±0.02（极微）

**问题**：放铳的惩罚信号来自分数差（失去点数），但通过 MC 回报传播到每一步时被大幅稀释（γ^n 衰减 + 与其他步骤信号混合）。没有 rank_bonus 意味着没有"垫底会被额外惩罚"的信号，模型不学习风险规避。

### 4.3 自博弈 baseline 更新经验

| BL# | Step | 更新后恢复速度 | 分析 |
|-----|------|------------|------|
| #1 | 20k | 快（5k步回升） | baseline 极弱（随机→初级），差距小 |
| #2 | 38k | 中（12k步恢复） | baseline 有基本能力，差距适中 |
| #3 | 55k | 慢（15k+未完全恢复） | baseline 显著更强，模型防守弱点暴露 |

**结论**: 随着 baseline 变强，恢复时间指数增长。过早/过频繁更新会导致模型长期处于追赶状态。

### 4.4 训练实况总结（Step 70k-240k）

#### Phase 2 奖励塑形实验 (70k-180k)

70k 切换 Phase 2 配置（rank_bonus=true, epsilon 0.15, temp 0.20）后：
- BL#4 (100k) 和 BL#5 (130k) 更新后，模型陷入长期追赶（perpetual chasing）
- 110k 步的 rank_bonus + action_bonus 未能教会防守，放铳率从 33% 上升到 38-41%
- agari_bonus 反而激励"快攻小牌"（局部最优），和牌点数停滞在 ~5,500

#### 纯 score_diff 实验 (180k-235k，关闭所有奖励塑形)

180k 关闭所有奖励塑形（rank_bonus=false, action_bonus=false），55k 步纯 score_diff 结果：

| 指标 | 180k (关闭前) | 225k (最佳点) | 235k | 分析 |
|------|-------------|-------------|------|------|
| 和牌点数 | 5,256 | 6,501 | 6,373 | +21-33%，模型学会追大牌 |
| 放铳率 | 41.0% | 44.0% | 45.2% | 不降反升，防守未学会 |
| point_per_round | -314 | -80 | -341 | 高方差震荡，未能转正 |
| 和牌率 | 63.4% | 59.4% | 56.1% | 下降（追大牌的代价） |
| 4th 占比 | 26.8% | 28.4% | 30.2% | 恶化 |

**结论**：纯 score_diff 能教会手牌价值（和牌点数 +41%），但无法教会防守。MC 回报的信用分配太慢，模型进入"赌博模式"（高和牌点数 + 高放铳率）。

#### BL#6 更新实验 (236k-240k)

236k 更新 BL#6（激进型 235k 模型），240k 评估：

| 指标 | 235k (BL更新前) | 240k (BL更新后) | 变化 |
|------|-----------------|-----------------|------|
| 放铳率 | 45.2% | 39.2% | -6%! |
| 4th 占比 | 30.2% | 26.5% | -3.7% |
| avg_ranking | 2.590 | 2.531 | 改善 |
| 和牌点数 | 6,373 | 6,837 | 维持高位 |

**但存在"虚假进步"风险**：BL#6 本身放铳率 ~45%，3 个激进型对手互相喂分，模型只是利用了对手弱点而非真正学会防守。这是典型的**单一 baseline 过拟合**。

### 4.5 成功案例研究

调研了业界主要自博弈 AI 系统的训练策略：

#### AlphaGo Zero（围棋，零知识）
- MCTS 每步 1600 次模拟，提供强探索和精确价值估计
- Baseline 更新：新网络在 400 局中胜率 >55% 才替换
- 单一对手，但 MCTS 弥补了对手多样性不足

#### OpenAI Five（Dota 2，零知识）
- **80% 对局 vs 最新自身，20% vs 历史版本对手池**
- 历史版本按质量加权采样（更强对手被更频繁选中）
- 规模：256 GPU，每天 180 年等效对局量

#### AlphaStar（星际争霸 II）
- **联赛训练**：主力 + 主力剥削者 + 联赛剥削者，3 类 agent
- 优先虚拟自博弈（PFSP）选择对手
- 先监督学习人类数据，再 RL

#### Suphx（日麻，超人类）
- **非零知识**：Oracle Guiding（完美信息监督预训练）
- Global Reward Prediction 改善信用分配
- 运行时策略自适应

#### Mortal（日麻，本代码库原型）
- 监督预训练（人类对局数据）+ 自博弈 RL 强化

#### 核心洞察

我们的方案（纯零知识 + DQN 无 MCTS + 单一 baseline）比所有成功系统都更难。每个成功系统至少使用了以下之一：**MCTS、监督预训练、对手池、联赛训练**。

### 4.6 诊断总结与对手池方案

#### 三大瓶颈

1. **单一 Baseline 过拟合**：模型优化"打赢这个对手"而非"打好麻将"，策略循环
2. **纯 score_diff 无防守信号**：MC 回报信用分配太慢，放铳率锁死 44-46%
3. **Baseline 更新两难**：太频繁→追赶，太稀疏→过拟合

#### 对手池方案（V3 策略，待实施）

参考 OpenAI Five 的 80/20 策略，引入对手池：

```
[baseline.pool]
enabled = true
pool_dir = '/data/mortal/baseline_pool'    # 存放 5-8 个历史检查点
reload_every = 1                            # 每次 train_play 迭代重选 baseline
newest_weight = 3.0                         # 最新检查点采样权重 3x
```

**代码改动**（3 个文件）：
- `player.py`：添加 `_select_from_pool()` 和 `reload_baseline_from_pool()`
- `client.py`：主循环中周期性重载 baseline
- `config.toml`：新增 `[baseline.pool]` 配置段

**配合措施**：
- 重新启用 `houjuu_penalty = -0.2`（仅放铳惩罚，不开 agari_bonus 和 rank_bonus）
- 每 20k 步保存检查点到对手池，不等 point_per_round > 0
- 初始池：130k（保守型）、180k、225k、235k（激进型）

---

## 5. 训练阶段

### Phase 1: 基础探索 [已完成] (Step 0 — 70k)

**目标**: 从随机策略学会基础出牌逻辑（和牌、定缺、避免花猪）

#### 配置 (config.fresh_start.toml)

| 参数 | 值 | 说明 |
|------|-----|------|
| `boltzmann_epsilon` | **0.3** | 高探索率 |
| `boltzmann_temp` | **0.3** | 高温度，策略分布平滑 |
| `gamma` | **0.99** | 远视折扣 |
| `rank_bonus_enabled` | **false** | 纯分数差学习 |
| `action_bonus_enabled` | **true** | 和牌/放铳信号 (±0.1) |
| `ding_que_ce_weight` | **1.0** | 定缺监督适度 |
| `opp_wait_enabled` | **true** | 对手听牌预测 |
| `peak` LR | **5e-4** | 高学习率加速 |
| `weight_decay` | **0.01** | 适度正则化 |
| `batch_size` | **4096** | 4090 24GB 可承载 |
| `randomize_fan_config` | **false** | 标准规则 |

#### 实际达成情况

| 指标 | 初始值 | 达成值 (70k) | 说明 |
|------|--------|-------------|------|
| 定缺准确率 | ~33% | 98.5% | MODEL-03b 修复后快速收敛 |
| 基本出牌 | 随机 | 初步掌握 | 碰杠判断基本合理 |
| 副露率 | ~95% | ~77% | 学会控制副露 |
| 放铳率 | ~25% | ~33.8% | ⚠️ 未学会防守 |
| 和牌平均点数 | — | ~5,500 | 主要做 1-2 番小牌 |

#### Baseline 更新记录

| BL# | Step | 恢复速度 | 说明 |
|-----|------|---------|------|
| #1 | 20k | 快 (5k步) | baseline 极弱，差距小 |
| #2 | 38k | 中 (12k步) | baseline 有基本能力 |
| #3 | 55k | 慢 (15k+) | baseline 显著更强，防守弱点暴露 |

#### 启动命令

```bash
# 终端 1: Server
cd mortal && python server.py

# 终端 2-8: Self-play Clients (每张 GPU 一个)
CUDA_VISIBLE_DEVICES=1 python client.py  # 重复 7 次，GPU 1-7

# 终端 0: Trainer
python train.py
```

⚠️ **Phase 1 核心瓶颈**: 高探索率 (30%) 产生嘈杂数据 + 无排名奖励 → 模型无法学习防守

---

### Phase 2: 防守与收敛 (Step 70k — 355k) [已完成]

**目标**: 学会基本防守，提升手牌价值

#### 配置修改 (修改 config.toml 后重启 train.py + 所有 client.py)

```toml
# 降低探索，减少噪声数据
boltzmann_epsilon = 0.15    # 0.30 → 0.15
boltzmann_temp = 0.20       # 0.30 → 0.20

# 定向防守信号（V3 策略：仅 houjuu_penalty，不开 rank_bonus 和 agari_bonus）
[reward_shaping]
rank_bonus_enabled = false
action_bonus_enabled = true
agari_bonus = 0.0
houjuu_penalty = -0.2       # 放铳直接惩罚，改善信用分配

# 提高定缺监督
[aux]
ding_que_ce_weight = 2.0    # 1.0 → 2.0
```

#### 实际达成情况 (355k)

| 指标 | 入口值 (70k) | 达成值 (355k) | 目标 | 评价 |
|------|-------------|-------------|------|------|
| 放铳率 | ~33.8% | ~32% | < 35% | 达标 |
| 和牌点数 | ~5,500 | ~7,100 | > 6,000 | 超额达成 |
| `avg_ranking` | — | ~2.49 | < 2.45 | 接近但未达标（plateau） |
| `point_per_round` | — | ~-20 | > 0 | 接近零线但未转正 |
| `ding_que/dqn_match_rate` | 98.5% | 99.5% | >99% | 达标 |

#### Phase 2 关键事件

- **70k-180k**: rank_bonus + action_bonus 未能教会防守，放铳率上升至 38-41%
- **180k**: 关闭所有奖励塑形，纯 score_diff 驱动
- **180k-235k**: 和牌点数 +41%，但放铳率恶化至 45%（"赌博模式"）
- **249k**: V3 实施 — 对手池 + houjuu_penalty=-0.2，多样化对手训练
- **249k-355k**: 放铳率从 45% 降至 32%，和牌点数维持 ~7,100，但 355k 处出现 plateau

#### Phase 2 瓶颈分析 (355k plateau)

1. **对手池陈旧**：最新检查点为 295k，与当前 355k 差距 60k 步
2. **探索率偏高**：epsilon=0.15 在 355k 时产生过多噪声
3. **houjuu_penalty 过强**：-0.2 使模型过度保守，牺牲进攻机会
4. **学习率偏高**：peak=5e-4 在精调阶段导致参数震荡
5. **Phase 2 超期**：原计划 300k 结束，实际运行到 355k

#### Baseline 更新策略 (V3 修订：对手池)

**V3 策略**：引入对手池，替代单一 baseline 更新。

对手池管理：
1. 每 20k 步将当前检查点保存到 `/data/mortal/baseline_pool/`
2. 保留最近 5-8 个检查点，删除更旧的
3. 每次 train_play 迭代从池中加权随机选取 baseline（最新 3x 权重）

```bash
# 保存检查点到对手池
cp /data/mortal/mortal.pth /data/mortal/baseline_pool/mortal_XXXk.pth
# 如果未实施对手池代码，仍使用传统方式：
cp /data/mortal/mortal.pth /data/mortal/baseline.pth
# 重启所有 client.py
```

---

### Phase 3: 进攻/防守平衡 (Step 355k — 485k) [已完成]

**目标**: 在维持防守的前提下提升和牌质量和整体胜率，突破 355k plateau

#### 配置修改 (Step 355k 切换)

```toml
# 进一步降低探索
boltzmann_epsilon = 0.08    # 0.15 → 0.08
boltzmann_temp = 0.15       # 0.20 → 0.15

# 减弱放铳惩罚，让分数差主导
[reward_shaping]
rank_bonus_enabled = false
action_bonus_enabled = true
agari_bonus = 0.0
houjuu_penalty = -0.1       # -0.2 → -0.1（减弱，分数差已可接力）

# 降低学习率
[optim.scheduler]
peak = 3e-4                 # 5e-4 → 3e-4
final = 5e-5                # 1e-4 → 5e-5
warm_up_steps = 3000        # 2000 → 3000（平滑过渡）
```

#### 对手池状态 (355k 切换时)

| 检查点 | 风格 | 备注 |
|--------|------|------|
| 130k | 保守型 | Phase 2 早期 |
| 200k | 中间型 | 纯 score_diff 初期 |
| 249k | 激进型 | V3 实施点 |
| 295k | 平衡型 | V3 运行中 |
| 355k | 当前最新 | Phase 3 起点，newest_weight=3.0 |

#### 监控目标

| 指标 | 入口值 (355k) | 目标 |
|------|-------------|------|
| 放铳率 | ~32% | < 25% |
| 和牌点数 | ~7,100 | > 7,500 |
| `avg_ranking` | ~2.49 | < 2.30 |
| `point_per_round` | ~-20 | > 0 (稳定转正) |

#### Baseline 更新策略

对手池持续运作，每 20-30k 步添加新检查点。

---

### Phase 4: 纯博弈 (Step 485k — 580k) [已完成]

**目标**: 关闭所有奖励塑形，纯 score_diff 驱动

#### 配置修改

```toml
boltzmann_epsilon = 0.05    # 0.06 → 0.05
boltzmann_temp = 0.10       # 0.12 → 0.10

[reward_shaping]
rank_bonus_enabled = false
action_bonus_enabled = false  # 全部关闭
agari_bonus = 0.0
houjuu_penalty = 0.0          # -0.05 → 0
```

#### 实际达成情况 (580k)

| 指标 | 入口值 (485k) | 达成值 (580k) | 评价 |
|------|-------------|-------------|------|
| 放铳率 | ~34% | ~34% | 稳定，防守已内化 |
| 和牌点数 | ~6,500 | ~6,200 | 下降（低番速胡退化） |
| point_per_round | ~-260 | ~-391 | 恶化 |
| agari_rate | ~60% | ~62% | 偏高 |

#### Phase 4 瓶颈分析 (580k plateau)

1. **对手池过乱**：130k-485k 共 7 个检查点，67% 对局面对远弱于自己的对手
2. **低番速胡策略退化**：agari_point 从 6666(510k) 降至 5964(555k)，模型学会"打弱对手"
3. **梯度分布严重失衡**：审计发现 next_rank_loss 占总梯度 54.5%，DQN 仅占 43%
4. **隐藏奖励塑形**：ding_que_bonus (±0.02) 仍在 Q-target 上活跃

**对手池清理** (550k)：删除 130k/200k/249k/295k/355k，仅保留 420k/485k/550k

---

### Phase 5: 梯度净化 (Step 580k+) [当前阶段]

**目标**: 消除辅助任务对 RL 梯度的干扰，让 DQN 主导编码器学习

#### 问题诊断

审计发现 Brain 编码器梯度分布严重失衡：

| 损失项 | 加权值 | 占比 | 问题 |
|--------|--------|------|------|
| dqn_loss | 0.229 | 43.2% | RL 主目标，应占 90%+ |
| next_rank_loss × 0.25 | 0.289 | **54.5%** | 辅助任务反客为主 |
| ding_que_ce × 2.0 | 0.006 | 1.1% | |
| opp_wait × 0.1 | 0.005 | 0.9% | |
| ding_que_dqn_ce × 0.1 | 0.0004 | 0.08% | DQN 竞争梯度 |

血战到底单局结算，`next_rank` 预测排名的辅助任务价值有限，却消耗了超过一半的编码器学习能力。

#### 配置修改 (Step 580k)

```toml
[aux]
next_rank_weight = 0.0         # 0.25 → 0: 关闭，血战到底无需跨局排名预测
ding_que_aux_enabled = false   # true → false: 去掉 Q-target 上的隐藏奖励 (±0.02)
ding_que_ce_weight = 1.0       # 2.0 → 1.0: 定缺已收敛 (CE=0.003)
ding_que_dqn_ce_weight = 0.0   # 0.1 → 0: 消除竞争梯度
```

#### 变更后梯度分布

| 损失项 | 加权值 | 占比 |
|--------|--------|------|
| dqn_loss | 0.229 | **95.4%** |
| ding_que_ce × 1.0 | 0.003 | 1.3% |
| opp_wait × 0.1 | 0.005 | 2.1% |

#### 监控目标

| 指标 | 入口值 (580k) | 目标 |
|------|-------------|------|
| 放铳率 | ~34% | < 30% |
| point_per_round | ~-300 | > 0 (稳定转正) |
| avg_ranking | ~2.53 | < 2.40 |
| agari_point | ~6,200 | > 6,500 |

#### 对手池状态

| 检查点 | 选中概率 | 备注 |
|--------|---------|------|
| 420k | 20% | 强对手，不同风格 |
| 485k | 20% | 接近当前水平 |
| 550k | 60% (newest) | 自我对弈 |

#### 风险与回退

- **风险**: 关闭 next_rank 可能导致短期波动（编码器特征重组）
- **回退条件**: 30k 步后 point_per_round < -500，恢复 next_rank_weight = 0.05

---

### Phase 6: Target Network 引入 (Step ~800k — ~1.2M)

**目标**: 引入 Target Network，使 TD(λ<1) 生效，解决信用分配瓶颈

#### 问题诊断

当前 TD(λ=1.0) 等价于 Monte Carlo 回报。一局 ~20 步出牌，仅 1-2 次关键决策产生分数变化，
MC 将 reward 均匀回传到所有前序动作：

- 1 番荣和 (1000点) → reward = 0.1
- 回传 20 步后，第 1 步信号 ≈ 0.1 × 0.99^19 ≈ 0.083
- 每个动作有效梯度信号 ≈ 0.004

**结果**: 模型无法精确定位「是哪步出牌导致放铳」，防守学习极慢。

#### 改动要点

1. **添加 Target Network**（Brain + DQN 的延迟副本）
2. **软更新**：每 1000 步，τ=0.005
3. **降低 td_lambda**：1.0 → 0.95（启用 V(s') bootstrap）
4. **降低探索**：epsilon 0.05→0.03, temp 0.10→0.08

```toml
boltzmann_epsilon = 0.03    # 0.05 → 0.03
boltzmann_temp = 0.08       # 0.10 → 0.08

[env]
td_lambda = 0.95            # 1.0 → 0.95 (Target Network 启用后)

[target_network]
enabled = true
tau = 0.005                 # 软更新系数
update_every = 1000         # 每 N 步更新

[optim.scheduler]
peak = 1e-4                 # 3e-4 → 1e-4
final = 1e-5
```

#### 代码改动（3 个文件）

| 文件 | 改动 |
|------|------|
| `train.py` | 创建 target_brain/target_dqn；软更新循环；TD(λ) 传入 V(s') |
| `dataloader.py` | `compute_td_lambda_returns()` 接收 V(s') 参数做 bootstrap |
| `config.toml` | 新增 `[target_network]` 配置段 |

#### 预期效果

| 指标 | 入口 | 目标 | 原理 |
|------|------|------|------|
| 放铳率 | ~30% | < 25% | 信用分配精确到单步，防守行为快速收敛 |
| point_per_round | ~0 | > +200 | V(s') 减少 MC 方差，策略稳定 |
| agari_point | ~6500 | > 7000 | 更好地评估追番 vs 速胡的 EV |
| avg_ranking | ~2.40 | < 2.25 | 综合提升 |

#### 风险与回退

- **风险**: Target Network 初始化不当可能导致 V(s') 估计偏差
- **回退**: 若 30k 步后 dqn_loss 爆炸，恢复 td_lambda=1.0 并检查 target 更新频率

---

### Phase 7: 超人类突破 (Step ~1.2M+)

**目标**: 引入推理时搜索或 Oracle Guiding，突破单次前向传播的天花板

#### 选项 A: Oracle Guiding（参考 Suphx）

1. 训练 Oracle 模型：输入包含所有玩家手牌（完美信息），学习理论最优策略
2. Oracle 监督预训练：用 Oracle 策略做 CE 监督，让当前模型学会「理想出牌」
3. 切回自博弈 RL：在 Oracle 初始化的基础上继续强化

| 阶段 | 训练方式 | 预计步数 |
|------|---------|---------|
| Oracle 训练 | 自博弈（完美信息，收敛快） | ~200k |
| Oracle 蒸馏 | CE(student, oracle) | ~100k |
| RL 强化 | 自博弈（不完美信息） | ~500k+ |

#### 选项 B: 推理时搜索（轻量版）

不改训练流程，只在推理时增强：

1. 对每个合法动作，用当前模型 rollout N 局（N=100-500）
2. 取平均回报最高的动作
3. 可选：信息集采样（随机猜测对手手牌，多次采样取平均）

| 参数 | 值 | 说明 |
|------|-----|------|
| rollout_per_action | 100-500 | 每个动作模拟局数 |
| info_set_samples | 10-50 | 对手手牌采样次数 |
| 推理延迟 | ~100ms/步 | 可接受（人类思考 1-5 秒） |

#### 选项 C: Population-Based Training（参考 AlphaStar）

维护 5-10 个独立 agent，各自与不同对手组合训练：

- **主力 agent**：正常自博弈
- **剥削者**：专门针对主力弱点训练
- **防守型**：优化放铳率目标
- **进攻型**：优化 agari_point 目标

所有 agent 组成联赛，互相对弈产生数据。

#### 推荐路径

**Phase 7A (Oracle Guiding)** > Phase 7B (搜索) > Phase 7C (Population)

理由：
- Oracle Guiding 实现复杂度适中，效果已在 Suphx 验证
- 血战到底规则简单，Oracle（完美信息）训练快
- 搜索需要改推理流程，延迟增加；Population 需要大幅增加计算资源

#### 监控目标

| 指标 | 入口 | 超人类目标 |
|------|------|-----------|
| avg_ranking | < 2.25 | < 2.05 |
| 放铳率 | < 25% | < 18% |
| agari_point | > 7000 | > 7500 |
| point_per_round | > +200 | > +500 |

---

## 6. 阶段转换决策树

```
Step 0 ──────────── Phase 1 [已完成] ───┐
                                         │  70k steps, 防守弱(33.8%)
                                         ▼
                    Phase 2 [已完成] ────┐
                    防守与收敛 (70k-355k)  │  对手池 + houjuu_penalty
                    放铳率 ~32% ─────────▼
                    Phase 3 [已完成] ────┐
                    进攻/防守 (355k-485k)  │  减弱惩罚, 降 LR
                    放铳率 ~34% ─────────▼
                    Phase 4 [已完成] ────┐
                    纯博弈 (485k-580k)    │  关闭所有奖励塑形
                    plateau, 梯度审计 ───▼
                    Phase 5 [当前]       │
                    梯度净化 (580k-800k)   │  DQN 梯度 43%→95%
                    point_per_round > 0? ─▼
                    Phase 6 [计划]       │
                    Target Network (800k-1.2M) │  TD(λ=0.95), 信用分配改善
                    放铳率 < 25%? ────────▼
                    Phase 7 [计划]
                    超人类突破 (1.2M+)
                    Oracle Guiding / 搜索 / Population
```

**阶段转换操作**:
1. 修改 `mortal/config.toml` 中对应参数
2. 手动更新 baseline: `cp mortal.pth baseline.pth`
3. 重启 `train.py`（scheduler 自动适配新参数）
4. 重启所有 `client.py`（加载新 baseline）

**自博弈 Baseline 更新规则** (V3 修订：对手池策略):

#### 历史经验

- V1（Phase 1）：手动更新，15-20k 间隔，适用于弱 baseline
- V2（Phase 2 初期）：要求 point_per_round > 0 再更新，但模型 55k 步未达标
- **单一 baseline 的根本问题**：模型过拟合于特定对手风格（策略循环），更新时机无论如何选择都有缺陷

#### V3 策略：对手池

参考 OpenAI Five 的 80/20 对手池策略，用对手多样性替代单一 baseline 的更新时机问题。

| 阶段 | 池更新间隔 | 池大小 | 最新权重 |
|------|----------|--------|---------|
| Phase 2 | 20k 步 | 5-8 个 | newest_weight = 3.0 |
| Phase 3 | 20-30k 步 | 5-8 个 | newest_weight = 3.0 |
| Phase 4 | 30-50k 步 | 5-8 个 | newest_weight = 2.0 |
| Phase 5 | 50-80k 步 | 5-8 个 | newest_weight = 2.0 |

> 如果对手池代码尚未实施，退回 V2 策略：手动更新单一 baseline，间隔 ≥ 20k 步。

---

## 7. 关键配置速查表

### 7.1 全阶段配置对比

| 参数 | Phase 1 (完成) | Phase 2 (完成) | Phase 3 (完成) | Phase 4 (完成) | Phase 5 (当前) | Phase 6 (计划) | Phase 7 (计划) |
|------|---------------|---------------|----------------|----------------|----------------|----------------|----------------|
| epsilon | 0.30 | 0.15 | 0.08 | 0.05 | **0.05** | 0.03 | 0.02 |
| temp | 0.30 | 0.20 | 0.15 | 0.10 | **0.10** | 0.08 | 0.05 |
| rank_bonus | false | false | false | false | **false** | false | false |
| agari_bonus | 0.1 | 0.0 | 0.0 | 0.0 | **0** | 0 | 0 |
| houjuu_penalty | -0.1 | -0.2 | -0.1 | 0 | **0** | 0 | 0 |
| next_rank_weight | 0.25 | 0.25 | 0.25 | 0.25 | **0** | 0 | 0 |
| ding_que_aux | true | true | true | true | **false** | false | false |
| ding_que_ce | 1.0 | 2.0 | 2.0 | 2.0 | **1.0** | 1.0 | 1.0 |
| ding_que_dqn_ce | 0.1 | 0.1 | 0.1 | 0.1 | **0.01** | 0.01 | 0.01 |
| td_lambda | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** | 0.95 | 0.95 |
| target_network | 无 | 无 | 无 | 无 | **无** | 有 (τ=0.005) | 有 |
| peak LR | 5e-4 | 5e-4 | 3e-4 | 3e-4 | **3e-4** | 1e-4 | 1e-4 |
| final LR | 1e-4 | 1e-4 | 5e-5 | 5e-5 | **5e-5** | 1e-5 | 1e-5 |
| BL策略 | 手动 15-20k | 对手池 20k | 对手池 20-30k | 对手池 30-50k | **滑动窗口 3个** | 滑动窗口 | Oracle+滑动窗口 |
| 特殊技术 | — | — | — | — | **梯度净化** | Target Network | Oracle/搜索 |

### 7.2 学习率进度表

| 阶段 | peak | final | warmup | max_steps |
|------|------|-------|--------|-----------|
| Phase 1-2 | 5e-4 | 1e-4 | 2000 | 1000000 |
| Phase 3-5 | 3e-4 | 5e-5 | 3000 | 1000000 |
| Phase 6 | 1e-4 | 1e-5 | 5000 | 2000000 |
| Phase 7 | 1e-4 | 1e-5 | 5000 | 3000000 |

---

## 8. 监控与故障排除

### 8.1 TensorBoard 监控

```bash
tensorboard --logdir /data/mortal/logs --bind_all
```

关键面板:
- `dqn_loss`: 应持续下降后趋于稳定
- `avg_rank`: 应持续下降（越低越好，理论最优 1.0）
- `avg_pt`: 应持续上升（理论最优 4.0）
- `next_rank_loss`: 辅助损失，应下降
- `opp_wait_loss`: 对手听牌预测损失
- `ding_que_ce_loss`: AuxNet 定缺分类损失（应快速下降）
- `ding_que_dqn_ce_loss`: DQN 定缺弱 CE 损失（应逐步下降）
- `ding_que/aux_match_rate`: AuxNet 定缺分类准确率（应快速上升至 >90%）
- `ding_que/dqn_match_rate`: DQN 实际定缺决策准确率（应逐步上升至 >70%）
- `lr`: 学习率曲线，验证 scheduler 行为

### 8.2 常见故障

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| dqn_loss 爆炸 (NaN) | LR 过高 / AMP 溢出 | 确认 `enable_amp = false`; 降低 peak LR |
| avg_rank 不降 | self-play 数据不足 | 检查 client 是否运行; 检查 `/data/mortal/buffer` |
| avg_rank 突然跳升 | baseline 与模型差距过大 | 更新 baseline: `cp mortal.pth baseline.pth` |
| DataLoader 僵死 | BUG-01 (PyO3/rayon) | 自动重启机制已内置; 若持续则手动重启 train.py |
| 内存溢出 (OOM) | batch_size 过大 | 降低 batch_size: 4096→2048→1024 |
| `dqn_match_rate` 卡在 ~33% | DQN 定缺信号不足 | 提高 `ding_que_dqn_ce_weight` 至 0.2-0.5 |
| `aux_match_rate` 卡在 ~33% | AuxNet 监督信号不足 | 提高 `ding_que_ce_weight` 至 2-3 |
| 和牌率低 | 探索不足 / 奖励信号弱 | 提高 epsilon/temp; 增大 agari_bonus |
| 训练速度慢 | CPU 瓶颈 | 增加 client 数; 提高 num_workers |

### 8.3 磁盘空间管理

```bash
# 每 ~50k 步清理旧 train_play 数据（保留最近 20k 步即可）
find /data/mortal/train_play -name "*.json.gz" -mtime +3 -delete

# 检查磁盘占用
du -sh /data/mortal/*
```

---

## 9. 数据流与损失函数

### 9.1 主损失: DQN (Bellman)

```
L_dqn = MSE(Q(s,a), r + γ·max_a' Q(s',a'))
```

- 当前使用 TD(λ=1.0) ≡ Monte Carlo: `target = Σ γ^t · r_t`
- 未来引入 target network 后可回调 λ=0.95

### 9.2 辅助损失

```
L_total = L_dqn
        + next_rank_weight × CE(rank_pred, rank_label)              [0.0, Phase 5 关闭]
        + opp_wait_weight  × BCE(wait_pred, wait_label)              [0.1]
        + ding_que_ce_weight × CE(auxnet_dq_logits, dq_label)       [1.0, AuxNet 主分类头]
        + ding_que_dqn_ce_weight × CE(dqn_q[31:34], dq_label)       [0.0, Phase 5 关闭]
        + min_q_weight × CQL_penalty                                  [0.0, 关闭]
```

### 9.3 奖励构成

```
每步奖励 = 分数差(Δscore) / 10000
         + action_bonus (和牌+0.1 / 放铳-0.1)
         + rank_bonus (仅最后一步, 按最终排名)
```

量级参考（Phase 2 配置）:
- 1 番和牌 (1,000 点) → reward ≈ 0.1
- 2 番和牌 (2,000 点) → reward ≈ 0.2
- 5 番封顶 (16,000 点) → reward ≈ 1.6
- agari_bonus: +0.1 / houjuu_penalty: -0.1
- rank_bonus: 1st=+0.3, 2nd=+0.1, 3rd=-0.1, 4th=-0.3

---

## 10. 重要约束与已知限制

### 10.1 不可更改

| 项 | 值 | 原因 |
|-----|-----|------|
| `enable_amp` | false | FP16 溢出，Q-target 范围 ±16000 (PERF-02) |
| `td_lambda` | 1.0 | 无 target network 时 λ<1 仅衰减 (BUG-06) |
| `version` | 4 | obs_shape=423, 唯一支持版本 |
| DDP | 不使用 | 瓶颈在 self-play CPU，非 GPU (PERF-05) |

### 10.2 SPCalculator 已知不足 (AUDIT-08)

- 杠上花/杠上炮 番型在 SP 期望值计算中始终为 false
- 影响: 极微（杠场景稀少，EV 差异 <1%）
- 状态: 已知 TODO，暂不修复

### 10.3 阶段切换注意事项

- 修改 config.toml 后只需重启 `train.py`，**不需要删除 checkpoint**
- Scheduler 自动适配新参数（TRAIN-04 设计）
- 但 **client.py 必须全部重启**（不会自动 reload baseline）
- Baseline 需手动更新: `cp mortal.pth baseline.pth`

---

## 11. 预计时间线

基于实际观测: ~0.5 秒/步，4090×8 (7 client + 1 trainer)

| 阶段 | Step 范围 | 预计耗时 | 累计 |
|------|-----------|----------|------|
| Phase 1 [已完成] | 0–70k | ~10h | ~10h |
| Phase 2 [已完成] | 70k–355k | ~40h | ~50h |
| Phase 3 [已完成] | 355k–485k | ~18h | ~68h |
| Phase 4 [已完成] | 485k–580k | ~13h | ~81h |
| Phase 5 [当前] | 580k–~800k | ~30h | ~111h (~5天) |
| Phase 6 (Target Network) | 800k–1.2M | ~55h | ~166h (~7天) |
| Phase 7 (超人类突破) | 1.2M–3M | ~250h | ~416h (~17天) |

**总计**: 约 2.5-3 周达到强业余水平 (1.2M 步)。超人类需要 Phase 7 架构改进，预计 4-5 周。

---

## 12. 超人类可行性深度分析 (680k 步)

> 基于 680k 步训练数据、奖励系统审计、番数系统分析、成功案例研究综合评估。
> 分析时间: 2026-02-15

### 12.1 结论概要

| 目标 | 可行性 | 需要的改动 |
|------|--------|-----------|
| 击败随机玩家 | **已达到** | — |
| 击败普通玩家 | 高概率 (Phase 5→6) | 继续训练 + Target Network |
| 击败强业余玩家 | 有条件 | + Target Network + 更长训练 |
| 超人类水平 | 当前架构极难 | + Oracle Guiding 或推理时搜索 |

### 12.2 有利因素

#### 血战到底比日麻简单 ~10 倍

| 维度 | 血战到底 | 日本麻将 | 对 AI 影响 |
|------|---------|---------|-----------|
| 牌数 | 108 (3花色×9×4) | 136 (含字牌) | 搜索空间小 |
| 吃牌 | 无 | 有 | 动作空间小 |
| 役种 | ~17种，叠加简单 | ~40+种，复杂互锁 | 手牌评估简单 |
| 宝牌/立直 | 无 | 有 | 隐藏信息少 |
| 番数上限 | 5番封顶 (16000) | 无限（役满 32000+） | 价值范围窄 |
| 结束条件 | 3人和牌 | 东风/半庄局数 | 无跨局策略 |

Suphx 在更复杂的日麻中实现超人类，理论上血战到底更容易达到。

#### 观测特征丰富 (423×27)

- SP 期望值表（100 通道）：预计算和牌概率、期望点数
- 向听数 one-hot（5 通道）
- 对手防守特征（9 通道）
- 对手听牌预测辅助任务（81 维 opp_wait）
- 杠选择、副露计数、禁手、FanConfig 等

特征工程质量高于大多数麻将 AI 论文，**不是瓶颈**。

#### 模型容量足够

9.1M 参数 (30-block ResNet 192ch + Dueling DQN) 对 108 张牌游戏足够。
SuitAwareConv1d 花色隔离是良好的领域先验。

#### 核心规则已学会 (680k 步)

- 定缺准确率 99.94%
- 放铳率从 45% 降至 32.5%（有实质学习）
- 基本碰/杠判断合理
- 和牌点数 ~5915（平均 ~2.2 番）

### 12.3 结构性瓶颈

#### 瓶颈 #1: 无推理时搜索（最大天花板）

所有超人类系统都有推理时搜索：

| 系统 | 推理时搜索 | 结果 |
|------|-----------|------|
| AlphaZero | MCTS 1600 次模拟/步 | 超人类围棋/象棋 |
| Suphx | 运行时策略自适应 (RTPA) | 超人类日麻 |
| Pluribus | 在线 MCCFRM | 超人类 6 人德州 |
| **本系统** | **单次前向传播 → argmax** | ? |

单次前向传播需要同时完成手牌评估、危险度判断、收益风险权衡。
搜索可以显式枚举可能结果并评估。没有搜索，模型天花板受限于泛化能力。

#### 瓶颈 #2: MC 回报信用分配差

TD(λ=1.0) ≡ Monte Carlo，reward 均匀回传到所有前序动作：

```
场景: 1 番荣和 (1000点), reward = 0.1, 一局 20 步, γ=0.99
第 1 步信号: 0.1 × 0.99^19 ≈ 0.083
每个动作有效梯度信号 ≈ 0.004
```

模型无法精确定位「是哪步出牌导致放铳」。这是防守学习慢（680k 步仍 32.5%）的根本原因。

**解决**: Target Network + TD(λ=0.95)，使 V(s') bootstrap 精确到单步信用分配。

#### 瓶颈 #3: DQN 无博弈论基础

DQN 把 4 人游戏当单智能体 MDP：`max_a Q(s,a)`，假设环境固定。
实际对手在变化，导致：
1. 策略循环（A 打败 B → C 打败 A → C 输给 B）
2. DQN 输出确定性 Q 值，Boltzmann 只是近似随机化
3. 确定性策略容易被观察模式后针对

Pluribus 用 CFR 解决，Suphx 用 RTPA 运行时适应。我们缺乏对应机制。

#### 瓶颈 #4: 训练规模不足

| 系统 | 计算规模 | 等效对局量 |
|------|---------|-----------|
| AlphaZero | 5000 TPU × 3天 | 数十亿局 |
| OpenAI Five | 256 GPU × 10月 | 每天 180 年等效 |
| Pluribus | 64核 CPU × 8天 | ~万亿次迭代 |
| **本系统** | 8× 4090 × ~12天 | **~4.35M 局** |

### 12.4 奖励系统深度审计

#### 当前奖励状态 (Phase 5, 580k+)

```python
# train.py L315: Q-target 的唯一输入
q_target = returns + ding_que_bonus

# 其中:
# returns = TD(λ=1.0) 回报 = Σ γ^t × (score_diff_t / 10000)  ← 纯分数差
# ding_que_bonus = 0  (ding_que_aux_enabled = false)
```

**当前是真正的「纯 score_diff」驱动**。所有奖励塑形和隐藏奖励均已关闭：

| 奖励通道 | 当前状态 | 代码位置 |
|---------|---------|---------|
| score_diff / 10000 | **唯一启用** | dataloader.py L232 |
| rank_bonus | 关闭 (Phase 2 后) | reward_calculator.py L43 |
| agari_bonus | 关闭 (Phase 4) | reward_calculator.py L62 |
| houjuu_penalty | 关闭 (Phase 4) | reward_calculator.py L62 |
| ding_que_bonus | 关闭 (Phase 5) | dataloader.py L252-260 |

#### 奖励演化全史与教训

| 阶段 | 奖励配置 | 效果 | 教训 |
|------|---------|------|------|
| Phase 1 (0-70k) | score_diff + agari=0.1 + houjuu=-0.1 | 学会基础出牌，放铳 33% | action_bonus 量级太小，被 score_diff 淹没 |
| Phase 2a (70-180k) | +rank_bonus [0.3,0.1,-0.1,-0.3] | 放铳反而升至 38-41% | rank_bonus 在最后一步才给，MC 回传后几乎无信号 |
| Phase 2b (180-249k) | 纯 score_diff | 和牌点数 +41%，放铳 45% | 学会追番，但彻底不防守（「赌博模式」） |
| Phase 2c (249-355k) | score_diff + houjuu=-0.2 | 放铳 45%→32% | **唯一有效的奖励塑形**：直接惩罚放铳行为 |
| Phase 3a (355-420k) | houjuu=-0.1 | 放铳维持 ~32% | 减半后防守保持 |
| Phase 3b (420-485k) | houjuu=-0.05 | 放铳 ~34% | 进一步减弱，略有退化 |
| Phase 4 (485-580k) | 纯 score_diff | 放铳 ~34%，agari_point 下降 | 防守内化但不再进步；低番速胡退化 |
| Phase 5 (580k+) | 纯 score_diff (去 ding_que_bonus) | 放铳 33%，缓慢改善中 | 梯度净化释放 DQN 学习能力 |

#### 奖励量级分析

以一局典型场景分析各奖励通道的信号强度：

```
场景: 玩家 A 做清一色自摸 (3番=4000点), 一局 20 步

(1) score_diff 奖励:
    自摸 4000点 → reward = 4000/10000 = 0.4
    被对手自摸 -4000 → reward = -0.4
    荣和 4000 → reward = 0.4 (赢家), -0.4 (放铳者)
    
(2) 历史 houjuu_penalty = -0.2:
    被放铳 → 额外 -0.2 (相当于额外 2000 点惩罚)
    占 score_diff 的 50% (对 1 番 1000 点) ~ 12.5% (对 3 番 4000 点)
    
(3) 历史 rank_bonus = 0.3 (1st):
    仅在游戏最后一步给予
    MC 回传 20 步后: 0.3 × 0.99^19 = 0.245
    但分散到 20 步: 每步 ~0.012
    
(4) 历史 agari_bonus = 0.1:
    和牌时额外 +0.1 (相当于额外 1000 点)
    对 1 番: 翻倍; 对 5 番: 仅 +6%
    扭曲了番数的自然比例关系
```

#### 纯 score_diff 的优势与根本局限

**优势（Phase 5 当前状态）:**

1. **零偏差**: 不引入人为偏好，模型学到的就是「最大化分数」
2. **自然激励高番**: 5番(16000) 的 reward 1.6 vs 1番(1000) 的 0.1 = 16:1
3. **攻防平衡内置**: 放铳丢分 = 负 reward，自然学防守
4. **番数比例保持**: 不像 agari_bonus 扭曲不同番数间的相对价值

**根本局限:**

1. **信号稀疏**: 一局 ~20 步出牌，只有和牌/放铳时产生 reward（~1-2 步）。其余 18 步 reward = 0
2. **信用分配失败**: MC 把这 1-2 步的 reward 均匀传播到 20 步，无法识别关键决策点
3. **防守信号被稀释**:
   ```
   放铳 1000 点 → reward = -0.1
   MC 回传 15 步 → 每步信号 = -0.1 × 0.99^14 / 15 ≈ -0.0006
   vs DQN loss 量级 ~0.2 → 梯度比 ~0.3%
   ```
   这就是 680k 步后放铳率仍在 33% 的数学原因。
4. **杠收益难学习**:
   - 杠的即时收益（明杠 2000、暗杠 2000×n）在 score_diff 中有体现
   - 但杠的间接成本（暴露信息、减少手牌灵活性）通过 MC 回传极难学到
5. **多次和牌博弈难学习**:
   - 血战到底 3 人和牌才结束，先和者可以继续得分
   - 「先快速和 1 番锁定安全」vs「追大牌博最终排名」的决策需要全局 EV 评估
   - MC 回报只看最终 score_diff，无法区分这两种策略的细微差异

#### 奖励塑形能否帮助达到超人类？

| 奖励塑形方案 | 可行性 | 问题 |
|-------------|--------|------|
| 重新启用 houjuu_penalty | 低价值 | 防守已内化(33%)；额外惩罚会引入偏差，阻碍攻防平衡 |
| 加入向听改善 reward | 危险 | 鼓励速胡，扭曲追番策略；引入大量人为偏好 |
| 加入牌效率 reward | 危险 | 难以精确定义「好的弃牌」；可能与最终得分不一致 |
| 加入对手放铳预测 reward | 中等 | 鼓励读牌，但 opp_wait 已在做类似事情 |
| **不加奖励塑形，改用 Target Network** | **最优** | **从根本解决信用分配，不引入偏差** |

**结论**: 奖励塑形已经走到尽头。从 Phase 1 到 Phase 5 的实验证明：

- 奖励塑形能加速早期学习（houjuu_penalty 教会了防守基础）
- 但最终必须关闭（引入偏差 → 局部最优 → 阻碍超越人类）
- **当前瓶颈不在「奖励不够」而在「信用分配不准」**
- Target Network 是正确解法：不改变奖励，而是改善「哪步贡献了多少」的归因精度

#### 奖励系统对超人类路线图的影响

```
当前: score_diff/10000 → MC 回报 → 信号稀释 → 学习慢 → 天花板: 中等业余

Phase 6 改进:
score_diff/10000 → TD(λ=0.95) + V(s') → 单步信用分配 → 学习快 → 天花板: 强业余

Phase 7 改进 (Oracle Guiding):
Oracle 策略蒸馏 → score_diff RL → 搜索 → 天花板: 超人类
```

关键洞察：**奖励函数本身（score_diff）已经是正确的**。问题出在回报计算方式（MC vs TD）和推理架构（无搜索）。

---

### 12.5 番数系统对学习的影响

```
点数 = 1000 × 2^(番数-1)，封顶 5 番 = 16000
```

| 策略 | 平均番 | 平均点数 | reward | 分析 |
|------|-------|---------|--------|------|
| 速胡 (1番平胡) | 1 | 1000 | 0.1 | 低风险低收益 |
| 追 2 番 (自摸/门清) | 2 | 2000 | 0.2 | 性价比最高 |
| 追 3 番 (七对/清一色) | 3 | 4000 | 0.4 | 风险加倍，收益翻倍 |
| 追 5 番封顶 | 5 | 16000 | 1.6 | 稀有但 reward 极强 |

指数增长天然激励追番（5番 reward 是 1番的 16 倍），但关键的**何时追番、何时速胡**决策需要精确信用分配——恰好是当前系统的弱项。

当前 agari_point ≈ 5915（~2.2 番），人类高手 ~2.5-3 番。差距不大，但需要更好的 EV 评估。

### 12.6 成功案例对比

#### Suphx（日麻超人类）的关键技术

1. **Oracle Guiding**: 完美信息教师模型监督预训练 — 我们**完全缺失**
2. **Global Reward Prediction**: 专门网络预测全局回报 — 我们用原始 MC
3. **RTPA**: 推理时根据实时信息调整策略 — 我们是静态策略

#### Mortal（本代码库原型）

1. **监督预训练**: 天凤人类高手对局数据做 supervised learning
2. **RL 强化**: 在人类水平起点上自博弈

我们是**从零开始**，没有人类数据。需要自己发现所有策略，包括基本常识。

#### Pluribus（6人德州超人类）

- MCCFRM 自博弈（无人类数据）
- 在线搜索（推理时做 CFR 迭代）
- 8 天，64 核 CPU，$144 成本

Pluribus 证明了无人类数据也能超人类，但它有在线搜索。

### 12.7 当前水平 vs 超人类差距

| 能力 | 当前 (680k) | 超人类要求 | 差距 |
|------|------------|-----------|------|
| 防守 (houjuu_rate) | 32.5% | < 18% | 需要减半 |
| 和牌质量 (agari_point) | 5915 | > 7500 | 需要 +27% |
| 综合 (point_per_round) | -344 | > +500 | 需要翻转 |
| 对手建模 | opp_wait 辅助 | 实时推断手牌 | 缺乏 |
| 推理深度 | 单次前向传播 | 搜索/自适应 | 完全缺失 |

### 12.8 改进路线图（优先级排序）

#### 优先级 1: Target Network（Phase 6, 中等难度, 高收益）

- **改动**: 添加 target brain/dqn 延迟副本，软更新 τ=0.005
- **效果**: TD(λ=0.95) 生效，信用分配精确到单步
- **预期**: 防守学习速度 3-5 倍提升，放铳率 32% → 25%
- **代码量**: ~100 行 (train.py + dataloader.py)
- **风险**: 低（成熟技术，DQN 标配）

#### 优先级 2: Oracle Guiding（Phase 7A, 高难度, 极高收益）

- **改动**: 训练完美信息 Oracle → 蒸馏到当前模型 → 继续 RL
- **效果**: 模型从「理论最优」起点出发，而非从随机策略
- **预期**: 跳过「学习基本常识」阶段，直接优化高级策略
- **代码量**: ~500 行 (新 oracle 训练脚本 + 蒸馏逻辑)
- **风险**: 中（完美信息→不完美信息的迁移可能有 gap）

#### 优先级 3: 推理时搜索（Phase 7B, 高难度, 高收益）

- **改动**: 推理时对每个合法动作 rollout N 局，取最佳
- **效果**: 突破单次前向传播天花板
- **预期**: 复杂决策质量大幅提升（追番 vs 防守 vs 速胡）
- **代码量**: ~300 行 (推理引擎改造)
- **风险**: 高（延迟增加，信息集采样复杂）

#### 优先级 4: Population-Based Training（Phase 7C, 极高难度）

- **改动**: 5-10 个独立 agent + 联赛系统
- **效果**: 消除策略循环，发现鲁棒策略
- **预期**: 解决 DQN 无博弈论基础问题
- **代码量**: ~1000 行 (新训练框架)
- **风险**: 高（计算资源需求翻倍，工程复杂）

### 12.9 最终判定

**当前架构 (纯 DQN 自博弈 + MC 回报 + 无搜索) 的天花板是「中等业余」水平。**

突破路径：
1. +Target Network → 「强业余」（可行，推荐立即规划）
2. +Oracle Guiding → 「准超人类」（需要开发投入，推荐中期）
3. +推理时搜索 → 「超人类」（需要推理架构重构，推荐长期）

血战到底的简单性是最大优势。在同样的技术栈下，血战到底比日麻更容易达到超人类。
关键是补上 Suphx 使用的 Oracle Guiding 和 RTPA 中的至少一个。

---

## 附录 A: 完整文件清单

| 文件 | 用途 |
|------|------|
| `mortal/config.toml` | 当前活跃配置（从 config.fresh_start.toml 复制） |
| `mortal/config.fresh_start.toml` | Phase 1 初始配置模板 |
| `mortal/train.py` | 训练主循环 |
| `mortal/server.py` | Online self-play 数据服务器 |
| `mortal/client.py` | Self-play 客户端 |
| `mortal/model.py` | Brain + DQN + AuxNet 模型定义 |
| `mortal/engine.py` | 推理引擎（Boltzmann + top-p 采样）|
| `mortal/dataloader.py` | 数据加载 + TD(λ) 回报计算 |
| `mortal/reward_calculator.py` | 奖励塑形计算 |
| `mortal/lr_scheduler.py` | Cosine Annealing + Warmup |
| `rules.md` | 血战到底完整规则 |
| `ISSUES.md` | 所有已修复/已知问题记录 |

## 附录 B: 常用运维命令

```bash
# 查看训练进度
tail -f /data/mortal/logs/events.out.*

# 查看 self-play 产出
ls -la /data/mortal/train_play/ | tail

# 更新 baseline（avg_pt 稳定达标后）
cp /data/mortal/mortal.pth /data/mortal/baseline.pth

# 备份 checkpoint
cp /data/mortal/mortal.pth /data/mortal/backup/mortal_step_XXXk.pth

# 1v3 评估（可选，对比 challenger vs baseline）
cp /data/mortal/mortal.pth /data/mortal/challenger.pth
python one_vs_three.py

# 清理旧数据
find /data/mortal/train_play -name "*.json.gz" -mtime +3 -delete
```

---

## 附录 C: Baseline 更新日志

| 时间 | Step | avg_ranking | avg_pt | dqn_match_rate | 备注 |
|------|------|------------|--------|---------------|------|
| 2025-02-15 | 20k | 1.618 | 3.057 | — (修复前) | Phase 1 首次更新；同步部署 MODEL-03b（DQN 弱 CE + 定缺指标拆分）|
| 2025-02-15 | 38k | 1.819 | 2.651 | 98.1% | Phase 1 第二次更新；旧 baseline 过弱，提前刷新 |
| 2025-02-15 | 55k | 2.264 | 2.010 | 98.5% | Phase 1 第三次更新；纯自博弈频繁刷新提升数据质量；暂不切 Phase 2 |

> 更新后指标回调至 avg_ranking=1.915, avg_pt=2.520（对更强 baseline 的正常表现）。
> dqn_match_rate 在修复后 2k 步内从 ~33% 飙升至 97.6%，验证修复有效。

**Phase 2 切换** (Step 70k):
- 配置变更: epsilon 0.30→0.15, temp 0.30→0.20, rank_bonus_enabled true, ding_que_ce_weight 2.0
- 目标: 放铳率从 33% 降至 <15%
- BL#3 (55k) 继续使用，不更新 baseline

| 2025-02-15 | 100k | 2.476 | 1.772 | 99.1% | Phase 2 首次 BL 更新 (BL#4)；30k 步训练后对 BL#3 开始拉开差距 |
| 2025-02-16 | 130k | 2.547 | 1.680 | 99.2% | BL#5; 对 BL#4 恢复缓慢 (30k步仅 avg_ranking -0.02)，放铳率 38%，强制更新 |
| 2025-02-16 | 180k | 2.551 | 1.685 | 99.4% | 关闭所有奖励塑形 (rank_bonus=false, action_bonus=false); 纯 score_diff 驱动 |
| 2025-02-16 | 236k | 2.590 | 1.655 | 99.5% | BL#6; 纯 score_diff 55k 步后更新；和牌点数 +41% 但放铳率恶化至 45% |
| 2025-02-16 | 240k | 2.531 | 1.719 | 99.5% | BL#6 后效果：放铳率 39.2%、4th=26.5%（可能是"虚假进步"） |

**V3 训练策略制定** (Step 240k):
- 诊断：单一 baseline 过拟合 + 纯 score_diff 无法教防守
- 案例研究：AlphaGo Zero、OpenAI Five、AlphaStar、Suphx、Mortal
- 决策：引入对手池（参考 OpenAI Five 80/20 策略）+ 定向 houjuu_penalty=-0.2
- 状态：249k 已实施并部署

**Phase 3 切换** (Step 355k):
- Phase 2 plateau 分析：对手池陈旧、探索过高、houjuu 过强、LR 偏高
- 配置变更: epsilon 0.15→0.08, temp 0.20→0.15, houjuu -0.2→-0.1, peak LR 5e-4→3e-4, final LR 1e-4→5e-5, warmup 2000→3000
- 355k 检查点加入对手池（池内: 130k, 200k, 249k, 295k, 355k）
- 目标: 放铳率 <25%, avg_ranking <2.30, point_per_round 稳定转正
