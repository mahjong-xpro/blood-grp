# 血战到底麻将 AI 训练计划

> 基于全部代码审计（ISSUES.md 77 文件）和历史训练经验制定。
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

## 4. 训练阶段

### Phase 1: 探索期 (Step 0 — 100k)

**目标**: 从随机策略学会基础出牌逻辑（和牌、定缺、避免花猪）

#### 配置 (已在 config.fresh_start.toml 中设置)

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

#### 监控指标与目标

| 指标 | 初始值 | 目标 | 观察方法 |
|------|--------|------|----------|
| `avg_rank` | ~2.50 (随机) | < 2.35 | TensorBoard |
| `avg_pt` | ~1.75 (随机) | > 2.0 | TensorBoard |
| `dqn_loss` | 高波动 | 稳步下降 | TensorBoard |
| `ding_que_match` | ~33% (随机) | > 70% | 日志 |
| 和牌率 | ~10% | > 18% | test_play 日志 |
| 放铳率 | ~25% | < 20% | test_play 日志 |

#### 启动命令

```bash
# 终端 1: Server
cd mortal && python server.py

# 终端 2-8: Self-play Clients (每张 GPU 一个)
CUDA_VISIBLE_DEVICES=1 python client.py  # 重复 7 次，GPU 1-7

# 终端 0: Trainer
python train.py
```

#### 里程碑与判断

- **Step ~5k**: dqn_loss 开始下降 → 正常
- **Step ~20k**: avg_rank 降到 ~2.45 → 模型开始学习
- **Step ~50k**: 定缺准确率 > 60% → 定缺辅助生效
- **Step ~100k**: avg_rank < 2.35, avg_pt > 2.0 → **进入 Phase 2**

⚠️ **异常信号**: dqn_loss 持续上升或 avg_rank 不降 → 检查 self-play 是否正常产出数据

---

### Phase 2: 收敛期 (Step 100k — 300k)

**目标**: 策略初步收敛，学会基础防守与进攻平衡

#### 配置修改 (修改 config.toml 后重启 train.py)

```toml
# 降低探索
boltzmann_epsilon = 0.15
boltzmann_temp = 0.2

# ⚠️ 关键变更：开启排名奖励（Phase 1 为 false，此处必须改为 true）
[reward_shaping]
rank_bonus_enabled = true
rank_bonuses = [0.3, 0.1, -0.1, -0.3]
# action_bonus 保持不变 (agari=0.1, houjuu=-0.1)

# 提高定缺监督
ding_que_ce_weight = 2.0

# Scheduler 自动重启（TRAIN-04: 无需改 scheduler 参数，offset 机制保证连续性）
```

#### 监控目标

| 指标 | 入口 | 目标 |
|------|------|------|
| `avg_rank` | < 2.35 | < 2.20 |
| `avg_pt` | > 2.0 | > 2.3 |
| `ding_que_match` | > 70% | > 85% |
| 和牌率 | > 18% | > 22% |
| 放铳率 | < 20% | < 16% |

#### Baseline 更新

```bash
# 当 avg_pt 稳定 > 2.5 时手动更新
cp /data/mortal/mortal.pth /data/mortal/baseline.pth
# 重启所有 client.py（client 不会自动 reload baseline）
```

---

### Phase 3: 精调期 (Step 300k — 600k)

**目标**: 策略精细化，学会番数价值评估与大牌追求

#### 配置修改

```toml
# 低探索，精细调优
boltzmann_epsilon = 0.05
boltzmann_temp = 0.1

# 排名奖励保持开启（Phase 2 已开启，此处不变）
[reward_shaping]
rank_bonus_enabled = true
rank_bonuses = [0.3, 0.1, -0.1, -0.3]

# 微调动作奖励（减小剂量，让分数差主导）
action_bonus_enabled = true
agari_bonus = 0.05
houjuu_penalty = -0.05

# 可选: 降低 gamma 缩短视野（聚焦当前局）
[env]
gamma = 0.98

# Scheduler: 降低 peak LR
[optim.scheduler]
peak = 3e-4
final = 5e-5
warm_up_steps = 3000
max_steps = 1000000
```

#### 监控目标

| 指标 | 入口 | 目标 |
|------|------|------|
| `avg_rank` | < 2.20 | < 2.10 |
| `avg_pt` | > 2.3 | > 2.5 |
| `ding_que_match` | > 85% | > 90% |
| 自摸率 | | > 12% |
| 平均和牌番数 | | > 1.8 |

---

### Phase 4: 深度精调期 (Step 600k — 1M)

**目标**: 策略接近最优，精细化对手建模与防守策略

#### 配置修改

```toml
boltzmann_epsilon = 0.08   # 略微回升探索，避免局部最优
boltzmann_temp = 0.15

# 极轻量奖励
[reward_shaping]
rank_bonus_enabled = true
rank_bonuses = [0.0, 0.0, 0.0, -0.15]  # 仅惩罚 4th

action_bonus_enabled = true
agari_bonus = 0.05
houjuu_penalty = -0.05

# 更频繁测试
test_every = 3000

# LR 重启
[optim.scheduler]
peak = 2e-4
final = 1e-5
warm_up_steps = 5000
max_steps = 1350000
```

#### 监控目标

| 指标 | 入口 | 目标 |
|------|------|------|
| `avg_rank` | < 2.10 | < 2.05 |
| `avg_pt` | > 2.5 | > 2.7 |

---

### Phase 5 (可选): 多规则训练 (Step 1M+)

**目标**: 验证 FanConfig 条件化策略泛化能力

#### 前提

- Phase 4 完成，标准规则下 avg_rank < 2.10
- FANCFG-02 短训验证通过

#### 配置修改

```toml
[rules]
randomize_fan_config = true  # 开启多规则随机
```

#### 验证清单

1. 开启后 10k 步内无报错
2. Loss 曲线无异常跳变
3. 标准规则 avg_rank 无明显退化（允许 +0.05）
4. 不同 FanConfig 下决策有可观察差异

---

## 5. 阶段转换决策树

```
Step 0 ──────────── Phase 1 ────────────┐
                                         │
avg_rank < 2.35 && avg_pt > 2.0 ────────▼
                                  Phase 2
                                         │
avg_rank < 2.20 && avg_pt > 2.3 ────────▼
                                  Phase 3
                                         │
avg_rank < 2.10 && avg_pt > 2.5 ────────▼
                                  Phase 4
                                         │
avg_rank < 2.05 && 验证 FANCFG-02 ─────▼
                                  Phase 5 (可选)
```

**阶段转换操作**:
1. 修改 `mortal/config.toml` 中对应参数
2. 手动更新 baseline: `cp mortal.pth baseline.pth`
3. 重启 `train.py`（scheduler 自动适配新参数）
4. 重启所有 `client.py`（加载新 baseline）

---

## 6. 关键配置速查表

### 6.1 探索参数进度表

| 阶段 | Step 范围 | epsilon | temp | 说明 |
|------|-----------|---------|------|------|
| Phase 1 | 0–100k | 0.30 | 0.30 | 大范围探索 |
| Phase 2 | 100k–300k | 0.15 | 0.20 | 收敛 |
| Phase 3 | 300k–600k | 0.05 | 0.10 | 精调 |
| Phase 4 | 600k–1M | 0.08 | 0.15 | 微回升避免局部最优 |

### 6.2 学习率进度表

| 阶段 | peak | final | warmup | max_steps |
|------|------|-------|--------|-----------|
| Phase 1 | 5e-4 | 1e-4 | 2000 | 1000000 |
| Phase 3+ | 3e-4 | 5e-5 | 3000 | 1000000 |
| Phase 4 | 2e-4 | 1e-5 | 5000 | 1350000 |

### 6.3 奖励塑形进度表

| 阶段 | rank_bonus_enabled | rank_bonuses | agari | houjuu | 说明 |
|------|-------------------|--------------|-------|--------|------|
| Phase 1 | **false** | — | 0.1 | -0.1 | 纯分数差 + 动作信号 |
| Phase 2 | **true** ⬆️ | [0.3, 0.1, -0.1, -0.3] | 0.1 | -0.1 | 开启全排名奖励 |
| Phase 3 | true | [0.3, 0.1, -0.1, -0.3] | 0.05 | -0.05 | 保持排名奖励，减弱动作奖励 |
| Phase 4 | true | [0, 0, 0, -0.15] | 0.05 | -0.05 | 仅惩罚 4th |

---

## 7. 监控与故障排除

### 7.1 TensorBoard 监控

```bash
tensorboard --logdir /data/mortal/logs --bind_all
```

关键面板:
- `dqn_loss`: 应持续下降后趋于稳定
- `avg_rank`: 应持续下降（越低越好，理论最优 1.0）
- `avg_pt`: 应持续上升（理论最优 4.0）
- `next_rank_loss`: 辅助损失，应下降
- `opp_wait_loss`: 对手听牌预测损失
- `ding_que_ce_loss`: 定缺分类损失
- `lr`: 学习率曲线，验证 scheduler 行为

### 7.2 常见故障

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| dqn_loss 爆炸 (NaN) | LR 过高 / AMP 溢出 | 确认 `enable_amp = false`; 降低 peak LR |
| avg_rank 不降 | self-play 数据不足 | 检查 client 是否运行; 检查 `/data/mortal/buffer` |
| avg_rank 突然跳升 | baseline 与模型差距过大 | 更新 baseline: `cp mortal.pth baseline.pth` |
| DataLoader 僵死 | BUG-01 (PyO3/rayon) | 自动重启机制已内置; 若持续则手动重启 train.py |
| 内存溢出 (OOM) | batch_size 过大 | 降低 batch_size: 4096→2048→1024 |
| ding_que 准确率卡在 ~33% | 监督信号不足 | 提高 ding_que_ce_weight 至 2-3 |
| 和牌率低 | 探索不足 / 奖励信号弱 | 提高 epsilon/temp; 增大 agari_bonus |
| 训练速度慢 | CPU 瓶颈 | 增加 client 数; 提高 num_workers |

### 7.3 磁盘空间管理

```bash
# 每 ~50k 步清理旧 train_play 数据（保留最近 20k 步即可）
find /data/mortal/train_play -name "*.json.gz" -mtime +3 -delete

# 检查磁盘占用
du -sh /data/mortal/*
```

---

## 8. 数据流与损失函数

### 8.1 主损失: DQN (Bellman)

```
L_dqn = MSE(Q(s,a), r + γ·max_a' Q(s',a'))
```

- 当前使用 TD(λ=1.0) ≡ Monte Carlo: `target = Σ γ^t · r_t`
- 未来引入 target network 后可回调 λ=0.95

### 8.2 辅助损失

```
L_total = L_dqn
        + next_rank_weight × CE(rank_pred, rank_label)        [0.25]
        + opp_wait_weight  × BCE(wait_pred, wait_label)        [0.1]
        + ding_que_ce_weight × CE(dq_pred, dq_label)           [1.0]
        + ding_que_aux_scale × CE(dq_aux_pred, dq_aux_label)   [0.02]
        + min_q_weight × CQL_penalty                            [0.0, 关闭]
```

### 8.3 奖励构成

```
每步奖励 = 分数差(Δscore) / 240000
         + action_bonus (和牌+0.1 / 放铳-0.1)
         + rank_bonus (仅最后一步, 按最终排名)
```

---

## 9. 重要约束与已知限制

### 9.1 不可更改

| 项 | 值 | 原因 |
|-----|-----|------|
| `enable_amp` | false | FP16 溢出，Q-target 范围 ±16000 (PERF-02) |
| `td_lambda` | 1.0 | 无 target network 时 λ<1 仅衰减 (BUG-06) |
| `version` | 4 | obs_shape=423, 唯一支持版本 |
| DDP | 不使用 | 瓶颈在 self-play CPU，非 GPU (PERF-05) |

### 9.2 SPCalculator 已知不足 (AUDIT-08)

- 杠上花/杠上炮 番型在 SP 期望值计算中始终为 false
- 影响: 极微（杠场景稀少，EV 差异 <1%）
- 状态: 已知 TODO，暂不修复

### 9.3 阶段切换注意事项

- 修改 config.toml 后只需重启 `train.py`，**不需要删除 checkpoint**
- Scheduler 自动适配新参数（TRAIN-04 设计）
- 但 **client.py 必须全部重启**（不会自动 reload baseline）
- Baseline 需手动更新: `cp mortal.pth baseline.pth`

---

## 10. 预计时间线

基于 4090×8 服务器（7 client + 1 trainer）估算:

| 阶段 | Step 范围 | 预计耗时 | 累计 |
|------|-----------|----------|------|
| Phase 1 | 0–100k | ~3-5 天 | 3-5 天 |
| Phase 2 | 100k–300k | ~5-7 天 | 8-12 天 |
| Phase 3 | 300k–600k | ~7-10 天 | 15-22 天 |
| Phase 4 | 600k–1M | ~10-14 天 | 25-36 天 |
| Phase 5 | 1M+ | 开放 | — |

**总计**: 约 1-1.5 个月达到接近最优策略。

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
