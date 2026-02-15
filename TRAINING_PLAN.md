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

### Phase 2: 防守与收敛 (Step 70k — ~200k) [当前阶段]

**目标**: 学会防守，放铳率从 33% 降至 15% 以下

#### 配置修改 (修改 config.toml 后重启 train.py + 所有 client.py)

```toml
# 降低探索，减少噪声数据
boltzmann_epsilon = 0.15    # 0.30 → 0.15
boltzmann_temp = 0.20       # 0.30 → 0.20

# ⚠️ 关键变更：开启排名奖励（Phase 1 为 false，此处必须改为 true）
[reward_shaping]
rank_bonus_enabled = true   # false → true (防守学习的关键信号)
rank_bonuses = [0.3, 0.1, -0.1, -0.3]
# action_bonus 保持不变 (agari=0.1, houjuu=-0.1)

# 提高定缺监督
[aux]
ding_que_ce_weight = 2.0    # 1.0 → 2.0
```

#### 监控目标

| 指标 | 入口值 | 目标 | 重要性 |
|------|--------|------|--------|
| 放铳率 | ~33% | < 15% | ⭐ 最关键 |
| `avg_ranking` | — | < 2.35 (对当前 BL) | 高 |
| `avg_pt` | — | > 2.0 (对当前 BL) | 高 |
| `ding_que/dqn_match_rate` | 98.5% | 保持 >95% | 低 |
| 和牌率 | — | > 20% | 中 |

#### Baseline 更新策略

**每 30k 步强制更新**（不等指标达标）。纯自博弈中数据质量比指标稳定性更重要。

```bash
# 每 30k 步执行
cp /data/mortal/mortal.pth /data/mortal/baseline.pth
# 重启所有 client.py（client 不会自动 reload baseline）
```

---

### Phase 3: 进攻/防守平衡 (Step ~200k — ~500k)

**目标**: 在维持低放铳率的前提下提升和牌质量

#### 配置修改

```toml
# 进一步降低探索
boltzmann_epsilon = 0.08    # 0.15 → 0.08
boltzmann_temp = 0.15       # 0.20 → 0.15

# 减弱动作奖励，让分数差主导
[reward_shaping]
rank_bonus_enabled = true
rank_bonuses = [0.3, 0.1, -0.1, -0.3]
agari_bonus = 0.05          # 0.1 → 0.05
houjuu_penalty = -0.05      # -0.1 → -0.05

# 降低学习率
[optim.scheduler]
peak = 3e-4                 # 5e-4 → 3e-4
final = 5e-5                # 1e-4 → 5e-5
```

#### 监控目标

| 指标 | 入口 | 目标 |
|------|------|------|
| 放铳率 | < 15% | < 12% |
| 和牌点数 | ~5,500 | > 8,000 |
| `avg_ranking` | — | < 2.20 |

#### Baseline 更新策略

每 40-50k 步或当 avg_ranking < 2.30 时更新。

---

### Phase 4: 精细优化 (Step ~500k — ~1M)

**目标**: 策略接近最优，最大化胜率

#### 配置修改

```toml
boltzmann_epsilon = 0.05    # 0.08 → 0.05
boltzmann_temp = 0.10       # 0.15 → 0.10

# 弱化正奖励，保留 4th 惩罚
[reward_shaping]
rank_bonus_enabled = true
rank_bonuses = [0.1, 0.0, 0.0, -0.15]  # 弱化正奖励，保留4th惩罚
agari_bonus = 0.02          # 0.05 → 0.02
houjuu_penalty = -0.02      # -0.05 → -0.02

[optim.scheduler]
peak = 2e-4                 # 3e-4 → 2e-4
final = 1e-5                # 5e-5 → 1e-5
```

#### 监控目标

| 指标 | 入口 | 目标 |
|------|------|------|
| `avg_ranking` | < 2.20 | < 2.10 |
| `avg_pt` | — | > 2.5 |
| 放铳率 | < 12% | < 10% |

#### Baseline 更新策略

每 50-80k 步，当 avg_ranking 稳定 < 2.25 时更新。

---

### Phase 5: 超人类 (Step 1M+)

**目标**: 移除几乎所有奖励塑形，让纯分数差驱动策略

#### 配置修改

```toml
boltzmann_epsilon = 0.03    # 0.05 → 0.03
boltzmann_temp = 0.08       # 0.10 → 0.08

# 关闭所有辅助奖励，纯 score_diff / 10000 驱动
[reward_shaping]
rank_bonus_enabled = false
action_bonus_enabled = false

[optim.scheduler]
peak = 1e-4                 # 2e-4 → 1e-4
final = 1e-5
```

#### 可选增强

- `randomize_fan_config = true`（多规则泛化）
- 引入 target network + 降低 td_lambda（减少 MC 方差）
- Population-based training（多 baseline 池）

#### Baseline 更新策略

每 100k 步，当 avg_ranking < 2.20 时更新。

---

## 5. 阶段转换决策树

```
Step 0 ──────────── Phase 1 [已完成] ───┐
                                         │  70k steps, 防守弱(33.8%)
                                         ▼
                    Phase 2 [当前] ──────┐
                    防守与收敛             │
                    放铳率 < 15%? ────────▼
                                  Phase 3
                    进攻/防守平衡          │
                    放铳率 < 12%           │
                    和牌点数 > 8000? ─────▼
                                  Phase 4
                    精细优化               │
                    avg_ranking < 2.10? ──▼
                                  Phase 5
                    超人类 (纯分数差驱动)
```

**阶段转换操作**:
1. 修改 `mortal/config.toml` 中对应参数
2. 手动更新 baseline: `cp mortal.pth baseline.pth`
3. 重启 `train.py`（scheduler 自动适配新参数）
4. 重启所有 `client.py`（加载新 baseline）

**自博弈 Baseline 更新规则**:

纯自博弈的 baseline 更新是最关键的超参数之一：
- **过于频繁**: 模型持续处于追赶状态，无法积累有效策略
- **过于稀疏**: 数据质量低，学习效率差

| 阶段 | 更新间隔 | 条件 |
|------|---------|------|
| Phase 2 | 30k 步 | 强制更新（数据质量优先） |
| Phase 3 | 40-50k 步 | avg_ranking < 2.30 或到达间隔上限 |
| Phase 4 | 50-80k 步 | avg_ranking < 2.25 |
| Phase 5 | 100k 步 | avg_ranking < 2.20 |

---

## 6. 关键配置速查表

### 6.1 全阶段配置对比

| 参数 | Phase 1 (完成) | Phase 2 (当前) | Phase 3 | Phase 4 | Phase 5 |
|------|---------------|---------------|---------|---------|---------|
| epsilon | 0.30 | **0.15** | 0.08 | 0.05 | 0.03 |
| temp | 0.30 | **0.20** | 0.15 | 0.10 | 0.08 |
| rank_bonus | false | **true** | true | true | false |
| rank_bonuses | — | [.3,.1,-.1,-.3] | [.3,.1,-.1,-.3] | [.1,0,0,-.15] | — |
| agari_bonus | 0.1 | 0.1 | 0.05 | 0.02 | 0 |
| houjuu_penalty | -0.1 | -0.1 | -0.05 | -0.02 | 0 |
| peak LR | 5e-4 | 5e-4 | 3e-4 | 2e-4 | 1e-4 |
| ding_que_ce | 1.0 | **2.0** | 2.0 | 1.0 | 1.0 |
| BL更新间隔 | ~15-20k | 30k | 40-50k | 50-80k | 100k |

### 6.2 学习率进度表

| 阶段 | peak | final | warmup | max_steps |
|------|------|-------|--------|-----------|
| Phase 1-2 | 5e-4 | 1e-4 | 2000 | 1000000 |
| Phase 3 | 3e-4 | 5e-5 | 3000 | 1000000 |
| Phase 4 | 2e-4 | 1e-5 | 5000 | 1350000 |
| Phase 5 | 1e-4 | 1e-5 | 5000 | 2000000 |

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
- `ding_que_ce_loss`: AuxNet 定缺分类损失（应快速下降）
- `ding_que_dqn_ce_loss`: DQN 定缺弱 CE 损失（应逐步下降）
- `ding_que/aux_match_rate`: AuxNet 定缺分类准确率（应快速上升至 >90%）
- `ding_que/dqn_match_rate`: DQN 实际定缺决策准确率（应逐步上升至 >70%）
- `lr`: 学习率曲线，验证 scheduler 行为

### 7.2 常见故障

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
        + next_rank_weight × CE(rank_pred, rank_label)              [0.25]
        + opp_wait_weight  × BCE(wait_pred, wait_label)              [0.1]
        + ding_que_ce_weight × CE(auxnet_dq_logits, dq_label)       [1.0, AuxNet 主分类头]
        + ding_que_dqn_ce_weight × CE(dqn_q[31:34], dq_label)       [0.1, DQN 弱 CE]
        + min_q_weight × CQL_penalty                                  [0.0, 关闭]
```

### 8.3 奖励构成

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

基于实际观测: ~0.5 秒/步，4090×8 (7 client + 1 trainer)

| 阶段 | Step 范围 | 预计耗时 | 累计 |
|------|-----------|----------|------|
| Phase 1 [已完成] | 0–70k | ~10h | ~10h |
| Phase 2 | 70k–200k | ~18h | ~28h |
| Phase 3 | 200k–500k | ~42h | ~70h (~3天) |
| Phase 4 | 500k–1M | ~70h | ~140h (~6天) |
| Phase 5 | 1M–2M | ~140h | ~280h (~12天) |

**总计**: 约 2 周达到接近超人类水平 (2M 步)。如果性能瓶颈在某阶段出现，可能需要更长。

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
