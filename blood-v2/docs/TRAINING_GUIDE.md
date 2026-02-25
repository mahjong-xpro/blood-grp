# Blood-V2 训练步骤指南

> 基于 DEEP_REVIEW_2026_02_24 评审后的架构重构，需从头训练。
> 目标环境：Ubuntu 22.04 · CUDA 12.1 · RTX 4090

---

## 前置条件

训练前必须完成以下准备工作：

```bash
# 1. 激活 conda 环境
conda activate blood

# 2. 进入项目目录
cd ~/Mahjong/blood/blood-v2

# 3. 重新编译 Rust 引擎（架构变更后必须）
./scripts/manage.sh build

# 4. 安装 Python 包（editable 模式）
pip install -e python/

# 5. 验证引擎和模块导入
export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"
python -c "
from blood._engine import RustMahjongEnv
from blood.training.runner import register_blood_components
register_blood_components()
print('All imports OK')
"

# 6. 验证 GPU
python -c "import torch; print('GPU:', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

> ⚠️ 由于 TileAttention 4层 + LSTM 2层 架构变更，旧 checkpoint 不兼容，必须从头训练。

---

## 五阶段训练流水线

总计约 57.5M 环境步数，分 5 个阶段递进训练。

### 一键运行（推荐）

```bash
tmux new -s blood_train
./scripts/manage.sh train pipeline
```

pipeline 模式会自动按顺序执行全部 5 个阶段，阶段间自动传递 checkpoint。

### 分阶段手动运行

如需逐阶段控制，按以下顺序执行。

---

### 阶段 1：Warmup（基础策略学习）

| 参数 | 值 |
|------|-----|
| 步数 | 2M |
| 对手 | RuleBot |
| LSTM | ❌ 禁用 |
| 学习率 | 5e-4 |
| 熵系数 | 0.01 |
| 目标 | 学会定缺、基本出牌、简单胡牌模式 |

```bash
./scripts/manage.sh train warmup
```

关键配置（`configs/warmup.yaml`）：
- `warmup_reward_shaping: true` — 启用定缺奖励、胡牌奖励、危险牌惩罚
- `oracle_distill_weight: 0.1` — Oracle 策略蒸馏从第一步开始
- `aux_shanten_weight: 2.0` — 高权重辅助任务加速向听学习
- LSTM 架构已定义（`rnn_num_layers: 2`）但 `use_rnn: false`，确保 checkpoint 结构兼容

监控指标：
- `reward` 应稳步上升
- `aux_loss` 应持续下降
- 预期 warmup 结束时 reward ≈ 5~8

---

### 阶段 1.5：Warmup Transition（LSTM 稳定化）

| 参数 | 值 |
|------|-----|
| 步数 | 500K |
| 对手 | RuleBot（保持不变） |
| LSTM | ✅ 启用（2层 512-dim） |
| 学习率 | 2e-4（中间值） |
| gamma | 0.995（中间值） |
| 目标 | LSTM 在稳定环境中学会时序建模 |

```bash
./scripts/manage.sh train warmup_transition
```

过渡策略：
- 只改变 2 个变量（LSTM 启用 + gamma/lr 渐变），避免同时变更 6 个超参数导致崩溃
- 对手保持 RuleBot，让 LSTM 先在稳定环境中适应
- 辅助任务权重渐降：`aux_shanten_weight: 1.5`（从 2.0 降低）

监控指标：
- reward 短暂下降后应快速恢复（LSTM 初始化冲击）
- 若 reward 持续下降超过 200K 步，检查 LSTM 梯度是否爆炸

---

### 阶段 2a：Competitive（自博弈 + Oracle 价值头训练）

| 参数 | 值 |
|------|-----|
| 步数 | 1M |
| 对手 | 自博弈（联赛） |
| LSTM | ✅ |
| 学习率 | 1e-4 |
| 熵系数 | 0.01→0.05（线性预热） |
| 目标 | Oracle 价值头收敛（loss < 0.1） |

```bash
./scripts/manage.sh train competitive
```

关键变化：
- 对手从 RuleBot 切换为自博弈联赛（`opponent_mode: selfplay`）
- `warmup_reward_shaping: false` — 移除辅助奖励塑形
- 启用排名奖励（`reward_rank_bonus: 0.15`）和安全弃牌奖励（`reward_safe_discard: 0.015`）
- 启用向听番数加权（`shanten_fan_bonus_scale: 0.15`）
- `oracle_value_distill_weight: 0.0` — 价值蒸馏暂不启用，先让 Oracle 价值头自行收敛
- `oracle_value_head_loss_weight: 1.0` — 监督 MSE 损失训练 Oracle 价值头
- 联赛每 50K 步添加快照，20% 概率自对弈

监控指标：
- `blood/oracle_value_head_loss` 应降至 < 0.1（进入下一阶段的前提）
- `reward` 可能因对手变强而波动，属正常现象
- 熵系数从 0.01 线性预热到 0.05，防止策略过早收敛

---

### 阶段 2b：Competitive Distill（Oracle 价值蒸馏）

| 参数 | 值 |
|------|-----|
| 步数 | 4M |
| 对手 | 自博弈（联赛） |
| LSTM | ✅ |
| 学习率 | 1e-4 |
| 熵系数 | 0.05 |
| 目标 | 学生 Critic 从 Oracle 完美信息价值估计中学习 |

```bash
./scripts/manage.sh train competitive_distill
```

关键变化：
- `oracle_value_distill_weight: 0.1` — 启用价值蒸馏（核心变化）
- `oracle_value_head_loss_weight: 0.5` — 降低（Oracle 价值头已收敛）
- 其余参数与 competitive 保持一致

监控指标：
- `distill_loss` 应持续下降
- `reward` 应稳步提升（Critic 质量改善 → 更好的优势估计 → 更好的策略）
- `blood/oracle_value_head_loss` 应保持在低位

---

### 阶段 3：Elite（精英训练）

| 参数 | 值 |
|------|-----|
| 步数 | 50M |
| 对手 | 自博弈（联赛） |
| LSTM | ✅ |
| 学习率 | 1e-4（KL 自适应） |
| 熵系数 | 0.02→0.005（余弦退火） |
| 目标 | 最大化 Elo，冲击超人类水平 |

```bash
./scripts/manage.sh train elite
```

关键特性：
- 训练规模大幅提升（50M 步，占总训练量 87%）
- 启用 RTPA（运行时策略适应）和 ISMCE（信息集蒙特卡洛评估）
- 超参数动态调度：
  - 熵系数：余弦退火 0.02→0.005（40M 步内）
  - 优势裁剪：线性收紧 5.0→3.0（10M~40M 步）
- 向听奖励衰减：前 30M 步从 100% 线性衰减到 30%
- 排名奖励提升至 0.2（最终目标是排名）
- 联赛每 25K 步添加快照（更细粒度的对手多样性）

监控指标：
- `blood/elo_current` — 当前模型 Elo（核心指标）
- `blood/elo_best` — 历史最佳 Elo
- `blood/elo_pool_mean` — 联赛池平均 Elo
- `ppo_policy_loss` 应稳定下降
- `reward` 在自博弈中可能趋于零和，关注 Elo 而非 reward 绝对值

---

## 监控与调试

### 启动 TensorBoard

```bash
# 另开 tmux 窗口
tmux new -s blood_monitor
./scripts/manage.sh monitor
# 浏览器访问: http://<服务器IP>:6006
```

### 关键 TensorBoard 指标一览

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `reward` | 每步平均奖励 | 逐步上升 |
| `ppo_policy_loss` | PPO 策略损失 | 稳定下降 |
| `extra_loss_total` | 辅助损失总和 | 逐步下降 |
| `aux_loss` | 向听/听牌辅助任务 | 逐步下降 |
| `distill_loss` | Oracle 蒸馏损失 | 逐步下降 |
| `blood/oracle_value_head_loss` | Oracle 价值头损失 | < 0.1 后进入 distill |
| `blood/elo_current` | 当前 Elo | 持续上升 |
| `exploration_loss` | 策略熵 | 不应归零 |

### GPU 监控

```bash
watch -n 1 nvidia-smi
htop
```

---

## 显存不足 (OOM) 处理

如果训练时出现 `CUDA out of memory` 错误：

```bash
# 方法 1：设置 PYTORCH_ALLOC_CONF 减少碎片（manage.sh 已自动设置）
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 方法 2：进一步减小 batch_size（修改对应阶段的 yaml）
# warmup/transition: batch_size 2048 → 1024
# competitive/distill/elite: batch_size 1024 → 512

# 方法 3：减少 num_envs_per_worker（当前已降至 16）
# 可进一步降至 8，但会降低数据吞吐量

# 方法 4：确认无其他进程占用 GPU
nvidia-smi
# 如有其他进程，kill 掉或指定空闲 GPU：
# CUDA_VISIBLE_DEVICES=1 ./scripts/manage.sh train <phase>
```

当前默认配置已针对单张 24GB GPU（RTX 4090）优化：
- warmup / warmup_transition: `batch_size=2048`, `num_envs_per_worker=16`
- competitive / competitive_distill / elite: `batch_size=1024`, `num_envs_per_worker=16`

---

## 训练中断与恢复

SF2 自动保存 checkpoint，中断后直接恢复：

```bash
# 恢复当前阶段（自动加载最新 checkpoint）
./scripts/manage.sh train <phase> --resume

# 示例：恢复 elite 训练
./scripts/manage.sh train elite --resume
```

---

## 查看训练状态

```bash
./scripts/manage.sh status
```

输出各阶段 checkpoint 数量、联赛池大小、TensorBoard 日志等信息。

---

## 训练完成后

### 录制回放验证

```bash
# 录制 50 局回放
./scripts/manage.sh record --games 50

# 启动回放查看器
./scripts/manage.sh replay
# 浏览器访问: http://<服务器IP>:5001
```

### 导出 ONNX 模型

```bash
./scripts/manage.sh export \
    --checkpoint checkpoints/blood_v2_elite/checkpoint_best.pth \
    --quantize
```

### 运行评估

```bash
./scripts/manage.sh eval
```

---

## 阶段总览

```
warmup (2M)  →  warmup_transition (500K)  →  competitive (1M)  →  competitive_distill (4M)  →  elite (50M)
  RuleBot         RuleBot+LSTM               自博弈+Oracle头       Oracle价值蒸馏              RTPA+ISMCE精调
  基础策略         LSTM稳定化                  价值头收敛             Critic蒸馏                  冲击超人类
```

跨阶段一致参数（不可修改）：
- `blood_num_tile_attn_layers: 4`
- `blood_tile_attn_heads: 4`
- `rnn_num_layers: 2`
- `rnn_size: 512`
- `blood_obs_channels: 470`
- `blood_conv_channels: 256`
- `blood_num_res_blocks: 20`
