# Blood-V2 训练流程指南

## 总览

```
Phase 1 ──→ Phase 1.5 ──→ Phase 2a ──→ Phase 2b ──→ Phase 3
warmup      transition     competitive   distill      elite
2M steps    500K steps     1M steps      4M steps     200M steps
RuleBot     RuleBot        Self-play     Self-play    Self-play
LSTM ON     gamma/LR ramp  + League      + Oracle KD  + OpponentPredictor
                                         + OppPredictor  + TurnAttention
                                                      + ISMCE(推理)
─────────────────────────────────────────────────────────────
总计: ~207.5M env steps
```

---

## 前置准备

```bash
cd /Users/twosson/Mahjong/blood/blood-v2

# 1. 编译 Rust 引擎
./scripts/manage.sh build

# 2. 确认环境
source .venv/bin/activate
python -c "import blood._engine; print('Rust engine OK')"
python -c "from blood.model.factory import register_blood_model; print('Model OK')"

# 3. 启动 TensorBoard (另开终端)
./scripts/manage.sh monitor
```

---

## Phase 1: Warmup (2M steps)

**目标:** 从零学习基本打牌规则，建立 LSTM 时序记忆

| 参数 | 值 |
|------|-----|
| 对手 | RuleBot (Rust 规则引擎) |
| LSTM | 2 层 × 512 |
| 向听奖励 | 0.01 (强引导) |
| 熵系数 | 0.05 |
| 编码器 | SpatialPoolingProj (3层) |

```bash
./scripts/manage.sh train warmup
```

### 监控指标

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `policy_loss` | 下降趋势 | PPO 策略损失 |
| `entropy` | 0.8 → 0.5 | 自然过渡，不应骤降 |
| `value_loss` | 下降趋势 | 价值函数收敛 |
| `shanten_progress` | > 0 | 模型学会减少向听 |

### 过渡标准
- ✅ `policy_loss` 稳定
- ✅ `value_loss < 0.5`
- ✅ 能稳定和牌（从 TensorBoard 观察 win_rate）

---

## Phase 1.5: Warmup Transition (500K steps)

**目标:** 平滑过渡到竞技参数 (gamma 0.995→0.999, LR 渐变)

```bash
./scripts/manage.sh train warmup_transition
# 自动加载 Phase 1 best checkpoint
```

### 过渡标准
- ✅ `value_loss` 无剧烈跳升
- ✅ 指标曲线平滑

---

## Phase 2a: Competitive (1M steps)

**目标:** 切换到自对弈，建立联赛对手池

| 参数 | 值 |
|------|-----|
| 对手 | 自对弈 (LeagueManager) |
| 联赛池 | 最大 200 个 checkpoint |
| 冻结窗口 | 最近 3 个 checkpoint 不采样 |
| 熵系数 | cosine 0.05 → 0.03 |
| 向听奖励 | 衰减中 |

```bash
./scripts/manage.sh train competitive
```

### 监控指标

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `elo` | 上升趋势 | 联赛 Elo 评分 |
| `entropy` | 0.5 → 0.4 | 逐渐收敛 |
| `fraction_clipped` | < 10% | PPO clip 比例 |
| `league_pool_size` | 增长中 | 对手多样性 |

### 过渡标准
- ✅ `elo` 稳定上升
- ✅ `entropy` > 0.35
- ✅ 联赛池 ≥ 20 个 checkpoint

---

## Phase 2b: Competitive Distill (4M steps)

**目标:** 启用 Oracle 知识蒸馏 + OpponentPredictor 训练

| 参数 | 值 |
|------|-----|
| Oracle 蒸馏 | KD(0.05) + CE(0.1) |
| OpponentPredictor | ✅ 启用 (weight=0.1) |
| 熵系数 | cosine 0.05 → 0.02 |

```bash
./scripts/manage.sh train distill
```

### 新增监控指标

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `distill_loss` | 下降趋势 | Oracle KD 损失 |
| `oracle_ce_loss` | 下降趋势 | 优势加权 CE |
| `opponent_hand_loss` | < 0.5 | OpponentPredictor BCE |

### 过渡标准
- ✅ `distill_loss` 收敛
- ✅ `opponent_hand_loss < 0.4`
- ✅ `elo` 持续上升

---

## Phase 3: Elite (200M steps)

**目标:** 超人类精英训练 — 所有功能全开

| 参数 | 值 |
|------|-----|
| OpponentPredictor | ✅ weight=0.05 |
| TurnAttention | ✅ 4 heads |
| 熵系数 | cosine 0.02 → 0.01 (200M) |
| 熵下限 | 0.009 |
| 向听衰减 | 120M 步到 30% |
| 排名奖励 | 0.2 (score-weighted) |
| ISMCE (推理) | 96 worlds × depth 8 |
| RTPA (推理) | 攻击 0.8 / 防守 1.5 |

```bash
./scripts/manage.sh train elite
```

### 关键监控指标

| 指标 | 健康范围 | 警告阈值 | 说明 |
|------|---------|---------|------|
| `entropy` | 0.02 → 0.01 | < 0.009 触发 floor | 不应过早收敛 |
| `elo` | 持续上升 | 连续 5M 步停滞 | 核心能力指标 |
| `grad_norm` | < 3.0 | > 10.0 | 梯度爆炸预警 |
| `fraction_clipped` | < 8% | > 15% | PPO 健康度 |
| `kl_divergence` | ~0.002 | > 0.01 | 策略更新幅度 |
| `lr` | 自适应 | 锁死下限 | KL adaptive |

### PBT 超参搜索 (可选)
```bash
./scripts/run_pbt.sh --population 4 --elite-config configs/elite.yaml
```

---

## 一键全流程

```bash
# 全自动 5 阶段流水线（每阶段结束自动传递 checkpoint）
./scripts/manage.sh train pipeline

# 多 GPU
./scripts/manage.sh train pipeline --num-policies 4
```

---

## 常用操作

```bash
# 查看状态
./scripts/manage.sh status

# 中断后恢复
./scripts/manage.sh train elite --resume

# 评估
./scripts/manage.sh eval

# 录制回放
./scripts/manage.sh record --games 50

# 查看回放
./scripts/manage.sh replay

# 导出模型
./scripts/manage.sh export --checkpoint train_dir/blood_v2_elite/checkpoint_p0/checkpoint_best.pth
```

---

## 预估资源与时间

| 阶段 | Steps | 预估时间 (单GPU) |
|------|-------|-----------------|
| warmup | 2M | ~2 小时 |
| transition | 500K | ~30 分钟 |
| competitive | 1M | ~1 小时 |
| distill | 4M | ~4 小时 |
| elite | 200M | ~5-7 天 |
| **总计** | **207.5M** | **~6-8 天** |

> [!TIP]
> 使用 `--num-policies 4` 可将 elite 阶段缩短到 2-3 天 (4 GPU)。
