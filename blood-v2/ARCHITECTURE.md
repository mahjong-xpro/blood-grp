# 血战到底 V2 — 基于 Sample Factory v2 的超人类 AI 架构

> **目标**：基于 Sample Factory v2 高吞吐 RL 框架，构建完全自包含的血战到底超人类 AI。  
> **核心优势**：Sample Factory 提供生产级分布式 PPO 基础设施（异步 rollout / 推理 / 训练），我们专注于领域建模。  
> **原则**：完全独立项目，不依赖 V1 代码（mortal/libblood）。

---

## 目录

1. [为什么选择 Sample Factory v2](#1-为什么选择-sample-factory-v2)
2. [系统架构总览](#2-系统架构总览)
3. [组件详解](#3-组件详解)
4. [游戏引擎 (Rust)](#4-游戏引擎-rust)
5. [环境层 (PyO3 + Gymnasium)](#5-环境层-pyo3--gymnasium)
6. [神经网络](#6-神经网络)
7. [动作与观测空间](#7-动作与观测空间)
8. [训练算法与策略](#8-训练算法与策略)
9. [Oracle 蒸馏](#9-oracle-蒸馏)
10. [League 与 Self-play](#10-league-与-self-play)
11. [训练阶段规划](#11-训练阶段规划)
12. [训练稳定性](#12-训练稳定性)
13. [评估协议](#13-评估协议)
14. [目录结构](#14-目录结构)
15. [交付里程碑](#15-交付里程碑)
16. [与 V1 架构对比](#16-与-v1-架构对比)

---

## 1. 为什么选择 Sample Factory v2

### V1 的问题

V1（mortal/ + libblood/）采用 **Dueling DQN + 自研训练循环**，存在以下瓶颈：

| 问题 | 影响 |
|------|------|
| DQN 离散 Q-value 估计 | 信用分配不精确，exploration 受限 |
| 单机单 GPU 训练 | 吞吐量受限（~200 局/秒） |
| 手写分布式架构 | 需自建 gRPC 推理服务、Redis 轨迹缓存、DDP 训练 |
| Boltzmann 探索 | 接近最优时易陷入局部最优 |

### Sample Factory v2 解决了什么

| 能力 | Sample Factory 提供 | 我们需要做的 |
|------|---------------------|-------------|
| 分布式 PPO | 异步 Rollout / Inference / Batcher / Learner | 零额外代码 |
| 高吞吐 | 共享内存零拷贝、双缓冲采样 | 只需实现 `env.step()` |
| 多策略 Self-play | `--num_policies=N` + PBT | 配置即用 |
| 自定义模型 | Encoder / Core / Decoder 工厂注册 | 实现 SuitAwareEncoder |
| GPU 推理 | Inference Worker 自动 batch | 零额外代码 |
| 序列化调试 | `--serial_mode=True` | 单进程调试 |

### 性能对比预估

| 指标 | V1 (DQN + mortal) | V2 (SF + PPO) |
|------|-------------------|---------------|
| 算法 | Dueling DQN | PPO + GAE |
| 吞吐量（单机 1×4090） | ~200 局/秒 | ~3,000 局/秒 |
| 吞吐量（8×4090） | 不支持 | ~15,000 局/秒 |
| 探索机制 | Boltzmann (ε=0.05) | 策略熵自然探索 |
| 信用分配 | TD(λ) + Target Network | GAE(λ) 原生支持 |
| 离策略修正 | 无 | V-trace 内置 |
| Self-play | 手写历史池 | SF 原生多策略 |

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sample Factory v2 Runtime                     │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Rollout Workers  │    │ Inference Workers │                   │
│  │  (N 个进程)       │◄──►│  (M 个 GPU)       │                   │
│  │                   │    │                   │                   │
│  │  BloodMahjongEnv │    │  SuitAwareEncoder │                   │
│  │  ┌─────────────┐ │    │  + PPO Policy     │                   │
│  │  │ Rust Engine  │ │    │                   │                   │
│  │  │ (PyO3 FFI)  │ │    └────────┬──────────┘                   │
│  │  └─────────────┘ │             │                              │
│  └──────────────────┘             │ SharedMemory                 │
│                                   ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │     Batcher       │───►│     Learner       │                   │
│  │  (轨迹组装)       │    │  (PPO SGD on GPU) │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Runner: 日志 / TensorBoard / Checkpoint / PBT 控制       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

1. **Rollout Worker** 持有 `BloodMahjongEnv` 实例，调用 Rust 引擎的 `step()`
2. 产生 `obs` 通过共享内存发送给 **Inference Worker**
3. Inference Worker 用 GPU 运行 `SuitAwareEncoder → Policy Head`，返回 `action`
4. Rollout Worker 收到 action，推进环境，每 `--rollout` 步产生一条轨迹
5. **Batcher** 收集轨迹，组装成训练 batch
6. **Learner** 用 PPO 在 GPU 上执行 SGD，更新权重通过共享内存广播

---

## 3. 组件详解

### 3.1 Rollout Worker

- 数量：`--num_workers=N`（推荐：CPU 核数 / 2）
- 每个 Worker 持有 `--num_envs_per_worker` 个环境实例
- 双缓冲采样（`--worker_num_splits=2`）：当一半环境等待推理时，另一半继续执行
- 环境完全运行在 CPU 上（Rust 引擎），不占 GPU

```
单机 32 核 + 1×4090 推荐配置：
  --num_workers=12
  --num_envs_per_worker=8
  → 96 个并行环境实例
```

### 3.2 Inference Worker

- 数量：`--policy_workers_per_policy=1`（单 GPU 时 1 个即可）
- 自动 batch：收集多个 Worker 的 obs → 一次 GPU forward
- 权重在 Learner 每次 SGD 后通过共享内存同步

### 3.3 Batcher + Learner

- Batcher 与 Learner 默认在同一进程（减少 CUDA context）
- `--batch_size` 控制 mini-batch 大小
- `--num_batches_per_epoch` × `--num_epochs` 控制每个数据集的 SGD 步数
- PPO clip + GAE(λ) + 熵正则化 + 梯度裁剪

### 3.4 多 GPU 扩展

```
单机 8×4090：
  --num_policies=4 (4个策略互相对弈)
  每个策略占 2 张 GPU (1 学习 + 1 推理)
  --num_workers=48
  --num_envs_per_worker=8
  → 384 个并行环境
```

---

## 4. 游戏引擎 (Rust)

### 4.1 已完成（engine/ 目录）

| 模块 | 功能 | 状态 |
|------|------|------|
| `tile.rs` | 27 种牌型定义、洗牌、花色工具 | ✅ 完成 |
| `hand.rs` | 手牌操作、向听计算、副露类型 | ✅ 完成 |
| `win.rs` | 和牌判定、17 种番型、计分 | ✅ 完成 |
| `game.rs` | 完整 FSM（751 行）、合法动作生成 | ✅ 完成 |
| `actions.rs` | 38 种动作编解码、合法 mask | ✅ 完成 |
| `obs.rs` | 67 通道观测编码 + Oracle 28 通道 | ✅ 完成 |

### 4.2 FSM 状态机

```
[*] → Deal → DingQue → Draw → SelfCheck → Discard → Reaction → PostWin → Scoring → [*]
                                    ↑                      │
                                    └──────────────────────┘ (血战续战)
```

### 4.3 规则完整性

| 规则 | 状态 |
|------|------|
| 定缺 (必须缺一门) | ✅ |
| 血战续战 (有人胡后继续) | ✅ |
| 一炮多响 | ✅ |
| 杠上开花 / 杠上炮 | ✅ |
| 海底捞月 | ✅ |
| 抢杠胡 | ⚠️ TODO (game.rs:308) |
| 查花猪 / 查大叫 | ✅ |
| 5 番封顶 (16000) | ✅ |
| 杠即时支付 | ⚠️ 待加入 |

---

## 5. 环境层 (PyO3 + Gymnasium)

### 5.1 架构

```
Python (Sample Factory)
  │
  │  gymnasium.Env API
  ▼
sf_blood/env.py  (BloodMahjongEnv)
  │
  │  PyO3 FFI
  ▼
env_core/  (Rust cdylib)
  │
  │  直接调用
  ▼
engine/  (纯 Rust 引擎)
```

### 5.2 多智能体建模

血战到底是 4 人游戏，但在 Sample Factory 中我们采用 **单智能体视角** 建模：

- 每个环境实例控制 **1 个玩家**
- 其他 3 个玩家由 **同策略不同实例** 或 **历史策略** 控制
- SF 的 `--num_policies` 和 `--pbt_mix_policies_in_one_env` 自动处理策略混合

```python
class BloodMahjongEnv(gymnasium.Env):
    """
    单玩家视角的血战到底环境。
    
    内部维护一个完整 4 人牌局，暴露 player_id=0 的观测/动作。
    其他 3 家由内置策略（规则 Bot / 历史模型）或 SF 多策略控制。
    """
    observation_space = spaces.Dict({
        "obs":  spaces.Box(0, 1, shape=(67, 27), dtype=np.float32),
        "mask": spaces.Box(0, 1, shape=(38,), dtype=np.bool_),
    })
    action_space = spaces.Discrete(38)
```

### 5.3 step() 逻辑

```
step(action) {
    1. 将 action 应用到 player_0
    2. 推进 FSM 直到下一个需要 player_0 决策的时刻
       （中间其他玩家的决策由内置 AI 立即处理）
    3. 返回 (obs, reward, terminated, truncated, info)
}
```

关键点：
- 每次 `step()` 可能推进多个 FSM 步骤（跳过非自己回合）
- `reward` = Δscore / 16000（归一化到 [-1, 1]）
- `terminated` = 牌局结束
- `info` 携带额外统计（番数、胡牌类型、向听数等）

### 5.4 PyO3 绑定

```rust
// env_core/src/lib.rs
#[pyclass]
struct RustMahjongGame {
    state: GameState,
    opponent_policy: OpponentPolicy,
}

#[pymethods]
impl RustMahjongGame {
    #[new]
    fn new(seed: u64, opponent_mode: &str) -> Self { ... }
    
    fn reset(&mut self, seed: u64) -> PyResult<(PyObs, PyMask)> { ... }
    fn step(&mut self, action: usize) -> PyResult<StepResult> { ... }
    fn get_oracle_obs(&self) -> PyResult<Vec<f32>> { ... }
}
```

---

## 6. 神经网络

### 6.1 SuitAwareEncoder（自定义 SF Encoder）

核心思想：麻将牌按花色分 3 组（万/筒/条 各 9 张），卷积核 kernel=3 自然隔离花色边界。

```
输入 obs: (B, 67, 27)
    │
    ▼
SuitAwareConv1d(67 → 128, k=3, pad=1)  ← 阻止 9m↔1p, 9p↔1s 跨界
    │ BN + Mish
    ▼
SuitAwareConv1d(128 → 256, k=3, pad=1)
    │ BN + Mish
    ▼
ResBlock × 8 (256ch, SuitAwareConv)
    │ 每块: BN→Mish→Conv→BN→Mish→Conv + ChannelAttention
    ▼
GlobalAvgPool per suit → (B, 256, 3) → (B, 768)
    │
    ▼
MLP(768 → 512) + LayerNorm + Mish
    │
    ▼
encoder_output: (B, 512)  → 传给 SF 的 Core → Decoder → Actor/Critic
```

参数量：约 **3.5M**（比 V1 的 10.5M 更精简，因为 PPO 不需要 DQN 的双头）

### 6.2 ChannelAttention

```python
class ChannelAttention(nn.Module):
    """Squeeze-Excitation 变体，允许花色间信息交换。"""
    # GlobalAvgPool → FC(C→C//4) → Mish → FC(C//4→C) → Sigmoid → Scale
```

### 6.3 Actor-Critic Head

```
encoder_output (512)
    │
    ├─── Actor Head ──→ Linear(512 → 38) → masked_softmax → π(a|s)
    │
    └─── Critic Head ─→ Linear(512 → 256) → Mish → Linear(256 → 1) → V(s)
```

SF 默认支持 `--actor_critic_share_weights=True`（共享 Encoder）或 `False`（独立 Encoder）。推荐先共享，后期如果 Actor/Critic 学习目标冲突再分离。

### 6.4 动作 mask

Sample Factory 原生支持 action masking。在 `obs_dict` 中提供 `"mask"` key：

```python
obs_dict = {
    "obs": np.array(67×27, dtype=float32),
    "mask": np.array(38, dtype=bool),  # True = 合法
}
```

SF 内部在 softmax 前将非法动作的 logit 设为 -∞。

---

## 7. 动作与观测空间

### 7.1 动作空间 (action_dim = 38)

| 编号 | 动作 | 适用阶段 |
|------|------|---------|
| 0-26 | 弃牌 (tile_0 ~ tile_26) | Discard |
| 27 | 自摸 (Tsumo) | SelfCheck |
| 28 | 荣和 (Ron) | Reaction |
| 29 | 碰 (Pon) | Reaction |
| 30 | 明杠 (MinKan) | Reaction |
| 31 | 暗杠 (AnKan) | SelfCheck |
| 32 | 加杠 (KaKan) | SelfCheck |
| 33 | 过 (Pass) | SelfCheck / Reaction |
| 34-36 | 定缺 (万/筒/条) | DingQue |
| 37 | 自动弃牌 (已胡透明) | 内部 |

### 7.2 Student 观测空间 (67 × 27)

| 通道组 | 通道数 | 描述 |
|--------|--------|------|
| 自己手牌 | 4 | 持有 1/2/3/4 张 one-hot |
| 自己副露 | 4 | 碰/明杠/暗杠/加杠 |
| 3 家舍牌 | 12 | 每家 4 通道 |
| 3 家副露 | 12 | 每家 4 通道 |
| 3 家是否已胡 | 3 | 布尔广播 |
| 全场可见牌 | 4 | 全局计数 |
| 自己定缺 | 3 | one-hot |
| 对手定缺推断 | 9 | 3 人 × 3 花色 |
| 当前触发牌 | 1 | Reaction 阶段 |
| 牌墙进度 | 1 | [0,1] |
| 分数差 | 3 | 归一化 |
| 向听估计 | 1 | [0,1] |
| 巡目编码 | 4 | 分桶 one-hot |
| 庄家标记 | 4 | 偏移 one-hot |
| RTPA 预留 | 2 | 风格参数 |
| **合计** | **67** | |

### 7.3 Oracle 额外观测 (+28 通道)

| 通道组 | 通道数 |
|--------|--------|
| 3 家真实手牌 | 12 |
| 牌墙剩余 | 4 |
| 对手真实定缺 | 9 |
| 对手向听数 | 3 |
| **合计** | **28** |

---

## 8. 训练算法与策略

### 8.1 PPO 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| 算法 | PPO | Sample Factory 原生 |
| GAE λ | 0.95 → 0.97 | 课程调整 |
| Clip ratio ε | 0.2 → 0.1 | 精英期收紧 |
| γ (discount) | 0.995 | 麻将一局 ~20 步 |
| 熵系数 | 0.03 → 0.003 | 逐步减少探索 |
| 梯度裁剪 | max_norm=0.5 | 防止梯度爆炸 |
| 学习率 | 3e-4 → 1e-5 | cosine annealing |
| Mini-batch | 512 → 1024 | 随训练规模增大 |
| Rollout 长度 | 32 | 约 1.5 局完整交互 |
| Num epochs | 2 | 每个 batch 训练 2 遍 |

### 8.2 V-trace 离策略修正

当 Inference Worker 使用的权重滞后于 Learner 时，SF 自动应用 V-trace：

```
ρ_t = min(ρ̄, π_new(a|s) / π_old(a|s))
V_trace = V(s) + Σ γ^t (Π c_i) × δ_t
```

无需任何额外代码。

### 8.3 奖励设计

```python
reward = Δscore / 16000  # 归一化到 [-1, 1]
```

- 每次有人胡牌产生即时 reward（非回合末一次性结算）
- 终局查花猪 / 查大叫的赔付也计入
- **热身期额外奖励**（50k 步后关闭）：
  - 定缺清空：+0.05
  - 胡牌：+0.1
  - 放铳：-0.05

### 8.4 花色置换数据增强

万/筒/条完全对称，每条轨迹可生成 6 种排列。在环境层随机应用：

```python
def augment_observation(obs, permutation):
    """将 obs 中万/筒/条按 permutation 重排。"""
    # permutation ∈ {(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)}
    new_obs = rearrange_suits(obs, permutation)
    return new_obs
```

---

## 9. Oracle 蒸馏

### 9.1 两阶段训练

```
阶段 1: 训练 Oracle（看到完美信息）
  - observation: (67+28, 27) = (95, 27)
  - 独立 PPO 训练至收敛
  - 目标: avg_ranking < 1.5

阶段 2: Student 蒸馏
  - Student loss = PPO_loss + α × KL(π_student ‖ π_oracle)
  - α: 1.0 → 0.1 线性衰减（500k 步）
  - Oracle 冻结权重，只提供 soft target
```

### 9.2 实现方式

```python
class OracleDistillCallback(sf.AlgoObserver):
    """SF 训练回调，在每个 batch 后注入蒸馏 loss。"""
    
    def on_train_step(self, learner, batch):
        oracle_obs = batch["oracle_obs"]  # 来自环境 info
        with torch.no_grad():
            oracle_logits = self.oracle_model(oracle_obs)
        student_logits = batch["action_logits"]
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(oracle_logits / T, dim=-1),
            reduction="batchmean"
        ) * T * T
        learner.extra_loss += self.alpha * kl_loss
```

---

## 10. League 与 Self-play

### 10.1 Sample Factory 原生 Self-play

```bash
python sf_blood/train.py \
    --env=blood_mahjong \
    --num_policies=4 \
    --pbt_mix_policies_in_one_env=True \
    --with_pbt=True
```

4 个策略在同一牌局中对弈，PBT 自动淘汰弱策略并继承强策略的超参数。

### 10.2 League System

```
Main Agent ──── 50% 对自己 + 30% 历史池 + 20% Exploiter
Exploiter  ──── 专打 Main Agent 弱点
历史池     ──── 每 5k 步保存一份 checkpoint
```

通过 SF 的 `AgentPolicyMapping` 自定义实现：

```python
class LeagueMapping:
    def get_policy_for_agent(self, agent_idx, env_idx):
        if agent_idx == 0:
            return MAIN_POLICY
        r = random()
        if r < 0.5:
            return sample_from_history_pool()
        elif r < 0.8:
            return MAIN_POLICY  # self-play
        else:
            return EXPLOITER_POLICY
```

---

## 11. 训练阶段规划

### Phase A: 游戏引擎 (✅ 已完成)

- Rust 引擎实现全部血战到底规则
- 1000 局随机对弈无 panic
- 全番型测试通过

### Phase B: 环境封装

- PyO3 绑定 → Gymnasium 环境
- `env.step()` 正确返回 obs/reward/done/mask
- 100 局 Python 端完整对弈测试

### Phase C: 单机训练（热身期）

| 参数 | 值 |
|------|------|
| 框架 | Sample Factory v2 |
| 对手 | 规则 Bot → Self-play |
| Workers | 12 |
| Envs/worker | 8 |
| LR | 3e-4 |
| Entropy | 0.03 |
| 步数 | 0 - 50k |
| 验收 | avg_ranking < 2.8 |

### Phase D: 竞争期

| 参数 | 值 |
|------|------|
| 对手 | Self-play + 历史池 |
| --num_policies | 2 → 4 |
| LR | 1e-4 → 3e-5 cosine |
| Entropy | 0.01 |
| 步数 | 50k - 500k |
| 验收 | avg_ranking < 2.3 |

### Phase E: Oracle 蒸馏

| 参数 | 值 |
|------|------|
| Oracle 训练 | 独立 PPO，完美信息 |
| 蒸馏 α | 1.0 → 0.1 |
| 步数 | 500k - 1M |
| 验收 | deal_in_rate < 20% |

### Phase F: 精英期

| 参数 | 值 |
|------|------|
| League System | 开启 |
| PBT | 开启 |
| LR | 3e-5 → 1e-5 |
| Entropy | 0.003 |
| 步数 | 1M+ |
| 验收 | avg_ranking < 1.8 |

### Phase G: 部署

- ONNX 导出（SF 原生支持）
- WebSocket 对弈网关
- ISMCE 推理增强

---

## 12. 训练稳定性

| 机制 | 触发条件 | 动作 |
|------|---------|------|
| Policy loss 爆炸 | policy_loss > 10 | 跳过 batch |
| 梯度 NaN | 参数出现 NaN | 回滚 checkpoint |
| ELO 崩塌 | 连续 3 次评估恶化 | 回滚 + LR × 0.5 |
| Entropy 坍缩 | entropy < 0.1 | entropy_coef × 2 |
| KL 发散 | 蒸馏 KL > 10 | α × 0.5 |
| 定期备份 | 每 1000 步 | 完整 checkpoint |
| 最佳保留 | avg_ranking 创新低 | 存为 best.pt |
| PBT 淘汰 | 排名末尾 20% | 继承顶部策略 |

---

## 13. 评估协议

| 指标 | 超人类目标 |
|------|----------|
| avg_ranking (2000 局) | < 1.8 |
| point_per_round | > +500 |
| win_rate | 28-35% |
| deal_in_rate | < 18% |
| mean_fan | > 2.2 |

评测方式：Main Agent vs League Pool 最强 3 版本，打 2000 局取均值。

---

## 14. 目录结构

```
blood-v2/
├── Cargo.toml                      # Rust workspace (engine + env_core)
├── pyproject.toml                  # Python 项目配置 (uv/pip)
├── ARCHITECTURE.md                 # ← 本文档
│
├── engine/                         # [Phase A ✅] Rust 游戏引擎
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── tile.rs                 # 27 种牌型、洗牌
│       ├── hand.rs                 # 手牌、向听、副露
│       ├── win.rs                  # 和牌判定、番数、计分
│       ├── game.rs                 # FSM 状态机 (751 行)
│       ├── actions.rs              # 38 动作编解码
│       └── obs.rs                  # 67+28 通道观测编码
│
├── env_core/                       # [Phase B] PyO3 Rust → Python 绑定
│   ├── Cargo.toml                  # PyO3 cdylib
│   └── src/
│       ├── lib.rs                  # 模块根
│       ├── pybind.rs               # #[pyclass] / #[pymethods]
│       └── opponent.rs             # 内置对手策略 (规则 Bot)
│
├── sf_blood/                       # [Phase C-G] Python 训练代码
│   ├── __init__.py
│   ├── env.py                      # BloodMahjongEnv (gymnasium.Env)
│   ├── encoder.py                  # SuitAwareEncoder (SF Encoder)
│   ├── model.py                    # 完整 ActorCritic (可选)
│   ├── train.py                    # 训练入口
│   ├── enjoy.py                    # 评估入口
│   ├── cfg.py                      # 自定义参数 + 默认覆盖
│   ├── reward.py                   # 奖励计算工具
│   ├── oracle.py                   # Oracle 蒸馏回调
│   ├── league.py                   # League 策略映射
│   ├── augment.py                  # 花色置换增强
│   └── eval.py                     # 1v3 评估脚本
│
├── configs/                        # 各阶段训练配置
│   ├── warmup.yaml                 # Phase C 热身
│   ├── competitive.yaml            # Phase D 竞争
│   └── elite.yaml                  # Phase F 精英
│
└── tests/
    ├── test_env.py                 # 环境正确性测试
    ├── test_model.py               # 模型形状测试
    └── test_engine.py              # Rust 引擎测试
```

---

## 15. 交付里程碑

| 阶段 | 交付物 | 验收标准 | 预计工期 |
|------|--------|---------|---------|
| **A** ✅ | Rust 游戏引擎 | 全规则测试通过 + 1000 局无 panic | 已完成 |
| **B** | PyO3 环境 | Python 端 100 局完整对弈 + obs/mask 维度正确 | 1 周 |
| **C** | 单机训练 | 50k 步后 avg_ranking < 2.8 | 1 周 |
| **D** | 竞争训练 | 500k 步后 avg_ranking < 2.3 | 2 周 |
| **E** | Oracle 蒸馏 | deal_in_rate < 20% | 2 周 |
| **F** | 精英训练 | avg_ranking < 1.8 | 4 周 |
| **G** | 部署 | WebSocket 对弈可用 | 1 周 |

> [!IMPORTANT]
> **基础不牢，地动山摇** — Phase B 的 PyO3 绑定必须确保 obs / mask / reward 完全正确后才进入训练阶段。

---

## 16. 与 V1 架构对比

| 维度 | V1 (mortal + libblood) | V2 (Sample Factory) |
|------|----------------------|---------------------|
| RL 算法 | Dueling DQN | PPO + GAE |
| 训练框架 | 手写 train.py | Sample Factory v2 |
| 分布式 | 手写 server/client + gRPC | SF 内置异步组件 |
| 共享内存 | 无（进程间序列化） | SF 零拷贝共享内存 |
| 模型 | 30-block ResNet (10.5M) | SuitAware 8-block (3.5M) |
| 探索 | Boltzmann (ε, T) | 策略熵自然探索 |
| Self-play | 手写历史池 | SF 多策略 + PBT |
| Oracle | 共享 Optimizer | 独立训练 + 蒸馏回调 |
| 观测 | 423 通道 (含 SP Table) | 67 通道 (更简洁) |
| 调试 | 手动日志 | SF TensorBoard + serial_mode |
| 导出 | 手动 TorchScript | SF 原生 ONNX |
| 代码量 | ~5000 行 Python | ~1000 行 Python + SF |

V2 的核心理念：**让框架做框架的事，我们专注领域知识。**
