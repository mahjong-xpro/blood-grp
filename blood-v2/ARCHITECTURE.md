# Blood-V2 Architecture

> 四川血战麻将 AI 系统 — Rust 引擎 + PyTorch 神经网络 + PPO 强化学习

## 1. 系统概览

Blood-V2 是一个混合 Rust/Python 系统：
- **Rust 引擎** (`crates/engine/`): 游戏逻辑、观测编码、SP Table 计算、ISMCE 搜索
- **PyO3 绑定** (`crates/pybind/`): Rust↔Python 桥接
- **Python 训练** (`python/blood/`): 神经网络、PPO 训练、评估、联赛系统
- **配置** (`configs/`): 5 阶段课程学习 YAML 配置

## 2. 目录结构

```
blood-v2/
├── crates/
│   ├── engine/src/
│   │   ├── consts.rs          # 全局常量 (470通道, 34动作, 27牌种)
│   │   ├── obs/
│   │   │   ├── student.rs     # 学生观测编码 (470×27)
│   │   │   └── oracle.rs      # Oracle 观测编码 (+52通道, SP缓存)
│   │   ├── algo/
│   │   │   ├── sp/calc.rs     # SP Table 计算 (MAX_SAMPLES=5)
│   │   │   └── ismce.rs       # ISMCE 搜索 (约束采样+防守rollout)
│   │   └── state/
│   │       └── board.rs       # 游戏状态机 (7阶段)
│   └── pybind/src/
│       ├── env.rs             # Gym 环境包装 (oracle SP缓存, winners字段)
│       ├── ismce_py.rs        # ISMCE Python 接口
│       └── lib.rs             # PyO3 模块注册 (22个通道常量导出)
├── python/blood/
│   ├── model/
│   │   ├── encoder.py         # SuitAwareResNetEncoder (循环式segments架构)
│   │   ├── oracle.py          # OracleEncoder (同架构)
│   │   ├── factory.py         # BloodActorCritic (SF2集成)
│   │   ├── heads.py           # AuxHead (Focal Loss)
│   │   └── inference.py       # PolicyModel (推理/自对弈)
│   ├── eval/
│   │   ├── evaluate.py        # NeuralAgent (RTPA+ISMCE协作)
│   │   ├── rtpa.py            # 实时策略调整 (多信号听牌)
│   │   ├── ismce.py           # ISMCE 搜索器
│   │   ├── arena.py           # 1v3 评估 (随机座位, 平均排名)
│   │   └── elo.py             # Elo 评分追踪
│   ├── training/
│   │   ├── runner.py          # SF2 训练入口 (YAML注入, 纯函数)
│   │   ├── callbacks.py       # BloodObserver (调度器+Elo+联赛)
│   │   ├── losses.py          # BloodLossComputer (Oracle CE+辅助损失)
│   │   ├── league.py          # LeagueManager (Elo加权采样)
│   │   └── scheduler.py       # HyperparamScheduler (动态调度)
│   └── env/
│       ├── blood_env.py       # Rust引擎包装 (冷却恢复)
│       └── selfplay_env.py    # 自对弈环境 (奖励塑形)
└── configs/
    ├── warmup.yaml            # 阶段1: RuleBot, 无LSTM, 2M步
    ├── warmup_transition.yaml # 阶段1.5: RuleBot, LSTM启用, 500K步
    ├── competitive.yaml       # 阶段2: 自对弈, 1M步
    ├── competitive_distill.yaml # 阶段2.5: Oracle蒸馏, 4M步
    └── elite.yaml             # 阶段3: 精英训练, 50M步
```

## 3. 观测空间

### 学生观测 (470 × 27)

| Section | 起始通道 | 数量 | 描述 |
|---------|---------|------|------|
| 1 手牌 | 0 | 5 | one-hot(4) + 最后摸牌(1) |
| 2 游戏上下文 | 5 | 13 | 分数/排名/庄家/进度/分差 |
| 3 定缺 | 18 | 17 | 自己+对手定缺状态 |
| 4 游戏状态 | 35 | 5 | 牌山/听牌/振听/岭上/杠数 |
| 5 自己牌河 | 40 | 58 | 位置编码+衰减 |
| 6 对手牌河 | 98 | 174 | 3×58 |
| 7 可见牌 | 272 | 48 | 牌河/副露概览 |
| 8 防守 | 320 | 9 | 对手花色比例 |
| 9 派生 | 329 | 11 | 剩余牌/门前/副露数 |
| 10 手牌分析 | 340 | 6 | 待牌+向听 |
| 11 动作上下文 | 346 | 12 | 最后打牌/候选/可操作 |
| 12 SP Table | 358 | 99 | 每牌EV+最佳回合概率 |
| 13 番配置 | 457 | 7 | 7个番种标志 |
| 14 对手手牌信息 | 464 | 6 | 手牌数(3)+副露来源(3) |

### Oracle 额外观测 (+52 通道)
- 对手 SP Table 摘要 (3×3=9ch, 带缓存)
- 对手危险度分数 (3ch)
- 对手完整手牌 (3×9=27ch)
- 对手定缺完成状态 (3ch)
- 其他完美信息 (10ch)

## 4. 动作空间

34 个离散动作：
- 0-26: 打牌 (27种牌)
- 27: 碰
- 28-30: 杠 (明杠/暗杠/加杠)
- 31: 和 (荣和/自摸)
- 32-33: 定缺选择

## 5. 神经网络架构 (~37M 参数)

### 编码器 (SuitAwareResNetEncoder)

```
输入: (B, 470, 27)
  ↓ SuitAwareConv1d(470→256, k=3) + GroupNorm + Mish
  ↓ SuitPositionalEncoding(256)
  ↓ [Segment 0: BottleneckBlock×5 → TileAttention(256, 4heads)]
  ↓ [Segment 1: BottleneckBlock×5 → TileAttention(256, 4heads)]
  ↓ [Segment 2: BottleneckBlock×5 → TileAttention(256, 4heads)]
  ↓ [Segment 3: BottleneckBlock×5 → TileAttention(256, 4heads)]
  ↓ Flatten → (B, 6912)
  ↓ LayerNorm + Linear(6912→1024)
输出: (B, 1024)
```

- **SuitAwareConv1d**: 将 27 位置重塑为 3×9（万/筒/条），共享卷积核，强制花色隔离
- **BottleneckBlock**: 1×1 降维 → SuitAwareConv3 → 1×1 升维 + SE 通道注意力
- **TileAttention**: 27 位置多头自注意力，唯一的跨花色交互机制
- **4 个 Segment**: 20 个 BottleneckBlock 均匀分配到 4 组，每组后接 TileAttention

### 时序建模

```
编码器输出: (B, 1024)
  ↓ LSTM(1024→512, 2层)
输出: (B, 512)
```

### 解耦 Actor-Critic

```
Actor: PreNorm MLP(512→512→512) → Linear(512→34)
Critic: PreNorm MLP(512→512→512) → Linear(512→1)
AuxHead: Shanten(3×5) + Waits(81) + OppTenpai(Focal Loss)
```

### Oracle 编码器 (~14M 参数)
同架构（4 TileAttention 层），输入为 (470+52)×27，无 LSTM。

## 6. 训练系统

### PPO + Oracle 蒸馏
- Sample Factory 2 (SF2) 框架
- 猴子补丁 `Learner._calculate_losses` 注入自定义损失
- Oracle CE 损失: softmax(advantages) 加权（连败时不归零）
- 辅助损失: 向听预测 + 待牌预测 + 对手听牌预测(Focal Loss)

### 超参数动态调度
- `HyperparamScheduler`: 支持 linear/cosine/cyclic/step 调度
- Elite 阶段: 熵系数 cosine 退火 0.02→0.005, adv_clip 线性收紧 5.0→3.0
- 通过 `BloodObserver.on_training_step()` 回调应用

### 联赛系统
- 最多 50 个 checkpoint 池
- 多项式衰减采样 (α=2.0) + 均匀下限 + 自对弈概率
- 可选 Elo 加权采样 (高斯 σ=200)

### Elo 追踪
- 多人配对 Elo (K=32/64 自适应)
- JSON 持久化 (原子写入)
- TensorBoard 日志: `blood/elo_*`

## 7. 评估系统

### RTPA (实时策略调整)
- 6 信号对手听牌估计: 副露数/摸切率/端张比/花色集中/回合进度/终盘加速
- 线性终盘放大

### ISMCE (信息集蒙特卡洛评估)
- `evaluate_discards_full()`: 约束采样 + 危险度计算 + 防守感知 rollout
- 对手定缺约束采样
- 多因子危险度: 现物/副露/向听估计/安全牌模式/终盘乘数

### Arena
- 1v3 评估，随机座位
- 平均排名（平局感知）
- Elo 评分更新

## 8. 奖励系统

- 基础: Δscore / 32000
- 排名奖励: [1.0, 0.3, -0.3, -1.0] × 权重
- 安全打牌奖励: 0.015 (competitive)
- 向听奖励: 番数加权 (fan_bonus_scale) + 衰减调度
- Warmup 塑形: 与结构化奖励互斥防护

## 9. 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `blood_num_tile_attn_layers` | 4 | TileAttention 层数 (跨阶段一致) |
| `blood_tile_attn_heads` | 4 | 注意力头数 |
| `rnn_num_layers` | 2 | LSTM 层数 (跨阶段一致) |
| `rnn_size` | 512 | LSTM 隐藏维度 |
| `NUM_STUDENT_CHANNELS` | 470 | 学生观测通道数 |
| `NUM_ORACLE_CHANNELS` | 522 | Oracle 观测通道数 |
| `ACTION_SPACE` | 34 | 动作空间大小 |
| `REWARD_NORM` | 32000 | 奖励归一化因子 |
