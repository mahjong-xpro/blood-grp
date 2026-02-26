# Blood v2 超人类水平改造技术设计文档

## 1. 背景与目标

Blood v2 是面向四川血战麻将的深度强化学习系统，基于 SuitAwareResNet + LSTM + PPO 架构，通过 5 阶段课程学习训练。本文档描述已实现的 6 项改进，目标是将系统从"强 AI"提升至超人类水平。

### 1.1 系统现状

| 指标 | 改造前 | 改造后 |
|------|--------|--------|
| 学生观测通道 | 470 | **473** (Section 15: 現物标记 3ch) |
| Oracle 观测通道 | 522 | **525** (473 + 52) |
| ISMCE rollout 深度 | 4 | **8** |
| ISMCE 采样世界数 | 64 | **96** (elite 配置) |
| 危险度评分 | 基础版 | **增强版** (定缺+副露+向听估计+打牌模式) |
| 对手建模 | 无 | **OpponentHandPredictor** (75ch→128ch, 6 blocks) |
| 时序注意力 | 无 | **TurnAttention** (零初始化 residual) |
| 超参搜索 | 手动 | **PBTController** (10 个可调参数) |
| elite 训练步数 | 50M | **200M** |

### 1.2 常量定义

```rust
// crates/engine/src/consts.rs
pub const NUM_STUDENT_CHANNELS: usize = 473;
pub const NUM_ORACLE_EXTRA_CHANNELS: usize = 52;
pub const NUM_ORACLE_CHANNELS: usize = 525;  // 473 + 52
pub const ACTION_SPACE: usize = 34;
pub const NUM_TILE_TYPES: usize = 27;
pub const REWARD_NORM: i32 = 32_000;
pub const MAX_FAN: u8 = 6;
pub const MAX_TURNS: usize = 28;
```

```python
# python/blood/consts.py — 与 Rust 保持同步
NUM_STUDENT_CHANNELS = 473
NUM_ORACLE_CHANNELS = 525
```

---

## 2. Phase A: 搜索与评估深化

### A1. ISMCE 搜索深化

**文件**: `crates/engine/src/algo/ismce.rs`

ISMCE (Information Set Monte Carlo Evaluation) 从信息集中采样一致的对手手牌，评估每个候选弃牌在 `depth` 回合内达到听牌/和牌的概率。

#### 核心数据结构

```rust
// crates/engine/src/algo/ismce.rs:16-34
struct RolloutResult {
    won: bool,
    tenpai: bool,
    score: f64,       // 和牌时的期望得分
}

pub struct IsmceScore {
    pub tile: Tile,
    pub win_rate: f64,
    pub expected_score: f64,          // 番数加权期望得分
    pub tenpai_rate: f64,
    pub tenpai_value: f64,            // 听牌质量 (待数×剩余轮归一化)
    pub danger_cost: f64,             // 防守代价
    pub avg_shanten_improvement: f64,
    pub win_rate_raw: f64,            // 向后兼容
}
```

#### 配置

```rust
// crates/engine/src/algo/ismce.rs:37-51
pub struct IsmceConfig {
    pub num_worlds: usize,     // 默认 64, elite 配置 96
    pub rollout_depth: usize,  // v2: 4 → 8
    pub base_seed: u64,
}
```

#### 三层评估 API

1. `evaluate_discards` — 基础版，无约束采样，无防守
2. `evaluate_discards_constrained` — 约束采样（尊重对手定缺），无防守
3. **`evaluate_discards_full`** — 推荐入口，组合三项能力:
   - 约束世界采样 (`sample_world_constrained`)
   - 增强危险度评分 (`danger_scores_enhanced`)
   - 防守感知 rollout (`simulate_draws_with_opponents`)

#### 番数估算

```rust
// crates/engine/src/algo/ismce.rs:255-298
fn estimate_fan_quick(hand, melds_count, is_tsumo) -> u8
// 检查: 清一色(+4), 对对和(+1), 门清(+1)
// 返回 1-6 (capped at MAX_FAN)
```

#### 对手荣和概率估算

```rust
// crates/engine/src/algo/ismce.rs:547-578
fn estimate_ron_probability(discard_tile, danger_tiles, opponent_danger_info) -> f32
// danger < 0.3 → 0.0
// 检查对手是否可能听牌 (est_shanten < 1.0)
// 缩放: (danger - 0.3) * 0.35 → 范围 [0.0, 0.25]
```

#### PyO3 绑定

**文件**: `crates/pybind/src/ismce_py.rs`

返回 7-tuple: `(tile, win_rate, tenpai_rate, improvement, expected_score, tenpai_value, danger_cost)`

三个导出函数:
- `ismce_evaluate` — 基础版
- `ismce_evaluate_full` — 完整版（约束采样+防守）
- `ismce_danger` — 增强危险度评分

#### Python 评分公式

```python
# python/blood/eval/ismce.py:138
combined = expected_score / 32000.0 + 0.3 * tenpai_value - 0.5 * danger_cost
```

Python 端 `ISMCESearcher` 将 policy logits 与 ISMCE 分数在 log 空间做标准化后混合:
- 默认权重: `policy_weight=0.7`, `ismce_weight=0.3`
- 两个信号分别在候选牌上做零均值+单位方差标准化

### A2. 防守建模精细化

**文件**: `crates/engine/src/algo/ismce.rs:714-810`

#### OpponentDangerInfo 结构

```rust
pub struct OpponentDangerInfo {
    pub ding_que: Option<Suit>,
    pub melds_count: usize,
    pub discard_count: usize,
}
```

#### danger_scores_enhanced 算法

对每张牌计算综合危险度 [0, 1]，考虑所有对手状态:

1. **定缺花色检查**: 对手定缺的花色 → 对该对手完全安全 (continue)
2. **基础危险度**: 对手打过此牌 → 0.05 × unseen/total; 未打过 → 0.25 + 0.15 × unseen/total
3. **副露加成**: melds × 0.1, 上限 0.35
4. **向听数估计**: est_shanten = max(0, 4 - melds - discards/5)
   - 听牌(0) → +0.3, 一向听(1) → +0.15, 二向听(2) → +0.05
5. **打牌模式**: 最近 3 巡安全牌比例 × 0.1
6. **终盘加成**: wall_remaining < 20 时, × (1.0 + (20 - remaining)/20 × 0.5)

### A3. 对手手牌推断

**文件**: `python/blood/model/opponent_model.py`

#### OpponentHandPredictor 架构

```
Input: (B, 75, 27) — 对手特定公开特征
  ↓
SuitAwareConv1d(75, 128, k=3) + GroupNorm + Mish
  ↓
6 × BottleneckBlock(128)
  ↓
1 × TileAttention(128, heads=4)
  ↓
GroupNorm + Conv1d(128, 1, 1) + Sigmoid
  ↓
Output: (B, 27) — P(tile_t ∈ opponent_hand)
```

输入 75 通道组成:
- 对手牌河 (58 ch, 来自 Section 6)
- 可见牌 (4 ch, 来自 Section 7)
- 游戏上下文 (5 ch: wall_remaining, turn_progress 等)
- 副露信息 (8 ch)

训练方式: BCE loss, 使用 Oracle 观测中的对手手牌作为 ground truth 标签。

#### 集成到训练

**文件**: `python/blood/model/factory.py:131-140`

```python
# BloodActorCritic.__init__
self.opponent_predictor_enabled = getattr(cfg, "opponent_predictor_enabled", False)
if self.opponent_predictor_enabled:
    self.opponent_predictor = OpponentHandPredictor(conv_ch=128, num_blocks=6)
    self.opponent_predictor_weight = getattr(cfg, "opponent_predictor_weight", 0.1)
```

**文件**: `python/blood/training/losses.py:121-127`

```python
# BloodLossComputer.compute
if getattr(ac, "opponent_predictor_enabled", False):
    opp_loss = self._opponent_hand_loss(ac, obs)
    if opp_loss is not None:
        extra_loss = extra_loss + opp_weight * opp_loss
```

#### 信息引导采样 (Rust 端)

**文件**: `crates/engine/src/algo/ismce.rs:158-225`

```rust
pub fn sample_world_informed(
    info: &PlayerInfo,
    opponents: &[OpponentInfo],
    opponent_hand_probs: &[[f32; NUM_TILE_TYPES]; 3],
    rng_seed: u64,
) -> (Vec<Vec<Tile>>, Vec<Tile>)
```

按预测概率降序排列候选牌，优先将高概率牌分配给对应对手。同时尊重定缺约束（定缺花色概率置零）。

> **注意**: `sample_world_informed` 的 Python 调用路径尚未完全连通。`evaluate_discards_full` 当前使用 `sample_world_constrained`，未接入 OpponentHandPredictor 的概率输出。

### A4. 观测空间扩展

**文件**: `crates/engine/src/obs/student.rs`

#### Section 14: 对手手牌信息 (6 ch, offset 464-469)

```
ch 464-466: 3 个对手手牌数量 (hand_count / 13.0)
ch 467-469: 3 个对手最近副露来源的相对位置 (source / 3.0)
```

#### Section 15: 現物/Genbutsu 安全牌 (3 ch, offset 470-472)

```
ch 470-472: 每个对手一个通道，该对手弃过的牌标记为 1.0
```

血战无振听规则，对手弃过的牌永远不会被该对手荣和。显式编码避免网络从 174ch 牌河中自行学习此规则。

---

## 3. Phase B: 架构改进

### B1. TurnAttention

**文件**: `python/blood/model/factory.py:22-53`

#### 架构

```python
class TurnAttention(nn.Module):
    def __init__(self, dim=512, num_heads=4, max_turns=32):
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_turns, dim))
        # 零初始化输出投影 → 初始行为等价于纯 LSTM
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
```

#### 关键设计决策

1. **零初始化 residual**: `out_proj` 权重和偏置初始化为零，确保训练初期 TurnAttention 输出为零，模型行为与纯 LSTM 完全一致。随着训练推进，注意力机制逐渐学习有用的跨回合模式。

2. **位置编码**: 可学习的位置嵌入 `pos_embed`，截断正态初始化 (std=0.02)。

3. **训练模式 BPTT 支持** (`factory.py:182-205`):
   - 训练时 core_output 为 (B×T, dim)，reshape 为 (B, T, dim)
   - 每个位置 t 只关注 [0, t] 的历史（因果注意力）
   - 推理时退化为自注意力

4. **推理模式** (`inference.py:148-157`):
   - 维护 `_memory_buffer` 列表，存储最近 `max_turns` 个 LSTM 输出
   - 每步追加当前 features，截断超出窗口的历史

#### 配置

```yaml
# elite.yaml
turn_attention_enabled: false   # 需显式启用
turn_attention_heads: 4
```

默认 `max_turns` 取自 `recurrence` 参数（默认 32）。

### B2. 训练规模扩展

**文件**: `configs/elite.yaml`

| 参数 | 改造前 | 改造后 |
|------|--------|--------|
| `train_for_env_steps` | 50M | **200M** |
| `ismce_num_worlds` | 64 | **96** |
| `ismce_rollout_depth` | 4 | **8** |
| `league_max_pool_size` | 50 | **200** |
| `league_add_every` | 50000 | **100000** |
| `blood_arena_eval_every` | 500K | **1M** |
| `blood_arena_eval_games` | 50 | **100** |

#### 动态超参调度

```yaml
# Entropy: cosine 退火 0.02 → 0.01, 全 200M 步
blood_schedule_entropy: "cosine,0.02,0.01,0,200000000"
# Advantage clip: 线性收紧 5.0 → 3.0, 10M-160M 步
blood_schedule_adv_clip: "linear,5.0,3.0,10000000,160000000"
# Entropy floor 安全网
blood_entropy_floor: 0.009
```

#### 向听奖励衰减

```yaml
shanten_reward_decay_steps: 120000000   # 前 120M 步线性衰减
shanten_reward_min_ratio: 0.3           # 衰减到原始值的 30%
shanten_fan_bonus_scale: 0.3            # 番数加权
```

---

## 4. Phase C: PBT 超参搜索

**文件**: `python/blood/training/pbt.py`

### PBTController

管理 N 个并行训练实例，周期性评估、选择和变异超参数。

#### 搜索空间 (10 个可调参数)

```python
PBT_SEARCH_SPACE = {
    "exploration_loss_coeff": (0.005, 0.05),
    "oracle_distill_weight": (0.005, 0.1),
    "oracle_ce_weight": (0.01, 0.2),
    "reward_tsumo_bonus": (0.02, 0.2),
    "reward_deal_in_penalty": (0.01, 0.1),
    "reward_safe_discard": (0.0, 0.03),
    "reward_shanten_progress": (0.001, 0.01),
    "reward_rank_bonus": (0.05, 0.4),
    "ppo_clip_ratio": (0.1, 0.25),
    "learning_rate": (3e-5, 3e-4),
}
```

#### 核心流程

```
initialize_population(base_hyperparams)
  → 对每个 member: 在搜索空间内随机扰动 (factor ∈ [1/perturb, perturb])
  → JSON 持久化到 pbt_runs/pbt_state.json

step()  (每 eval_every 步调用)
  → 按 Elo 降序排列
  → Exploit: bottom fraction 复制 top fraction 的权重
  → Explore: 变异超参数 (×1/1.2, ×1.0, ×1.2 三选一)
```

#### 默认配置

```python
PBTController(
    population_size=4,
    eval_every=1_000_000,
    exploit_fraction=0.2,
    perturb_factor=1.2,
    work_dir="pbt_runs",
)
```

#### 启动脚本

**文件**: `scripts/run_pbt.sh`

```bash
./scripts/run_pbt.sh [population_size] [base_config]
# 默认: population_size=4, base_config=configs/elite.yaml
```

---

## 5. 实施路线图

```
Phase A (搜索与评估)          Phase B (架构)           Phase C (超参)
┌─────────────────────┐    ┌──────────────────┐    ┌──────────────┐
│ A1. ISMCE 深化       │    │ B1. TurnAttention│    │ C1. PBT      │
│   - rollout_depth=8  │    │   - 零初始化      │    │   - 10 参数   │
│   - num_worlds=96    │    │   - BPTT 支持     │    │   - JSON 持久 │
│   - 番数估算         │    │   - 推理 memory   │    │   - 4 members │
│                      │    │                  │    │              │
│ A2. 防守精细化       │    │ B2. 训练扩展      │    │              │
│   - danger_enhanced  │    │   - 200M steps   │    │              │
│   - 对手行为模型     │    │   - cosine 退火   │    │              │
│                      │    │   - 向听衰减      │    │              │
│ A3. 对手推断         │    │                  │    │              │
│   - OpponentPredictor│    │                  │    │              │
│   - sample_informed  │    │                  │    │              │
│                      │    │                  │    │              │
│ A4. 观测扩展 473ch   │    │                  │    │              │
│   - Section 14 (6ch) │    │                  │    │              │
│   - Section 15 (3ch) │    │                  │    │              │
└─────────────────────┘    └──────────────────┘    └──────────────┘
         ↓                          ↓                      ↓
    已实现到代码              已实现到代码             已实现到代码
    默认启用                 默认关闭(需显式启用)      需手动运行
```

### 依赖关系

- A4 (观测扩展 473ch) 是所有其他改进的前提 — 模型必须重新训练
- A3 (OpponentPredictor) 依赖 Oracle 标签 → 需在 competitive_distill 之后启用
- B1 (TurnAttention) 独立于 Phase A，可在任何阶段启用
- C1 (PBT) 在 elite 阶段运行，依赖 A1-A4 和 B2 的基础配置

### 启用方式

```yaml
# 启用 OpponentHandPredictor
opponent_predictor_enabled: true
opponent_predictor_conv_ch: 128
opponent_predictor_num_blocks: 6
opponent_predictor_weight: 0.1

# 启用 TurnAttention
turn_attention_enabled: true
turn_attention_heads: 4
```

---

## 6. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 473ch 观测不兼容旧检查点 | 高 | 必须从 warmup 重新训练完整流水线 |
| ISMCE depth=8 计算开销翻倍 | 中 | 仅推理时启用；训练时不使用 ISMCE |
| TurnAttention 训练不稳定 | 中 | 零初始化 residual 确保平滑过渡 |
| OpponentPredictor 特征提取不完整 | 中 | `_opponent_hand_loss` 中部分通道使用 zeros placeholder (标注 TODO) |
| `sample_world_informed` 未连通 | 低 | Rust 端已实现，Python 调用路径待补全 |
| PBT 4 members 搜索空间有限 | 低 | 可扩展 population_size；搜索空间已覆盖关键参数 |
| 200M 步训练资源需求 | 中 | 16 workers × 16 envs = 256 并行环境 |

---

## 7. 验证计划

### 7.1 单元验证

```bash
# Rust 引擎测试 (含 ISMCE)
cd blood-v2 && cargo test --release

# Python 测试
cd blood-v2 && PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}" python -m pytest tests/ -v
```

### 7.2 集成验证

```bash
# 冒烟测试: 端到端流水线
cd blood-v2 && ./scripts/manage.sh train smoke_test

# 验证观测通道数
python -c "from blood.consts import NUM_STUDENT_CHANNELS, NUM_ORACLE_CHANNELS; \
           assert NUM_STUDENT_CHANNELS == 473; assert NUM_ORACLE_CHANNELS == 525"
```

### 7.3 性能验证

| 检查项 | 方法 | 预期 |
|--------|------|------|
| ISMCE depth=8 延迟 | `ismce_evaluate` 基准测试 | < 50ms/候选牌 (96 worlds) |
| TurnAttention 训练收敛 | TensorBoard entropy 曲线 | 不低于 entropy_floor (0.009) |
| OpponentPredictor 精度 | `opponent_pred_loss` 监控 | BCE < 0.3 after 10M steps |
| 整体 Elo | Arena 评估 vs RuleBot | 持续上升，无回退 |

### 7.4 文件清单

| 文件 | 改动类型 |
|------|----------|
| `crates/engine/src/consts.rs` | 修改: 473ch, Section 14/15 offsets |
| `crates/engine/src/algo/ismce.rs` | 修改: RolloutResult, IsmceScore, depth=8, 防守 rollout, sample_world_informed |
| `crates/engine/src/obs/student.rs` | 修改: Section 14 (6ch) + Section 15 (3ch) |
| `crates/pybind/src/ismce_py.rs` | 修改: 7-tuple 返回, ismce_evaluate_full |
| `python/blood/consts.py` | 修改: 473/525 同步 |
| `python/blood/model/opponent_model.py` | **新增**: OpponentHandPredictor |
| `python/blood/model/factory.py` | 修改: TurnAttention, OpponentPredictor 集成 |
| `python/blood/model/inference.py` | 修改: TurnAttention 推理支持 |
| `python/blood/eval/ismce.py` | 修改: 7-tuple 解析, 新评分公式 |
| `python/blood/training/losses.py` | 修改: _opponent_hand_loss |
| `python/blood/training/pbt.py` | **新增**: PBTController |
| `python/blood/cfg.py` | 修改: 新增配置参数 |
| `configs/elite.yaml` | 修改: 200M steps, 96 worlds, depth 8, Phase A/B 参数 |
| `scripts/run_pbt.sh` | **新增**: PBT 启动脚本 |
