# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

Blood 是一个面向四川血战麻将的深度强化学习系统，改编自 Mortal 项目（日本立直麻将），针对四川血战玩法重新设计。当前活跃开发在 `blood-v2/`；V1 系统（`mortal/` + `libblood/`）为遗留代码。

权威规则文档为 `rules.md`。与日麻的核心区别：108 张牌（无字牌）、必须定缺、无吃、3 人和牌即结束、无流局。

## 构建与开发命令

### V2（活跃开发）

```bash
# 构建 Rust 引擎 + PyO3 绑定
cd blood-v2 && maturin develop --release

# 或使用管理脚本
cd blood-v2 && ./scripts/manage.sh build

# 所有 Python 命令需设置 PYTHONPATH
export PYTHONPATH="$(pwd)/blood-v2/python:${PYTHONPATH:-}"
```

### V1（遗留）

```bash
cargo build --release  # 从根 workspace 构建 libblood + exe-wrapper
conda env create -f environment.yml  # conda 环境名: mortal
```

## 测试命令

```bash
# V2 Rust 测试
cd blood-v2 && cargo test --release

# V2 Python 测试
cd blood-v2 && PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}" python -m pytest tests/ -v

# V2 冒烟测试（端到端流水线验证）
cd blood-v2 && ./scripts/manage.sh train smoke_test

# V1 Rust 测试（CI 使用）
cargo test --workspace --no-default-features --features flate2/zlib
```

## 代码检查与格式化

```bash
# Rust
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -Dwarnings

# Python
ruff check blood-v2/python/

# 拼写检查（配置见 typos.toml）
typos
```

## 训练流水线

统一通过 `blood-v2/scripts/manage.sh` 管理：

```bash
./scripts/manage.sh train warmup                     # 阶段1: RuleBot 对手, 2M 步
./scripts/manage.sh train warmup_transition --resume  # 阶段1.5: gamma/lr 渐变, 500K 步
./scripts/manage.sh train competitive --resume        # 阶段2a: 自对弈, 2.5M 步
./scripts/manage.sh train distill --resume            # 阶段2b: Oracle 蒸馏
./scripts/manage.sh train elite --resume              # 阶段3: RTPA+ISMCE, 50M 步
./scripts/manage.sh train pipeline                    # 完整 5 阶段流水线
./scripts/manage.sh monitor                           # TensorBoard 监控
./scripts/manage.sh eval                              # 模型评估
./scripts/manage.sh status                            # 检查点状态
./scripts/manage.sh export --checkpoint <path>        # ONNX 导出
```

## 架构

### 双语言系统

- **Rust** (`blood-v2/crates/engine/`): 游戏引擎、观测编码、和牌判定、向听计算、SP Table、ISMCE 搜索
- **PyO3 绑定** (`blood-v2/crates/pybind/`): Rust↔Python 桥接，编译为 `blood._engine`
- **Python** (`blood-v2/python/blood/`): 神经网络、PPO 训练（基于 Sample Factory 2）、评估、联赛系统

### 神经网络（约 20M 参数）

SuitAwareResNetEncoder: 输入 (470×27) → SuitAwareConv1d → 4 个 segment [5×BottleneckBlock + TileAttention] → LSTM(2 层, 512) → 解耦 Actor-Critic 头 + 辅助头。

核心设计：SuitAwareConv1d 将 27 个牌位重塑为 3×9（万/筒/条），强制花色隔离。TileAttention 是唯一的跨花色交互机制。

### 观测空间

- 学生观测: 473×27 张量（手牌、游戏上下文、定缺、牌河、可见牌、防守、SP Table、番种配置、現物安全牌）
- Oracle 观测: 额外 52 通道（对手手牌、危险度、SP 摘要）→ 525×27

### 动作空间

34 个离散动作：0-26 打牌、27 碰、28-30 杠（明杠/暗杠/加杠）、31 和、32-33 定缺选择。

### 训练

5 阶段课程学习，使用 Sample Factory 2 PPO + Oracle 价值蒸馏。通过猴子补丁 `Learner._calculate_losses` 注入自定义损失。联赛系统最多 50 个检查点池 + Elo 加权采样。

### 评估

- RTPA: 实时策略调整，基于 6 个对手听牌信号
- ISMCE: 信息集蒙特卡洛评估，约束采样 + 防守感知 rollout
- Arena: 1v3 评估，随机座位 + Elo 追踪

## 关键常量

| 常量 | 值 | 位置 |
|------|-----|------|
| `NUM_STUDENT_CHANNELS` | 473 | `crates/engine/src/consts.rs` |
| `NUM_ORACLE_CHANNELS` | 525 | `crates/engine/src/consts.rs` |
| `ACTION_SPACE` | 34 | `crates/engine/src/consts.rs` |
| `TILE_TYPES` | 27 | `crates/engine/src/consts.rs` |
| `REWARD_NORM` | 32000 | configs |

## 工作范围规则

- 所有规则逻辑必须以 `rules.md` 为准，并与引擎实现保持一致
- 不引入日本麻将特有规则（立直、宝牌、流局、吃牌等）
- 涉及计分/和牌/番数逻辑时，参考 `rules.md` 和 `blood-v2/crates/engine/src/algo/`
