# 血战到底麻将训练系统部署指南 (Bloody Battle Mahjong Training Deployment Guide)

本文档旨在指导如何部署、配置并运行血战到底麻将的强化学习训练系统。

## 1. 环境准备 (Prerequisites)

### 1.1 系统要求
- **OS**: Linux (推荐 Ubuntu 20.04+) 或 macOS
- **Python**: 3.9+
- **Rust**: Latest stable (用于编译 `libblood`)
- **CUDA**: 11.x+ (推荐使用 GPU 训练)

### 1.2 核心组件编译
训练开始前，必须编译 Rust 核心库 `libblood` 并生成 Python 绑定。

```bash
# 进入 libblood 目录
cd libblood

# 编译并生成 wheel 包 (mortal 目录下)
# 注意：生成的 whl 文件名可能因平台而异，请根据实际输出调整
python3 -m maturin build --release --out ../mortal

# 安装生成的包
pip3 install --force-reinstall ../mortal/libblood-*.whl
```

## 2. 配置文件 (Configuration)

系统提供了一个配置检查工具，确保您在开始训练前配置正确。

### 2.1 初始化配置
```bash
cd mortal
# 复制示例配置
cp config.example.toml config.toml
```

### 2.2 编辑配置 (`config.toml`)
重点修改以下字段：
- `[control].device`: 设置为 `'cuda:0'` (GPU) 或 `'cpu'`。
- `[control].version`: **必须设置为 4** (启用血战到底优化)。
- `[dataset].globs`: 设置训练数据路径 (自博弈模式下会自动生成)。
- `[train_play.default].log_dir`: 设置自对局日志保存路径。

### 2.3 验证配置
使用提供的脚本自动检查配置项和环境依赖：
```bash
./scripts/check-config.sh
```
如果输出全绿 (Green)，则说明环境准备就绪。

## 3. 训练模式 (Training Modes)

本系统支持 **离线训练 (Offline)** 和 **在线训练 (Online)** 两种模式。对于初学者或单机环境，推荐使用 **离线训练 + 自博弈**。

### 3.1 离线训练 (Offline Training) - **推荐**
适用于单机环境，系统会自动进行“训练 -> 自博弈生成数据 -> 训练”的闭环迭代。

**启动命令**:
```bash
./scripts/blood-train.sh offline
```

**流程说明**:
1. 加载现有数据（如果配置了 dataset）。
2. 如果无数据，自动启动自博弈 (Self-Play) 生成初始批次。
3. 执行 Epoch 训练。
4. 训练结束后，执行新一轮自博弈生成数据。
5. 循环步骤 3-4。

### 3.2 在线训练 (Online Training)
适用于多机分布式环境，支持大规模并发生成数据。需要启动 Server 和多个 Client。

**步骤 1: 启动参数服务器**
```bash
./scripts/blood-server.sh start
```

**步骤 2: 启动训练客户端 (可多台机器)**
```bash
cd mortal
python3 client.py
```

**步骤 3: 启动训练主进程**
```bash
./scripts/blood-train.sh online
```

## 4. 监控与评估 (Monitoring & Evaluation)

### 4.1 TensorBoard 监控
训练过程中，损失函数、Q值分布、自对局胜率等指标会实时记录。

```bash
./scripts/blood-tensorboard.sh
```
访问 `http://localhost:6006` 查看图表。

**关键指标**:
- `test_play/avg_pt`: 平均得点 (越高越好)。
- `test_play/avg_ranking`: 平均排名 (越低越好，范围 1.0-4.0)。
- `loss/dqn_loss`: TD 误差。

### 4.2 日志回放 (Replay Viewer)
可视化查看 AI 的打牌表现。

**启动回放服务器**:
1. 确保日志文件 (JSON) 已生成 (在 `train_play` 或 `test_play` 目录下)。
2. 启动 Viewer:
   ```bash
   cd log-viewer
   python3 -m http.server 8000
   ```
3. 浏览器访问 `http://localhost:8000/replay.html?log=../(path_to_json)`。

## 5. 推理与模型使用 (Inference)

训练好的模型保存为 `.pth` 文件 (默认 `mortal/mortal.pth`)。
使用 `mortal/mortal.py` 或 `scripts/blood-mortal.sh` 加载模型进行推理。

```bash
# 使用脚本启动 AI 玩家 (ID: 0-3)
./scripts/blood-mortal.sh 0
```
