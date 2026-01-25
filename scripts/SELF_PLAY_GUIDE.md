# 自博弈训练指南

本指南介绍如何从零开始进行自博弈训练（无需人类对战数据）。

## 自博弈训练原理

自博弈训练流程：

```
1. 初始化模型（随机或预训练）
   ↓
2. 使用当前模型进行自对局生成数据
   ↓
3. 使用生成的数据训练模型
   ↓
4. 使用训练后的模型作为新的 baseline
   ↓
5. 重复步骤 2-4
```

## 准备工作

### 1. 创建配置文件

```bash
cd mortal
cp config.example.toml config.toml
```

### 2. 配置自博弈训练

编辑 `config.toml`，关键配置如下：

```toml
[control]
version = 4
online = false  # 离线训练模式
state_file = 'mortal.pth'  # 模型保存路径
best_state_file = 'best.pth'  # 最佳模型路径
tensorboard_dir = 'logs'  # TensorBoard 日志目录
device = 'cuda:0'  # 或 'cpu'

# 训练参数
batch_size = 512
opt_step_every = 1
save_every = 400
test_every = 20000
submit_every = 400

# 自对局参数
[train_play.default]
games = 800  # 每次自对局局数（建议 800-2000）
log_dir = 'train_play'  # 自对局日志保存目录
boltzmann_epsilon = 0.005  # 探索率（初期可以设置更高，如 0.01）
boltzmann_temp = 0.05  # 温度参数
top_p = 1.0
repeats = 1

# 数据集配置（自博弈模式下，数据来自 train_play）
[dataset]
# 不需要设置 globs，系统会自动使用 train_play 生成的数据
globs = []  # 留空，或指向 train_play 目录
file_index = 'file_index.pth'
file_batch_size = 15
reserve_ratio = 0.0
num_workers = 4  # 根据 CPU 核心数调整
num_epochs = 1
enable_augmentation = false
augmented_first = false

# Baseline 配置（用于自对局）
[baseline.train]
device = 'cuda:0'
enable_compile = false
state_file = 'baseline.pth'  # Baseline 模型路径

[baseline.test]
device = 'cuda:0'
enable_compile = false
state_file = 'baseline.pth'
```

### 3. 初始化模型

#### 选项 A：从随机模型开始（推荐）

系统会自动创建随机初始化的模型。只需确保配置文件中的路径正确：

```toml
[control]
state_file = 'mortal.pth'  # 如果文件不存在，会自动创建随机模型
best_state_file = 'best.pth'

[baseline.train]
state_file = 'baseline.pth'  # 第一次训练时，baseline 会使用当前模型
```

#### ⚠️ 重要：GRP 模型

**GRP 模型需要先创建或训练**，因为训练主模型时需要 GRP 来计算奖励。

**快速创建随机 GRP 模型**：
```bash
cd mortal
python ../scripts/create-initial-grp.py
```

**或使用自对局数据训练 GRP**：
```bash
# 1. 先进行少量自对局生成数据
# 2. 训练 GRP
cd mortal
python train_grp.py
```

详见：`scripts/GRP_TRAINING_GUIDE.md`

#### 选项 B：使用预训练模型

如果有预训练模型，设置路径：

```toml
[control]
state_file = '/path/to/pretrained.pth'
```

### 4. 配置数据路径

**关键**：需要将数据集路径指向自对局生成的数据目录：

```toml
[dataset]
# 指向 train_play 目录（自对局数据保存位置）
globs = ['train_play/**/*.json.gz']
file_index = 'file_index.pth'
```

**如果 train_play 目录还没有数据**：
- 可以先设置 `globs = []`（空列表）
- 训练脚本会自动进行自对局生成初始数据
- 或者手动运行一次自对局（见下方）

## 开始训练

### 第一次训练（从零开始）

1. **确保配置文件正确**

```bash
cd mortal
# 检查 config.toml 已创建并配置
```

2. **开始训练**

```bash
./scripts/blood-train.sh offline
```

训练流程：

1. **初始化阶段**：
   - 如果 `state_file` 不存在，创建随机初始化的模型
   - 如果 `baseline.train.state_file` 不存在，使用当前模型作为 baseline

2. **第一轮自对局**：
   - 使用当前模型（可能是随机的）进行自对局
   - 生成对局数据保存到 `train_play/` 目录

3. **训练阶段**：
   - 使用生成的数据训练模型
   - 定期保存检查点

4. **迭代循环**：
   - 每 `save_every` 步后，进行新的自对局
   - 使用训练后的模型作为新的 baseline
   - 继续训练

### 训练过程说明

训练会自动执行以下循环：

```
步骤 0: 初始化模型
   ↓
步骤 1: 自对局生成数据（train_play）
   ↓
步骤 2: 训练模型（使用生成的数据）
   ↓
步骤 3: 保存模型检查点
   ↓
步骤 4: 更新 baseline（使用当前模型）
   ↓
步骤 5: 重复步骤 1-4
```

## 关键配置说明

### 自对局参数

```toml
[train_play.default]
games = 800  # 每次自对局局数
```

- **初期**：可以设置较小值（如 400-800）快速迭代
- **后期**：可以设置较大值（如 2000-4000）生成更多数据

```toml
boltzmann_epsilon = 0.005  # 探索率
```

- **初期**：可以设置更高（如 0.01-0.02）增加探索
- **后期**：可以降低（如 0.001-0.005）更注重利用

### Baseline 更新策略

系统会自动使用当前训练中的模型作为 baseline。这意味着：

- **第一轮**：使用随机模型 vs 随机模型（或预训练模型）
- **后续轮**：使用训练后的模型 vs 之前的模型版本

这种策略可以：
- 逐步提升模型水平
- 避免模型退化
- 保持训练稳定性

## 监控训练

### 1. TensorBoard

```bash
tensorboard --logdir mortal/logs
```

查看指标：
- `loss/dqn_loss` - Q 网络损失
- `loss/cql_loss` - CQL 损失（离线训练）
- `loss/next_rank_loss` - 排名预测损失
- `hparam/lr` - 学习率
- `q_predicted` / `q_target` - Q 值分布

### 2. 日志文件

训练日志会输出到控制台，包括：
- 自对局进度
- 训练进度
- 模型保存信息
- 测试结果（如果配置了）

### 3. 检查点文件

- `mortal.pth` - 最新模型
- `best.pth` - 最佳模型（根据测试结果）
- `train_play/` - 自对局数据

## 进阶配置

### 多阶段训练

可以分阶段调整参数：

**阶段 1：快速探索（初期）**
```toml
[train_play.default]
games = 400
boltzmann_epsilon = 0.02
boltzmann_temp = 0.1
```

**阶段 2：稳定训练（中期）**
```toml
[train_play.default]
games = 800
boltzmann_epsilon = 0.005
boltzmann_temp = 0.05
```

**阶段 3：精细优化（后期）**
```toml
[train_play.default]
games = 2000
boltzmann_epsilon = 0.001
boltzmann_temp = 0.01
```

### 使用多个 Baseline

可以配置多个 baseline 模型进行对局，增加数据多样性。

## 常见问题

### 1. 训练初期模型表现很差

**正常现象**：
- 随机模型初期表现很差是正常的
- 需要几轮迭代才能看到改善
- 建议先进行 5-10 轮快速迭代（games=400）

### 2. 自对局数据太少

**解决方案**：
- 增加 `games` 参数
- 增加 `repeats` 参数（重复使用相同配置）
- 检查 `log_dir` 是否有数据生成

### 3. 训练速度慢

**优化建议**：
- 使用 GPU：`device = 'cuda:0'`
- 启用混合精度：`enable_amp = true`
- 增加 `num_workers`（但不要超过 CPU 核心数）
- 启用 `enable_cudnn_benchmark = true`

### 4. 内存不足

**解决方案**：
- 减小 `batch_size`
- 减小 `file_batch_size`
- 设置 `num_workers = 1`
- 定期清理旧的训练数据

### 5. Baseline 模型不存在

**解决方案**：
- 第一次训练时，系统会自动使用当前模型作为 baseline
- 确保 `[baseline.train]` 和 `[baseline.test]` 的 `state_file` 路径正确
- 如果路径不存在，系统会使用 `[control].state_file` 作为 baseline

## 训练检查清单

- [ ] 配置文件已创建（`mortal/config.toml`）
- [ ] 模型保存路径已设置
- [ ] Baseline 路径已设置（或留空让系统自动处理）
- [ ] 自对局参数已配置（`games`, `log_dir` 等）
- [ ] GPU/CPU 设备配置正确
- [ ] 有足够的磁盘空间（自对局数据可能很大）
- [ ] TensorBoard 可以访问（可选）

## 预期训练时间

- **初期迭代**：每轮约 1-2 小时（取决于硬件和 games 数量）
- **完整训练**：可能需要数天到数周
- **建议**：先进行小规模测试（games=200），确认流程正常

## 下一步

训练一段时间后：

1. **评估模型**：使用测试集评估性能
2. **1v3 测试**：运行 `one_vs_three.py` 进行对战测试
3. **调整参数**：根据结果调整训练参数
4. **继续迭代**：持续训练直到达到目标性能

---

## 快速开始命令

```bash
# 1. 准备配置
cd mortal
cp config.example.toml config.toml
# 编辑 config.toml（至少设置 device 和路径）

# 2. 开始自博弈训练
./scripts/blood-train.sh offline

# 3. 监控训练（另一个终端）
tensorboard --logdir mortal/logs
```

---

**注意**：自博弈训练是一个迭代过程，需要耐心。初期模型表现可能很差，但随着训练进行会逐步改善。
