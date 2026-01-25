# 血战到底麻将完整训练流程（从零开始）

## 📋 文档概述

本文档提供**从零开始**训练血战到底麻将AI的**完整流程**，适用于**没有任何人类对战数据**的情况。

**核心特点**：
- ✅ 支持自博弈训练（Self-Play）
- ✅ 无需准备任何数据
- ✅ 系统自动生成数据并迭代改进
- ✅ 完整的步骤说明和验证方法

---

## 🎯 训练流程总览

```
准备阶段
├─ 1. 环境准备
├─ 2. 配置文件设置
├─ 3. 创建GRP模型（必需）
└─ 4. 验证配置

训练阶段
├─ 5. 开始自博弈训练
├─ 6. 监控训练进度
└─ 7. 迭代优化
```

---

## 📦 第一部分：环境准备

### 1.1 系统要求

**硬件要求**：
- **CPU**：多核（建议4核以上）
- **内存**：至少8GB（建议16GB+）
- **GPU**：可选但强烈推荐（NVIDIA GPU，支持CUDA）
- **存储**：至少10GB可用空间（用于模型和数据）

**软件要求**：
- Python 3.8+
- PyTorch（支持CUDA如果使用GPU）
- Rust（用于编译libblood）
- 其他依赖见项目README

### 1.2 项目结构

确保项目结构如下：

```
blood/
├── libblood/          # Rust核心库
├── mortal/            # Python训练代码
│   ├── config.toml    # 配置文件（需要创建）
│   └── ...
├── scripts/           # 工具脚本
│   ├── create-initial-grp.py
│   ├── check-config.sh
│   └── ...
└── ...
```

### 1.3 编译Rust库

```bash
cd libblood
cargo build --lib --release
```

**验证**：
```bash
cd ..
python -c "import libblood; print('libblood available')"
# 如果成功，说明编译完成
```

---

## ⚙️ 第二部分：配置文件设置

### 2.1 创建配置文件

```bash
cd mortal
cp config.example.toml config.toml
```

### 2.2 最小配置（自博弈训练）

编辑 `mortal/config.toml`，至少修改以下部分：

#### 2.2.1 控制配置

```toml
[control]
version = 4                    # 模型版本
online = false                 # 离线训练模式（自博弈）
state_file = '/data/mortal/mortal.pth'           # 主模型保存路径
best_state_file = '/data/mortal/best.pth'        # 最佳模型路径
tensorboard_dir = '/data/mortal/logs'            # TensorBoard日志目录
device = 'cuda:0'              # 设备：'cuda:0'（GPU）或 'cpu'
enable_cudnn_benchmark = true  # 启用CUDNN基准（GPU）
enable_amp = true              # 启用混合精度训练（GPU）
enable_compile = false         # 模型编译（可选）

# 训练参数
batch_size = 512               # 批次大小（根据GPU内存调整）
opt_step_every = 1             # 每N步优化一次
save_every = 400               # 每N步保存一次（保存后会进行新的自对局）
test_every = 20000             # 每N步测试一次
submit_every = 400             # 每N步提交一次（在线训练用）
```

#### 2.2.2 自对局配置

```toml
[train_play.default]
games = 800                    # 每次自对局局数
                                # 建议：初期400-800（快速迭代），后期800-2000（更多数据）
log_dir = '/data/mortal/train_play'  # 自对局数据保存目录
boltzmann_epsilon = 0.005      # 探索率（初期可以设置更高，如0.01）
boltzmann_temp = 0.05         # 温度参数
top_p = 1.0                    # Top-p采样
repeats = 1                    # 重复次数
```

#### 2.2.3 数据集配置

```toml
[dataset]
# 自博弈训练：指向自对局生成的数据目录
globs = ['/data/mortal/train_play/**/*.json.gz']  # 数据文件路径（支持glob模式）
file_index = '/data/mortal/file_index.pth'        # 文件索引缓存
file_batch_size = 15           # 文件批次大小（根据内存调整）
reserve_ratio = 0.0            # 保留比例
num_workers = 4                # 数据加载工作进程数（建议为CPU核心数）
player_names_files = []        # 玩家名称文件（自博弈训练留空）
num_epochs = 1                 # 每个数据集的训练轮数
enable_augmentation = false    # 数据增强
augmented_first = false        # 是否先使用增强数据
```

#### 2.2.4 Baseline配置

```toml
[baseline.train]
device = 'cuda:0'              # Baseline设备
name = 'baseline'              # Baseline名称
state_file = '/data/mortal/baseline.pth'  # Baseline模型路径
                                # 如果不存在，系统会自动使用当前模型
stochastic_latent = false      # 是否使用随机潜在变量
enable_compile = false         # 是否编译模型
enable_amp = true              # 是否启用混合精度
enable_rule_based_agari_guard = true  # 是否启用规则基础的和牌保护

[baseline.test]
device = 'cuda:0'
name = 'baseline'
state_file = '/data/mortal/baseline.pth'
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true
```

#### 2.2.5 GRP配置

```toml
[grp]
# ⚠️ 重要：GRP模型需要先创建，训练主模型时需要GRP来计算奖励
state_file = '/data/mortal/grp.pth'

[grp.network]
hidden_size = 64               # GRU隐藏层大小
num_layers = 2                 # GRU层数

[grp.control]
device = 'cuda:0'              # GRP训练设备
enable_cudnn_benchmark = false # 是否启用CUDNN基准
tensorboard_dir = '/data/mortal/grp_logs'  # GRP TensorBoard目录
batch_size = 512               # GRP训练批次大小
save_every = 2000              # 每N步保存一次
val_steps = 400                # 验证步数

[grp.dataset]
# 自博弈训练：可以指向自对局生成的数据
train_globs = [
    '/data/mortal/train_play/**/*.json.gz',
]
val_globs = [
    '/data/mortal/test_play/**/*.json.gz',  # 如果有测试数据
]
file_index = '/data/mortal/grp_file_index.pth'
file_batch_size = 50

[grp.optim]
lr = 1e-5                      # 学习率
```

#### 2.2.6 其他配置

```toml
# 环境配置
[env]
gamma = 1                      # 折扣因子
pts = [6.0, 4.0, 2.0, 0.0]     # 排名对应的分数（1位、2位、3位、4位）

# ResNet模型配置
[resnet]
conv_channels = 192            # 卷积通道数
num_blocks = 40                # ResNet块数

# CQL配置
[cql]
min_q_weight = 0.1            # CQL最小Q值权重

# 辅助网络配置
[aux]
next_rank_weight = 0.1        # 下一排名预测权重

# 优化器配置
[optim]
lr = 1e-4                      # 学习率
eps = 1e-8                     # 优化器epsilon
betas = [0.9, 0.999]          # 优化器betas
weight_decay = 1e-4           # 权重衰减
max_grad_norm = 1.0           # 最大梯度范数

[optim.scheduler]
warmup_steps = 1000           # 预热步数
max_steps = 1000000           # 最大步数
```

### 2.3 使用配置脚本（推荐）

如果使用服务器环境（数据目录在 `/data/mortal`，源码在 `/data/blood`）：

```bash
./scripts/setup-config.sh /data/mortal /data/blood
```

这会自动：
- 创建必要的目录
- 复制配置文件
- 替换路径占位符

### 2.4 验证配置

```bash
./scripts/check-config.sh
```

**检查项**：
- ✅ 配置文件存在
- ✅ TOML语法正确
- ✅ 关键配置项存在
- ✅ 路径配置正确
- ✅ 设备配置正确
- ⚠️ libblood模块可用（如果未编译会警告，但可以继续）

---

## 🎲 第三部分：创建GRP模型（必需）

### 3.1 为什么需要GRP模型？

**GRP（Global Ranking Prediction）** 用于：
- 预测最终排名概率
- 计算训练数据的奖励（reward）
- 训练主模型时必需

**如果没有GRP模型，训练会报错！**

### 3.2 创建随机GRP模型

```bash
cd mortal
python ../scripts/create-initial-grp.py
```

**预期输出**：
```
============================================================
创建初始 GRP 模型
============================================================
GRP 模型文件: /data/mortal/grp.pth
网络配置: {'hidden_size': 64, 'num_layers': 2}

创建随机初始化的 GRP 模型...
============================================================
✓ GRP 模型创建成功！
============================================================
文件路径: /data/mortal/grp.pth

注意:
  - 这是一个随机初始化的模型，预测可能不准确
  - 建议先用少量数据训练 GRP，或使用自对局数据训练
  - 训练 GRP: cd mortal && python train_grp.py

下一步:
  1. 进行初始自对局生成数据（可选）
  2. 训练 GRP 模型（推荐，使用自对局数据）
  3. 开始主模型训练: ./scripts/blood-train.sh offline
============================================================
```

### 3.3 验证GRP模型

```bash
# 检查文件是否存在
ls -lh /data/mortal/grp.pth

# 应该看到类似输出：
# -rw-r--r-- 1 user user 2.5M Jan 25 10:00 /data/mortal/grp.pth
```

---

## 🚀 第四部分：开始训练

### 4.1 训练模式选择

#### 模式A：离线自博弈训练（推荐，无数据）

```bash
./scripts/blood-train.sh offline
```

**特点**：
- ✅ 自动进行自对局生成数据
- ✅ 自动训练模型
- ✅ 自动迭代改进
- ✅ 无需准备数据

#### 模式B：在线训练（需要服务器）

```bash
./scripts/blood-server.sh start  # 启动服务器
./scripts/blood-train.sh online  # 启动客户端
```

**特点**：
- 需要训练服务器
- 多个客户端可以连接
- 适合分布式训练

### 4.2 训练流程详解

#### 第一次运行

**系统自动执行**：

1. **初始化模型**
   ```
   [INFO] mortal.pth not found, creating random model...
   [INFO] baseline.pth not found, using current model as baseline
   ```

2. **检测数据目录**
   ```
   [INFO] train_play directory is empty, starting self-play...
   ```

3. **进行自对局**
   ```
   [INFO] Running self-play: 800 games
   [INFO] Self-play progress: 100/800 games...
   [INFO] Self-play completed: 800 games, 800 files generated
   ```

4. **开始训练**
   ```
   [INFO] Building file index...
   [INFO] File list size: 800
   [INFO] Starting training...
   [INFO] Training step 1/400...
   ```

5. **保存检查点**
   ```
   [INFO] Saving checkpoint at step 400
   [INFO] Updating baseline model...
   [INFO] Starting new self-play session...
   ```

#### 后续运行

**系统自动执行**：

1. **加载模型**
   ```
   [INFO] Loading model from mortal.pth
   [INFO] Loading baseline from baseline.pth
   ```

2. **进行自对局**（使用训练后的模型）
   ```
   [INFO] Starting self-play: 800 games
   [INFO] Self-play completed: 800 games
   ```

3. **继续训练**
   ```
   [INFO] Starting training...
   [INFO] Training step 401/800...
   ```

### 4.3 训练参数说明

#### 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `games` | 每次自对局局数 | 初期：400-800<br>后期：800-2000 |
| `save_every` | 每N步保存一次 | 400（保存后会进行新的自对局） |
| `batch_size` | 批次大小 | GPU：512<br>CPU：128-256 |
| `num_workers` | 数据加载进程数 | CPU核心数（但不超过8） |
| `boltzmann_epsilon` | 探索率 | 初期：0.01<br>后期：0.005 |

#### 调整建议

**初期（前10轮）**：
- `games = 400`（快速迭代）
- `boltzmann_epsilon = 0.01`（更多探索）

**中期（10-50轮）**：
- `games = 800`（平衡数据量和速度）
- `boltzmann_epsilon = 0.005`（减少探索）

**后期（50+轮）**：
- `games = 1200-2000`（更多数据）
- `boltzmann_epsilon = 0.005`（稳定策略）

---

## 📊 第五部分：监控训练

### 5.1 训练日志

训练会在控制台输出详细日志：

```
[INFO] Self-play: 800 games
[INFO] Training step 1/400, loss: 0.1234
[INFO] Training step 2/400, loss: 0.1200
...
[INFO] Saving checkpoint at step 400
[INFO] Avg rank: 2.500000
[INFO] Avg pt: 0.000000
```

### 5.2 TensorBoard监控

**启动TensorBoard**：

```bash
# 另一个终端
tensorboard --logdir /data/mortal/logs
```

**访问**：
- 浏览器打开：`http://localhost:6006`

**查看指标**：
- `loss/dqn_loss` - DQN损失
- `loss/cql_loss` - CQL损失
- `loss/next_rank_loss` - 排名预测损失
- `hparam/lr` - 学习率
- `q_predicted` / `q_target` - Q值分布

### 5.3 文件监控

**检查生成的文件**：

```bash
# 检查自对局数据
ls -lh /data/mortal/train_play/ | head -10

# 检查模型文件
ls -lh /data/mortal/*.pth

# 检查日志
ls -lh /data/mortal/logs/
```

### 5.4 训练状态检查

**检查训练是否正常**：

1. **数据生成正常**
   ```bash
   # 应该看到 .json.gz 文件
   ls /data/mortal/train_play/*.json.gz | wc -l
   ```

2. **模型保存正常**
   ```bash
   # 应该看到模型文件
   ls -lh /data/mortal/mortal.pth
   ls -lh /data/mortal/baseline.pth
   ```

3. **训练损失下降**
   - 查看TensorBoard或日志
   - 损失应该逐步下降

---

## 🔄 第六部分：训练迭代流程

### 6.1 完整迭代周期

```
┌─────────────────────────────────────┐
│  1. 自对局生成数据                    │
│     - 1个训练模型 vs 3个baseline模型 │
│     - 生成800局对局数据              │
│     - 保存到 train_play/            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. 训练模型                         │
│     - 加载生成的数据                 │
│     - 训练400步                      │
│     - 计算损失并更新模型             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. 保存检查点                       │
│     - 保存模型到 mortal.pth          │
│     - 更新 baseline.pth             │
│     - 记录训练指标                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. 开始新的迭代                     │
│     - 回到步骤1                      │
│     - 使用改进的模型                 │
└─────────────────────────────────────┘
```

### 6.2 训练阶段

#### 阶段1：初始化（第1轮）

- **模型状态**：随机初始化
- **表现**：很差（正常）
- **关注点**：确保训练流程正常

#### 阶段2：快速改进（第2-10轮）

- **模型状态**：开始学习
- **表现**：逐步改善
- **关注点**：训练损失是否下降

#### 阶段3：稳定提升（第11-50轮）

- **模型状态**：持续改进
- **表现**：明显提升
- **关注点**：排名提升趋势

#### 阶段4：精细优化（第50+轮）

- **模型状态**：接近收敛
- **表现**：稳定在高水平
- **关注点**：性能优化和超参数调优

---

## 🛠️ 第七部分：训练优化

### 7.1 定期重新训练GRP

**为什么需要**：
- 随着主模型改进，对局模式会变化
- 使用新的对局数据训练GRP可以提高预测准确性
- 更准确的GRP可以改善主模型的训练效果

**何时重新训练**：
- 初期：每5-10轮重新训练一次
- 后期：每10-20轮重新训练一次

**如何重新训练**：

```bash
cd mortal
python train_grp.py
# 或
../scripts/blood-train-grp.sh
```

**注意**：
- 确保有足够的自对局数据（至少几百局）
- 训练GRP需要一些时间（取决于数据量）

### 7.2 调整训练参数

#### 根据硬件调整

**GPU内存不足**：
```toml
batch_size = 256        # 减小批次大小
file_batch_size = 10   # 减小文件批次大小
num_workers = 2        # 减少工作进程
```

**CPU训练**：
```toml
device = 'cpu'
batch_size = 128
enable_amp = false
num_workers = 4        # 根据CPU核心数调整
```

**多GPU训练**：
- 需要修改代码支持DataParallel或DistributedDataParallel
- 当前版本不支持，需要自行实现

#### 根据训练阶段调整

**初期（快速迭代）**：
```toml
[train_play.default]
games = 400            # 减少对局数，加快迭代
boltzmann_epsilon = 0.01  # 增加探索

[control]
save_every = 200       # 更频繁保存
```

**后期（精细训练）**：
```toml
[train_play.default]
games = 1200           # 增加对局数，更多数据
boltzmann_epsilon = 0.005  # 减少探索

[control]
save_every = 400       # 正常保存频率
```

### 7.3 超参数调优

**建议的调优顺序**：

1. **学习率**（最重要）
   ```toml
   [optim]
   lr = 1e-4  # 尝试：5e-5, 1e-4, 2e-4
   ```

2. **批次大小**
   ```toml
   batch_size = 512  # 尝试：256, 512, 1024
   ```

3. **探索率**
   ```toml
   boltzmann_epsilon = 0.005  # 尝试：0.01, 0.005, 0.001
   ```

4. **模型架构**（高级）
   ```toml
   [resnet]
   conv_channels = 192  # 尝试：128, 192, 256
   num_blocks = 40      # 尝试：30, 40, 50
   ```

---

## 📈 第八部分：评估训练效果

### 8.1 关键指标

#### 训练指标

1. **损失函数**
   - `dqn_loss`：应该逐步下降
   - `cql_loss`：应该稳定
   - `next_rank_loss`：应该下降

2. **Q值分布**
   - `q_predicted` 和 `q_target` 应该接近
   - 分布应该合理（不过大或过小）

3. **学习率**
   - 应该按照调度器变化
   - 初期可能较高，后期降低

#### 游戏指标

1. **平均排名**
   - 目标：接近2.0（4人游戏的平均排名）
   - 初期可能接近2.5（随机）
   - 后期应该低于2.0（表现更好）

2. **平均分数**
   - 目标：高于0（排名分数）
   - 初期可能接近0或负值
   - 后期应该为正且持续增长

### 8.2 测试评估

**运行测试对局**：

系统会在 `test_every` 步后自动运行测试：

```
[INFO] Running test play: 3000 games
[INFO] Test play completed
[INFO] Avg rank: 1.850000
[INFO] Avg pt: 15.500000
```

**手动测试**：

```bash
# 使用测试脚本（如果有）
cd mortal
python test_model.py  # 需要实现
```

### 8.3 模型对比

**对比不同检查点**：

```bash
# 查看模型文件时间
ls -lth /data/mortal/mortal*.pth

# 对比不同版本的性能
# 查看TensorBoard中的历史记录
```

---

## 🐛 第九部分：常见问题与解决

### 9.1 训练启动问题

#### 问题1：GRP模型不存在

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: '/data/mortal/grp.pth'
```

**解决方法**：
```bash
cd mortal
python ../scripts/create-initial-grp.py
```

#### 问题2：配置文件不存在

**错误信息**：
```
FileNotFoundError: config.toml not found
```

**解决方法**：
```bash
cd mortal
cp config.example.toml config.toml
# 然后编辑配置文件
```

#### 问题3：libblood模块不可用

**错误信息**：
```
ModuleNotFoundError: No module named 'libblood'
```

**解决方法**：
```bash
# 编译Rust库
cd libblood
cargo build --release

# 安装Python绑定
cd ..
pip install -e .  # 如果有setup.py
# 或确保PYTHONPATH包含项目根目录
```

### 9.2 训练过程问题

#### 问题1：内存不足

**症状**：
- 训练过程中断
- OOM（Out of Memory）错误

**解决方法**：
```toml
batch_size = 256        # 减小批次大小
file_batch_size = 10    # 减小文件批次大小
num_workers = 2         # 减少工作进程
```

#### 问题2：训练速度慢

**症状**：
- 每步训练时间很长
- GPU利用率低

**解决方法**：
```toml
device = 'cuda:0'              # 确保使用GPU
enable_cudnn_benchmark = true   # 启用CUDNN基准
enable_amp = true               # 启用混合精度
num_workers = 4                 # 增加数据加载进程
```

#### 问题3：训练损失不下降

**症状**：
- 损失值很高且不下降
- 模型表现没有改善

**可能原因和解决方法**：

1. **学习率太高**
   ```toml
   [optim]
   lr = 5e-5  # 降低学习率
   ```

2. **批次大小太小**
   ```toml
   batch_size = 512  # 增加批次大小
   ```

3. **数据质量问题**
   - 检查自对局数据是否正常
   - 确保数据文件可以正常加载

4. **模型初始化问题**
   - 尝试重新创建模型
   - 检查模型参数是否合理

### 9.3 数据生成问题

#### 问题1：自对局数据为空

**症状**：
- `train_play` 目录为空
- 训练无法开始

**解决方法**：
- 系统应该自动检测并生成数据
- 如果未自动生成，手动运行：
  ```bash
  cd mortal
  python ../scripts/generate-initial-data.py
  ```

#### 问题2：自对局速度慢

**症状**：
- 自对局生成数据很慢
- 每局对局时间很长

**解决方法**：
```toml
[train_play.default]
games = 400  # 减少对局数（初期）

[control]
device = 'cuda:0'  # 确保使用GPU
```

### 9.4 模型保存问题

#### 问题1：模型文件损坏

**症状**：
- 无法加载模型
- 训练中断后无法恢复

**解决方法**：
- 检查磁盘空间
- 检查文件权限
- 定期备份模型文件

#### 问题2：无法恢复训练

**症状**：
- 训练中断后无法继续

**解决方法**：
- 检查 `mortal.pth` 是否存在
- 检查 `baseline.pth` 是否存在
- 确保配置文件路径正确

---

## 📝 第十部分：训练检查清单

### 训练前检查

- [ ] Rust库已编译（`libblood`可用）
- [ ] 配置文件已创建（`mortal/config.toml`）
- [ ] 配置文件已验证（`./scripts/check-config.sh`）
- [ ] GRP模型已创建（`/data/mortal/grp.pth`存在）
- [ ] 数据目录已创建（`/data/mortal/train_play`等）
- [ ] 设备配置正确（GPU/CPU）
- [ ] 磁盘空间充足（至少10GB）

### 训练中检查

- [ ] 自对局数据正常生成（`train_play/`目录有文件）
- [ ] 训练损失正常下降（查看TensorBoard或日志）
- [ ] 模型文件正常保存（`mortal.pth`、`baseline.pth`）
- [ ] 内存/GPU使用正常（无OOM错误）
- [ ] 训练速度合理（每步时间稳定）

### 训练后检查

- [ ] 模型文件已保存
- [ ] 训练指标已记录（TensorBoard）
- [ ] 可以正常加载模型
- [ ] 可以继续训练（恢复训练）

---

## 🎓 第十一部分：训练最佳实践

### 11.1 初期训练策略

**目标**：快速验证训练流程，确保系统正常工作

**策略**：
1. 使用较小的 `games`（400）
2. 使用较高的探索率（`boltzmann_epsilon = 0.01`）
3. 更频繁的保存（`save_every = 200`）
4. 观察5-10轮，确保损失下降

### 11.2 中期训练策略

**目标**：稳定提升模型性能

**策略**：
1. 使用正常的 `games`（800）
2. 使用正常的探索率（`boltzmann_epsilon = 0.005`）
3. 定期重新训练GRP（每5-10轮）
4. 监控训练指标，及时调整

### 11.3 后期训练策略

**目标**：精细优化，达到最佳性能

**策略**：
1. 使用更多的 `games`（1200-2000）
2. 降低探索率（`boltzmann_epsilon = 0.005`或更低）
3. 超参数调优
4. 长期训练（可能需要数周）

### 11.4 数据管理

**自对局数据**：
- 每次自对局前会清空旧数据
- 只使用最新的自对局数据训练
- 避免数据过时

**模型备份**：
- 定期备份重要检查点
- 保存最佳模型（`best.pth`）
- 记录训练配置和超参数

### 11.5 监控和维护

**日常监控**：
- 检查训练日志
- 查看TensorBoard
- 检查系统资源使用

**定期维护**：
- 清理旧的日志文件
- 备份重要模型
- 检查磁盘空间

---

## 📚 第十二部分：参考文档

### 核心文档

- `scripts/SELF_PLAY_GUIDE.md` - 自博弈训练详细指南
- `scripts/QUICK_START_SELF_PLAY.md` - 快速开始指南
- `scripts/TRAINING_GUIDE.md` - 通用训练指南
- `scripts/NO_DATA_TRAINING.md` - 无数据训练指南

### GRP相关

- `scripts/GRP_TRAINING_GUIDE.md` - GRP训练指南
- `scripts/GRP_PRINCIPLE.md` - GRP原理说明
- `scripts/GRP_ANALYSIS.md` - GRP实现分析
- `scripts/GRP_OPTIMIZATION_NOTES.md` - GRP优化记录

### 配置相关

- `scripts/CONFIG_SETUP.md` - 配置设置指南
- `scripts/CONFIG_CHECKLIST.md` - 配置检查清单

### 工具脚本

- `scripts/create-initial-grp.py` - 创建初始GRP模型
- `scripts/check-config.sh` - 验证配置文件
- `scripts/setup-config.sh` - 自动设置配置
- `scripts/generate-initial-data.py` - 生成初始数据
- `scripts/blood-train.sh` - 训练脚本
- `scripts/blood-train-grp.sh` - GRP训练脚本

---

## 🎯 快速参考

### 完整训练命令序列

```bash
# 1. 编译Rust库（如果未编译）
cd libblood && cargo build --release && cd ..

# 2. 创建配置文件（如果不存在）
cd mortal && cp config.example.toml config.toml && cd ..

# 3. 设置配置（服务器环境）
./scripts/setup-config.sh /data/mortal /data/blood

# 4. 验证配置
./scripts/check-config.sh

# 5. 创建GRP模型（必需）
cd mortal && python ../scripts/create-initial-grp.py && cd ..

# 6. 开始训练（自动生成数据）
./scripts/blood-train.sh offline
```

### 关键路径

| 项目 | 路径 |
|------|------|
| 配置文件 | `mortal/config.toml` |
| 主模型 | `/data/mortal/mortal.pth` |
| Baseline模型 | `/data/mortal/baseline.pth` |
| GRP模型 | `/data/mortal/grp.pth` |
| 自对局数据 | `/data/mortal/train_play/` |
| 训练日志 | `/data/mortal/logs/` |

### 关键配置项

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `online` | `[control]` | `false` = 离线自博弈训练 |
| `device` | `[control]` | `'cuda:0'` 或 `'cpu'` |
| `games` | `[train_play.default]` | 每次自对局局数 |
| `save_every` | `[control]` | 每N步保存（保存后会自对局） |
| `globs` | `[dataset]` | 数据文件路径 |

---

## ✅ 总结

### 核心要点

1. **无需准备数据**：系统支持自博弈训练，自动生成数据
2. **GRP模型必需**：训练前必须先创建GRP模型
3. **自动迭代**：系统会自动进行自对局→训练→自对局的循环
4. **持续改进**：模型会逐步提升，需要耐心

### 立即开始

```bash
# 1. 创建GRP模型
cd mortal && python ../scripts/create-initial-grp.py

# 2. 开始训练
cd .. && ./scripts/blood-train.sh offline
```

### 预期时间线

- **第1轮**：1-2小时（包括自对局）
- **前10轮**：每轮1-2小时
- **10-50轮**：每轮1-2小时
- **50+轮**：每轮1-2小时

### 成功标志

- ✅ 训练损失逐步下降
- ✅ 平均排名逐步改善
- ✅ 模型文件正常保存
- ✅ 可以持续训练

---

**祝你训练顺利！** 🎉

如有问题，请参考相关文档或检查常见问题部分。
