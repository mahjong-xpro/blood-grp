# 训练指南

本指南介绍如何开始训练 Bloody Battle Mahjong AI。

## 训练模式

项目支持两种训练模式：

1. **离线训练（Offline Training）**：使用已有的对局数据训练
2. **在线训练（Online Training）**：通过自对局生成数据并训练

**重要**：如果没有人类对战数据，系统支持**自博弈训练**（从零开始）。详见：`scripts/SELF_PLAY_GUIDE.md`

## 前置准备

### 1. 配置文件

复制示例配置文件并修改：

```bash
cd mortal
cp config.example.toml config.toml
```

编辑 `config.toml`，设置必要的路径和参数。

### 2. 准备数据（离线训练需要）

- **训练数据**：对局日志文件（JSON格式，gzip压缩）
- **文件路径**：在 `config.toml` 的 `[dataset]` 部分配置

```toml
[dataset]
globs = ['/path/to/dataset/**/*.json.gz']
file_index = '/path/to/file_index.pth'
```

### 3. 初始模型（可选）

如果有预训练模型，设置路径：

```toml
[control]
state_file = '/path/to/mortal.pth'  # 初始模型
best_state_file = '/path/to/best.pth'  # 最佳模型
```

如果没有，训练会从头开始。

---

## 方式一：离线训练（推荐新手）

### 步骤 1：准备配置文件

编辑 `mortal/config.toml`：

```toml
[control]
version = 4
online = false  # 离线训练
state_file = 'mortal.pth'
best_state_file = 'best.pth'
tensorboard_dir = 'logs'
device = 'cuda:0'  # 或 'cpu'

[dataset]
globs = ['/path/to/your/dataset/**/*.json.gz']
file_index = 'file_index.pth'
file_batch_size = 15
num_workers = 4
num_epochs = 1

[train_play.default]
games = 800
log_dir = 'train_play'
boltzmann_epsilon = 0.005
boltzmann_temp = 0.05
```

### 步骤 2：开始训练

```bash
cd mortal
python train.py
```

### 训练流程

训练会自动执行以下循环：

1. **加载数据**：从数据集加载对局数据
2. **训练模型**：使用强化学习算法更新模型
3. **自对局**：使用当前模型进行自对局生成新数据
4. **测试评估**：定期测试模型性能
5. **保存模型**：定期保存检查点

### 监控训练

使用 TensorBoard 查看训练进度：

```bash
tensorboard --logdir logs
```

然后在浏览器打开 `http://localhost:6006`

---

## 方式二：在线训练（分布式训练）

在线训练适合多机器分布式训练场景。

### 架构

```
训练服务器 (server.py)
    ↓
分发模型参数
    ↓
多个训练客户端 (client.py)
    ↓
收集对局数据
    ↓
返回给服务器
```

### 步骤 1：启动训练服务器

```bash
# 终端 1：启动服务器
./scripts/blood-server.sh start

# 或手动启动
cd mortal
python server.py
```

服务器会：
- 监听配置的端口（默认 5000）
- 分发模型参数给客户端
- 收集客户端提交的对局数据

### 步骤 2：配置在线训练

编辑 `mortal/config.toml`：

```toml
[control]
online = true  # 启用在线训练

[online.remote]
host = '127.0.0.1'  # 服务器地址
port = 5000

[online.server]
buffer_dir = '/path/to/buffer'
drain_dir = '/path/to/drain'
capacity = 1600
```

### 步骤 3：启动训练客户端

在多个终端或机器上启动客户端：

```bash
# 终端 2-N：启动训练客户端
cd mortal
python client.py
```

客户端会：
- 从服务器获取最新模型参数
- 使用模型进行自对局
- 将对局数据提交回服务器

### 步骤 4：启动训练主程序

```bash
# 终端 M：启动训练主程序
cd mortal
python train.py
```

训练主程序会：
- 从服务器获取对局数据
- 训练模型
- 将更新后的模型参数提交回服务器

---

## 方式三：GRP 训练（排名预测模型）

GRP（Global Ranking Prediction）用于预测最终排名。

### 步骤 1：准备数据

```toml
[grp.dataset]
train_globs = ['/path/to/train/**/*.json.gz']
val_globs = ['/path/to/val/**/*.json.gz']
file_index = 'grp_file_index.pth'
```

### 步骤 2：开始训练

```bash
cd mortal
python train_grp.py
```

---

## 训练参数说明

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批次大小 | 512 |
| `opt_step_every` | 每N个批次更新一次 | 1 |
| `save_every` | 每N步保存一次 | 400 |
| `test_every` | 每N步测试一次 | 20000 |
| `device` | 计算设备 | 'cuda:0' |

### 学习率调度

```toml
[optim.scheduler]
peak = 1e-4      # 峰值学习率
final = 1e-4     # 最终学习率
warm_up_steps = 0
max_steps = 0    # 0 表示不限制
```

### 自对局参数

```toml
[train_play.default]
games = 800              # 每次自对局局数
boltzmann_epsilon = 0.005  # 探索率
boltzmann_temp = 0.05    # 温度参数
top_p = 1.0             # Top-p 采样
```

---

## 训练脚本

为了方便使用，可以创建训练脚本：

### 离线训练脚本

创建 `scripts/blood-train.sh`：

```bash
#!/bin/bash
cd "$(dirname "$0")/../mortal"
python train.py
```

### 在线训练脚本

创建 `scripts/blood-train-online.sh`：

```bash
#!/bin/bash
# 启动训练服务器
./scripts/blood-server.sh start

# 等待服务器启动
sleep 5

# 启动训练主程序
cd "$(dirname "$0")/../mortal"
python train.py
```

---

## 常见问题

### 1. 内存不足

- 减小 `batch_size`
- 减小 `file_batch_size`
- 使用 `num_workers = 1`

### 2. GPU 内存不足

- 启用混合精度训练：`enable_amp = true`
- 减小 `batch_size`
- 使用梯度累积（调整 `opt_step_every`）

### 3. 训练速度慢

- 启用 `enable_cudnn_benchmark = true`
- 增加 `num_workers`
- 使用更快的存储（SSD）

### 4. 没有训练数据

- 可以先使用随机模型进行自对局生成数据
- 或从在线平台下载对局数据

---

## 训练检查清单

- [ ] 配置文件已创建并正确设置
- [ ] 数据路径正确（离线训练）
- [ ] GPU/CPU 设备配置正确
- [ ] 有足够的磁盘空间保存模型和日志
- [ ] TensorBoard 可以访问（可选）
- [ ] 服务器已启动（在线训练）
- [ ] 客户端已启动（在线训练）

---

## 下一步

训练完成后：

1. **评估模型**：使用测试集评估性能
2. **1v3 测试**：运行 `one_vs_three.py` 进行对战测试
3. **部署模型**：将模型用于实际对局

---

## 参考文档

- `scripts/SELF_PLAY_GUIDE.md` - **自博弈训练指南**（从零开始，无需人类数据）⭐
- `mortal/config.example.toml` - 完整配置示例
- `mortal/train.py` - 训练主程序
- `mortal/client.py` - 在线训练客户端
- `mortal/server.py` - 在线训练服务器
