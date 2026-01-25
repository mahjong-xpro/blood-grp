# 配置文件设置说明

## 已创建的配置文件

已根据服务器路径创建了配置文件：`mortal/config.toml`

### 路径配置

- **数据目录**: `/data/mortal`
- **源码目录**: `/data/blood`
- **配置文件**: `/data/blood/mortal/config.toml`

## 目录结构

配置文件会自动创建以下目录结构：

```
/data/mortal/
├── mortal.pth          # 当前模型
├── best.pth            # 最佳模型
├── baseline.pth        # Baseline 模型
├── challenger.pth      # 挑战者模型（1v3测试）
├── file_index.pth       # 数据文件索引
├── grp.pth             # GRP 模型
├── grp_file_index.pth  # GRP 数据索引
├── train_play/         # 自对局数据目录
├── test_play/          # 测试对局数据目录
├── 1v3/                # 1v3 测试数据目录
├── logs/               # TensorBoard 日志
├── buffer/             # 在线训练缓冲区
├── drain/              # 在线训练数据提取目录
└── dataset/            # 数据集目录（如果有）
    ├── train/
    └── val/
```

## 关键配置说明

### 1. 计算设备

```toml
device = 'cuda:0'  # 如果有 GPU
# 或
device = 'cpu'     # 如果只有 CPU
```

**检查 GPU**：
```bash
nvidia-smi  # 查看 GPU 信息
```

### 2. 自对局配置

```toml
[train_play.default]
games = 800  # 每次自对局局数
log_dir = '/data/mortal/train_play'
boltzmann_epsilon = 0.005  # 探索率
```

**建议**：
- 初期：`games = 400-800`，`boltzmann_epsilon = 0.01-0.02`
- 后期：`games = 2000+`，`boltzmann_epsilon = 0.001-0.005`

### 3. 数据集配置

```toml
[dataset]
globs = ['/data/mortal/train_play/**/*.json.gz']  # 自对局数据
file_index = '/data/mortal/file_index.pth'
num_workers = 4  # 根据 CPU 核心数调整
```

### 4. 训练参数

```toml
batch_size = 512  # 根据 GPU 显存调整
enable_amp = true  # 混合精度，节省显存
```

**显存不足时**：
- 减小 `batch_size`（如 256 或 128）
- 确保 `enable_amp = true`

## 快速设置脚本

使用提供的脚本快速设置：

```bash
# 设置环境变量（可选）
export DATA_DIR=/data/mortal
export SOURCE_DIR=/data/blood

# 运行设置脚本
./scripts/setup-config.sh
```

脚本会：
1. 创建所有必要的目录
2. 从示例配置创建配置文件
3. 更新所有路径为服务器路径

## 手动设置

如果需要手动设置：

```bash
cd /data/blood/mortal
cp config.example.toml config.toml

# 编辑配置文件
vim config.toml  # 或使用其他编辑器
```

然后更新以下路径：
- `state_file` → `/data/mortal/mortal.pth`
- `best_state_file` → `/data/mortal/best.pth`
- `tensorboard_dir` → `/data/mortal/logs`
- `train_play.log_dir` → `/data/mortal/train_play`
- `dataset.globs` → `['/data/mortal/train_play/**/*.json.gz']`
- 其他所有 `/path/to/` → `/data/mortal/`

## 验证配置

### 1. 检查配置文件语法

```bash
cd /data/blood/mortal
python3 -c "import tomli; tomli.load(open('config.toml', 'rb'))" && echo "✓ Config is valid"
```

### 2. 检查目录权限

```bash
# 确保有写入权限
mkdir -p /data/mortal/{train_play,test_play,logs}
touch /data/mortal/test_write && rm /data/mortal/test_write && echo "✓ Write permission OK"
```

### 3. 检查 Python 模块

```bash
cd /data/blood/mortal
python3 -c "import libblood; print('✓ libblood module OK')"
```

## 开始训练

配置完成后，可以开始训练：

```bash
cd /data/blood
./scripts/blood-train.sh offline
```

## 根据硬件调整配置

### GPU 训练（推荐）

```toml
device = 'cuda:0'
enable_cudnn_benchmark = true
enable_amp = true
batch_size = 512
num_workers = 4  # 根据 CPU 核心数
```

### CPU 训练

```toml
device = 'cpu'
enable_cudnn_benchmark = false
enable_amp = false
batch_size = 128  # CPU 训练建议减小批次
num_workers = 2   # CPU 训练建议减少工作进程
```

### 显存不足

```toml
batch_size = 256  # 或更小
enable_amp = true  # 必须启用
file_batch_size = 10  # 减小文件批次
```

## 环境变量

可以通过环境变量覆盖配置：

```bash
export MORTAL_CFG=/data/blood/mortal/config.toml
export PYTHON_CMD=python3
```

## 故障排除

### 配置文件找不到

```bash
# 设置环境变量
export MORTAL_CFG=/data/blood/mortal/config.toml
```

### 权限问题

```bash
# 确保目录有写入权限
sudo chown -R $USER:$USER /data/mortal
chmod -R 755 /data/mortal
```

### 路径不存在

```bash
# 创建所有必要的目录
mkdir -p /data/mortal/{train_play,test_play,1v3,logs,buffer,drain,dataset/{train,val}}
```

## 下一步

1. ✅ 配置文件已创建：`/data/blood/mortal/config.toml`
2. ⏭️ 检查并调整参数（特别是 `device`）
3. ⏭️ 开始训练：`./scripts/blood-train.sh offline`
