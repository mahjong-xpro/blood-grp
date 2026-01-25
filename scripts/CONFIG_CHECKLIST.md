# 配置文件检查清单

## 快速检查

运行检查脚本：

```bash
./scripts/check-config.sh
```

## 手动检查项

### 1. 基本配置 ✓

- [x] `version = 4` - 模型版本
- [x] `online = false` - 离线训练模式
- [x] `device = 'cuda:0'` 或 `'cpu'` - 计算设备

### 2. 路径配置 ✓

- [x] `state_file = '/data/mortal/mortal.pth'` - 模型保存路径
- [x] `best_state_file = '/data/mortal/best.pth'` - 最佳模型路径
- [x] `tensorboard_dir = '/data/mortal/logs'` - 日志目录
- [x] `train_play.log_dir = '/data/mortal/train_play'` - 自对局数据目录
- [x] `dataset.globs = ['/data/mortal/train_play/**/*.json.gz']` - 数据集路径
- [x] `baseline.train.state_file = '/data/mortal/baseline.pth'` - Baseline 路径

### 3. 训练参数 ✓

- [x] `batch_size = 512` - 批次大小（根据显存调整）
- [x] `save_every = 400` - 保存频率
- [x] `test_every = 20000` - 测试频率
- [x] `num_workers = 4` - 数据加载工作进程数

### 4. 自对局配置 ✓

- [x] `train_play.default.games = 800` - 每次自对局局数
- [x] `train_play.default.log_dir` - 数据保存目录
- [x] `train_play.default.boltzmann_epsilon = 0.005` - 探索率

### 5. 优化器配置 ✓

- [x] `optim.eps = 1e-8` - Adam 优化器 epsilon
- [x] `optim.betas = [0.9, 0.999]` - Adam 优化器 betas
- [x] `optim.weight_decay = 0.1` - 权重衰减
- [x] `optim.scheduler.peak = 1e-4` - 峰值学习率
- [x] `optim.scheduler.final = 1e-4` - 最终学习率

### 6. 模型配置 ✓

- [x] `resnet.conv_channels = 192` - ResNet 通道数
- [x] `resnet.num_blocks = 40` - ResNet 块数
- [x] `cql.min_q_weight = 5` - CQL 权重
- [x] `aux.next_rank_weight = 0.2` - 排名预测权重

## 根据硬件调整

### GPU 训练（推荐）

```toml
device = 'cuda:0'
enable_cudnn_benchmark = true
enable_amp = true
batch_size = 512
num_workers = 4
```

### CPU 训练

```toml
device = 'cpu'
enable_cudnn_benchmark = false
enable_amp = false
batch_size = 128
num_workers = 2
```

### 显存不足

```toml
batch_size = 256  # 或更小
enable_amp = true  # 必须启用
file_batch_size = 10
```

## 常见问题

### 路径不存在

**问题**: 配置文件中的路径不存在

**解决**: 训练时会自动创建目录，或手动创建：
```bash
mkdir -p /data/mortal/{train_play,test_play,logs,buffer,drain,1v3}
```

### 权限问题

**问题**: 无法写入目录

**解决**: 
```bash
chmod -R 755 /data/mortal
chown -R $USER:$USER /data/mortal
```

### 设备配置错误

**问题**: GPU 不可用但配置为 `cuda:0`

**解决**: 
```bash
# 检查 GPU
nvidia-smi

# 如果没有 GPU，修改配置
# device = 'cpu'
```

## 验证命令

```bash
# 1. 检查配置文件语法
python3 -c "import tomli; tomli.load(open('mortal/config.toml', 'rb'))"

# 2. 检查 Python 模块
cd mortal && python3 -c "import libblood; print('OK')"

# 3. 运行完整检查
./scripts/check-config.sh
```
