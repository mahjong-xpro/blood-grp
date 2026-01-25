# 快速开始：自博弈训练（从零开始）

如果你没有人类对战数据，可以按照以下步骤开始自博弈训练。

## 快速开始（4步）

### 步骤 1：创建配置文件

```bash
cd mortal
cp config.example.toml config.toml
# 编辑 config.toml，设置路径等
```

### 步骤 2：创建初始 GRP 模型（必需）

**重要**：GRP 模型是训练必需的，需要先创建。

```bash
cd mortal
python ../scripts/create-initial-grp.py
```

这会创建一个随机初始化的 GRP 模型。后续可以用自对局数据重新训练 GRP。

### 步骤 3：编辑配置文件（最小配置）

编辑 `mortal/config.toml`，至少修改以下部分：

```toml
[control]
version = 4
online = false
state_file = 'mortal.pth'  # 模型保存路径
best_state_file = 'best.pth'
tensorboard_dir = 'logs'
device = 'cuda:0'  # 如果有 GPU，否则用 'cpu'

# 自对局配置
[train_play.default]
games = 800  # 每次自对局局数
log_dir = 'train_play'  # 自对局数据保存目录

# 数据集配置（指向自对局生成的数据）
[dataset]
# 方式1：指向 train_play 目录（推荐）
globs = ['train_play/**/*.json.gz']
# 方式2：如果 train_play 目录还没有数据，可以先留空，然后手动运行一次自对局
# globs = []
file_index = 'file_index.pth'
file_batch_size = 15
num_workers = 4

# Baseline 配置（第一次训练时，系统会自动使用当前模型作为 baseline）
[baseline.train]
device = 'cuda:0'
state_file = 'baseline.pth'  # 如果不存在，会使用当前模型

[baseline.test]
device = 'cuda:0'
state_file = 'baseline.pth'
```

**重要**：如果 `train_play` 目录还没有数据，需要先进行一次自对局生成初始数据（见步骤 2.5）。

### 步骤 2.5：生成初始数据（如果 train_play 目录为空）

如果 `train_play` 目录还没有数据，需要先进行一次自对局生成初始数据。

**方法 1：使用训练脚本（推荐）**

训练脚本会自动处理：如果数据目录为空，会先进行自对局。直接运行：

```bash
./scripts/blood-train.sh offline
```

**方法 2：使用提供的脚本（推荐）**

使用项目提供的脚本：

```bash
cd mortal
python ../scripts/generate-initial-data.py
```

这个脚本会：
- 检查 baseline 模型是否存在
- 如果不存在，创建随机初始化的模型
- 进行自对局生成初始数据
- 显示生成的文件数和统计信息

### 步骤 5：开始训练

```bash
./scripts/blood-train.sh offline
```

**注意**：如果 GRP 模型不存在，训练会报错。请确保已完成步骤 2。

**注意**：如果 `train_play` 目录为空，训练脚本会自动进行自对局生成初始数据。

## 训练流程说明

系统会自动执行以下流程：

1. **初始化**：
   - 如果 `mortal.pth` 不存在，创建随机初始化的模型
   - 如果 `baseline.pth` 不存在，使用当前模型作为 baseline

2. **自对局生成数据**：
   - 使用当前模型进行自对局（1v3：1个训练模型 vs 3个baseline模型）
   - 生成对局数据保存到 `train_play/` 目录

3. **训练模型**：
   - 使用生成的数据训练模型
   - 定期保存检查点

4. **迭代循环**：
   - 每 `save_every` 步后，进行新的自对局
   - 使用训练后的模型更新 baseline
   - 继续训练

## 关键点

### ✅ 不需要准备数据
- 系统会自动进行自对局生成数据
- 不需要手动准备人类对战数据

### ✅ 自动迭代
- 训练过程会自动进行自对局 → 训练 → 自对局的循环
- 模型会逐步提升

### ✅ Baseline 自动更新
- 第一次训练时，baseline 会使用当前模型（可能是随机的）
- 后续会自动使用训练后的模型作为新的 baseline

## 监控训练

### 查看训练日志

训练会在控制台输出日志，包括：
- 自对局进度
- 训练进度
- 模型保存信息

### 使用 TensorBoard

```bash
# 另一个终端
tensorboard --logdir mortal/logs
```

然后在浏览器打开 `http://localhost:6006`

## 预期结果

- **初期**：模型表现可能很差（随机模型），这是正常的
- **几轮后**：模型会逐步改善
- **持续训练**：模型会持续提升

## 常见问题

### Q: 训练初期模型表现很差怎么办？

**A**: 这是正常的。随机模型初期表现很差，需要几轮迭代才能看到改善。建议：
- 先进行 5-10 轮快速迭代（可以设置 `games = 400`）
- 观察训练损失是否下降
- 如果损失下降，说明训练正常

### Q: 需要等多久才能看到效果？

**A**: 
- 初期：每轮约 1-2 小时（取决于硬件）
- 建议：先训练 5-10 轮观察趋势
- 完整训练：可能需要数天到数周

### Q: 如何加快训练速度？

**A**:
- 使用 GPU：`device = 'cuda:0'`
- 启用混合精度：`enable_amp = true`
- 增加 `num_workers`（但不要超过 CPU 核心数）
- 初期可以减少 `games` 快速迭代

### Q: 内存不足怎么办？

**A**:
- 减小 `batch_size`（如 256）
- 减小 `file_batch_size`（如 10）
- 设置 `num_workers = 1`

## 下一步

训练一段时间后：

1. **检查训练进度**：查看 TensorBoard 或日志
2. **评估模型**：运行测试评估性能
3. **调整参数**：根据结果调整训练参数
4. **继续训练**：持续迭代直到达到目标

## 详细文档

更多详细信息请查看：
- `scripts/SELF_PLAY_GUIDE.md` - 完整的自博弈训练指南
- `scripts/TRAINING_GUIDE.md` - 通用训练指南
