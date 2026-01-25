# 没有数据如何开始训练

## 核心答案

**系统支持自博弈训练（Self-Play）**，可以从零开始，**不需要任何人类对战数据**！

系统会自动：
1. 使用随机模型进行自对局生成数据
2. 使用生成的数据训练模型
3. 迭代改进

---

## 快速开始（3步）

### 步骤 1：创建 GRP 模型（必需）

**为什么需要**：训练主模型时需要 GRP 来计算奖励，GRP 模型必须先存在。

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
```

### 步骤 2：检查配置文件

确保 `mortal/config.toml` 配置正确：

```toml
[control]
online = false  # 离线训练模式（自博弈）
device = 'cuda:0'  # 或 'cpu'

[train_play.default]
games = 800  # 每次自对局局数
log_dir = '/data/mortal/train_play'  # 自对局数据保存目录

[dataset]
globs = ['/data/mortal/train_play/**/*.json.gz']  # 指向自对局数据
```

### 步骤 3：开始训练

```bash
./scripts/blood-train.sh offline
```

**系统会自动**：
1. 如果 `mortal.pth` 不存在 → 创建随机模型
2. 如果 `baseline.pth` 不存在 → 使用当前模型作为 baseline
3. 如果 `train_play` 目录为空 → **自动进行自对局生成数据**
4. 使用生成的数据开始训练

---

## 详细流程说明

### 自博弈训练的工作原理

```
初始化阶段：
├─ 创建随机主模型（如果不存在）
├─ 创建随机 GRP 模型（必需，手动创建）
└─ 使用主模型作为 baseline

训练循环：
├─ 步骤 1：自对局生成数据
│   └─ 1个训练模型 vs 3个baseline模型
│   └─ 生成对局数据保存到 train_play/
│
├─ 步骤 2：训练模型
│   └─ 使用生成的数据训练
│   └─ 定期保存检查点
│
└─ 步骤 3：更新 baseline
    └─ 使用训练后的模型更新 baseline
    └─ 回到步骤 1
```

### 第一次训练的特殊处理

**如果 `train_play` 目录为空**：
- 训练脚本会**自动检测**并先进行自对局
- 生成初始数据后再开始训练
- **无需手动准备数据**

**如果 `mortal.pth` 不存在**：
- 系统会**自动创建**随机初始化的模型
- 无需手动创建

**如果 `baseline.pth` 不存在**：
- 系统会**自动使用**当前模型（可能是随机的）作为 baseline
- 第一次自对局时，训练模型和 baseline 都是随机模型

---

## 完整示例

### 场景：完全从零开始

```bash
# 1. 确保配置文件存在
cd mortal
ls config.toml  # 应该存在

# 2. 创建 GRP 模型（必需）
python ../scripts/create-initial-grp.py

# 3. 检查配置（可选）
../scripts/check-config.sh

# 4. 开始训练（会自动生成数据）
cd ..
./scripts/blood-train.sh offline
```

**训练脚本会自动**：
1. 检测到 `mortal.pth` 不存在 → 创建随机模型
2. 检测到 `baseline.pth` 不存在 → 使用随机模型作为 baseline
3. 检测到 `train_play` 目录为空 → 进行自对局生成数据
4. 使用生成的数据开始训练

---

## 训练过程示例

### 第一次运行

```
[INFO] mortal.pth not found, creating random model...
[INFO] baseline.pth not found, using current model as baseline
[INFO] train_play directory is empty, starting self-play...
[INFO] Running self-play: 800 games
[INFO] Self-play completed: 800 games, 800 files generated
[INFO] Starting training with 800 games of data...
[INFO] Training step 1/400...
[INFO] Training step 2/400...
...
[INFO] Saving checkpoint at step 400
[INFO] Updating baseline model...
[INFO] Starting new self-play session...
```

### 后续运行

```
[INFO] Loading model from mortal.pth
[INFO] Loading baseline from baseline.pth
[INFO] Starting self-play: 800 games
[INFO] Self-play completed: 800 games
[INFO] Starting training...
```

---

## 关键配置说明

### 自对局配置

```toml
[train_play.default]
games = 800  # 每次自对局局数
# 建议：
# - 初期：400-800（快速迭代）
# - 后期：800-2000（更多数据）

log_dir = '/data/mortal/train_play'  # 数据保存目录
boltzmann_epsilon = 0.005  # 探索率（初期可以设置更高）
```

### 数据集配置

```toml
[dataset]
globs = ['/data/mortal/train_play/**/*.json.gz']  # 指向自对局数据
# 如果目录为空，系统会自动生成数据，所以可以放心设置这个路径
```

### 训练配置

```toml
[control]
save_every = 400  # 每 400 步保存一次
# 每次保存后，会进行新的自对局并更新 baseline
```

---

## 常见问题

### Q: 第一次训练会很慢吗？

**A**: 
- 第一次自对局：取决于 `games` 数量（800局约需10-30分钟）
- 第一次训练：取决于硬件（GPU更快）
- 总体：第一次完整循环可能需要1-2小时

### Q: 随机模型表现很差，正常吗？

**A**: **完全正常**！
- 随机模型初期表现很差是预期的
- 需要几轮迭代才能看到改善
- 建议先训练5-10轮观察趋势

### Q: 如何加快第一次训练？

**A**:
1. 减少 `games`（如 400 而不是 800）
2. 使用 GPU（`device = 'cuda:0'`）
3. 启用混合精度（`enable_amp = true`）

### Q: 数据会累积吗？

**A**: 
- **不会**：每次自对局前会清空 `train_play` 目录
- 只使用最新的自对局数据训练
- 这样可以避免数据过时

### Q: 需要手动生成初始数据吗？

**A**: **不需要**！
- 训练脚本会自动检测并生成数据
- 但如果想手动生成，可以使用：
  ```bash
  cd mortal
  python ../scripts/generate-initial-data.py
  ```

---

## 验证训练是否正常

### 检查点 1：GRP 模型存在

```bash
ls -lh /data/mortal/grp.pth
# 应该存在且大小约几MB
```

### 检查点 2：自对局数据生成

```bash
ls /data/mortal/train_play/
# 训练开始后，应该看到 .json.gz 文件
```

### 检查点 3：训练日志

训练日志应该显示：
- 自对局进度
- 训练进度
- 模型保存信息

### 检查点 4：TensorBoard

```bash
# 另一个终端
tensorboard --logdir /data/mortal/logs
# 浏览器打开 http://localhost:6006
```

---

## 预期时间线

### 初期（前5轮）

- **每轮时间**：1-2小时
- **模型表现**：可能很差（正常）
- **关注点**：训练损失是否下降

### 中期（5-20轮）

- **每轮时间**：1-2小时
- **模型表现**：逐步改善
- **关注点**：排名提升趋势

### 后期（20+轮）

- **每轮时间**：1-2小时
- **模型表现**：持续提升
- **关注点**：性能优化

---

## 总结

**没有数据也能训练！**

只需要：
1. ✅ 创建 GRP 模型（必需）
2. ✅ 配置正确
3. ✅ 运行训练脚本

系统会自动：
- 生成数据
- 训练模型
- 迭代改进

**立即开始**：
```bash
cd mortal && python ../scripts/create-initial-grp.py
cd .. && ./scripts/blood-train.sh offline
```

---

## 相关文档

- `scripts/SELF_PLAY_GUIDE.md` - 完整的自博弈训练指南
- `scripts/QUICK_START_SELF_PLAY.md` - 快速开始指南
- `scripts/NEXT_STEPS.md` - 下一步行动清单
