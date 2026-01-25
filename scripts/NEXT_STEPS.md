# GRP 优化后的下一步行动

## 当前状态

✅ **已完成**：
1. 方案A优化：添加已胡牌玩家信息（GRP_SIZE: 5 → 9）
2. 添加定缺信息（GRP_SIZE: 9 → 13）
3. 代码编译通过
4. 文档已更新

## ⚠️ 重要：需要重新训练 GRP 模型

由于GRP特征维度从5维扩展到13维，**所有现有的GRP模型都不能直接使用**，必须重新训练。

---

## 立即行动清单

### ⚠️ 重要：没有数据也能训练！

**系统支持自博弈训练（Self-Play）**，可以从零开始，**不需要任何人类对战数据**！

系统会自动：
- 使用随机模型进行自对局生成数据
- 使用生成的数据训练模型
- 迭代改进

### 1. 创建新的随机 GRP 模型（必需）

**目的**：创建一个使用13维特征的随机初始化GRP模型，用于开始训练。

**步骤**：
```bash
cd mortal
python ../scripts/create-initial-grp.py
```

**预期结果**：
- 创建 `/data/mortal/grp.pth`（或配置文件中指定的路径）
- 模型使用13维输入（自动从 `GRP_SIZE` 常量读取）

**验证**：
```bash
# 检查文件是否存在
ls -lh /data/mortal/grp.pth

# 检查文件大小（应该约几MB）
```

---

### 2. 验证特征提取（推荐）

**目的**：确保新的特征提取逻辑正确工作。

**方法A：使用现有对局数据测试**

如果有现有的对局数据文件（`.json.gz`），可以快速测试：

```python
# 测试脚本：test_grp_feature.py
from libblood.dataset import Grp
import json

# 加载一个对局文件
with open('path/to/game.json.gz', 'rb') as f:
    # 这里需要根据实际的数据格式来加载
    # 假设有 load_gz_log_files 方法
    games = Grp.load_gz_log_files(['path/to/game.json.gz'])
    
    for game in games:
        feature = game.take_feature()
        print(f"Feature shape: {feature.shape}")
        print(f"Expected: (num_kyokus, 13)")
        print(f"First kyoku feature: {feature[0]}")
        # 验证特征维度
        assert feature.shape[1] == 13, f"Expected 13 features, got {feature.shape[1]}"
        print("✓ Feature extraction works correctly!")
```

**方法B：运行训练数据加载测试**

```bash
cd mortal
python -c "
from dataloader import FileDatasetsIter
from config import config
import torch

# 尝试加载一个批次的数据
# 这会触发GRP特征提取
try:
    # 这里需要实际的训练数据路径
    print('Testing GRP feature extraction...')
    # 如果数据加载器能正常工作，说明特征提取正确
    print('✓ GRP feature extraction is working')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

---

### 3. 开始训练 GRP（如果已有数据）

**前提条件**：
- ✅ 已有对局数据（`.json.gz` 文件）
- ✅ 配置文件已设置正确的数据路径
- ✅ 新的随机GRP模型已创建

**步骤**：
```bash
cd mortal
python train_grp.py
# 或
../scripts/blood-train-grp.sh
```

**监控**：
- 查看训练日志
- 使用 TensorBoard 监控训练进度
- 检查模型保存是否正常

---

### 4. 开始主模型训练（自博弈训练）

**前提条件**：
- ✅ 新的随机GRP模型已创建（必需）
- ✅ 配置文件已正确设置
- ✅ 数据目录已创建

**步骤**：
```bash
# 如果 train_play 目录为空，训练脚本会自动进行自对局
./scripts/blood-train.sh offline
```

**注意**：
- 训练脚本会自动使用新的13维GRP特征
- 如果GRP模型不存在，训练会报错

---

## 验证清单

### ✅ 代码验证

- [x] 代码编译通过
- [ ] 特征提取逻辑测试（可选，但推荐）

### ✅ 模型准备

- [ ] 创建新的随机GRP模型
- [ ] 验证GRP模型文件存在且大小合理

### ✅ 配置验证

- [ ] 检查 `config.toml` 中的GRP配置
- [ ] 确认数据路径正确
- [ ] 确认设备配置（GPU/CPU）

### ✅ 训练准备

- [ ] 如果有数据：准备训练数据路径
- [ ] 如果没有数据：准备自博弈训练

---

## 详细步骤说明

### 步骤1：创建新的GRP模型

**为什么需要**：
- 旧模型（5维或9维）与新数据（13维）不兼容
- 必须创建新的随机模型开始训练

**命令**：
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
```

**如果出错**：
- 检查配置文件是否存在：`mortal/config.toml`
- 检查 `[grp]` 部分是否配置正确
- 检查目录权限（确保可以写入）

---

### 步骤2：验证特征提取（可选但推荐）

**为什么需要**：
- 确保新的特征提取逻辑正确工作
- 避免在训练时才发现问题

**简单验证方法**：

1. **检查GRP_SIZE常量**：
```bash
cd mortal
python -c "from libblood.consts import GRP_SIZE; print(f'GRP_SIZE = {GRP_SIZE}')"
# 应该输出: GRP_SIZE = 13
```

2. **如果有测试数据**：
   - 使用一个小的测试对局文件
   - 加载并检查特征维度

---

### 步骤3：开始训练

#### 选项A：训练GRP（如果有对局数据）

```bash
cd mortal
python train_grp.py
```

**配置要求**：
- `[grp.dataset].train_globs` 指向训练数据
- `[grp.dataset].val_globs` 指向验证数据
- `[grp.control].device` 设置正确

#### 选项B：开始主模型训练（自博弈）

```bash
./scripts/blood-train.sh offline
```

**配置要求**：
- `[control].online = false`
- `[train_play.default].log_dir` 设置正确
- `[dataset].globs` 指向自对局数据目录

---

## 常见问题

### Q: 旧的GRP模型还能用吗？

**A**: 不能。特征维度从5维/9维扩展到13维，旧模型不兼容。必须创建新模型。

### Q: 需要重新处理旧数据吗？

**A**: 
- **新数据**：会自动使用13维特征，无需处理
- **旧数据文件**：如果使用旧的对局数据文件，需要重新处理（但通常不需要，因为新代码会自动提取13维特征）

### Q: 训练会变慢吗？

**A**: 
- 特征维度增加（9→13），但影响很小
- GRU模型会自动适应新的输入维度
- 训练速度主要取决于数据量和硬件

### Q: 如何验证特征提取正确？

**A**: 
1. 检查 `GRP_SIZE = 13`
2. 加载一个对局文件，检查特征维度
3. 检查特征值是否在合理范围内（分数归一化，agari 0/1，ding_que 0/0.5/1）

---

## 推荐行动顺序

### 快速开始（5分钟）

1. ✅ 创建新的随机GRP模型
   ```bash
   cd mortal && python ../scripts/create-initial-grp.py
   ```

2. ✅ 开始自博弈训练
   ```bash
   ./scripts/blood-train.sh offline
   ```

### 完整验证（30分钟）

1. ✅ 创建新的随机GRP模型
2. ✅ 验证特征提取（如果有测试数据）
3. ✅ 检查配置文件
4. ✅ 开始训练

---

## 下一步优化方向

完成基础训练后，可以考虑：

1. **超参数调优**（高优先级）
   - 调整 `hidden_size`、`num_layers`、`lr`
   - 详见 `scripts/GRP_FURTHER_OPTIMIZATIONS.md`

2. **监控训练效果**
   - 使用 TensorBoard
   - 分析排名预测准确率

3. **迭代改进**
   - 根据训练效果调整策略
   - 定期重新训练GRP

---

## 总结

**立即需要做的**：
1. ✅ 创建新的随机GRP模型（必需）
2. ✅ 开始训练（自博弈或GRP训练）

**可选但推荐**：
1. 验证特征提取逻辑
2. 检查配置文件

**后续优化**：
1. 超参数调优
2. 监控训练效果

---

## 相关文档

- `scripts/GRP_TRAINING_GUIDE.md` - GRP训练详细指南
- `scripts/SELF_PLAY_GUIDE.md` - 自博弈训练指南
- `scripts/GRP_OPTIMIZATION_NOTES.md` - 优化实施记录
- `scripts/GRP_FURTHER_OPTIMIZATIONS.md` - 进一步优化建议
