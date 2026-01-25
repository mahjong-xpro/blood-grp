# GRP 训练指南

## GRP 是什么？

**GRP (Global Ranking Prediction)** 是一个用于预测最终排名的模型。它在训练过程中用于计算奖励（reward），帮助主模型学习。

> 📖 **详细原理**：请查看 `scripts/GRP_PRINCIPLE.md` 了解 GRP 的技术原理和在血战到底麻将中的作用。

> 🔧 **优化更新**：
> - 已实现方案A优化，GRP特征从5维扩展到9维，添加了已胡牌玩家信息
> - 已添加定缺信息，GRP特征从9维扩展到13维
> - 详见 `scripts/GRP_OPTIMIZATION_NOTES.md`

## GRP 的作用

在训练流程中，GRP 用于：

1. **计算奖励**：根据当前局数和分数，预测最终排名概率
2. **计算奖励增量**：计算每局结束后的奖励变化
3. **训练数据生成**：在数据加载时计算每个训练样本的奖励

## 是否需要先训练 GRP？

### 答案：**需要先训练 GRP**

原因：
- 训练主模型时，数据加载器会**立即加载 GRP 模型**（`dataloader.py:42`）
- 如果 GRP 模型文件不存在，训练会**报错**
- GRP 用于计算训练数据的奖励，这是训练过程必需的

### 但是，对于自博弈训练：

你可以：

1. **先用随机 GRP 模型**：创建一个随机初始化的 GRP 模型开始训练
2. **先用少量数据训练基础 GRP**：用初始自对局数据训练一个简单的 GRP
3. **迭代改进**：随着训练进行，定期重新训练 GRP 模型

## 训练 GRP 的步骤

### 方法 1：创建随机初始化的 GRP（快速开始）

```python
# create_initial_grp.py
import torch
from model import GRP
from config import config

grp = GRP(**config['grp']['network'])
state = {
    'model': grp.state_dict(),
    'optimizer': torch.optim.AdamW(grp.parameters()).state_dict(),
    'steps': 0,
    'timestamp': torch.tensor(0.0),
}
torch.save(state, config['grp']['state_file'])
print(f"Created initial GRP model: {config['grp']['state_file']}")
```

### 方法 2：使用自对局数据训练 GRP

1. **先进行少量自对局**（生成初始数据）
2. **训练 GRP**：
   ```bash
   cd mortal
   python train_grp.py
   ```
3. **开始主模型训练**

### 方法 3：使用已有数据训练 GRP

如果有对局数据：

```bash
cd mortal
python train_grp.py
```

## 自博弈训练的工作流程

### 推荐流程

```
步骤 1: 创建随机 GRP 模型
   ↓
步骤 2: 进行初始自对局（生成数据）
   ↓
步骤 3: 训练 GRP 模型（使用生成的数据）
   ↓
步骤 4: 开始主模型训练（使用 GRP 计算奖励）
   ↓
步骤 5: 定期重新训练 GRP（使用新的自对局数据）
```

## 配置文件中的 GRP 设置

```toml
[grp]
state_file = '/data/mortal/grp.pth'  # GRP 模型文件路径

[grp.network]
hidden_size = 64
num_layers = 2

[grp.control]
device = 'cuda:0'
tensorboard_dir = '/data/mortal/grp_logs'
batch_size = 512
save_every = 2000
val_steps = 400

[grp.dataset]
# 对于自博弈训练，可以指向自对局数据
train_globs = [
    '/data/mortal/train_play/**/*.json.gz',
]
val_globs = [
    '/data/mortal/test_play/**/*.json.gz',
]
file_index = '/data/mortal/grp_file_index.pth'
file_batch_size = 50

[grp.optim]
lr = 1e-5
```

## 快速开始脚本

我创建了 `scripts/create-initial-grp.py` 脚本，可以快速创建随机初始化的 GRP 模型。

## 常见问题

### Q: 没有 GRP 模型可以训练主模型吗？

**A**: 不可以。训练主模型时，数据加载器会立即尝试加载 GRP 模型，如果文件不存在会报错。

### Q: 可以用随机 GRP 模型开始吗？

**A**: 可以。随机 GRP 模型虽然预测不准确，但可以让训练开始。随着训练进行，可以定期重新训练 GRP。

### Q: GRP 需要多少数据？

**A**: 
- 最少：几百局对局数据就可以开始
- 推荐：几千到几万局数据
- 随着训练进行，使用更多数据重新训练 GRP

### Q: 需要定期重新训练 GRP 吗？

**A**: 建议定期重新训练：
- 初期：每生成一批新数据后重新训练
- 后期：可以降低频率（如每 5-10 轮重新训练一次）
