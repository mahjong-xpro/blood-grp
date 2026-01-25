# GRP 优化实施记录（方案A）

## 实施日期
2026-01-25

## 优化内容
1. **方案A**：扩展GRP特征，添加已胡牌玩家信息（GRP_SIZE: 5 → 9）
2. **定缺信息**：添加定缺信息到GRP特征（GRP_SIZE: 9 → 13）

## 修改内容

### 1. 修改 GRP_SIZE 常量

**文件**: `libblood/src/consts.rs`

**修改前**:
```rust
pub const GRP_SIZE: usize = 5;  // [kyoku, score[0], score[1], score[2], score[3]]
```

**修改后**:
```rust
pub const GRP_SIZE: usize = 9;  // [kyoku, score[0], score[1], score[2], score[3], agari[0], agari[1], agari[2], agari[3]]
```

### 2. 修改特征提取逻辑

**文件**: `libblood/src/dataset/grp.rs`

**关键修改**:
1. 添加正向遍历，跟踪每个 `StartKyoku` 时已胡牌玩家的状态
2. 在反向遍历提取特征时，使用已记录的已胡牌状态
3. 特征格式从 5 维扩展到 9 维

**实现逻辑**:
- 正向遍历事件序列，遇到 `Event::Hora` 时标记对应玩家已胡牌
- 遇到 `Event::StartKyoku` 时，记录"在该局开始前"的已胡牌状态
- 反向遍历时，使用记录的状态构建特征

### 3. 更新 Python 代码注释

**文件**:
- `mortal/model.py` - 更新 GRP 模型输入格式注释
- `mortal/reward_calculator.py` - 更新特征索引注释
- `mortal/dataloader.py` - 更新特征提取注释

## 特征格式

### 新特征格式（13维）

```
[kyoku, score[0]/10000, score[1]/10000, score[2]/10000, score[3]/10000,
 agari[0], agari[1], agari[2], agari[3],
 ding_que[0], ding_que[1], ding_que[2], ding_que[3]]
```

其中：
- `kyoku`: 当前局数（从1开始）
- `score[i]/10000`: 玩家 i 的分数除以 10000（归一化）
- `agari[i]`: 1.0 如果玩家 i 已胡牌，0.0 否则
- `ding_que[i]`: 0.0=万子(Man), 0.5=筒子(Pin), 1.0=条子(Sou)（归一化）

### 索引说明

- 索引 0: `kyoku`
- 索引 1-4: `score[0]` 到 `score[3]`
- 索引 5-8: `agari[0]` 到 `agari[3]`
- 索引 9-12: `ding_que[0]` 到 `ding_que[3]`

## 优势

1. **准确预测3人胡牌结束**：GRP 现在知道哪些玩家已胡牌，能准确预测"只剩1个胡牌机会"的情况
2. **更好的奖励信号**：在已有2人胡牌时，能更准确地评估第4个玩家的排名概率
3. **保持向后兼容**：虽然特征维度改变，但模型结构（GRU）会自动适应

## 注意事项

### ⚠️ 重要：需要重新训练

1. **所有现有 GRP 模型都需要重新训练**
   - 旧模型（5维/9维输入）与新数据（13维）不兼容
   - 必须重新训练 GRP 模型

2. **训练数据格式**
   - 新代码会自动生成13维特征
   - 旧数据文件需要重新处理（如果使用旧数据）

3. **模型初始化**
   - 使用 `create-initial-grp.py` 创建的新模型会自动使用13维输入
   - 旧的 `grp.pth` 文件不能直接使用

## 验证步骤

1. **编译验证**
   ```bash
   cargo check --lib
   ```

2. **功能验证**
   - 运行测试确保特征提取正确
   - 验证已胡牌状态跟踪逻辑

3. **训练验证**
   - 创建新的随机 GRP 模型
   - 使用新数据训练 GRP
   - 验证训练过程正常

## 后续工作

1. ✅ 完成代码修改
2. ⏳ 更新文档（GRP_PRINCIPLE.md, GRP_TRAINING_GUIDE.md）
3. ⏳ 验证编译和功能
4. ⏳ 创建新的随机 GRP 模型进行测试
5. ⏳ 进行对比实验（如果可能）

## 更新记录

### 2026-01-25: 添加定缺信息
- 扩展 GRP_SIZE 从 9 到 13
- 添加定缺信息提取逻辑
- 更新所有相关注释和文档

### 2026-01-25: 方案A - 添加已胡牌信息
- 扩展 GRP_SIZE 从 5 到 9
- 添加已胡牌玩家信息提取逻辑

## 相关文件

- `libblood/src/consts.rs` - GRP_SIZE 常量
- `libblood/src/dataset/grp.rs` - 特征提取逻辑
- `mortal/model.py` - GRP 模型定义
- `mortal/reward_calculator.py` - 奖励计算
- `mortal/dataloader.py` - 数据加载
- `scripts/GRP_ANALYSIS.md` - 分析文档
- `scripts/GRP_OPTIMIZATION_NOTES.md` - 本文档
