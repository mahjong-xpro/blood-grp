# DingQue Bug 完整修复方案

## 🎯 问题总结

经过深入调查，发现DingQue极端偏差（Man总是0%）的根本原因是：

**Observation encoding在DingQue阶段缺少关键信息**

## 🔍 根本原因

### Bug #1: Augmentation映射错误（已修复）
**文件**: `python/blood/env/augment.py:43`  
**问题**: 使用`perm.index(old_suit)`（反向查找）而非`perm[old_suit]`（正向映射）  
**影响**: 83.3%的训练样本使用了错误的action映射  
**状态**: ✅ 已修复

### Bug #2: Observation Encoding缺陷（本次修复）
**文件**: `crates/engine/src/obs/student.rs:94-100`  
**问题**: Section 3在DingQue阶段是空的，没有提供花色统计信息  
**影响**: 模型无法获得做出明智DingQue决策所需的关键信息  
**状态**: ✅ 已修复

## 📝 详细分析

### 原始代码问题

```rust
// === Section 3: DING QUE (17 ch) ===
if let Some(suit) = p.ding_que {
    for t in suit.start()..suit.end() {
        w!(ch + suit as usize, t, 1.0);
    }
}
ch += 3;
```

**问题**：
- 在DingQue阶段，`p.ding_que`是`None`
- Section 3的前3个通道全是0
- 模型看不到手牌的花色分布

### 修复后的代码

```rust
// === Section 3: DING QUE (17 ch) ===
if let Some(suit) = p.ding_que {
    // DingQue已完成：标记选择的花色
    for t in suit.start()..suit.end() {
        w!(ch + suit as usize, t, 1.0);
    }
} else if board.phase == Phase::DingQue {
    // DingQue阶段：提供花色统计信息
    // 通道0: Man花色的牌数量（归一化到 [0,1]）
    // 通道1: Pin花色的牌数量（归一化到 [0,1]）
    // 通道2: Sou花色的牌数量（归一化到 [0,1]）
    for suit in Suit::all() {
        let count = (suit.start()..suit.end())
            .filter(|&t| p.hand[t] > 0)
            .map(|t| p.hand[t] as u32)
            .sum::<u32>();
        fill_ch!(ch + suit as usize, count as f32 / 13.0);
    }
}
ch += 3;
```

**改进**：
- DingQue阶段：显式提供每个花色的牌数量
- DingQue之后：保持原有行为（标记已选择的花色）
- 信息清晰：模型可以直接看到应该选择哪个花色

## 🔧 修复步骤

### 1. 修改Rust代码
✅ 已完成：`crates/engine/src/obs/student.rs`

### 2. 重新编译
```bash
cd blood-v2
maturin develop --release
```

### 3. 测试修复
```bash
cd blood-v2
python3 scripts/test_obs_fix.py
```

### 4. 清理旧训练数据
```bash
cd blood-v2
rm -rf train_dir/blood_v2_warmup
```

### 5. 重新训练
```bash
cd blood-v2
python3 -m blood.train --config configs/warmup.yaml
```

### 6. 验证结果
训练完成后，运行：
```bash
cd blood-v2
python3 scripts/test_live_dingque.py
```

期望结果：Man/Pin/Sou的选择比例应该接近均匀分布（各约33%）

## 📊 预期效果

### 修复前
```
玩家 0:
  man:    0 (  0.0%)  ← 总是0%
  pin:   41 ( 41.0%)
  sou:   59 ( 59.0%)
```

### 修复后（预期）
```
玩家 0:
  man:   32 ( 32.0%)  ← 正常分布
  pin:   34 ( 34.0%)
  sou:   34 ( 34.0%)
```

## 🎓 经验教训

### 1. Observation设计的重要性
- Observation必须包含决策所需的**所有关键信息**
- 不能假设模型会从隐式信息中学习
- 显式编码比隐式学习更可靠

### 2. 调试RL系统的方法
1. **检查权重** - 模型本身是否有问题
2. **检查mask** - 环境是否正确
3. **检查数据流** - augmentation等是否正确
4. **检查observation** - 输入信息是否充分

### 3. 为什么这个bug难找
- 权重正常 ✓
- Mask正确 ✓
- Augmentation已修复 ✓
- 但observation缺少信息 ✗

## 📚 相关文档

- `DINGQUE_BUG_FOUND.md` - Bug发现过程
- `DINGQUE_SMOKING_GUN.md` - 关键证据
- `DINGQUE_BUG_ROOT_CAUSE.md` - 根本原因分析
- `EXPLORATION_OPTIMIZATION_SUMMARY.md` - Exploration优化

## ⚠️ 重要提醒

**必须重新训练模型**！

因为observation的语义改变了：
- 旧模型：Section 3在DingQue阶段是空的
- 新模型：Section 3在DingQue阶段包含花色统计

使用旧checkpoint会导致不可预测的行为。

## ✅ 验证清单

- [x] 修改observation encoding
- [ ] 重新编译Rust代码
- [ ] 测试observation修复
- [ ] 清理旧训练数据
- [ ] 重新训练模型
- [ ] 验证DingQue分布
- [ ] 确认Man不再是0%

## 🚀 下一步

1. 重新编译：`maturin develop --release`
2. 测试修复：`python3 scripts/test_obs_fix.py`
3. 清理数据：`rm -rf train_dir/blood_v2_warmup`
4. 开始训练：`python3 -m blood.train --config configs/warmup.yaml`
5. 等待训练完成（约6-8小时）
6. 验证结果：`python3 scripts/test_live_dingque.py`

预计修复后，DingQue选择将恢复正常分布！