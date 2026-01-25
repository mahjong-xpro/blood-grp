# NaN问题修复

> 修复时间：2026-01-26  
> 问题：模型输出logits全是NaN

## 🐛 问题描述

**错误信息**：
```
RuntimeError: failed to execute `react_batch` on Python engine
ValueError: Expected parameter logits (Tensor of shape (800, 32)) of distribution Categorical(logits: torch.Size([800, 32])) to satisfy the constraint IndependentConstraint(Real(), 1), but found invalid values:
tensor([[nan, nan, nan, ...], ...], device='cuda:0')
```

**根本原因**：
1. 在定缺选择阶段，我们调用了Agent的`set_scene`和`get_reaction`
2. Agent需要处理观察（observation），但是模型从未训练过定缺选择阶段的观察
3. 定缺选择阶段的观察格式可能包含模型未见过的情况（例如`ding_que`为`None`）
4. 这导致模型输出NaN

---

## ✅ 修复方案

**核心思路**：在定缺选择阶段，不调用Agent，而是直接自动选择定缺。

**修复内容**：

1. **修改 `game.rs` 的 `poll()` 函数**：
   - 在定缺选择阶段，跳过调用Agent的`set_scene`
   - 只有在非定缺选择阶段才调用Agent

2. **修改 `game.rs` 的 `commit()` 函数**：
   - 在定缺选择阶段，设置所有玩家的`last_reactions`为`Event::None`
   - `step()`函数会自动为所有玩家选择定缺（基于手牌统计）
   - 只有在非定缺选择阶段才调用Agent的`get_reaction`

**修复位置**：
- `libblood/src/arena/game.rs:78-105` - `poll()`函数
- `libblood/src/arena/game.rs:173-200` - `commit()`函数

---

## 📊 修复效果

### 修复前：
- ❌ 在定缺选择阶段调用Agent
- ❌ Agent处理未训练过的观察状态
- ❌ 模型输出NaN，导致程序崩溃

### 修复后：
- ✅ 在定缺选择阶段不调用Agent
- ✅ 直接自动选择定缺（基于手牌统计）
- ✅ 模型不会处理未训练过的状态
- ✅ 不会产生NaN

---

## 🔄 工作流程

**定缺选择阶段**：
1. `poll()`返回`Poll::InGame`（因为`ding_que_phase == true`）
2. 跳过调用Agent的`set_scene`
3. `commit()`设置所有玩家的`last_reactions`为`Event::None`
4. `step()`检测到`Event::None`，自动为所有玩家选择定缺
5. 所有玩家选择定缺后，退出定缺选择阶段，开始正常游戏

**正常游戏阶段**：
1. `poll()`返回`Poll::InGame`（如果`can_act()`返回`true`）
2. 调用Agent的`set_scene`和`get_reaction`
3. Agent返回正常的反应（打牌、碰、杠等）
4. 游戏正常进行

---

## 🔗 相关文件

- `libblood/src/arena/game.rs` - 主要修复文件
- `libblood/src/arena/board.rs` - 定缺选择处理逻辑
- `DEADLOCK_FIX_V2.md` - 之前的死循环修复

---

**最后更新**: 2026-01-26  
**修复状态**: ✅ 已完成
