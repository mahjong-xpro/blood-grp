# 死循环Bug修复 V2

> 修复时间：2026-01-26  
> 问题：游戏在定缺选择阶段仍然卡住

## 🐛 问题分析

**根本原因**：
1. 在定缺选择阶段，`can_act()`返回`false`（因为玩家还没有选择定缺，所以不能打牌）
2. 在`game.rs`的`poll()`函数中，如果`can_act()`返回`false`，会跳过调用Agent的`set_scene`和`get_reaction`
3. 这导致`last_reactions`保持为`Event::None`（或之前的值）
4. 虽然我们在`step()`中会自动选择定缺，但是如果没有调用Agent，`last_reactions`可能没有被正确设置

**修复方案**：
1. 在`poll()`函数中，如果是在定缺选择阶段，即使`can_act()`返回`false`，也返回`Poll::InGame`
2. 在`game.rs`的`poll()`和`commit()`函数中，如果是在定缺选择阶段，即使`can_act()`返回`false`，也调用Agent的`set_scene`和`get_reaction`

---

## ✅ 修复内容

### 修复 #1: 修改 `poll()` 函数

**位置**：`libblood/src/arena/board.rs:109-132`

**修改**：
```rust
pub fn poll(&mut self, mut reactions: [EventExt; 4]) -> Result<Poll> {
    loop {
        let poll = self.step(&reactions)?;
        match poll {
            Poll::InGame => {
                // 在定缺选择阶段，即使can_act()返回false，也应该返回InGame
                // 因为我们需要等待Agent返回反应（或自动选择定缺）
                if self.ding_que_phase {
                    return Ok(poll);
                }
                // 正常游戏阶段，检查是否有玩家可以行动
                if self.player_states.iter().any(|c| c.last_cans().can_act()) {
                    return Ok(poll);
                }
            }
            ...
        };
        reactions = Default::default();
    }
}
```

### 修复 #2: 添加辅助方法

**位置**：`libblood/src/arena/board.rs:134-150`

**添加**：
```rust
#[inline]
pub const fn is_ding_que_phase(&self) -> bool {
    self.ding_que_phase
}

#[inline]
pub const fn ding_que_selected(&self, player_id: usize) -> bool {
    self.ding_que_selected[player_id]
}
```

### 修复 #3: 修改 `game.rs` 中的 `poll()` 函数

**位置**：`libblood/src/arena/game.rs:78-100`

**修改**：
```rust
Poll::InGame => {
    let ctx = self.board.agent_context();
    // 在定缺选择阶段，需要处理所有还没有选择定缺的玩家，即使can_act()返回false
    let in_ding_que_phase = self.board.is_ding_que_phase();
    
    for (player_id, state) in ctx.player_states.iter().enumerate() {
        // 在定缺选择阶段，如果玩家还没有选择定缺，需要处理（即使can_act()返回false）
        let needs_scene = if in_ding_que_phase {
            !self.board.ding_que_selected(player_id)
        } else {
            state.last_cans().can_act()
        };
        
        if !needs_scene {
            continue;
        }
        // 调用Agent的set_scene
        ...
    }
}
```

### 修复 #4: 修改 `game.rs` 中的 `commit()` 函数

**位置**：`libblood/src/arena/game.rs:173-188`

**修改**：
```rust
let ctx = self.board.agent_context();
// 在定缺选择阶段，需要处理所有还没有选择定缺的玩家，即使can_act()返回false
let in_ding_que_phase = self.board.is_ding_que_phase();

for (player_id, state) in ctx.player_states.iter().enumerate() {
    // 在定缺选择阶段，如果玩家还没有选择定缺，需要处理（即使can_act()返回false）
    let needs_reaction = if in_ding_que_phase {
        !self.board.ding_que_selected(player_id)
    } else {
        state.last_cans().can_act()
    };
    
    if !needs_reaction {
        continue;
    }
    // 调用Agent的get_reaction
    ...
}
```

---

## 📊 修复效果

### 修复前：
- ❌ 在定缺选择阶段，`can_act()`返回`false`
- ❌ Agent不会被调用，`last_reactions`保持为`Event::None`
- ❌ 虽然会自动选择定缺，但可能导致死循环

### 修复后：
- ✅ 在定缺选择阶段，即使`can_act()`返回`false`，也会调用Agent
- ✅ Agent返回反应（或`Event::None`），然后自动选择定缺
- ✅ 游戏可以正常进行，不会卡住

---

## 🔗 相关文件

- `libblood/src/arena/board.rs` - 主要修复文件
- `libblood/src/arena/game.rs` - 游戏循环修复
- `DEADLOCK_FIX.md` - 第一次修复文档

---

**最后更新**: 2026-01-26  
**修复状态**: ✅ 已完成
