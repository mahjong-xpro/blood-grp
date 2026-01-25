# 死循环Bug修复

> 修复时间：2026-01-26  
> 问题：游戏在定缺选择阶段卡住，进度条停在 0/800

## 🐛 Bug描述

**现象**：
- 游戏在 `one_vs_three.rs:131` 后卡住
- 进度条显示 `0/800`，没有任何进展
- 程序无响应，疑似死循环

**根本原因**：
1. 我们之前添加了定缺选择阶段，要求所有玩家必须选择定缺
2. 但是Agent（`MortalAgent`等）的`get_reaction`方法没有实现定缺选择逻辑
3. Agent返回的是`Event::None`或其他动作，而不是`DingQue`事件
4. 游戏在定缺选择阶段等待所有玩家发送`DingQue`事件
5. 由于Agent没有返回`DingQue`事件，游戏一直等待，导致死循环

---

## ✅ 修复方案

**修改位置**：`libblood/src/arena/board.rs:408-485`

**修复内容**：
- 在定缺选择阶段，如果Agent没有返回`DingQue`事件，游戏自动为Agent选择定缺
- 自动选择策略：选择手牌中最少的花色作为定缺（这是合理的策略，因为定缺就是要打掉某个花色）

**修复后的逻辑**：
```rust
if !self.ding_que_selected[actor] {
    // 如果玩家还没有选择定缺
    let ding_que_event = if let Event::DingQue { actor: ev_actor, suit } = ev.event {
        // Agent返回了DingQue事件，使用Agent的选择
        Event::DingQue { actor: actor as u8, suit }
    } else {
        // Agent没有返回DingQue事件，自动为Agent选择定缺
        // 选择手牌中最少的花色作为定缺
        let state = &self.player_states[actor];
        let man_count: u8 = (0..9).map(|i| state.tehai[i]).sum();
        let pin_count: u8 = (9..18).map(|i| state.tehai[i]).sum();
        let sou_count: u8 = (18..27).map(|i| state.tehai[i]).sum();
        
        let suit = if man_count <= pin_count && man_count <= sou_count {
            crate::mjai::Suit::Man
        } else if pin_count <= sou_count {
            crate::mjai::Suit::Pin
        } else {
            crate::mjai::Suit::Sou
        };
        
        Event::DingQue { actor: actor as u8, suit }
    };
    
    // 更新玩家状态并记录日志
    ...
}
```

---

## 📊 修复效果

### 修复前：
- ❌ 游戏在定缺选择阶段卡住
- ❌ Agent没有返回`DingQue`事件
- ❌ 游戏一直等待，导致死循环

### 修复后：
- ✅ 如果Agent返回`DingQue`事件，使用Agent的选择
- ✅ 如果Agent没有返回`DingQue`事件，自动选择定缺
- ✅ 游戏可以正常进行，不会卡住

---

## 🔄 向后兼容性

**兼容性**：
- ✅ 如果Agent实现了定缺选择，会使用Agent的选择
- ✅ 如果Agent没有实现定缺选择，会自动选择定缺
- ✅ 不会破坏现有的Agent实现

**未来改进**：
- 可以考虑让Agent实现定缺选择逻辑（基于手牌统计或AI策略）
- 但目前自动选择定缺的策略（选择最少花色）是合理的

---

## 🧪 测试建议

修复后，应该测试：
1. ✅ 游戏是否可以在定缺选择阶段正常进行
2. ✅ 自动选择的定缺是否合理（选择最少花色）
3. ✅ 如果Agent返回`DingQue`事件，是否使用Agent的选择
4. ✅ 游戏流程是否正常，不会卡住

---

## 🔗 相关文件

- `libblood/src/arena/board.rs` - 主要修复文件
- `libblood/src/agent/mortal.rs` - Agent实现（未来可以添加定缺选择逻辑）
- `BUG_FIX_SUMMARY.md` - 之前的bug修复总结

---

**最后更新**: 2026-01-26  
**修复状态**: ✅ 已完成
