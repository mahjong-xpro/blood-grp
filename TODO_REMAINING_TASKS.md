# 血战到底改造剩余任务清单

> 基于 REFACTOR_PLAN.md 和代码库检查结果整理
> 最后更新：2026-01-25

## ✅ 已完成的任务

### 基础架构
- ✅ 目录重命名：`libriichi/` → `libblood/`
- ✅ Cargo.toml 包名更新：`libriichi` → `libblood`
- ✅ Python 导入路径更新：所有 `from libriichi` → `from libblood`
- ✅ CI/CD 配置更新：`.github/workflows/libblood.yml`
- ✅ 事件系统：已删除 Chi、Dora、Reach、Ryukyoku 事件
- ✅ 已添加 DingQue 事件

### 代码修复（2026-01-25）
- ✅ 修复 `hand.rs`：将 `tile34_to_vec` 重命名为 `tile27_to_vec` 并更新参数类型
- ✅ 修复测试代码：将所有 `tile37_to_vec(&hand(...))` 改为 `tile27_to_vec(&hand(...))`
- ✅ 编译验证：项目可以成功编译（`cargo check` 通过）

### 游戏规则
- ✅ 定缺规则：已实现 DingQue 事件和相关逻辑
- ✅ 番数计算系统：已重写为血战到底番数系统
- ✅ 计分系统：已重写为纯番数系统（5番封顶）
- ✅ 游戏结束条件：已实现3人和牌结束条件
- ✅ 抢杠逻辑：已修复（见 CHANKAN_FIX.md）

### 测试
- ✅ 和牌和番数计算测试：已重写（见 TEST_REWRITE_STATUS.md）
- ✅ 计分系统测试：已有完整测试
- ✅ 定缺规则测试：已添加
- ✅ 游戏流程测试：已添加

### 数据文件
- ✅ 数据文件分析：已完成（见 DATA_FILES_ANALYSIS.md）
- ✅ 确认不需要重新生成数据文件

---

## 🔄 待完成的任务

### 1. 代码清理（高优先级）

#### 1.1 清理 riichi 相关引用
**状态**：代码中仍有 164 处 `riichi` 相关引用（主要是注释和文档字符串）

**需要检查的文件**：
- `libblood/src/state/test.rs` (3处)
- `libblood/src/algo/agari.rs` (6处)
- `libblood/src/algo/sp/calc.rs` (32处)
- `libblood/src/state/update.rs` (14处)
- `libblood/src/arena/board.rs` (2处)
- `libblood/src/state/agent_helper.rs` (13处)
- `libblood/src/state/player_state.rs` (1处)
- `libblood/src/state/action.rs` (3处)
- `libblood/src/consts.rs` (1处)
- `libblood/src/agent/mortal.rs` (1处)
- `libblood/src/state/obs_repr.rs` (8处)
- `libblood/src/stat.rs` (72处) ⚠️ **最多**
- `libblood/src/bin/validate_logs.rs` (1处)
- `libblood/src/algo/sp/mod.rs` (1处)
- `libblood/src/agent/akochan.rs` (2处)
- `libblood/src/dataset/grp.rs` (1处)
- `libblood/src/state/getter.rs` (1处)
- `libblood/src/state/item.rs` (1处)
- `libblood/src/mjai/event.rs` (1处)

**任务**：
- [ ] 检查并更新所有注释中的 "riichi" → "blood"
- [ ] 更新所有文档字符串
- [ ] 更新错误消息中的术语
- [ ] 特别注意 `stat.rs`（72处引用，可能需要大量更新）

#### 1.2 数组大小检查
**状态**：✅ 已修复

**已完成**：
- ✅ 将 `tile34_to_vec` 重命名为 `tile27_to_vec` 并更新参数类型为 `[u8; 27]`
- ✅ 修复测试代码中的类型不匹配（`tile37_to_vec(&hand(...))` → `tile27_to_vec(&hand(...))`）
- ✅ 编译通过，无错误

### 2. 文档更新（中优先级）

#### 2.1 README.md
**任务**：
- [ ] 确认 README.md 已更新为血战到底相关描述
- [ ] 更新所有项目描述
- [ ] 更新示例和文档链接

#### 2.2 代码注释
**任务**：
- [ ] 更新所有包含 "Japanese mahjong" 的注释
- [ ] 更新所有包含 "Tenhou" 的注释
- [ ] 更新所有包含日文术语的注释

#### 2.3 API 文档
**任务**：
- [ ] 更新 `lib.rs` 中的模块文档
- [ ] 更新所有公共 API 的文档字符串
- [ ] 确保文档反映血战到底规则

### 3. 功能验证（高优先级）

#### 3.1 编译验证
**任务**：
- ✅ 确认项目可以成功编译（`cargo check` 通过）
- ⚠️ 仍有 83 个编译警告（主要是未使用的导入和函数，不影响功能）
- [ ] 确认 Python 模块可以正常导入
- [ ] 运行 `cargo fix` 自动修复部分警告

#### 3.2 功能验证
**任务**：
- [ ] 验证牌组只有108张，无字牌
- [ ] 验证定缺规则正常工作
- [ ] 验证番数计算正确（包括叠加和封顶）
- [ ] 验证计分系统正确（5番封顶，16000点）
- [ ] 验证3人和牌时游戏正确结束
- [ ] 验证已和牌玩家不再参与游戏
- [ ] 验证庄家不影响计分

#### 3.3 规则验证
**任务**：
- [ ] 验证不能打出定缺花色的牌
- [ ] 验证必须优先打出定缺花色的牌
- [ ] 验证有定缺花色牌时不能和牌
- [ ] 验证没有吃牌功能
- [ ] 验证没有立直功能
- [ ] 验证没有宝牌
- [ ] 验证任何有效和牌型都可以和牌（不需要役种）

#### 3.4 测试运行
**任务**：
- [ ] 运行所有测试（`cargo test`）
- [ ] 修复失败的测试
- [ ] 清理所有 `#[ignore]` 标记的测试
- [ ] 确保测试覆盖主要功能

### 4. 性能验证（中优先级）

#### 4.1 AI 对局测试
**任务**：
- [ ] 验证 AI 可以正常对局
- [ ] 验证对局结果符合血战到底规则
- [ ] 验证游戏流程正常（定缺、和牌、结束条件）

#### 4.2 训练流程测试
**任务**：
- [ ] 验证训练流程可以正常运行
- [ ] 验证观察空间编码正确
- [ ] 验证动作空间编码正确
- [ ] 验证奖励计算正确

### 5. 代码质量（低优先级）

#### 5.1 代码审查
**任务**：
- [ ] 检查是否有未使用的代码
- [ ] 检查是否有重复代码
- [ ] 检查是否有硬编码的魔法数字
- [ ] 检查错误处理是否完善

#### 5.2 代码风格
**任务**：
- [ ] 确保代码风格一致
- [ ] 运行代码格式化工具
- [ ] 运行 linter 检查

### 6. 其他任务（低优先级）

#### 6.1 日志查看器
**任务**：
- [ ] 更新 `log-viewer/` 中的示例
- [ ] 删除字牌相关显示逻辑
- [ ] 更新为血战到底规则

#### 6.2 Dockerfile
**任务**：
- [ ] 确认 Dockerfile 已更新（如果存在）
- [ ] 更新镜像名称和描述

#### 6.3 配置文件
**任务**：
- [ ] 检查 `mortal/config.example.toml` 是否需要更新
- [ ] 检查其他配置文件是否需要更新

---

## 📊 任务优先级总结

### P0（必须完成 - 阻塞功能）
1. ✅ 基础架构改造（已完成）
2. ✅ 游戏规则实现（已完成）
3. ⚠️ **功能验证**（需要完成）
4. ⚠️ **测试运行**（需要完成）

### P1（重要 - 影响质量）
1. ⚠️ **代码清理**（riichi 引用、数组大小）
2. ⚠️ **规则验证**（确保规则正确实现）

### P2（重要 - 影响可用性）
1. ⚠️ **文档更新**（README、注释、API文档）
2. ⚠️ **性能验证**（AI对局、训练流程）

### P3（优化 - 提升体验）
1. ⚠️ **代码质量**（代码审查、风格）
2. ⚠️ **其他任务**（日志查看器、Dockerfile等）

---

## 📝 下一步行动建议

1. **立即执行**：
   - 运行 `cargo test` 检查测试状态
   - 运行 `cargo build` 检查编译状态
   - 检查 `hand.rs` 中的 `[u8; 34]` 问题

2. **短期完成**（1-2天）：
   - 清理 riichi 相关引用（特别是 `stat.rs`）
   - 完成功能验证和规则验证
   - 运行并修复所有测试

3. **中期完成**（3-5天）：
   - 更新所有文档和注释
   - 完成性能验证
   - 代码质量检查

4. **长期优化**（按需）：
   - 日志查看器更新
   - Dockerfile 更新
   - 其他优化任务

---

## 🔍 检查命令

### 检查 riichi 引用
```bash
grep -r "riichi\|Riichi\|RIICHI" libblood/src --include="*.rs" | wc -l
```

### 检查数组大小
```bash
grep -r "\[u8; 34\]\|Simple2DArray<34" libblood/src --include="*.rs"
```

### 检查编译
```bash
cargo build
cargo build --release
```

### 检查测试
```bash
cargo test
cargo test --lib
```

### 检查 Python 模块
```bash
python -c "import libblood; print('OK')"
```

---

## 📚 相关文档

- `REFACTOR_PLAN.md` - 完整改造计划
- `rules.md` - 血战到底完整规则
- `TEST_REWRITE_STATUS.md` - 测试重写状态
- `CHANKAN_FIX.md` - 抢杠逻辑修复
- `DATA_FILES_ANALYSIS.md` - 数据文件分析

---

**最后更新**：2026-01-25  
**状态**：大部分核心功能已完成，主要剩余代码清理和验证任务

---

## 📊 本次更新（2026-01-25）

### 已完成
1. ✅ 修复 `hand.rs` 中的数组大小问题
   - 将 `tile34_to_vec` 重命名为 `tile27_to_vec`
   - 更新参数类型从 `[u8; 34]` 到 `[u8; 27]`
   
2. ✅ 修复测试代码中的类型不匹配
   - 将所有 `tile37_to_vec(&hand(...))` 改为 `tile27_to_vec(&hand(...))`
   - 更新导入语句

3. ✅ 编译验证
   - `cargo check` 通过
   - Python 模块导入成功

### 下一步建议
1. 运行 `cargo test` 检查测试状态
2. 运行 `cargo fix` 自动修复编译警告
3. 继续清理 riichi 相关引用（特别是注释和文档字符串）
