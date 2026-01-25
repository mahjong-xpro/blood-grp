# 系统全面检查与清理计划

## 目标
全面检查、修复、清理日本麻将残留代码，确保血战到底规则的一致性。

## 检查范围
- libblood/src/ 下的所有模块
- mortal/ 下的 Python 代码
- 文档和配置文件

## 模块列表

### 核心模块
1. **consts.rs** - 常量定义
2. **tile.rs** - 牌定义
3. **hand.rs** - 手牌处理
4. **macros.rs** - 宏定义

### state/ 模块（游戏状态）
5. **state/player_state.rs** - 玩家状态
6. **state/update.rs** - 状态更新
7. **state/obs_repr.rs** - 观察编码
8. **state/action.rs** - 动作定义
9. **state/agent_helper.rs** - Agent 辅助函数
10. **state/getter.rs** - 状态获取
11. **state/item.rs** - 牌河项
12. **state/sp_tables.rs** - 单玩家表
13. **state/test.rs** - 测试

### algo/ 模块（算法）
14. **algo/agari.rs** - 和牌判断
15. **algo/shanten.rs** - 向听数计算
16. **algo/point.rs** - 计分
17. **algo/sp/** - 单玩家算法
   - algo/sp/calc.rs
   - algo/sp/candidate.rs
   - algo/sp/state.rs
   - algo/sp/tile.rs

### arena/ 模块（竞技场）
18. **arena/board.rs** - 游戏板
19. **arena/game.rs** - 游戏逻辑
20. **arena/result.rs** - 结果处理
21. **arena/one_vs_three.rs** - 1v3 模式
22. **arena/two_vs_two.rs** - 2v2 模式

### dataset/ 模块（数据集）
23. **dataset/gameplay.rs** - 游戏数据
24. **dataset/grp.rs** - GRP 数据
25. **dataset/invisible.rs** - 不可见状态

### agent/ 模块（AI代理）
26. **agent/mortal.rs** - Mortal 代理
27. **agent/akochan.rs** - Akochan 代理
28. **agent/tsumogiri.rs** - 随机代理
29. **agent/batchify.rs** - 批处理
30. **agent/mjai_log.rs** - mjai 日志
31. **agent/py_agent.rs** - Python 代理
32. **agent/defs.rs** - 代理定义

### 其他模块
33. **mjai/** - mjai 协议
34. **stat.rs** - 统计
35. **rankings.rs** - 排名
36. **array.rs** - 数组工具
37. **vec_ops.rs** - 向量操作
38. **py_helper.rs** - Python 辅助

### Python 代码
39. **mortal/** - Python 训练代码

## 检查清单

### 关键词检查
- [ ] riichi / 立直
- [ ] dora / 宝牌
- [ ] aka / 红5
- [ ] jihai / 字牌
- [ ] bakaze / 场风
- [ ] jikaze / 自风
- [ ] honba / 本场
- [ ] kyotaku / 供托
- [ ] chi / 吃
- [ ] tedashi / 手出
- [ ] uradora / 里宝
- [ ] yakuman / 役满
- [ ] fu / 符
- [ ] yaku / 役

### 功能检查
- [ ] 牌组是否正确（108张，无字牌）
- [ ] 计分系统是否正确（纯番数，5番封顶）
- [ ] 定缺逻辑是否正确
- [ ] 3人和牌结束条件
- [ ] 观察编码是否正确
- [ ] 动作空间是否正确（无立直、无吃）

## 检查计划

### 阶段1：核心模块检查
1. consts.rs
2. tile.rs
3. hand.rs
4. macros.rs

### 阶段2：state/ 模块检查
5-13. state/ 下所有文件

### 阶段3：algo/ 模块检查
14-17. algo/ 下所有文件

### 阶段4：arena/ 模块检查
18-22. arena/ 下所有文件

### 阶段5：dataset/ 模块检查
23-25. dataset/ 下所有文件

### 阶段6：agent/ 模块检查
26-32. agent/ 下所有文件

### 阶段7：其他模块检查
33-38. 其他模块

### 阶段8：Python 代码检查
39. mortal/ 下所有文件

## 修复原则

1. **完全删除**日本麻将相关代码，不保留兼容性代码
2. **确保功能正确**，修复后必须通过测试
3. **更新注释**，确保注释反映血战到底规则
4. **保持代码风格**一致

## 执行顺序

按模块顺序逐个检查，每个模块：
1. 搜索关键词
2. 阅读代码
3. 识别问题
4. 制定修复方案
5. 执行修复
6. 验证修复
