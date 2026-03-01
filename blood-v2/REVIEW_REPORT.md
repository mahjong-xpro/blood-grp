# Blood-V2 深度评审报告

## 评审范围
6 大模块，~23K 行代码（Rust 7.5K + Python 6.9K + Tests 4K + Configs/Scripts 4.9K）

---

## Critical (1)

### C1. Mixed Precision GradScaler 创建但从未使用
- **文件**: `training/learner_patch.py`
- **描述**: `torch.amp.GradScaler` 被创建，`autocast` 被应用，但 scaler 从未用于 `scale(loss).backward()` / `unscale_()` / `step()` / `update()`。若 `use_mixed_precision=True`，forward 在 FP16 但 backward 用未缩放的 FP16 梯度，会导致训练不稳定或 NaN。
- **影响**: 当前默认 False，为休眠 bug。一旦启用即触发。
- **修复**: 完整实现 AMP workflow，或在 learner_patch 中集成 scaler 到 optimizer step。

---

## High (10)

### H1. SP Table iishanten EV 始终为零
- **文件**: `algo/sp/calc.rs:278-281`
- **描述**: `fill_lookahead_candidate` 将 14 张听牌手牌传给 `get_win_score`，但 `calc_fan` 期望完整和牌（向听=-1），听牌手牌（向听=0）返回 None → score=0。所有 iishanten 候选的期望值为零，SP Table 退化为仅按概率排序。
- **影响**: SP Table 无法按期望得分区分 iishanten 弃牌，降低观测质量。

### H2. SF2 内部 API 依赖无版本锁定
- **文件**: `training/learner_patch.py`
- **描述**: 猴子补丁依赖 `Learner._calculate_losses` 的 7 元组返回签名和 `self.epoch_actor_losses` 属性。SF2 任何内部 API 变更都会静默破坏训练。
- **修复**: 在 `pyproject.toml` 中锁定 SF2 版本，补丁入口加版本断言。

### H3. ISMCE 未释放 GIL
- **文件**: `pybind/ismce_py.rs`
- **描述**: 4 个 ISMCE 函数在计算期间持有 GIL，阻塞所有 Python 线程。修复仅需一行 `py.allow_threads(|| { ... })`。

### H4. ISMCE 无集成测试
- **文件**: `algo/ismce.rs`（1034 行）
- **描述**: 最大最复杂的模块仅有 3 个辅助函数单元测试。核心评估函数、采样策略、rollout 逻辑完全未测试。

### H5. 游戏引擎 discard 空候选静默卡死
- **文件**: `state/board.rs:543-563`
- **描述**: `do_discard` 在 `discard_candidates()` 为空时返回但不推进阶段，游戏永久卡在 Discard 阶段。正常游戏不应触发，但牌数损坏时会静默死循环。

### H6. 抢杠和（chankan）无测试
- **文件**: `state/board.rs:451-496`
- **描述**: 引擎最复杂的代码路径（加杠→对手荣和→加杠回退→牌恢复→即时雨退款），零测试覆盖。

### H7. 明杠（MinKan）执行无测试
- **文件**: `state/board.rs:734-783`
- **描述**: 明杠的支付逻辑和岭上摸牌完全未测试。

### H8. TurnAttention batch 对齐假设
- **文件**: `model/factory.py:228-241`
- **描述**: 训练时假设 `total % recurrence == 0`，不对齐的 batch 静默跳过 TurnAttention，造成训练-推理不一致。

### H9. 观测编码无通道值级别测试
- **文件**: `tests/test_obs.rs`
- **描述**: 仅测试形状和范围，无测试验证特定游戏状态产生特定通道值。编码 bug 无法被检测。

### H10. 观测编码无中局测试
- **文件**: `tests/test_obs.rs`
- **描述**: 所有测试用初始状态或刚定缺后的状态。牌河、防守、現物通道在测试中全为零。

---

## Medium (20)

| ID | 模块 | 描述 |
|----|------|------|
| M1 | algo/agari | chitoi 路径未更新 `best_fan` 变量（当前无害但脆弱） |
| M2 | algo/agari | 缺少金钩钓（jingoudiao）测试 |
| M3 | algo/agari | 缺少组合番型叠加测试 |
| M4 | algo/ismce | `estimate_fan_quick` 缺少一条龙（yitiaolong）检测，ISMCE rollout 低估番数 |
| M5 | algo/ismce | 定缺约束回退创建非法世界（对手持有定缺花色牌） |
| M6 | algo/ismce | 64 worlds 方差较大（10% 胜率的 95% CI ≈ [2.5%, 17.5%]） |
| M7 | obs/student | 硬编码魔法数字（13.0, 32000.0, 55.0）应引用命名常量 |
| M8 | obs/oracle | SP 缓存键省略 `tiles_seen`，结果可能略微过时 |
| M9 | obs/mask | 无 Reaction/SelfCheck/KanSelect 阶段掩码测试 |
| M10 | pybind/env | 奖励 sqrt 压缩在 Rust 和 Python 两侧重复实现，有分歧风险 |
| M11 | pybind/env | `np.array()` 复制了 PyO3 已零拷贝提供的数据，每步浪费 ~250KB |
| M12 | pybind/ismce | 对手信息向量长度未校验，静默使用默认值 |
| M13 | selfplay_env | 无 Rust 引擎调用超时保护（不同于 BloodMahjongEnv） |
| M14 | eval/arena | `np.random.choice` 使用全局 RNG，评估不可复现 |
| M15 | selfplay_env | `_score_delta_to_fan` 对不匹配的分数差返回 0，fan 统计可能偏低 |
| M16 | state/board | Scoring→Done 需外部调用 `finalize_scoring()`，API 契约隐式 |
| M17 | state/board | 加杠即时雨检测依赖脆弱的 `last_drawn_tile` 状态 |
| M18 | state/board | 缺少暗杠、过手加番、查花猪/查大叫、岭上摸牌耗尽墙的测试 |
| M19 | state/board | CLAUDE.md 动作空间描述与代码不一致（文档说 3 个杠动作，实际 1 个） |
| M20 | model/factory | 跨阶段 checkpoint 加载用 `strict=False`，无架构匹配校验 |

---

## Low (25+)

关键 Low 项摘要：
- 联赛池采样、Elo K-factor 平均、PBT 乘法变异等设计合理但有边界情况
- RTPA 通道偏移计算脆弱，危险乘数硬编码
- 增强分布不均匀（50% identity vs 10% 每个非 identity 排列）
- 自博弈中 agent 始终在座位 0，无座位随机化
- 全局可变冷却状态无锁（multiprocessing 下安全但脆弱）
- 各模块多处代码风格问题（未使用变量、冗余 `if True:`、占位符注释）

---

## 按优先级排序的修复建议

### 立即修复（影响训练正确性）
1. **C1**: 完整实现 AMP 或禁用 GradScaler 创建
2. **H1**: 修复 SP Table iishanten 评分（从听牌手牌正确构造完整手牌再评分）
3. **H2**: 锁定 SF2 版本 + 加版本断言
4. **H3**: ISMCE 函数加 `py.allow_threads()`（一行修复，显著提升推理性能）

### 短期修复（提升可靠性）
5. **H5**: `do_discard` 空候选时 panic 或强制进入 Scoring
6. **H8**: TurnAttention 处理非对齐 batch（pad 或 fallback）
7. **M11**: `np.array()` → `np.asarray()` 消除不必要拷贝
8. **M7**: 魔法数字替换为命名常量
9. **M14**: 评估用 seeded RNG 替代全局 RNG

### 中期修复（补测试）
10. **H4/H6/H7/H9/H10/M18**: 补充 ISMCE 集成测试、chankan/minkan/ankan 测试、观测通道值测试
11. **M4**: `estimate_fan_quick` 加一条龙检测
