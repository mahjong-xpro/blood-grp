# Blood-V2 深度评审报告

> 评审日期: 2026-02-24 | 最后更新: 2026-02-25
> 状态: **全部完成** — 64 项问题已处理，系统就绪待训练

---

## 1. 执行摘要

Blood-V2 是四川血战麻将 AI 系统，采用 Rust 引擎 + PyTorch 神经网络 + PPO 强化学习架构。

**初始评审发现**: 35 项问题（2 S0 严重 / 5 S1 高 / 10 S2 中 / 18 S3 低），主要瓶颈为 ISMCE 评估管道失效、SP Table 退化、训练规模不足。

**补充审查发现**: 29 项额外问题（#36-#64），包括推理模型架构不匹配、测试编译失败、文档过时等。

**最终状态**: 64 项全部处理完成。系统需要重新编译 Rust 并从头训练（架构变更）。

---

## 2. 系统架构概览（更新后）

```
┌─────────────────────────────────────────────────────────┐
│                    Blood-V2 System                       │
├──────────────┬──────────────────┬────────────────────────┤
│  Rust Engine │  Neural Network  │   Training Pipeline    │
│  crates/     │  python/blood/   │   python/blood/        │
│              │  model/          │   training/            │
│ • 7阶段状态机│ • ~37M 参数      │ • SF2 PPO框架          │
│ • 470ch 学生 │ • 4层TileAttn    │ • 5阶段课程学习        │
│ • +52ch Oracle│ • 2层LSTM(512)  │ • 超参数动态调度       │
│ • SP Table   │ • 解耦Actor-Critic│ • Elo追踪系统         │
│ • ISMCE搜索  │ • Focal Loss     │ • 联赛Elo加权采样      │
│ • SP缓存     │ • Oracle蒸馏     │ • TensorBoard监控      │
│ • 22常量导出 │ • 循环式segments │ • 纯函数配置注入       │
└──────────────┴──────────────────┴────────────────────────┘
```

---

## 3. 各子系统评审

### 3.1 Rust 引擎 — ✅ 优秀

| 项目 | 评审结果 | 修复状态 |
|------|---------|---------|
| 游戏逻辑 | 7阶段状态机，17种番型正确 | 无需修复 |
| SP Table | MAX_SAMPLES 原为 0，iishanten 退化 | ✅ 已修复: MAX_SAMPLES=5 |
| ISMCE 搜索 | 原无防守逻辑，采样不约束 | ✅ 已修复: `evaluate_discards_full()` 统一入口 |
| Oracle 观测 | 每步重算 3 对手 SP（性能瓶颈） | ✅ 已修复: FxHashMap SP 缓存 |
| 通道常量 | Python 端硬编码，布局变更脆弱 | ✅ 已修复: 22 个 CH_* 常量从 Rust 导出 |
| 学生观测 | 464 通道，缺少对手手牌信息 | ✅ 已修复: 470 通道 (Section 14) |
| Oracle 观测 | 12 冗余通道 | ✅ 已修复: 替换为 SP 摘要+危险度 |
| 地胡定义 | 可能与地方规则不符 | ✅ 已验证: 正确实现 |
| 花猪/大叫 | 罚分计算可能有误 | ✅ 已验证: `finalize_scoring()` 正确 |
| Env info dict | 缺少 winners/scores 字段 | ✅ 已修复: step() 返回 winners + scores |
| assert 级别 | Oracle 用 debug_assert（release 跳过） | ✅ 已修复: 改为 assert_eq! |

### 3.2 神经网络 — ✅ 优秀（已重构）

| 项目 | 评审结果 | 修复状态 |
|------|---------|---------|
| SuitAwareConv1d | 花色隔离卷积，设计亮点 | 无需修复 |
| TileAttention | 原仅 2 层，跨花色交互不足 | ✅ 已重构: 循环式架构，支持任意层数，默认 4 层 |
| BottleneckBlock | SE 注意力，设计合理 | 无需修复 |
| enc_proj 压缩 | 原 6.75x 压缩过大 | ✅ 已修复: 可选 2 层渐进压缩 MLP |
| LSTM | 原单层 512-dim | ✅ 已升级: 2 层 512-dim |
| Action mask | `-1e9` 在 float16 溢出 | ✅ 已修复: `torch.finfo(dtype).min` |
| oracle_num_blocks | cfg(25) vs yaml(20) 不匹配 | ✅ 已修复: 统一为 20 |
| 听牌预测 | BCE 不平衡 | ✅ 已修复: Focal Loss (α=0.25, γ=2.0) |
| 推理模型 | 架构与训练编码器不匹配 | ✅ 已重构: PolicyModel 使用相同循环式架构 |
| ONNX 导出 | 常量过时 + 返回值不匹配 | ✅ 已修复: 动态导入 + tuple 解包 |
| __init__.py | 新模块未导出 | ✅ 已修复: 所有关键类已导出 |

### 3.3 奖励系统与环境 — ✅ 良好

| 项目 | 评审结果 | 修复状态 |
|------|---------|---------|
| terminated 双重计算 | 排名奖励被跳过 | ✅ 已修复: finalize_scoring() 后统一计算 |
| _rust_engine_ok | 一旦 False 永不恢复 | ✅ 已修复: 100 步冷却 + 3 次永久禁用 |
| 排名奖励 | 默认禁用 | ✅ 已修复: competitive/elite 启用 |
| 安全打牌奖励 | 默认禁用 | ✅ 已修复: competitive 0.015 |
| 向听奖励 | 可能导致贪婪追求 | ✅ 已修复: 番数加权 + 衰减调度 |
| 全局步数追踪 | 每 worker 独立计数，衰减 512× 过慢 | ✅ 已修复: 按并行 env 数校正 |
| 排名平局 | 训练环境不处理平局 | ✅ 已修复: 平均排名算法 |
| 双重计算防护 | Warmup 塑形+结构化奖励可叠加 | ✅ 已修复: 代码级互斥防护 |

### 3.4 课程学习与 PPO — ✅ 良好

| 项目 | 评审结果 | 修复状态 |
|------|---------|---------|
| Warmup→Competitive | 同时变更 6 个超参数 | ✅ 已修复: 新增 warmup_transition 过渡阶段 |
| 联赛多样性 | α=3.0 过于集中 | ✅ 已修复: α=2.0 + 均匀下限 + 自对弈 |
| Checkpoint 排序 | 使用 st_mtime（不可靠） | ✅ 已修复: 文件名正则排序 |
| 训练规模 | 17M 步不足 | ✅ 已修复: elite 50M，总 ~57.5M |
| 熵系数 | 各阶段静态，无动态调度 | ✅ 已修复: HyperparamScheduler (cosine/linear/cyclic/step) |
| adv_clip | Elite 5.0 过于宽松 | ✅ 已修复: 线性收紧 5.0→3.0 |
| Warmup LSTM | 禁用 | ⚙️ 设计决策: 通过 transition 阶段缓解 |
| Elo 追踪 | 无持久化评分系统 | ✅ 已修复: EloTracker + TensorBoard + Elo 加权采样 |
| sys.argv 污染 | 全局修改不恢复 | ✅ 已修复: 纯函数 + try/finally |
| log_softmax | 每 minibatch 不必要计算 | ✅ 已修复: 每 100 步节流 |

### 3.5 评估系统 — ✅ 良好（原有严重缺陷已修复）

| 项目 | 评审结果 | 修复状态 |
|------|---------|---------|
| ISMCE 参数缺失 | hand/tiles_seen 未传入，搜索从不执行 | ✅ 已修复: _extract_ismce_state() |
| RTPA/ISMCE 互斥 | if/elif 导致无法协作 | ✅ 已修复: 顺序执行 RTPA→ISMCE |
| ISMCE 防守 | rollout 仅贪婪向听，无防守 | ✅ 已修复: evaluate_discards_full() 全链路 |
| 对手听牌估计 | 仅用副露比≥0.5 | ✅ 已修复: 6 信号多因子评分 |
| 终盘放大 | 阶跃函数 | ✅ 已修复: 线性斜坡 |
| ISMCE 采样 | 忽略对手行为模型 | ✅ 已修复: 约束采样（尊重定缺） |
| log-space 混合 | 尺度不匹配 | ✅ 已修复: 标准化 |
| Arena 座位 | 固定位置 0 | ✅ 已修复: 随机座位 |
| Arena 排名 | 平局乐观 | ✅ 已修复: 平均排名 |
| Arena 胜利检测 | 座位轮换时不可靠 | ✅ 已修复: Rust env 返回 winners 列表 |
| 通道偏移量 | 硬编码魔法数字 | ✅ 已修复: 从 consts 导入 |
| Oracle CE 加权 | 连败时归零 | ✅ 已修复: softmax 加权 |
| 缓存计数器 | checkpoint 重载后脆弱 | ✅ 已修复: 精确匹配 + 重置方法 |

---

## 4. 完整问题清单（64 项）

### S0 严重 — 功能失效

| # | 问题 | 状态 |
|---|------|------|
| 1 | ISMCE 调用缺少 hand/tiles_seen 参数 | ✅ |
| 2 | RTPA 和 ISMCE 互斥 (if/elif) | ✅ |
| 36 | 推理模型编码器架构与训练不匹配 | ✅ |
| 37 | evaluate.py 回退路径硬编码 -1e9 | ✅ |

### S1 高 — 训练/决策质量

| # | 问题 | 状态 |
|---|------|------|
| 3 | SP Table MAX_SAMPLES=0 | ✅ |
| 4 | terminated 双重计算 | ✅ |
| 5 | _rust_engine_ok 永不恢复 | ✅ |
| 6 | Action mask -1e9 float16 溢出 | ✅ |
| 7 | oracle_num_blocks 不匹配 | ✅ |
| 38 | inference.py get_action() 硬编码 -1e9 | ✅ |
| 39 | _cache_gen 计数器 checkpoint 重载脆弱 | ✅ |
| 40 | 每 worker _global_env_steps 衰减 512× 过慢 | ✅ |
| 41 | Oracle obs 每步 3 对手 SP Table（3× 成本） | ✅ |

### S2 中 — 性能/准确性

| # | 问题 | 状态 |
|---|------|------|
| 8 | Warmup→Competitive 同时变更 6 超参数 | ✅ |
| 9 | 联赛多样性不足 | ✅ |
| 10 | Checkpoint 排序用 st_mtime | ✅ |
| 11 | Arena 座位固定 | ✅ |
| 12 | 对手听牌仅用副露比 | ✅ |
| 13 | 终盘放大阶跃函数 | ✅ |
| 14 | log-space 混合尺度不匹配 | ✅ |
| 15 | ISMCE 采样忽略对手行为 | ✅ |
| 16 | ISMCE rollout 无防守逻辑 | ✅ |
| 17 | 训练规模 17M 不足 | ✅ |
| 42 | Python 通道偏移量未从 Rust 导出 | ✅ |
| 43 | evaluate.py 重复硬编码通道偏移 | ✅ |
| 44 | Arena 胜利检测回退不可靠 | ✅ |
| 45 | Oracle CE 优势加权连败归零 | ✅ |
| 46 | 训练排名不处理平局 | ✅ |
| 47 | Warmup 塑形+结构化奖励双重计算 | ✅ |
| 48 | _OPP_KAWA_STRIDE 脆弱耦合 | ✅ |
| 57 | export_onnx.py OBS_CHANNELS=464 过时 | ✅ |
| 58 | export_onnx.py OnnxWrapper 返回值不匹配 | ✅ |

### S3 低 — 改进建议

| # | 问题 | 状态 |
|---|------|------|
| 18 | 缺少对手手牌数编码 | ✅ |
| 19 | 缺少副露来源信息 | ✅ |
| 20 | Oracle 12 冗余通道 | ✅ |
| 21 | TileAttention 仅 2 层 | ✅ 重构为 4 层 |
| 22 | enc_proj 6.75x 压缩 | ✅ |
| 23 | LSTM 单层 512-dim | ✅ 升级为 2 层 |
| 24 | 听牌预测 BCE 不平衡 | ✅ Focal Loss |
| 25 | 向听奖励贪婪追求 | ✅ 番数加权 |
| 26 | 排名奖励默认禁用 | ✅ |
| 27 | 安全打牌奖励禁用 | ✅ |
| 28 | Warmup 禁用 LSTM | ⚙️ 设计决策 |
| 29 | 熵系数无动态调度 | ✅ HyperparamScheduler |
| 30 | Elite adv_clip 5.0 过宽 | ✅ 线性收紧 |
| 31 | 无 Elo/胜率追踪 | ✅ EloTracker |
| 32 | 平局排名乐观 | ✅ |
| 33 | 地胡定义 | ✅ 已验证正确 |
| 34 | 花猪/大叫罚分 | ✅ 已验证正确 |
| 35 | danger_scores 纯启发式 | ✅ 多因子评分 |
| 49 | sys.argv 全局污染 | ✅ |
| 50 | log_softmax 每 minibatch 不必要 | ✅ 节流 |
| 51 | elo.py get_leaderboard(0) 返回空 | ✅ |
| 52 | oracle.rs debug_assert 替代 assert_eq | ✅ |
| 53 | ismce.py 热路径内 import | ✅ |
| 54 | test_model.py 引用旧编码器属性 | ✅ |
| 55 | test_model.py 旧格式 checkpoint 测试 | ✅ |
| 56 | test_obs.rs encode_oracle_obs() 参数数量 | ✅ |
| 59 | oracle.rs 用 HashMap 而非 FxHashMap | ✅ |
| 60 | eval/__init__.py 未导出新模块 | ✅ |
| 61 | training/__init__.py 为空 | ✅ |
| 62 | ARCHITECTURE.md 严重过时 | ✅ 完全重写 |
| 63 | model/__init__.py 未导出 PolicyModel | ✅ |
| 64 | test_smoke.py 硬编码 OBS_CHANNELS | ✅ |

---

## 5. 关键架构变更

### 5.1 ISMCE 防守逻辑全链路

```
evaluate.py: NeuralAgent.__call__()
  → _extract_opponent_state()        # 从 obs 提取对手状态
  → ISMCESearcher.select_action()    # 传入对手信息
    → ismce_evaluate_full()          # PyBind 新接口
      → danger_scores_enhanced()     # 多因子危险度（一次性）
      → sample_world_constrained()   # 尊重对手定缺
      → simulate_draws_with_defense() # 同向听优先安全牌
```

### 5.2 编码器循环式架构

```
输入 (B, 470, 27)
  → SuitAwareConv1d(470→256) + GroupNorm + Mish
  → SuitPositionalEncoding(256)
  → [Segment 0: BottleneckBlock×5 → TileAttention(4heads)]
  → [Segment 1: BottleneckBlock×5 → TileAttention(4heads)]
  → [Segment 2: BottleneckBlock×5 → TileAttention(4heads)]
  → [Segment 3: BottleneckBlock×5 → TileAttention(4heads)]
  → Flatten(6912) → LayerNorm → Linear(1024)
  → LSTM(1024→512, 2层)
  → 解耦 Actor(512→34) / Critic(512→1)
```

### 5.3 超参数动态调度

```yaml
# elite.yaml 示例
blood_schedule_entropy: "cosine,0.02,0.005,0,40000000"
blood_schedule_adv_clip: "linear,5.0,3.0,10000000,40000000"
```

支持 linear / cosine / cyclic / step 四种调度类型。

### 5.4 Elo 追踪系统

- 多人配对 Elo（4人麻将每对贡献更新，K/(n-1) 缩放）
- 自适应 K 因子（新玩家 64，成熟 32）
- JSON 原子持久化
- 可选 Elo 加权对手采样（高斯 σ=200）
- TensorBoard: `blood/elo_current`, `blood/elo_best`, `blood/elo_pool_mean`

### 5.5 Oracle SP 缓存

- `OracleSpCache = FxHashMap<(usize, OppSpCacheKey), OppSpCacheEntry>`
- 缓存键: `(tehai, num_melds, tiles_left)` — 故意省略 tiles_seen（±1 影响极小）
- `reset()` 时清空，每局最多 ~150 条目
- 预计减少 ~50% SP 计算量

---

## 6. 训练配置（更新后）

| 阶段 | 步数 | 对手 | LSTM | 熵系数 | 特殊 |
|------|------|------|------|--------|------|
| warmup | 2M | RuleBot | ❌ | 0.01 | 基础策略学习 |
| warmup_transition | 500K | RuleBot | ✅ | 0.01 | LSTM 稳定化 |
| competitive | 1M | 自对弈 | ✅ | 0.01→0.05 | 熵预热 |
| competitive_distill | 4M | 自对弈 | ✅ | 0.05 | Oracle 价值蒸馏 |
| elite | 50M | 自对弈 | ✅ | 0.02→0.005 | RTPA+ISMCE, adv_clip 5→3 |

**跨阶段一致参数**:
- `blood_num_tile_attn_layers: 4`
- `blood_tile_attn_heads: 4`
- `rnn_num_layers: 2`
- `rnn_size: 512`

---

## 7. 超人类路径分析

### 当前水平评估

修复后预计达到**强竞技水平**。三大原始瓶颈（SP Table 退化、评估管道失效、训练规模不足）已全部解决。

### 进一步提升方向

| 维度 | 当前状态 | 提升方向 |
|------|---------|---------|
| 搜索深度 | ISMCE 约束采样+防守 rollout | 更深的 rollout + MCTS |
| 训练规模 | 57.5M 步 | 100M+ 步 |
| 对手建模 | 6 信号听牌估计 | 神经网络对手模型 |
| 防守 | 启发式危险度 | 学习型防守策略 |
| 时序推理 | 2 层 LSTM | Transformer 替代 |
| 番型规划 | 向听+番数加权 | 显式番型目标网络 |

---

## 8. 重要注意事项

1. **⚠️ 需要从头训练**: TileAttention 4 层 + LSTM 2 层改变了模型架构，旧 checkpoint 不兼容
2. **⚠️ 需要重新编译 Rust**: SP 缓存、常量导出、winners 字段、assert_eq 等修改了 Rust 代码
3. **⚠️ 配置一致性**: 所有 YAML 配置已统一架构参数，跨阶段必须保持一致
4. **推理模型兼容**: `PolicyModel.from_sf2_checkpoint()` 支持新旧两种 state_dict 格式

---

## 附录: 修改文件清单

### Rust (crates/)
- `engine/src/algo/ismce.rs` — evaluate_discards_full(), 防守 rollout
- `engine/src/obs/oracle.rs` — SP 缓存, FxHashMap, assert_eq
- `engine/src/obs/student.rs` — 470 通道 Section 14
- `engine/src/consts.rs` — 22 个 CH_* 常量
- `engine/tests/test_obs.rs` — encode_oracle_obs() 参数修复
- `pybind/src/env.rs` — winners/scores 字段, SP 缓存传递
- `pybind/src/ismce_py.rs` — ismce_evaluate_full()
- `pybind/src/lib.rs` — 常量导出

### Python — 模型 (python/blood/model/)
- `encoder.py` — 循环式 segments/tile_attns 架构
- `oracle.py` — 同上
- `factory.py` — oracle 参数传递, reset_cache_counters()
- `inference.py` — PolicyModel 重写, 旧格式兼容
- `heads.py` — Focal Loss

### Python — 训练 (python/blood/training/)
- `runner.py` — 纯函数配置注入, try/finally
- `callbacks.py` — 调度器+Elo 集成
- `losses.py` — softmax 加权, log_softmax 节流
- `league.py` — Elo 加权采样
- `scheduler.py` — 新建: HyperparamScheduler

### Python — 评估 (python/blood/eval/)
- `evaluate.py` — 对手状态提取, 常量导入, -1e38
- `rtpa.py` — 常量导入
- `ismce.py` — full evaluator 路由, 模块级 import
- `arena.py` — winners 检测, Elo 集成
- `elo.py` — 新建: EloTracker

### Python — 环境 (python/blood/env/)
- `selfplay_env.py` — 步数校正, 平局排名, 双重计算防护
- `blood_env.py` — 冷却恢复机制

### Python — 其他
- `cfg.py` — 新增 ~15 个配置参数
- `consts.py` — Rust 常量导入 + 回退
- `__init__.py` (eval/, training/, model/) — 模块导出

### 配置 (configs/)
- warmup.yaml, warmup_transition.yaml, competitive.yaml, competitive_distill.yaml, elite.yaml, default.yaml — 统一架构参数 + 调度配置

### 测试 (tests/)
- test_model.py — segments/tile_attns 断言, 新格式 checkpoint 测试
- test_smoke.py — 动态常量导入

### 脚本 (scripts/)
- export_onnx.py — 动态常量 + tuple 解包

### 文档
- ARCHITECTURE.md — 完全重写
- DEEP_REVIEW_2026_02_24.md — 本文档
