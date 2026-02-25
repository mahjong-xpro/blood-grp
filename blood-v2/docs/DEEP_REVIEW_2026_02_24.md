# Blood-V2 深度评审报告

> 评审日期：2026-02-24  
> 评审范围：Rust引擎、神经网络架构、奖励系统与环境设计、课程学习与PPO超参数、评估系统

---

## 1. 执行摘要

Blood-V2 是一个高质量的血战到底麻将AI系统，采用 Rust 引擎 + PyTorch 神经网络的混合架构，通过 PPO 强化学习与 Oracle 蒸馏进行训练。Rust 引擎规则实现完整正确，17种番型与分数计算均无误；神经网络约35M参数，花色感知编码器设计精巧。但评审发现2个严重Bug（ISMCE参数缺失、RTPA/ISMCE互斥）、多个中等问题（terminated双重计算、SP Table退化、训练规模不足），以及若干设计改进空间。当前系统距超人类水平的主要瓶颈在于：搜索质量（SP Table/ISMCE）、训练规模（17M步不足）、评估管线缺陷。

---

## 2. 系统架构概览

```
┌─────────────────────────────────────────────────────┐
│                    Blood-V2 系统                      │
├──────────────┬──────────────┬───────────────────────┤
│  Rust 引擎    │  神经网络     │  训练与评估            │
│  - 规则逻辑   │  - Student   │  - PPO + Oracle蒸馏   │
│  - 状态机     │    (~21M)    │  - 4阶段课程学习       │
│  - 观测编码   │  - Oracle    │  - 联赛系统            │
│  - SP Table  │    (~14M)    │  - RTPA + ISMCE       │
│  - ISMCE采样  │  - 辅助任务   │  - Arena评估          │
└──────────────┴──────────────┴───────────────────────┘
```

- **Rust引擎**：7阶段状态机，464通道学生观测 + 52通道Oracle观测，含防死锁机制
- **神经网络**：SuitAwareConv1d编码器 + BottleneckBlock/SE注意力 + TileAttention + 解耦Actor-Critic双头 + LSTM时序建模
- **训练管线**：PPO强化学习，KL自适应学习率，花色增强（S₃群），4阶段课程（Warmup→Competitive→Elite→Master）
- **评估系统**：RTPA温度自适应 + ISMCE信息集蒙特卡洛搜索 + Arena统计评估

---

## 3. 各子系统评审结论

### 3.1 Rust引擎

**评级：优秀（局部可改进）**

规则正确性方面表现出色：
- 17种番型计算完整正确
- 分数计算公式 `1000 × 2^(fan-1)`，6番封顶=32000，实现正确
- 定缺规则完整（强制先打定缺花色、定缺未完不能和牌/碰/杠）
- 终局结算（查花猪、查大叫）正确
- 过手加番、抢杠、杠上花/杠上炮、一炮多响均正确

状态机设计稳健：7阶段覆盖所有场景，多层防死锁机制（500次迭代上限、8次停滞检测）。

观测编码丰富但有改进空间：
- 464通道学生观测无信息泄露
- 缺少对手手牌数量显式编码和副露来源信息
- Oracle 52通道中有12通道冗余（对手定缺信息在学生obs中已有）
- `debug_assert` vs `assert` 使用策略合理（核心数据操作用 `debug_assert` 优先性能，观测编码用 `assert`）

**关键瓶颈**：SP Table 的 `MAX_SAMPLES=0` 导致一向听评估退化为粗略估计，这是当前最大的决策质量瓶颈。听牌状态精确计算，但二向听及以上使用粗略估计。

ISMCE（Rust端）存在设计局限：采样未考虑对手行为模型（均匀随机分配未见牌），rollout使用贪心向听最小化不考虑防守，`danger_scores` 纯启发式过于简单。

**争议点**：
- 地胡定义（当前为庄家第一张打出被荣和）可能与部分地方规则不一致
- 花猪/大叫罚分使用自摸计算（含自摸番），部分规则应按荣和计算

### 3.2 神经网络架构

**评级：良好（存在设计风险）**

模型总参数量约35M（Student ~21M + Oracle ~14M）。

编码器设计亮点：
- `SuitAwareConv1d` 花色感知设计正确，三花色同构共享权重
- `BottleneckBlock` + SE注意力适合麻将特征提取
- 推理延迟 CPU 单样本 <5ms，完全满足实时需求

Oracle蒸馏设计完整：
- 双通道蒸馏（策略KL + 价值MSE）
- 两阶段价值蒸馏（先训练oracle value head，再蒸馏）是亮点

**风险点**：
- `TileAttention`（27位置4头自注意力）仅2层，跨花色推理能力可能不足
- `464×27 → 6912 → 1024` 的 `enc_proj` 是6.75x压缩，存在信息瓶颈风险
- LSTM单层512维（competitive/elite阶段），时序建模能力偏弱
- 动作掩码 `-1e9` 在 float16 下会溢出
- `oracle_num_blocks` 在 `cfg.py`(25) 和 yaml(20) 之间不一致

**辅助任务**：
- 对手向听预测（3×5类CE）合理
- 对手听牌预测（81维BCE）在极端不平衡下可能退化，建议使用 Focal Loss

### 3.3 奖励系统与环境设计

**评级：良好（存在Bug）**

奖励设计合理：
- sqrt压缩将32:1线性比率压缩到5.6:1，有效降低方差
- 自摸bonus(0.1) > 放铳penalty(0.05)，鼓励进攻，符合血战特点
- 向听进退奖励是唯一的高密度信号源

花色增强实现正确：6种S₃群排列，obs/action/mask/labels一致变换。

**设计隐患**：
- 向听进退奖励可能导致贪心向听追求（忽略番数价值）
- 安全弃牌奖励默认关闭(0.0)
- 排名奖励默认关闭(0.0)，但血战最终目标是排名
- warmup塑形奖励中放铳判定较粗糙

**Bug**：
- `terminated` 双重计算（第319行 vs 第396行）可能导致排名奖励被跳过
- `_rust_engine_ok` 一旦设为 `False` 永远不恢复，可能降低训练吞吐量

### 3.4 课程学习与PPO超参数

**评级：良好（过渡风险高）**

课程设计：
- 4阶段课程整体合理
- gamma递增（0.99→0.998→0.999）是亮点，逐步扩展时间视野
- Warmup禁用LSTM值得商榷（丢失通用时序特征学习机会）
- **Warmup→Competitive过渡同时改变6个超参数，是训练崩溃高风险点**

PPO超参数：
- KL-adaptive LR 的 masked action space 校准体现深入理解
- entropy从0.01→0.05→0.01的非单调变化有理由但可能过高
- `adv_clip` 在Elite阶段反而放宽（2.0→5.0），需要监控

联赛系统存在问题：
- 多项式衰减采样（α=3.0）过于偏向最新checkpoint，有效多样性低
- 缺少Elo/胜率追踪
- checkpoint排序用 `st_mtime` 而非文件名数字，在NFS等环境可能出错

**训练规模不足**：总计17M env steps，大概率不足以达到超人类水平，建议至少50-100M步。

### 3.5 评估系统

**评级：存在严重缺陷**

RTPA温度自适应方向正确：
- 攻击降温、防御升温、分差调节、残局放大
- 但对手听牌推断使用副露比例(>=0.5)作为代理，准确性低
- 残局放大为阶跃函数，缺乏平滑过渡

**严重Bug**：
- `evaluate.py` 中 ISMCE 调用缺少 `hand`/`tiles_seen` 等关键参数，导致 ISMCE 搜索永远不会执行
- RTPA 与 ISMCE 互斥设计，无法同时生效
- log空间混合存在尺度不匹配风险

Arena评估：
- 座位固定在位置0未轮换，可能引入系统性偏差
- Bootstrap 10000次重采样，统计方法基本正确
- 同分排名计算偏乐观

---

## 4. Bug与问题汇总表

### 4.1 严重（S0）— 功能失效

| # | 子系统 | 问题描述 | 影响 |
|---|--------|----------|------|
| 1 | 评估系统 | `evaluate.py` 中 ISMCE 调用缺少 `hand`/`tiles_seen` 等关键参数 | ISMCE搜索永远不会执行，搜索增强完全失效 |
| 2 | 评估系统 | RTPA 与 ISMCE 互斥设计 | 两个增强机制无法协同，评估时只能二选一 |

### 4.2 高（S1）— 影响训练/决策质量

| # | 子系统 | 问题描述 | 影响 |
|---|--------|----------|------|
| 3 | Rust引擎 | SP Table `MAX_SAMPLES=0`，一向听评估退化 | 最大决策质量瓶颈，向听推进评估不准确 |
| 4 | 奖励系统 | `terminated` 双重计算（第319行 vs 第396行） | 排名奖励可能被跳过，影响训练信号 |
| 5 | 奖励系统 | `_rust_engine_ok` 设为 `False` 后永不恢复 | 训练吞吐量可能持续降低 |
| 6 | 神经网络 | 动作掩码 `-1e9` 在 float16 下溢出 | 混合精度训练时可能产生NaN |
| 7 | 神经网络 | `oracle_num_blocks` cfg.py(25) vs yaml(20) 不一致 | Oracle模型结构可能与预期不符 |

### 4.3 中（S2）— 性能/准确性受限

| # | 子系统 | 问题描述 | 影响 |
|---|--------|----------|------|
| 8 | 课程学习 | Warmup→Competitive过渡同时改变6个超参数 | 训练崩溃高风险 |
| 9 | 联赛系统 | 多项式衰减α=3.0，有效对手多样性低 | 自我博弈多样性不足，可能过拟合 |
| 10 | 联赛系统 | checkpoint排序用 `st_mtime` | NFS等环境下排序可能出错 |
| 11 | 评估系统 | Arena座位固定位置0未轮换 | 评估结果可能有系统性偏差 |
| 12 | 评估系统 | 对手听牌推断用副露比例>=0.5 | 推断准确性低，RTPA调温不准 |
| 13 | 评估系统 | 残局放大为阶跃函数 | 策略在阈值附近不连续 |
| 14 | 评估系统 | log空间混合尺度不匹配 | 搜索结果与网络输出混合可能失真 |
| 15 | Rust引擎 | ISMCE采样未考虑对手行为模型 | 信息集采样质量低 |
| 16 | Rust引擎 | ISMCE rollout贪心向听最小化 | rollout不考虑防守，评估偏乐观 |
| 17 | 训练规模 | 总计17M env steps | 大概率不足以达到超人类水平 |

### 4.4 低（S3）— 改进建议

| # | 子系统 | 问题描述 | 影响 |
|---|--------|----------|------|
| 18 | Rust引擎 | 缺少对手手牌数量显式编码 | 观测信息不完整 |
| 19 | Rust引擎 | 缺少副露来源信息 | 无法推断对手牌型关联 |
| 20 | Rust引擎 | Oracle 52通道中12通道冗余 | 计算资源浪费 |
| 21 | 神经网络 | TileAttention仅2层 | 跨花色推理能力可能不足 |
| 22 | 神经网络 | enc_proj 6.75x压缩 | 信息瓶颈风险 |
| 23 | 神经网络 | LSTM单层512维 | 时序建模能力偏弱 |
| 24 | 神经网络 | 对手听牌预测81维BCE不平衡 | 预测可能退化 |
| 25 | 奖励系统 | 向听奖励可能导致贪心追求 | 忽略番数价值 |
| 26 | 奖励系统 | 排名奖励默认关闭 | 与血战最终目标不一致 |
| 27 | 奖励系统 | 安全弃牌奖励默认关闭 | 缺少防守激励 |
| 28 | 课程学习 | Warmup禁用LSTM | 丢失通用时序特征学习机会 |
| 29 | 课程学习 | entropy非单调变化可能过高 | 探索过度 |
| 30 | 课程学习 | adv_clip Elite阶段放宽到5.0 | 需要监控 |
| 31 | 联赛系统 | 缺少Elo/胜率追踪 | 无法量化训练进展 |
| 32 | 评估系统 | 同分排名计算偏乐观 | 评估指标略有偏差 |
| 33 | Rust引擎 | 地胡定义可能与部分地方规则不一致 | 规则兼容性 |
| 34 | Rust引擎 | 花猪/大叫罚分使用自摸计算 | 部分规则应按荣和计算 |
| 35 | Rust引擎 | danger_scores纯启发式 | 危险度评估过于简单 |

---

## 5. 超人类路径分析

### 5.1 当前水平评估

系统具备完整的血战到底麻将规则实现和合理的RL训练框架，但由于评估管线存在严重Bug（ISMCE完全失效），当前无法准确衡量实际水平。基于架构分析，预估当前水平为**强业余级别**。

### 5.2 与超人类水平的差距

| 维度 | 当前状态 | 超人类要求 | 差距 |
|------|----------|-----------|------|
| 搜索深度 | SP Table退化，ISMCE失效 | 精确向听评估 + 有效搜索 | 🔴 巨大 |
| 训练规模 | 17M steps | 50-100M+ steps | 🔴 巨大 |
| 对手建模 | 均匀随机采样 | 基于行为的信念更新 | 🟡 显著 |
| 防守能力 | 安全弃牌奖励关闭，rollout无防守 | 攻防平衡决策 | 🟡 显著 |
| 时序推理 | 单层LSTM 512维 | 深层时序建模 | 🟢 中等 |
| 番数规划 | 向听贪心，无番数引导 | 番数-速度权衡 | 🟢 中等 |

### 5.3 关键瓶颈（按影响排序）

1. **搜索质量**：SP Table `MAX_SAMPLES=0` 使一向听评估退化，ISMCE评估端完全失效。搜索是从"强"到"超人类"的核心跳板。
2. **训练规模**：17M步远不足以充分探索血战到底的策略空间（4人×3花色定缺×多种番型组合）。
3. **评估管线**：ISMCE参数缺失 + RTPA/ISMCE互斥，导致无法验证搜索增强效果，形成改进盲区。
4. **对手建模**：均匀随机采样无法捕捉对手倾向，信息集搜索质量受限。
5. **奖励信号**：排名奖励关闭、向听贪心倾向、缺少番数引导，训练目标与实际博弈目标存在偏差。

---

## 6. 改进建议

### P0 — 立即修复（阻塞性Bug）

| # | 建议 | 预期收益 | 工作量 |
|---|------|----------|--------|
| P0-1 | 修复 `evaluate.py` 中 ISMCE 调用参数缺失 | 恢复搜索增强功能 | 小（1天） |
| P0-2 | 重构 RTPA/ISMCE 为可组合设计（非互斥） | 两种增强机制可协同 | 中（2-3天） |
| P0-3 | 修复 `terminated` 双重计算逻辑 | 排名奖励正常生效 | 小（半天） |
| P0-4 | 修复 `_rust_engine_ok` 恢复机制 | 训练吞吐量稳定 | 小（半天） |
| P0-5 | 修复动作掩码 `-1e9` 为 `-1e4`（兼容float16） | 混合精度训练安全 | 小（半天） |
| P0-6 | 统一 `oracle_num_blocks` 配置 | 消除配置歧义 | 小（半天） |

### P1 — 高优先级（显著提升决策质量）

| # | 建议 | 预期收益 | 工作量 |
|---|------|----------|--------|
| P1-1 | SP Table 设置合理的 `MAX_SAMPLES`（如1000-5000） | 一向听评估精度大幅提升 | 中（2-3天） |
| P1-2 | 训练规模扩展至50-100M steps | 策略充分收敛 | 大（计算资源） |
| P1-3 | 启用排名奖励并调优权重 | 训练目标对齐博弈目标 | 中（1-2天+调参） |
| P1-4 | Warmup→Competitive过渡拆分为2-3个子阶段 | 降低训练崩溃风险 | 中（2-3天） |
| P1-5 | Arena评估增加座位轮换 | 消除系统性偏差 | 小（1天） |
| P1-6 | 联赛系统增加Elo追踪 | 量化训练进展 | 中（2-3天） |

### P2 — 中优先级（进一步提升上限）

| # | 建议 | 预期收益 | 工作量 |
|---|------|----------|--------|
| P2-1 | ISMCE采样引入对手行为模型 | 信息集采样质量提升 | 大（1-2周） |
| P2-2 | ISMCE rollout加入防守逻辑 | rollout评估更准确 | 中（3-5天） |
| P2-3 | 增加对手手牌数量显式编码 | 观测信息更完整 | 小（1天） |
| P2-4 | TileAttention增加到4层 | 跨花色推理增强 | 中（2-3天+重训练） |
| P2-5 | LSTM升级为2层或引入Transformer | 时序建模增强 | 中（3-5天+重训练） |
| P2-6 | 对手听牌预测改用Focal Loss | 不平衡场景预测改善 | 小（1天） |
| P2-7 | 向听奖励加入番数加权 | 减少贪心向听倾向 | 中（2-3天） |
| P2-8 | 残局放大改为sigmoid平滑过渡 | 策略连续性改善 | 小（半天） |
| P2-9 | 联赛采样α降至1.5-2.0 | 对手多样性提升 | 小（半天） |

### P3 — 低优先级（锦上添花）

| # | 建议 | 预期收益 | 工作量 |
|---|------|----------|--------|
| P3-1 | 移除Oracle观测中12通道冗余 | 微量计算节省 | 小（1天） |
| P3-2 | 增加副露来源信息编码 | 对手牌型推断增强 | 中（2-3天） |
| P3-3 | enc_proj压缩比优化（如4x） | 降低信息瓶颈风险 | 中（需实验） |
| P3-4 | Warmup阶段启用LSTM | 时序特征早期学习 | 小（配置修改） |
| P3-5 | checkpoint排序改用文件名数字 | NFS兼容性 | 小（半天） |
| P3-6 | 启用安全弃牌奖励并调参 | 防守能力提升 | 中（需调参） |
| P3-7 | danger_scores改进为学习型评估 | 危险度评估更准确 | 大（1-2周） |
| P3-8 | 地胡定义增加可配置选项 | 规则兼容性 | 小（半天） |
| P3-9 | 花猪/大叫罚分增加荣和计算选项 | 规则兼容性 | 小（半天） |

---

## 7. 结论

Blood-V2 展现了扎实的工程基础和对血战到底麻将规则的深入理解。Rust引擎的规则实现质量高，神经网络架构设计合理，训练管线具备完整的课程学习和自我博弈框架。

然而，系统存在**2个严重Bug**阻碍了评估系统的正常运作，**5个高优先级问题**影响训练和决策质量。最关键的三个瓶颈是：

1. **SP Table退化**（`MAX_SAMPLES=0`）— 直接削弱每一步决策的评估质量
2. **评估管线失效**（ISMCE参数缺失 + RTPA/ISMCE互斥）— 搜索增强完全不可用
3. **训练规模不足**（17M步）— 策略无法充分收敛

建议的改进路径：**先修复P0阻塞性Bug → 落实P1搜索与训练扩展 → 逐步推进P2/P3优化**。预计完成P0+P1后，系统可达到强竞技水平；进一步完成P2中的对手建模和网络增强后，有望接近超人类水平。

总体评价：**架构设计优秀，实现质量高，但关键子系统存在阻塞性缺陷需要优先修复。**

---

## 八、已完成的修复与优化

> 本章记录截至 2026-02-24 已完成的所有修复和优化工作。

### P0 修复（严重Bug）

#### 1. ISMCE 集成重构 — `evaluate.py`

- **问题描述**：
  - ISMCE调用缺少 `hand`/`tiles_seen`/`melds_count`/`ding_que` 参数，搜索永远不执行
  - RTPA与ISMCE互斥（if/elif），无法同时生效
- **修改文件**：`python/blood/eval/evaluate.py`
- **修改内容**：新增 `_extract_ismce_state()` 从464×27观测张量提取游戏状态；重构 `__call__()` 为四步流水线（RTPA算温度→ISMCE用温度搜索→纯RTPA→纯策略网络），两者协同工作
- **影响范围**：评估系统核心流程，ISMCE搜索增强功能恢复，RTPA与ISMCE可协同生效

#### 2. terminated 双重计算 — `selfplay_env.py`

- **问题描述**：step()中 `terminated` 计算两次，`finalize_scoring()` 可能改变结果，导致排名奖励被跳过
- **修改文件**：`python/blood/env/selfplay_env.py`
- **修改内容**：`terminated` 只在 `finalize_scoring()` 之后计算一次
- **影响范围**：训练环境奖励计算，排名奖励信号恢复正常

#### 3. 动作掩码float16溢出 — `factory.py`

- **问题描述**：硬编码 `-1e9` 在float16下溢出（max≈65504）
- **修改文件**：`python/blood/model/factory.py`
- **修改内容**：改用 `torch.finfo(dtype).min` 自动适配精度
- **影响范围**：混合精度训练安全性，消除float16下潜在的NaN问题

### P1 优化（架构改进）

#### 4. enc_proj 信息瓶颈缓解 — `encoder.py`, `inference.py`, `cfg.py`, `test_model.py`

- **问题描述**：6912→1024单层Linear压缩比6.75x，信息损失风险高
- **修改文件**：
  - `python/blood/model/encoder.py`
  - `python/blood/model/inference.py`
  - `python/blood/cfg.py`
  - `tests/test_model.py`
- **修改内容**：新增2层渐进压缩MLP选项（6912→2048→1024），通过 `--blood_enc_proj_layers=2` 启用，默认1保持向后兼容；`inference.py` 自动检测新旧checkpoint格式
- **影响范围**：编码器信息保留能力提升，向后兼容现有checkpoint

#### 5. RTPA 对手听牌推断优化 — `rtpa.py`, `consts.py`

- **问题描述**：仅用副露比例>=0.5判断对手听牌，大量假阳性/假阴性
- **修改文件**：
  - `python/blood/eval/rtpa.py`
  - `python/blood/consts.py`
- **修改内容**：改为多信号评分系统（副露数0.25 + 摸切比例0.25 + 幺九弃牌0.15 + 花色集中度0.15 + 进度0.10 + 残局加成），已胡玩家直接排除；残局放大从阶跃函数改为线性渐变（wall_remaining 20→0，系数1.0→1.3）
- **影响范围**：RTPA温度调节准确性显著提升，残局策略连续性改善

#### 6. Arena 座位轮换 — `arena.py`

- **问题描述**：agent始终坐座位0（庄家位），引入系统性偏差
- **修改文件**：`python/blood/eval/arena.py`
- **修改内容**：每局随机分配座位0-3；同分排名改用平均排名
- **影响范围**：Arena评估结果消除座位偏差，排名计算更公平

#### 7. 配置不一致修复 — `cfg.py`, `default.yaml`

- **问题描述**：
  - `oracle_num_blocks`：`cfg.py` 默认值从25改为20，与所有yaml一致
  - `rnn_size`：保持 `default.yaml` 中1024，添加注释说明competitive/elite覆盖为512
- **修改文件**：
  - `python/blood/cfg.py`
  - `configs/default.yaml`
- **修改内容**：`oracle_num_blocks` 默认值统一为20；`rnn_size` 添加注释说明
- **影响范围**：消除配置歧义，Oracle模型结构与预期一致

### P2 优化（数值改进）

#### 8. ISMCE log空间尺度归一化 — `ismce.py`

- **问题描述**：策略logits和ISMCE评分尺度差异大，混合权重失真
- **修改文件**：`python/blood/eval/ismce.py`
- **修改内容**：混合前对两路信号分别做标准化（零均值+单位方差）
- **影响范围**：ISMCE搜索结果与网络输出混合质量提升

#### 9. Rust引擎冷却恢复机制 — `blood_env.py`

- **问题描述**：`_rust_engine_ok` 一旦 `False` 永不恢复，偶尔超时就永久禁用worker
- **修改文件**：`python/blood/env/blood_env.py`
- **修改内容**：100步冷却期+连续超时计数，冷却结束自动恢复，连续3次超时才永久禁用
- **影响范围**：训练吞吐量稳定性提升，避免因偶发超时永久损失worker

### S1 修复（决策质量瓶颈）

#### 10. SP Table 一向听精度恢复 — `crates/engine/src/algo/sp/calc.rs`

- **问题**：`MAX_SAMPLES=0` 导致一向听评估完全跳过实际采样，退化为硬编码估计（`avg_outs=4.0`, `avg_score=2000.0`），被评审标注为"最大决策质量瓶颈"
- **修复**：`MAX_SAMPLES` 从 0 提升到 5，对一向听候选的有效进牌按可用枚数降序排列后采样前5个；采样路径使用 `get_win_score()` → `calc_fan()` 计算实际番型得分；未采样牌的回退估计从硬编码改为基于有效牌总数的启发式 `(total_eff * 0.5).max(2.0)`
- **性能安全**：5采样 × 27次 calc_shanten ≈ 135次调用，SHANTEN_CACHE 缓存后实际约50-80次，远低于15秒超时阈值

### 修改文件清单

| 文件 | 修改类型 |
|------|---------|
| `python/blood/eval/evaluate.py` | P0 重构 |
| `python/blood/env/selfplay_env.py` | P0 修复 |
| `python/blood/model/factory.py` | P0 修复 |
| `python/blood/model/encoder.py` | P1 优化 |
| `python/blood/model/inference.py` | P1 适配 |
| `python/blood/eval/rtpa.py` | P1 优化 |
| `python/blood/eval/arena.py` | P1 优化 |
| `python/blood/eval/ismce.py` | P2 优化 |
| `python/blood/env/blood_env.py` | P2 优化 |
| `python/blood/cfg.py` | P1 配置修复 |
| `python/blood/consts.py` | P1 新增常量 |
| `configs/default.yaml` | P1 配置修复 |
| `tests/test_model.py` | 测试更新 |
| `crates/engine/src/algo/sp/calc.rs` | S1 修复 |

---

### 批次2：遗留问题深度优化

11. **Warmup→Competitive 过渡平滑化** — 新建 `configs/warmup_transition.yaml`
    - 问题：原流水线从 warmup 直接跳到 competitive，同时改变6个超参数（对手/LSTM/batch/gamma/lr/entropy），训练崩溃高风险
    - 修复：插入500K步过渡阶段，仅改变2个变量（启用LSTM + gamma/lr渐变），保持RuleBot对手和batch_size不变
    - 训练流水线更新为：warmup(2M) → warmup_transition(500K) → competitive(1M) → competitive_distill(4M) → elite(50M)

12. **奖励系统关键项启用** — `configs/competitive.yaml`, `competitive_distill.yaml`, `elite.yaml`, `default.yaml`
    - 问题：排名奖励和安全弃牌奖励默认关闭(0.0)，防守信号不足
    - 修复：competitive/distill阶段启用排名奖励(0.15)和安全弃牌奖励(0.015)；elite阶段排名奖励提升到0.2，向听奖励从0.003衰减到0.001

13. **训练规模扩大** — `configs/elite.yaml`
    - 问题：总计17M env steps不足以达到超人类水平
    - 修复：elite阶段从10M提升到50M步，总训练量约57.5M步

14. **联赛checkpoint排序改用文件名** — `python/blood/training/league.py`
    - 问题：`st_mtime` 在NFS/容器环境不可靠
    - 修复：正则提取文件名中的env_steps数字排序，fallback到st_mtime

15. **联赛对手多样性增强** — `python/blood/training/league.py`, `python/blood/cfg.py`
    - 问题：α=3.0多项式衰减有效多样性低，缺少自博弈
    - 修复：α降到2.0 + uniform_floor=0.1混合均匀分布保底；新增self_play_prob=0.2支持当前策略vs自身

16. **听牌预测Focal Loss** — `python/blood/model/heads.py`, `python/blood/cfg.py`
    - 问题：标准BCE在极度类别不平衡下退化为"全部预测不听"
    - 修复：新增sigmoid_focal_loss(alpha=0.25, gamma=2.0)替代BCE

### 批次2修改文件清单

| 文件 | 修改类型 |
|------|---------|
| `configs/warmup_transition.yaml` | 新建（过渡阶段配置） |
| `configs/competitive.yaml` | 奖励系统启用 |
| `configs/competitive_distill.yaml` | 奖励系统启用 |
| `configs/elite.yaml` | 训练规模+奖励衰减 |
| `configs/default.yaml` | 奖励参数声明 |
| `python/blood/training/league.py` | 联赛系统重构 |
| `python/blood/model/heads.py` | Focal Loss |
| `python/blood/cfg.py` | 新增配置参数 |

### 批次3：Rust引擎深层优化

17. **学生观测编码增强（464→470通道）** — `crates/engine/src/obs/student.rs`, `consts.rs`, `state/player.rs`, `state/board.rs`, `python/blood/consts.py`
    - 问题：缺少对手手牌数量和副露来源信息，限制防守推理能力
    - 修复：新增 Section 14（6通道）——ch 464-466 对手手牌数（hand_count/13.0），ch 467-469 最近副露来源相对位置（rel_pos/3.0）
    - player.rs 新增 `meld_from: Vec<Option<usize>>` 追踪副露来源；board.rs 在 Pon/MinKan/AnKan 处同步维护
    - 注意：此变更改变观测空间维度，需要重新训练模型

18. **Oracle冗余通道替换** — `crates/engine/src/obs/oracle.rs`
    - 问题：12通道冗余（对手定缺9ch + 定缺完成3ch 在学生观测中已有）
    - 修复：替换为对手SP Table摘要（9ch：最佳弃牌EV/听牌概率/胜率）+ 对手危险度评分（3ch），总通道数保持52不变

19. **ISMCE约束采样** — `crates/engine/src/algo/ismce.rs`
    - 问题：均匀随机分配未见牌，不考虑对手定缺约束
    - 修复：新增 `evaluate_discards_constrained()` 函数，分配未见牌时排除对手已定缺花色的牌，提高采样质量

20. **ISMCE增强危险度评估** — `crates/engine/src/algo/ismce.rs`
    - 问题：danger_scores 纯启发式，仅基于"对手是否打过该牌"和"未见牌数量"
    - 修复：新增 `danger_scores_enhanced()` 函数，综合考虑副露数量、定缺花色安全性、打牌模式分析（近期安全牌比例）、粗略向听估计

### 批次3修改文件清单

| 文件 | 修改类型 |
|------|---------|
| `crates/engine/src/consts.rs` | 通道数 464→470 |
| `crates/engine/src/obs/student.rs` | Section 14 新增 |
| `crates/engine/src/obs/oracle.rs` | 冗余通道替换 |
| `crates/engine/src/state/player.rs` | meld_from 字段 |
| `crates/engine/src/state/board.rs` | meld_from 维护 |
| `crates/engine/src/algo/ismce.rs` | 约束采样+增强危险度 |
| `python/blood/consts.py` | 常量同步 |
