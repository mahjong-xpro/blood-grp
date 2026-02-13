# Phase 7C: Minimal Guardrails (极轻量护栏)

> **Status**: EXECUTING
> **Current Step**: 875,000
> **From**: Phase 6 "Pure Greed" (Step 738k - 875k)
> **Strategy**: 最小剂量奖励塑形 + LR 重启，保留大牌追求天花板
> **Target**: Rank ≤ 2.43, Agari Point ≥ 5,500

---

## 1. Phase 6 失败诊断 (Post-Mortem)

### 1.1 核心数据对比

| 指标 | Phase 5 巅峰 (735k) | Phase 6 最新 (875k) | 变化 | 严重度 |
|:---|:---:|:---:|:---:|:---:|
| **Avg Rank** | 2.465 | 2.564 | +0.099 | 🔴 严重退化 |
| **Agari Rate** | 65.2% | 57.9% | -7.3% | 🔴 进攻崩塌 |
| **Houjuu Rate** | 38.7% | 43.5% | +4.8% | 🔴 防御崩溃 |
| **Fuuro Rate** | 65.0% | 54.9% | -10.1% | 🔴 策略混乱 |
| **Agari Point** | 5,125 | 6,241 | +1,116 (+22%) | 🟢 唯一亮点 |
| **Lose Val** | -2,316 | -2,648 | -332 | 🔴 放铳变大 |
| **4th Rate** | 24.0% | 29.2% | +5.2% | 🔴 输家率飙升 |
| **DQN Loss** | 0.145 | 0.078 | -0.067 (-46%) | ⚠️ 价值函数漂移 |
| **LR** | 7.0e-5 | 6.8e-5 | -0.2e-5 | ⚠️ 学习率接近底部 |

### 1.2 Phase 6 Fuuro 率震荡轨迹 (策略混乱的铁证)

```
Step  Fuuro    评价
750k  57.3%    下滑
780k  72.2%    短暂反弹 (Phase 6 唯一亮点)
790k  44.9%    暴跌
810k  48.2%    低迷
830k  39.6%    继续下滑
840k  33.2%    ← 历史最低！AI 几乎不鸣牌了
845k  59.7%    剧烈反弹
875k  54.9%    不稳定
```

**Fuuro 在 33%-72% 之间剧烈震荡**，标准差极大。对比 Phase 5 稳态 (63-68%)，策略已完全失控。

### 1.3 失败根因分析

| # | 根因 | 机制 | 影响 |
|:---:|:---|:---|:---|
| 1 | **移除所有奖励塑形** | 没有 agari_bonus → 无即时胡牌激励 | Agari Rate 从 65% 跌到 58% |
| 2 | **移除所有奖励塑形** | 没有 houjuu_penalty → 无即时放铳惩罚 | Houjuu Rate 从 39% 飙到 44% |
| 3 | **信号太稀疏** | 仅靠游戏结束分差学习，中间行为无反馈 | Fuuro 剧烈震荡 (33%-72%)，策略不收敛 |
| 4 | **无排名压力** | rank_bonus 全部为 0 | 4th Rate 飙升至 29.2% |
| 5 | **LR 过低** | 6.8e-5 接近 final 2e-5 | 无法快速修正行为偏差 |
| 6 | **DQN Loss 骤降** | 0.145 → 0.078 (-46%) | 价值函数对错误策略过度自信 |

### 1.4 Phase 6 的唯一遗产

> **Agari Point: 5,125 → 6,241 (+22%)**

移除 agari_bonus 后，AI 不再追求"小快灵"的低番和牌，转而等待高番大牌。
这是一个有价值的学习成果——AI 学会了**评估手牌价值**。
Phase 7 的目标是**保留这种大牌意识，同时恢复胡牌频率和防守能力**。

---

## 2. Baseline 状态与已知 Bug

### Baseline 更新历史
Baseline 一直由用户**手动更新** (`cp mortal.pth baseline.pth` + 重启 client)。
TensorBoard 仅记录自动触发的 2 次更新 (Step 5k, 20k)，不反映手动操作。

### 已知 Bug: 自动更新机制无效

`train.py` 的自动更新 (`shutil.copy`) 只更新磁盘文件，但自博弈 client 进程的
`TrainPlayer.__init__()` 仅在启动时加载一次 baseline 到内存，**不会自动 reload**。
online 模式下 client 是独立进程，服务端无法通知其重新加载。

```
缺陷链:
  train.py: shutil.copy(mortal.pth, baseline.pth)  ← 磁盘更新 ✓
  client:   self.baseline_engine = MortalEngine(...)  ← 内存中还是旧模型 ✗
```

**结论**: `baseline_update_threshold` 保持禁用 (4.0)，继续使用手动更新 + 重启 client 的方式。

### Phase 7 的 Baseline 策略

1. **手动更新**: 每当 Rank 连续 3 个检查点改善时，执行 `cp mortal.pth baseline.pth` + 重启 client
2. **建议频率**: 约每 30-50k 步更新一次
3. **Phase 7 启动时**: 立即手动更新一次

---

## 3. Phase 7 战略设计

### 3.1 核心哲学：三管齐下

```
┌──────────────────────────────────────────────────────┐
│              Phase 7: The Renaissance                │
│                                                      │
│  ① 恢复奖励塑形 (Restore Reward Shaping)            │
│     → 重建行为锚点，停止策略震荡                     │
│                                                      │
│  ② 定期手动更新 Baseline (Manual Self-Play)           │
│     → 每 30-50k 步更新一次，保持对手强度             │
│                                                      │
│  ③ LR 重启 (Learning Rate Restart)                   │
│     → 给模型足够的学习能力修正行为                   │
└──────────────────────────────────────────────────────┘
```

### 3.2 目标画像：Phase 7 理想 AI

| 维度 | Phase 5 峰值 | Phase 6 现状 | **Phase 7 目标** | 策略 |
|:---|:---:|:---:|:---:|:---|
| Fuuro Rate | 65.0% | 54.9% | **60-66%** | 恢复 agari_bonus → 鼓励副露进攻 |
| Agari Rate | 65.2% | 57.9% | **63-67%** | agari_bonus + 更强对手 → 被迫抢先胡牌 |
| Agari Point | 5,125 | 6,241 | **5,500-6,200** | 保留 Phase 6 的大牌意识 |
| Houjuu Rate | 38.7% | 43.5% | **34-37%** | houjuu_penalty + 强对手 → 精准防守 |
| Lose Val | -2,316 | -2,648 | **< -2,400** | 面对强对手学会避大牌 |
| Avg Rank | 2.465 | 2.564 | **≤ 2.43** | 综合提升 → 排名自然改善 |
| 4th Rate | 24.0% | 29.2% | **≤ 22%** | rank_bonuses 惩罚 4th → 避免垫底 |

---

## 4. 具体配置变更

### 4.1 config.toml 变更一览 (Option C: 极轻量)

```toml
# ═══════════════════════════════════════════════════════
# Phase 7C 配置: 极轻量护栏 (Step 875k+)
# 最小剂量奖励塑形 + LR 重启，保留大牌追求天花板
# ═══════════════════════════════════════════════════════

# ──── 探索参数 ────
boltzmann_epsilon = 0.08   # Phase 6: 0.05 → Phase 7C: 0.08 (微增)
boltzmann_temp = 0.18      # Phase 6: 0.15 → Phase 7C: 0.18 (微增)

# ──── 奖励塑形 (极轻量护栏) ────
[reward_shaping]
rank_bonus_enabled = true
rank_bonuses = [0.0, 0.0, 0.0, -0.15]  # 只惩罚 4th，不奖励 1st
                                         # 保持学习方向纯净

action_bonus_enabled = true
agari_bonus = 0.05                       # 极小: 3番以上偏差 < 4%
houjuu_penalty = -0.05                   # 极小: 仅防止防守完全崩溃

baseline_update_threshold = 4.0          # 禁用 (auto-update bug)

# ──── 学习率调度 (LR 重启) ────
[optim.scheduler]
peak = 2e-4                # 重启: 当前 6.8e-5 → warmup 到 2e-4
final = 1e-5               # 更低终点
warm_up_steps = 5000       # 5k 步缓慢 warmup
max_steps = 1350000        # 875k + 475k = 1350k

# ──── 测试频率 ────
test_every = 3000          # 更频繁监控
```

### 4.2 不变的参数 (保持 Phase 6)

| 参数 | 值 | 理由 |
|:---|:---|:---|
| batch_size | 4096 | 4090 单卡最优 |
| gamma | 0.99 | 长期规划不变 |
| td_lambda | 0.95 | TD(λ) 平衡不变 |
| conv_channels | 256 | 模型结构不变 |
| num_blocks | 50 | 模型结构不变 |
| opp_wait_enabled | true | 对手建模保留 |
| weight_decay | 0.1 | 正则化不变 |
| ding_que_ce_weight | 0.5 | 定缺监督不变 |
| next_rank_weight | 0.25 | 辅助任务不变 |

### 4.3 极轻量奖励的偏差分析

```
Phase 7C 奖励结构 (agari_bonus=0.05, houjuu_penalty=-0.05):
┌─────────────────────────────────────────────────────────┐
│ 事件          │ 分差信号    │ 额外奖励   │ 总信号  │ 偏差  │
├─────────────────────────────────────────────────────────┤
│ 1番荣和       │ +0.10      │ +0.05     │ +0.15  │ +50%  │
│ 1番自摸       │ +0.30      │ +0.05     │ +0.35  │ +17%  │
│ 3番自摸       │ +1.20      │ +0.05     │ +1.25  │ +4%   │
│ 5番自摸       │ +4.80      │ +0.05     │ +4.85  │ +1%   │ ← 几乎无影响
│ 典型放铳      │ -0.26      │ -0.05     │ -0.31  │ +19%  │
│ 大牌放铳      │ -1.60      │ -0.05     │ -1.65  │ +3%   │
│ 最终 4 位     │ (分差)     │ -0.15     │        │       │
└─────────────────────────────────────────────────────────┘
```

**设计核心**: 对 3 番以上手牌偏差 < 4%，大牌追求天花板不受限制。
仅对 1 番小牌有显著信号放大 (+50%)，作为"最低行为护栏"——
防止 Fuuro 暴跌到 33% 和 4th Rate 飙升到 29% 的极端情况。

---

## 5. 执行步骤

### Step 0: 备份 (安全第一)

```bash
# 在服务器上执行
cp /data/mortal/mortal.pth /data/mortal/backup_phase6_875k.pth
cp /data/mortal/config.toml /data/mortal/config_phase6_backup.toml
```

### Step 1: 更新 Baseline + 重启 Client

```bash
# 1. 停止所有自博弈 client
# 2. 用当前最新模型替换 baseline
cp /data/mortal/mortal.pth /data/mortal/baseline.pth
# 3. 重启所有自博弈 client (使其加载新 baseline)
```

> **重要**: 由于自动更新 Bug，每次更新 baseline 后必须重启 client 进程。
> 建议在 Phase 7 期间每 30-50k 步重复此操作。

### Step 2: 修改 config.toml

按照 §4.1 的配置进行修改。核心变更:
1. 恢复 `reward_shaping` (rank_bonus + action_bonus)
2. LR 重启 (peak=2e-4, max_steps=1350000)
3. 降低 baseline_update_threshold 到 1.9
4. 提高探索率 (epsilon=0.10, temp=0.20)
5. 加密测试频率 (test_every=3000)

### Step 3: 重启训练

```bash
# 重启训练进程，使新配置生效
```

### Step 4: 进入监控模式

---

## 6. 监控计划 (Monitoring Protocol)

### 6.1 Phase 7 时间线与预期

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段              │ Steps      │ 预期 Rank  │ 预期现象        │
├──────────────────────────────────────────────────────────────┤
│ ① 震荡期 (Shock)  │ 875k-900k │ 2.55-2.65  │ 面对强对手适应  │
│ ② 恢复期 (Recovery)│ 900k-950k │ 2.50-2.55  │ 奖励塑形生效   │
│ ③ 突破期 (Breakout)│ 950k-1050k│ 2.45-2.50  │ 超越 Phase 5   │
│ ④ 精调期 (Refine) │ 1050k-1200k│ 2.43-2.46  │ 收敛到新均衡   │
│ ⑤ 收尾期 (Final)  │ 1200k-1350k│ ≤ 2.43     │ LR 极低，微调  │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 每 3k 步检查清单

| # | 检查项 | 健康范围 | 危险信号 |
|:---:|:---|:---|:---|
| 1 | Fuuro Rate | 55-68% | < 50% (过度防守) 或 > 72% (无脑鸣牌) |
| 2 | Houjuu Rate | 34-42% | > 45% (防御崩溃) |
| 3 | Agari Rate | 58-68% | < 55% (进攻停滞) |
| 4 | Agari Point | 5000-6500 | < 4500 (大牌意识丧失) |
| 5 | DQN Loss | 0.06-0.15 | 持续上升 > 0.20 (训练不稳定) |
| 6 | 4th Rate | 22-30% | > 32% (垫底太频繁) |
| 7 | Pt/Round | > -200 | < -400 连续 3 个检查点 |
| 8 | DingQue | > 79% | < 75% (退化) |

### 6.3 阶段转换触发条件

#### 震荡期 → 恢复期 (预计 900k)
- **确认条件**: Fuuro Rate 稳定在 55-68% 且连续 3 个检查点
- **如果未满足**: 检查 agari_bonus 是否生效，考虑提高到 0.25

#### 恢复期 → 突破期 (预计 950k)
- **确认条件**: Rank ≤ 2.52 且 Houjuu ≤ 40%
- **如果未满足**: 考虑应用 §7.2 的 Priority 2 调整

#### 突破期 → 精调期 (预计 1050k)
- **确认条件**: Rank ≤ 2.46 连续 5 个检查点
- **如果未满足**: 考虑降低 epsilon 到 0.06，temp 到 0.12

---

## 7. 应急方案 (Contingency Plans)

### 7.1 场景 A: 震荡期过长 (> 40k 步未进入恢复期)

**症状**: Step 915k 时 Rank 仍 > 2.60
**诊断**: 新 Baseline 太强，奖励信号不足以引导恢复
**处方**:
```toml
agari_bonus = 0.30           # 提高胡牌激励
houjuu_penalty = -0.30       # 加强防守惩罚
rank_bonuses = [0.20, 0.05, -0.10, -0.35]  # 加重 4th 惩罚
```

### 7.2 场景 B: Fuuro 再次陷入震荡 (振幅 > 15%)

**症状**: Fuuro 在 40%-70% 之间大幅摆动
**诊断**: 探索率过高，策略分布太平坦
**处方**:
```toml
boltzmann_epsilon = 0.06     # 降低探索
boltzmann_temp = 0.12        # 降低温度，策略更确定
```

### 7.3 场景 C: Agari Point 暴跌 (< 4500)

**症状**: Phase 6 的大牌意识丧失
**诊断**: agari_bonus 过高导致 AI 回到"小快灵"模式
**处方**:
```toml
agari_bonus = 0.10           # 降低，减少小牌激励
# 同时考虑: 让 AI 自然恢复后再评估
```

### 7.4 场景 D: 训练不稳定 (DQN Loss 飙升)

**症状**: DQN Loss > 0.25 且持续上升
**诊断**: LR 重启太激进，价值函数震荡
**处方**:
```toml
[optim.scheduler]
peak = 1e-4                  # 降低 peak LR
warm_up_steps = 10000        # 更长的 warmup
```

### 7.5 Baseline 手动更新时机指南

| 条件 | 操作 |
|:---|:---|
| Rank 连续 3 个检查点改善 | 更新 Baseline + 重启 client |
| 距上次更新已过 50k 步 | 更新 Baseline + 重启 client |
| Rank 突然恶化 > 0.05 | **不要更新**，等待模型自行恢复 |
| 刚更新 Baseline < 15k 步 | **不要更新**，让模型适应新对手 |

### 7.6 核武选项: 回滚到 Phase 5 峰值

**触发条件**: Phase 7 运行 100k 步后 Rank 仍 > 2.55
**操作**:
```bash
# 如果存在 735k 检查点
cp /data/mortal/backup_step735k.pth /data/mortal/mortal.pth
# 使用 Phase 7 配置重新开始
# 但 LR peak 降到 1.5e-4 (更保守)
```

---

## 8. Phase 7 vs Phase 6 配置对比

| 参数 | Phase 5 | Phase 6 | **Phase 7C** | 理由 |
|:---|:---:|:---:|:---:|:---|
| epsilon | 0.08 | 0.05 | **0.08** | 微增探索 |
| temp | 0.15 | 0.15 | **0.18** | 微增平滑 |
| rank_bonus | false | false | **true** | 仅罚 4th |
| rank_bonuses | - | [0,0,0,0] | **[0,0,0,-0.15]** | 极轻量，只防垫底 |
| action_bonus | false | false | **true** | 极小剂量 |
| agari_bonus | 0.15 | 0.0 | **0.05** | 最小护栏，3番以上偏差<4% |
| houjuu_penalty | -0.3 | 0.0 | **-0.05** | 最小护栏，不强制防守 |
| baseline_threshold | 4.0 | 4.0 | **4.0** | 禁用 (auto-update bug) |
| LR peak | 5e-4 | 5e-4 | **2e-4** | LR 重启 |
| LR final | 5e-5 | 2e-5 | **1e-5** | 更低终点 |
| warmup | 2000 | 2000 | **5000** | 安全重启 |
| max_steps | 850k | 1100k | **1350k** | 475k 新步数 |
| test_every | 5000 | 5000 | **3000** | 密切监控 |

---

## 9. 预期效果与成功标准

### 短期 (875k → 950k): 止血恢复

| 指标 | 当前 (875k) | 目标 (950k) | 标志 |
|:---|:---:|:---:|:---|
| Fuuro Rate | 54.9% (震荡) | 58-65% (稳定) | 标准差 < 5% |
| Houjuu Rate | 43.5% | < 40% | 连续 3 个检查点 |
| Agari Rate | 57.9% | > 60% | 回到 60 线上方 |
| Rank | 2.564 | < 2.55 | 止跌回稳 |

### 中期 (950k → 1100k): 超越 Phase 5

| 指标 | Phase 5 峰 (735k) | 目标 (1100k) | 标志 |
|:---|:---:|:---:|:---|
| Rank | 2.465 | **≤ 2.45** | 新历史最佳 |
| Agari Point | 5,125 | **≥ 5,500** | 保留大牌意识 |
| Houjuu Rate | 38.7% | **≤ 36%** | Phase 史上首破 37% |
| 4th Rate | 24.0% | **≤ 22%** | 新历史最佳 |

### 长期 (1100k → 1350k): 终极形态

| 指标 | 目标 | 标志 |
|:---|:---:|:---|
| Rank | **≤ 2.43** | 终极目标 |
| Agari Rate | **≥ 64%** | 高频胡牌 |
| Houjuu Rate | **≤ 35%** | 精英防守 |
| Agari Point | **≥ 5,500** | 大牌意识 |
| 4th Rate | **≤ 21%** | 极少垫底 |
| Baseline 手动更新 | **≥ 5 次** | 持续自我进化 |

---

## 10. 核心洞察总结

### Phase 7 的本质

```
Phase 3-5: 手动 Baseline 更新 + 奖励塑形 → 稳步进化至 Rank 2.465
Phase 6:   移除所有奖励塑形 → 策略崩溃 (Rank 2.564, Fuuro 震荡 33-72%)

Phase 7:   吸取 Phase 6 教训 + 恢复优化的奖励 + LR 重启 + 自动 Baseline
           → 在 Phase 5 基础上继续突破
```

875k 步训练积累的价值：
1. 模型的特征提取网络 (Mortal ResNet) 已经学会了深层牌理
2. 辅助任务 (定缺 81%、对手听牌预测) 已经成熟
3. Phase 6 教会了 AI 追求高打点 (Agari Point 6241)
4. 我们对奖励参数的效果有了充分的经验数据

Phase 7 的目标是**恢复 Phase 5 的行为稳定性，同时用 LR 重启和自动 Baseline 更新推动模型突破 Rank 2.43 的新天花板**。

---

*Created: 2026-02-13*
*Baseline: Phase 6 @ Step 875k*
*Target: Rank ≤ 2.43 @ Step 1350k*
