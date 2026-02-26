# Blood V2 番数 & 奖励系统深度分析

> 日期: 2026-02-26 | 阶段: Phase 3 Elite @ 6.45M / 50M steps
> 代码: `crates/engine/src/algo/agari.rs`, `point.rs`, `consts.rs`, `python/blood/env/selfplay_env.py`
> 配置: `configs/elite.yaml`

---

## 一、番数系统 (Fan System)

### 1.1 完整番表

| 役种 | 番数 | 叠加规则 | 代码位置 |
|------|------|---------|---------|
| **平胡 (PingHu)** | +1 | 基础番，所有胡牌必有 | `agari.rs:153` |
| **自摸 (Tsumo)** | +1 | 非荣和时自动加 | `agari.rs:154` |
| **门清 (Menqing)** | +1 | 无明副露 (暗杠不破) | `agari.rs:155` |
| **七对 (QiDui)** | +2 | 门清限定，与对对胡互斥 | `agari.rs:156` |
| **对对胡 (ToiToi)** | +1 | 全刻子，与七对互斥 | `agari.rs:157` |
| **金钩钓 (JinGouDiao)** | +1 | 4副露+1张，与对对胡共存 | `agari.rs:158, 211-213` |
| **清一色 (QingYiSe)** | +2 | 全手同色 | `agari.rs:159` |
| **带幺九 (DaiYaoJiu)** | +3 | 每组含1/9，与断幺互斥 | `agari.rs:160` |
| **断幺 (DuanYaoJiu)** | +1 | 全2-8，与带幺九互斥 | `agari.rs:161` |
| **一条龙 (YiTiaoLong)** | +1 | 同花色123+456+789 | `agari.rs:162` |
| **夹心五 (JiaXinWu)** | +1 | 和牌5且在456顺中 | `agari.rs:163` |
| **根 (Gen/Root)** | +1/个 | 4归1，每组+1 | `agari.rs:164` |
| **杠上花 (GangShangHua)** | +1 | 杠后岭上自摸 | `agari.rs:168-170` |
| **杠上炮 (GangShangPao)** | +1 | 杠后打牌被荣和 | `agari.rs:171-174` |
| **抢杠 (ChanKan)** | +1 | 抢加杠荣和 | `agari.rs:175-178` |
| **海底 (HaiDi)** | +1 | 最后一张牌 | `agari.rs:181-184` |
| **天胡/地胡** | =6 | 覆盖所有番，直接设为MAX | `agari.rs:187-190` |

### 1.2 得分公式

```
score = 1000 × 2^(fan-1)，封顶 MAX_FAN=6 → 32000
```

| 番数 | 得分 | 荣和收入 | 自摸总收入 (×3人) |
|------|------|---------|-----------------|
| 1番 | 1,000 | 1,000 | 3,000 |
| 2番 | 2,000 | 2,000 | 6,000 |
| 3番 | 4,000 | 4,000 | 12,000 |
| 4番 | 8,000 | 8,000 | 24,000 |
| 5番 | 16,000 | 16,000 | 48,000 |
| 6番 | 32,000 | 32,000 | 96,000 |

### 1.3 番数系统评估: ✅ 合理

1. **指数增长** — `2^(fan-1)` 有效激励追求高番
2. **互斥规则正确** — 带幺九(+3)/断幺(+1)互斥、七对/对对胡互斥、金钩钓与对对胡共存(+1+1)
3. **天/地胡覆盖** — 直接设 MAX_FAN=6，避免超封顶
4. **6番封顶** — 符合四川血战标准，7+番牌极罕见，对训练无实质影响
5. **自摸经济激励** — +1番但收入×3，系统性鼓励自摸（符合血战策略核心）

---

## 二、奖励系统 (Reward System)

### 2.1 奖励计算流程

```
Rust: score_delta → Python: _r = Δ/32000 → sqrt压缩: sign(r)×√|r| → 结构化shaping叠加
```

### 2.2 Sqrt 压缩后的奖励映射

| 事件 | Δ分数 | Sqrt 压缩后 |
|------|------|------------|
| 1番荣和 | +1,000 | **+0.177** |
| 3番荣和 | +4,000 | **+0.354** |
| 6番荣和 | +32,000 | **+1.000** |
| 1番自摸 | +3,000 | **+0.306** |
| 6番自摸 | +96,000 | **+1.732** |
| 1番放铳 | -1,000 | **-0.177** |
| 6番放铳 | -32,000 | **-1.000** |

Sqrt 将 1番:6番 的比例从 1:32（线性）压缩为 1:5.6，大幅降低 PPO 奖励方差。

### 2.3 结构化奖励 (Elite Phase 当前值)

| 组件 | 值 | 相对1番荣和 | 目的 |
|------|------|-----------|------|
| tsumo_bonus | +0.1 × intensity | 18%→6% | 鼓励自摸 (score-weighted) |
| deal_in_penalty | -0.05 × intensity | 8%→3% | 惩罚放铳 (score-weighted) |
| shanten_progress | +0.003 (衰减中) | 1.7% | 向听引导 |
| shanten_regress | -0.001 (衰减中) | 0.6% | 惩罚退步 |
| safe_discard | +0.01/次 | 5.6% | 防守意识 (从0.015降低) |
| rank_bonus (1st) | +0.2 × intensity | 28%→12% | 排名激励 (score-weighted) |
| rank_bonus (4th) | -0.2 × intensity | 28%→12% | 排名惩罚 (score-weighted) |

---

## 三、已实施的优化

### 3.1 Score-weighted 所有 Shaping 组件 (2026-02-26)

**问题**: 固定 tsumo_bonus(0.1)/deal_in_penalty(0.05)/rank_bonus(0.2) 在低番局中主导信号。

**方案**: 所有 shaping 组件统一乘以得分强度系数：

```python
# selfplay_env.py — 统一 score-weighted 模式
intensity = clamp(sqrt(|score_delta| / 32000), 0.25, 1.0)
bonus *= intensity  # tsumo_bonus, deal_in_penalty, rank_bonus
```

**效果**:

| 组件 / 场景 | 1番 (Δ=1000) | 3番 (Δ=4000) | 6番 (Δ=32000) |
|------------|-------------|-------------|---------------|
| tsumo_bonus | 0.031 (was 0.100) | 0.035 | 0.100 |
| deal_in_penalty | 0.014 (was 0.050) | 0.018 | 0.050 |
| rank_bonus (1st) | 0.050 (was 0.200) | 0.071 | 0.200 |

### 3.2 safe_discard 降低 (2026-02-26)

`safe_discard` 从 0.015 降到 **0.01**（elite.yaml），缓解累积过度防守倾向和 entropy 下降。

**配置**: 全部涉及的 yaml 注释已同步更新。

---

## 四、待观察的问题

### 4.1 BloodMahjongEnv 线性 vs SelfPlayEnv sqrt 🟢 低优先级

`board.rs::get_rewards()` 使用线性归一化（Δ/32000），`selfplay_env.py` 使用 sqrt 压缩。当前 Phase 3 只用 `SelfPlayEnv`，不影响训练。

### 4.2 向听衰减调度终值 🟢 低优先级

30M 步后向听奖励衰减到 30%（0.003×0.3=0.0009/听）。如果 Elo 在 30M+ 步后停滞，可将 `min_ratio` 从 0.3 降到 0.1。

---

## 五、总结

| 维度 | 评估 |
|------|------|
| 番数系统 | ✅ 合理，18种役种完整，6番封顶标准 |
| 得分公式 | ✅ `1000×2^(fan-1)`，指数增长有区分度 |
| 基础奖励 | ✅ Sqrt 压缩有效降低方差 |
| tsumo_bonus | ✅ score-weighted，低番自摸信号自动衰减 |
| deal_in_penalty | ✅ score-weighted，低番放铳惩罚自动衰减 |
| 排名奖励 | ✅ score-weighted，低番局信号占比 28%(was 113%) |
| 向听引导 | ✅ 衰减调度 + 番数加权 |
| 防守激励 | ✅ safe_discard 从 0.015 降到 0.01 |

### 配置更新清单

| 配置文件 | tsumo/deal-in | rank_bonus | safe_discard | 状态 |
|---------|--------------|-----------|-------------|------|
| elite.yaml | ✅ 注释 | ✅ 注释 | ✅ 0.015→0.01 | 完成 |
| competitive.yaml | ✅ 注释 | ✅ 注释 | ✅ 注释 | 完成 |
| competitive_distill.yaml | ✅ 注释 | ✅ 注释 | ✅ 注释 | 完成 |
| default.yaml | ✅ 注释 | ✅ 注释 | - (0.0) | 完成 |
| warmup.yaml | ✅ 注释 | - | - | 完成 |
| warmup_transition.yaml | ✅ 注释 | ✅ 注释 | - (0.0) | 完成 |
| smoke_test.yaml | ✅ 注释 | ✅ 注释 | - (0.0) | 完成 |
