# RNN架构分析：LSTM vs GRU vs Transformer

## 当前架构

**2层LSTM (512 hidden size)**
- 配置: `rnn_type: lstm`, `rnn_num_layers: 2`, `rnn_size: 512`
- 参数量: ~4.2M (2 layers × 512 × 1024 × 4 gates)
- 训练速度: 基准 (1.0x)

---

## 架构对比

### 1. LSTM (当前使用)

#### 优势 ✅
1. **长期记忆能力强**
   - 遗忘门精确控制信息保留
   - 适合麻将的长序列（最多70+回合）
   
2. **训练稳定**
   - 梯度消失问题得到有效缓解
   - 大量成功案例（AlphaGo、Suphx）

3. **Sample Factory原生支持**
   - 无需额外实现
   - BPTT优化成熟

#### 劣势 ❌
1. **计算开销大**
   - 4个门（输入/遗忘/输出/候选）
   - 参数量是GRU的1.33倍

2. **训练速度慢**
   - 串行计算，无法并行化
   - BPTT需要展开整个序列

---

### 2. GRU (候选方案)

#### 优势 ✅
1. **参数效率高**
   - 只有2个门（重置/更新）
   - 参数量比LSTM少25%
   - **估算**: ~3.1M参数 (vs LSTM 4.2M)

2. **训练速度快**
   - 计算量少约25-30%
   - 内存占用更低

3. **性能相当**
   - 多数任务与LSTM持平
   - 短序列任务甚至更好

#### 劣势 ❌
1. **长期记忆稍弱**
   - 没有独立的遗忘门
   - 极长序列（>100步）可能不如LSTM

2. **麻将领域验证少**
   - Suphx使用LSTM
   - 缺少成功案例

#### 血战麻将适配性分析

**序列长度**: 平均40-50回合，最长70+回合
- ✅ GRU在此长度范围内表现良好
- ⚠️ 极端长局可能略逊于LSTM

**关键信息**:
- 对手弃牌历史（需要记忆）
- 自己手牌变化（需要记忆）
- 副露信息（需要记忆）

**结论**: GRU的记忆能力**足够**，但不如LSTM**保险**

---

### 3. Transformer (激进方案)

#### 优势 ✅
1. **并行计算**
   - 自注意力可以并行
   - 训练速度可能快2-3x

2. **全局建模**
   - 直接建模任意回合间关系
   - 不受序列长度限制

3. **表达能力强**
   - 多头注意力捕获复杂模式
   - 位置编码灵活

#### 劣势 ❌
1. **参数量爆炸**
   - 自注意力: O(L²×d)
   - **估算**: ~8-10M参数（2层，512维）
   - 是LSTM的2-2.5倍

2. **内存占用大**
   - 需要存储完整注意力矩阵
   - L=32时: 32×32×512 = 524K per head
   - 8 heads = 4.2M 额外内存

3. **训练不稳定**
   - RL中Transformer训练困难
   - 需要careful tuning（warmup、layer norm位置）

4. **Sample Factory不原生支持**
   - 需要自己实现
   - BPTT逻辑需要重写

5. **推理速度慢**
   - 自注意力计算量大
   - 单步推理需要完整历史

#### 血战麻将适配性分析

**计算复杂度**:
```
LSTM:    O(L × d²)        = 32 × 512² ≈ 8.4M ops
GRU:     O(L × d²)        = 32 × 512² ≈ 6.3M ops (少25%)
Transformer: O(L² × d)    = 32² × 512 ≈ 524K ops (注意力)
           + O(L × d²)    = 32 × 512² ≈ 8.4M ops (FFN)
           = 总计 ~9M ops
```

**结论**: Transformer在麻将场景下**没有明显优势**，反而更复杂

---

## 实验建议

### 方案A: 保持LSTM（保守）✅ 推荐

**理由**:
1. 已有成功案例（Suphx）
2. 训练稳定，风险低
3. 当前性能瓶颈不在RNN

**适用场景**: 
- 追求稳定性
- 时间紧迫
- 已有LSTM checkpoint

---

### 方案B: 尝试GRU（平衡）⚙️ 可选

**实施步骤**:
1. 修改配置: `rnn_type: gru`
2. 保持其他参数不变
3. 从头训练或fine-tune

**预期收益**:
- 训练速度提升25-30%
- 内存占用降低20%
- 性能持平或略降（<2%）

**风险**:
- 需要重新训练（无法加载LSTM checkpoint）
- 长序列性能可能略降

**验证方法**:
```bash
# 训练GRU版本
python -m blood.train --config=configs/warmup.yaml \
    --rnn_type=gru --experiment=blood_v2_warmup_gru

# 对比评估
python -m blood.eval.evaluate \
    --checkpoint=checkpoints/lstm/latest.pth \
    --num_games=1000

python -m blood.eval.evaluate \
    --checkpoint=checkpoints/gru/latest.pth \
    --num_games=1000
```

---

### 方案C: Transformer（激进）❌ 不推荐

**理由**:
1. 参数量翻倍，训练成本高
2. RL中训练不稳定
3. 推理速度慢
4. 需要大量工程实现

**仅在以下情况考虑**:
- 有充足GPU资源（>40GB）
- 有Transformer RL经验
- 愿意投入大量调试时间

---

## 混合方案：TurnAttention + LSTM ✅ 当前最优

**已实现**: [`factory.py`](blood-v2/python/blood/model/factory.py:22-73)

```python
class TurnAttention(nn.Module):
    """Turn-level cross-attention over LSTM history."""
    def __init__(self, dim=512, num_heads=4, max_turns=32):
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # ... 零初始化residual，训练初期等价于纯LSTM
```

**优势**:
1. **保留LSTM优势**: 稳定训练、长期记忆
2. **增强建模能力**: 跨回合注意力
3. **渐进式学习**: 零初始化确保平滑过渡
4. **计算开销可控**: 只在LSTM之上加一层注意力

**参数量**: LSTM 4.2M + TurnAttention 1.0M = 5.2M (增加24%)

**性能提升**: 预计1-3% Elo提升（基于AlphaGo Zero经验）

---

## 推荐方案

### 短期（当前训练）
✅ **保持LSTM + TurnAttention**
- 已在elite阶段启用
- 稳定且有理论支持
- 无需重新训练

### 中期（下一轮训练）
⚙️ **可选尝试GRU**
- 如果训练速度是瓶颈
- 如果GPU内存紧张
- 需要完整训练周期验证

### 长期（研究方向）
🔬 **探索Transformer变体**
- Linformer（线性复杂度）
- Performer（kernel attention）
- 仅在有充足资源时考虑

---

## 性能对比表

| 架构 | 参数量 | 训练速度 | 推理速度 | 长期记忆 | 稳定性 | 推荐度 |
|------|--------|----------|----------|----------|--------|--------|
| LSTM | 4.2M | 1.0x | 1.0x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 推荐 |
| GRU | 3.1M | 1.3x | 1.2x | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚙️ 可选 |
| Transformer | 8-10M | 0.5x | 0.3x | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ 不推荐 |
| LSTM+TurnAttn | 5.2M | 0.9x | 0.95x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 最优 |

---

## 实施建议

### 如果要切换到GRU

1. **修改配置**:
```yaml
# configs/warmup_gru.yaml
rnn_type: gru
rnn_size: 512
rnn_num_layers: 2
```

2. **从头训练**:
```bash
python -m blood.train --config=configs/warmup_gru.yaml \
    --experiment=blood_v2_warmup_gru
```

3. **对比评估**:
```bash
# 训练到相同步数后评估
python -m blood.eval.evaluate \
    --checkpoint=checkpoints/lstm/step_2M.pth \
    --num_games=2000 --output=lstm_results.json

python -m blood.eval.evaluate \
    --checkpoint=checkpoints/gru/step_2M.pth \
    --num_games=2000 --output=gru_results.json
```

4. **分析结果**:
```python
import json
lstm = json.load(open('lstm_results.json'))
gru = json.load(open('gru_results.json'))

print(f"LSTM Elo: {lstm['elo']:.1f}")
print(f"GRU Elo: {gru['elo']:.1f}")
print(f"Difference: {gru['elo'] - lstm['elo']:.1f}")
```

---

## 结论

### 当前最优方案
✅ **LSTM + TurnAttention**

理由:
1. 稳定性最高（Suphx验证）
2. 已有checkpoint可复用
3. TurnAttention提供额外建模能力
4. 训练成本可控

### 未来探索方向
⚙️ **GRU作为备选**

条件:
- 训练速度成为瓶颈
- GPU内存不足
- 愿意重新训练

### 不推荐
❌ **Transformer**

理由:
- 参数量翻倍
- 训练不稳定
- 推理速度慢
- 工程成本高
- 在麻将场景下无明显优势

**建议：保持当前LSTM+TurnAttention架构，专注于其他优化方向（如定缺系统、奖励塑形、搜索增强）。**