# DingQue修复验证报告

## 检查范围

检查所有定缺相关修复，确认没有引入新的bug。

## 修复1: 观察编码 (student.rs)

### 位置
[`blood-v2/crates/engine/src/obs/student.rs:94-113`](blood-v2/crates/engine/src/obs/student.rs:94-113)

### 代码审查
```rust
// === Section 3: DING QUE (17 ch) ===
if let Some(suit) = p.ding_que {
    // DingQue已完成：标记选择的花色
    for t in suit.start()..suit.end() {
        w!(ch + suit as usize, t, 1.0);
    }
} else if board.phase == Phase::DingQue {
    // DingQue阶段：提供花色统计信息
    for suit in Suit::all() {
        let count = (suit.start()..suit.end())
            .filter(|&t| p.hand[t] > 0)
            .map(|t| p.hand[t] as u32)
            .sum::<u32>();
        fill_ch!(ch + suit as usize, count as f32 / 13.0);
    }
}
ch += 3;
```

### 验证结果 ✅
- **逻辑正确**: 定缺完成和定缺阶段分支互斥
- **通道使用**: 使用3个通道 (ch+0, ch+1, ch+2)
- **归一化**: 除以13.0正确（初始手牌数）
- **无越界**: suit as usize ∈ {0,1,2}，安全
- **无副作用**: 不影响其他Section

## 修复2: 数据增强映射 (augment.py)

### 位置
[`blood-v2/python/blood/env/augment.py:56,62`](blood-v2/python/blood/env/augment.py:56,62)

### 代码审查
```python
def augment_action(action: int, perm) -> int:
    if action >= 27:
        if 31 <= action <= 33:
            old_suit = action - 31
            new_suit = perm[old_suit]  # 正向映射
            return 31 + new_suit
        return action
    
    old_suit = action // 9
    rank = action % 9
    new_suit = perm[old_suit]  # 正向映射
    return new_suit * 9 + rank
```

### 验证结果 ✅
- **映射方向**: 使用`perm[old_suit]`正向映射，正确
- **定缺动作**: 31-33映射正确
- **弃牌动作**: 0-26映射正确
- **其他动作**: 27-30, 34+保持不变，正确
- **一致性**: 两处使用相同映射方法

### 数学验证
```python
# 示例: perm=(2,0,1) 表示 Man→Sou, Pin→Man, Sou→Pin
# 定缺万(31): old_suit=0, new_suit=perm[0]=2, return 33 (定索) ✓
# 弃1万(0): old_suit=0, rank=0, new_suit=perm[0]=2, return 18 (1索) ✓
```

## 修复3: 均匀先验 (factory.py)

### 位置
[`blood-v2/python/blood/model/factory.py:260-271`](blood-v2/python/blood/model/factory.py:260-271)

### 代码审查
```python
# DingQue uniform prior
dq_mask = mask[:, 31:34]  # (B, 3)
if dq_mask.all():  # 所有样本都在定缺阶段
    dq_logits = action_distribution_params[:, 31:34]  # (B, 3)
    dq_mean = dq_logits.mean(dim=-1, keepdim=True)  # (B, 1)
    prior_strength = 0.3
    action_distribution_params[:, 31:34] = (
        dq_logits * (1.0 - prior_strength) +
        dq_mean * prior_strength
    )
```

### 验证结果 ✅
- **检测逻辑**: `dq_mask.all()` 检查所有样本是否在定缺阶段
- **索引范围**: 31:34正确对应定缺动作
- **均值计算**: `mean(dim=-1, keepdim=True)` 保持维度正确
- **混合比例**: 0.3先验强度合理
- **张量操作**: 形状匹配，无广播错误

### 潜在问题检查 ⚠️
**问题**: `dq_mask.all()` 要求**所有样本**都在定缺阶段才生效

**场景分析**:
- **Warmup阶段**: 对手是RuleBot，可能快速完成定缺
- **混合batch**: 如果batch中有些样本已完成定缺，`all()`返回False
- **结果**: 先验可能不生效

**建议修复**:
```python
# 改为逐样本检测
dq_mask = mask[:, 31:34]  # (B, 3)
is_dingque = dq_mask.all(dim=-1)  # (B,) 每个样本是否在定缺阶段

if is_dingque.any():  # 只要有样本在定缺阶段
    dq_logits = action_distribution_params[:, 31:34]
    dq_mean = dq_logits.mean(dim=-1, keepdim=True)
    prior_strength = 0.3
    
    # 仅对定缺样本应用先验
    mixed = dq_logits * (1.0 - prior_strength) + dq_mean * prior_strength
    action_distribution_params[:, 31:34] = torch.where(
        is_dingque.unsqueeze(-1),  # (B, 1)
        mixed,
        dq_logits
    )
```

## 修复4: Oracle定缺跳过 (losses.py)

### 位置
[`blood-v2/python/blood/training/losses.py:84-101`](blood-v2/python/blood/training/losses.py:84-101)

### 代码审查
```python
# Check if in DingQue phase
dq_mask = action_mask[:, 31:34] if action_mask is not None else None
is_dingque_phase = dq_mask is not None and dq_mask.all()

# KL distillation: skip during DingQue phase
if not is_dingque_phase:
    student_logits = getattr(action_dist, "raw_logits", None)
    if student_logits is None:
        log.warning(...)
    else:
        mask_bool = action_mask.bool() if action_mask is not None else None
        distill = self._oracle_distill_loss(...)
        extra_loss = extra_loss + ac.distill_weight * distill
        summaries["distill_loss"] = distill.detach()
```

### 验证结果 ✅
- **检测逻辑**: 检查31-33是否全部合法
- **跳过条件**: `not is_dingque_phase` 正确
- **其他损失**: Oracle CE和value distillation不受影响
- **日志记录**: distill_loss仅在非定缺阶段记录

### 与修复3相同的问题 ⚠️
**问题**: `dq_mask.all()` 要求batch中**所有样本**都在定缺阶段

**影响**: 
- 如果batch混合了定缺和非定缺样本，`all()`返回False
- Oracle distillation会对定缺样本生效，传播偏差

**建议修复**:
```python
# 改为逐样本检测
dq_mask = action_mask[:, 31:34] if action_mask is not None else None
if dq_mask is not None:
    is_dingque = dq_mask.all(dim=-1)  # (B,) 每个样本
    has_dingque = is_dingque.any()
    has_non_dingque = (~is_dingque).any()
    
    if has_non_dingque:  # 只要有非定缺样本
        student_logits = getattr(action_dist, "raw_logits", None)
        if student_logits is not None:
            # 创建掩码：仅对非定缺样本计算distillation
            non_dq_mask = ~is_dingque  # (B,)
            
            # 方案A: 仅对非定缺样本计算loss
            if non_dq_mask.any():
                mask_bool = action_mask.bool() if action_mask is not None else None
                distill = self._oracle_distill_loss(...)
                # 加权：仅非定缺样本贡献loss
                distill = distill * non_dq_mask.float().mean()
                extra_loss = extra_loss + ac.distill_weight * distill
                summaries["distill_loss"] = distill.detach()
```

## 总结

### 已验证的修复 ✅
1. **观察编码** (student.rs) - 无问题
2. **数据增强** (augment.py) - 无问题

### 需要改进的修复 ⚠️
3. **均匀先验** (factory.py:264) - 需要改为逐样本检测
4. **Oracle跳过** (losses.py:86) - 需要改为逐样本检测

### 问题根源
两处都使用了 `dq_mask.all()`，要求**整个batch**都在定缺阶段才生效。这在混合batch场景下会失效。

### 修复优先级
**高优先级**: 修复losses.py的Oracle跳过逻辑
- 影响: 直接导致Oracle偏差传播
- 场景: Competitive/Elite阶段启用Oracle时

**中优先级**: 修复factory.py的均匀先验逻辑  
- 影响: 先验可能不生效，但已有其他保护
- 场景: 所有训练阶段

### 建议行动
1. 立即修复losses.py的逐样本检测
2. 可选修复factory.py的逐样本检测
3. 重新训练验证分布