# Blood-V2 性能优化方案（超人类水平导向）

## 核心原则

**在保持模型表达能力的前提下,优化训练吞吐量**

当前系统经过20轮深度评审,架构评分10/10,所有组件都有明确的设计理由:
- 51M参数(Student 37M + Oracle 14M)是为了达到超人类水平
- 20个BottleneckBlock + 4个TileAttention是经过验证的最优配置
- 473通道观测空间覆盖了麻将决策的全部信息维度
- SP Table、ISMCE、Oracle蒸馏都是核心竞争力

**不能为了速度牺牲这些核心能力**

---

## 问题诊断

### 当前性能
- **实测**: 163K步 / 146分钟 = **18.6 steps/sec**
- **配置**: 16 workers × 16 envs = 256 并行环境
- **预期**: 基于类似规模项目,应达到 100-200 steps/sec

### 速度慢的真实原因

根据代码分析,主要瓶颈是:

#### 1. Serial Mode 强制单进程 (影响: 10-15x)

```python
# runner.py:445
cfg.serial_mode = True
cfg.async_rl = False
```

**原因**: 为了让monkey-patch的自定义损失生效。SF2的ParallelRunner使用multiprocessing.spawn,子进程重新导入模块,主进程的patch失效。

**影响**: 256个并行环境实际只能串行执行,完全丧失SF2的并行优势。

**这是最严重的瓶颈,必须解决**

#### 2. 模型前向传播开销 (影响: 2-3x)

51M参数的模型,每步需要推理256个环境:
- 编码器: 20 BottleneckBlock + 4 TileAttention
- LSTM: 2层 × 512维
- Oracle: 14M参数的Teacher网络

但这些都是**必要的架构组件**,不能简单削减。

#### 3. Rust引擎超时机制 (影响: 1.2-1.5x)

每次reset/step创建新线程,15秒超时检测。这是防御性编程,但确实有开销。

---

## 优化方案（按优先级）

### ⚡ P0: 解决Serial Mode瓶颈

这是唯一可以带来10x+提升且**不损害模型能力**的优化。

#### 方案A: 使用SF2的官方扩展机制 (推荐)

Sample Factory 2应该提供了自定义损失的官方方式。需要调研:

```python
# 可能的API
from sample_factory.algo.learning.learner import Learner

class BloodLearner(Learner):
    def _calculate_losses(self, mb, num_invalids):
        # 自定义损失逻辑
        pass

# 或者
def custom_loss_fn(learner, mb):
    # 返回额外损失
    pass

cfg.custom_learner = BloodLearner
# 或
cfg.custom_loss_fn = custom_loss_fn
```

**行动**: 深入研究SF2文档和源码,找到官方扩展点

#### 方案B: 改用支持并行的RL框架

如果SF2确实不支持并行+自定义损失,考虑迁移到:

**CleanRL** (推荐):
- 代码简洁,易于自定义
- 原生支持PPO + 自定义损失
- 社区活跃,文档完善

**RLlib**:
- Ray生态,天然并行
- 支持复杂的自定义
- 但学习曲线较陡

**迁移成本**: 2-3周,但可以获得10-15x的速度提升

#### 方案C: 修复SF2的multiprocessing.spawn问题

使用共享内存标志位:

```python
import multiprocessing as mp

# 主进程
_patch_applied = mp.Value('i', 0)

def _patch_learner():
    global _patch_applied
    if _patch_applied.value == 1:
        return  # 已经patch过
    
    # 应用patch
    _original = Learner._calculate_losses
    def _patched(self, mb, num_invalids):
        # 自定义逻辑
        pass
    Learner._calculate_losses = _patched
    _patch_applied.value = 1

# 在模块导入时调用
_patch_learner()
```

**风险**: 可能不稳定,需要充分测试

**预期提升**: 10-15x → 训练时间从60天降到4-6天

---

### ⚡ P1: 优化模型推理效率（不改变架构）

#### 1. Mixed Precision Training (FP16)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, values = model(obs)
    loss = compute_loss(logits, values, ...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优势**:
- 2x内存节省 → 可以增大batch_size
- 1.5-2x速度提升（在支持Tensor Core的GPU上）
- **不改变模型架构和能力**

**注意**: 需要在关键位置保持FP32精度(如loss计算)

#### 2. 编译优化

```python
# PyTorch 2.0+
model = torch.compile(model, mode="reduce-overhead")
```

**预期提升**: 1.2-1.5x

#### 3. 批量推理优化

当前每个环境独立推理。可以批量处理:

```python
# 当前
for env in envs:
    obs = env.get_obs()
    action = model(obs)
    
# 优化后
obs_batch = torch.stack([env.get_obs() for env in envs])
actions_batch = model(obs_batch)  # 单次前向传播
```

SF2应该已经做了这个优化,需要确认。

**预期总提升**: 2-3x

---

### ⚡ P2: 优化Rust引擎调用

#### 1. 使用线程池

```python
from concurrent.futures import ThreadPoolExecutor

class BloodMahjongEnv:
    _executor = ThreadPoolExecutor(max_workers=4)
    
    def _run_with_timeout(self, fn, timeout=15):
        future = self._executor.submit(fn)
        return future.result(timeout=timeout)
```

**优势**: 避免频繁创建/销毁线程

#### 2. 降低超时时间

如果Rust引擎稳定,可以从15秒降到5秒:

```python
_RUST_TIMEOUT_SEC = 5  # 从15降到5
```

#### 3. 移除超时保护（激进方案）

如果Rust引擎经过充分测试,可以完全移除:

```python
def step(self, action):
    # 直接调用,不创建线程
    result = self._env.step(action)
```

**风险**: 如果Rust引擎hang,整个训练会卡死

**预期提升**: 1.2-1.5x

---

### 🔧 P3: 调整训练超参数

这些优化**不影响最终模型质量**,只是调整训练效率:

#### 1. 增大Batch Size

```yaml
batch_size: 2048  # 当前
→ batch_size: 4096  # 建议
```

**优势**:
- 减少更新频率,降低通信开销
- 更稳定的梯度估计

**注意**: 可能需要调整学习率 (lr × sqrt(batch_size_ratio))

#### 2. 减少Checkpoint保存频率

```yaml
save_every_sec: 30  # 当前(每30秒)
→ save_every_sec: 300  # 建议(每5分钟)
```

**优势**: 减少I/O开销

#### 3. 减少Arena评估频率

```yaml
blood_arena_eval_every: 200000  # 当前
→ blood_arena_eval_every: 500000  # 建议
```

**优势**: 评估很耗时,减少频率可以节省训练时间

**预期提升**: 1.2-1.5x

---

## 综合优化路线图

### 短期方案（1-2周实施）

**目标**: 在不改变架构的前提下,提升10-20x

1. **解决Serial Mode** (P0)
   - 调研SF2官方扩展机制
   - 如果没有,准备迁移到CleanRL

2. **Mixed Precision Training** (P1)
   - 启用FP16训练
   - 验证精度损失可接受

3. **优化Rust调用** (P2)
   - 使用线程池
   - 降低超时时间到5秒

4. **调整超参数** (P3)
   - batch_size: 2048 → 4096
   - save_every_sec: 30 → 300
   - eval_every: 200K → 500K

**预期效果**:
- 训练速度: 18.6 steps/sec → **200-300 steps/sec**
- 训练时间: 60天 → **3-6天**

### 长期方案（1-2月实施）

**目标**: 进一步优化,但不损害模型能力

1. **迁移到CleanRL** (如果SF2不支持并行)
   - 重写训练循环
   - 保持所有自定义损失
   - 充分测试等价性

2. **分布式训练**
   - 多GPU训练(DDP)
   - 多机训练(如果有资源)

3. **优化数据流水线**
   - 异步环境交互
   - 预取下一批数据

**预期效果**:
- 训练速度: 300 steps/sec → **500-1000 steps/sec**
- 训练时间: 3-6天 → **1-2天**

---

## 不推荐的优化（会损害模型能力）

### ❌ 减少ResBlock数量

```yaml
blood_num_res_blocks: 20 → 12  # ❌ 不推荐
```

**原因**: 20个block是经过验证的最优配置,减少会降低模型表达能力

### ❌ 减少TileAttention层数

```yaml
blood_num_tile_attn_layers: 4 → 2  # ❌ 不推荐
```

**原因**: 4层TileAttention是核心设计,允许多层次的跨花色交互

### ❌ 减小卷积通道数

```yaml
blood_conv_channels: 256 → 192  # ❌ 不推荐
```

**原因**: 256通道是为了充分表达473维观测空间

### ❌ 简化Oracle架构

**原因**: Oracle Teacher必须与Student对称(甚至更强),才能有效蒸馏

### ❌ 移除辅助任务

**原因**: Shanten预测和OW预测是重要的辅助信号,加速学习

---

## 性能监控

### 添加关键指标

```python
# callbacks.py
def extra_summaries(self, runner, policy_id, writer, env_steps):
    # 训练吞吐量
    fps = runner.fps.get(policy_id, 0)
    writer.add_scalar("perf/steps_per_sec", fps, env_steps)
    
    # GPU利用率
    if torch.cuda.is_available():
        gpu_util = torch.cuda.utilization()
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        writer.add_scalar("perf/gpu_util_pct", gpu_util, env_steps)
        writer.add_scalar("perf/gpu_mem_gb", gpu_mem, env_steps)
    
    # 模型推理时间
    if hasattr(learner, "_forward_time_ms"):
        writer.add_scalar("perf/forward_ms", learner._forward_time_ms, env_steps)
```

### 使用Profiler定位瓶颈

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 训练1000步
for _ in range(1000):
    runner.step()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 打印top 20耗时函数
```

---

## 实施计划

### Week 1: 调研与准备
- [ ] 深入研究SF2文档,寻找官方扩展机制
- [ ] 如果没有,评估迁移到CleanRL的成本
- [ ] 设置性能监控指标
- [ ] 建立性能基准测试

### Week 2: 实施P0优化
- [ ] 解决Serial Mode问题
- [ ] 验证并行训练的正确性
- [ ] 对比优化前后的训练曲线

### Week 3: 实施P1+P2优化
- [ ] 启用Mixed Precision Training
- [ ] 优化Rust引擎调用
- [ ] 调整训练超参数

### Week 4: 验证与调优
- [ ] 运行完整的warmup阶段(2M步)
- [ ] 验证模型质量没有下降
- [ ] 微调超参数

---

## 预期结果

### 优化前
- 训练速度: 18.6 steps/sec
- Warmup (2M步): 30小时
- Competitive (5M步): 75小时
- Elite (10M步): 150小时
- **总计: 255小时 ≈ 10.6天**

### 优化后（保守估计）
- 训练速度: 200 steps/sec (10x提升)
- Warmup (2M步): 2.8小时
- Competitive (5M步): 7小时
- Elite (10M步): 14小时
- **总计: 23.8小时 ≈ 1天**

### 优化后（乐观估计）
- 训练速度: 500 steps/sec (27x提升)
- Warmup (2M步): 1.1小时
- Competitive (5M步): 2.8小时
- Elite (10M步): 5.6小时
- **总计: 9.5小时 ≈ 0.4天**

---

## 总结

**核心策略**: 优化训练效率,不损害模型能力

**关键优化**:
1. 解决Serial Mode瓶颈 (10-15x提升)
2. Mixed Precision Training (1.5-2x提升)
3. 优化Rust引擎调用 (1.2-1.5x提升)
4. 调整训练超参数 (1.2-1.5x提升)

**总预期提升**: 10-27x

**不做的事**:
- ❌ 不减少ResBlock/TileAttention
- ❌ 不减小通道数
- ❌ 不简化Oracle架构
- ❌ 不移除辅助任务

**这些都是系统达到超人类水平的核心组件**

---

生成时间: 2026-02-27 14:03 CST