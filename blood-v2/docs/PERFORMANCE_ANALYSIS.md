# Blood-V2 训练性能分析报告

## 问题概述

**当前性能**: 163K 步 / 146 分钟 = **18.6 steps/sec**  
**预期性能**: 应该达到 **200-500 steps/sec** (基于类似规模的 RL 项目)

训练速度慢了 **10-25倍**,这会导致:
- Phase 1.5 (500K步) 需要 **7.5小时** (当前) vs 预期 25-40分钟
- 完整训练周期 (58M步) 需要 **60天** vs 预期 1-3天

---

## 性能瓶颈分析

### 🔴 关键瓶颈 #1: Serial Mode 强制单进程训练

**位置**: [`runner.py:445`](blood-v2/python/blood/training/runner.py:445)

```python
cfg.serial_mode = True
cfg.async_rl = False
```

**影响**: 
- Sample Factory 2 的并行优势完全丧失
- Learner、RolloutWorker、EnvRunner 全部运行在主进程
- 无法利用多核 CPU 并行采样环境交互
- 当前配置 `num_workers: 16, num_envs_per_worker: 16` = 256 并行环境,但实际只能串行执行

**原因**: 
为了让 monkey-patch 的自定义损失函数生效。SF2 的 `ParallelRunner` 使用 `multiprocessing.spawn` 启动子进程,子进程会重新导入模块,导致主进程的 monkey-patch 失效。

**预估影响**: **10-20x 速度损失**

---

### 🟡 瓶颈 #2: 过大的神经网络架构

**编码器参数量**: ~37M (Student) + ~14M (Oracle) = **51M 总参数**

**架构细节**:
- **20个 BottleneckBlock** (每个 ~1.8M 参数)
- **4个 TileAttention 层** (每个 ~260K 参数)
- **LSTM**: 2层 × 512维 = ~4M 参数
- **SpatialPoolingProj**: 注意力池化 (4 queries × 256ch)

**对比**:
- AlphaZero (围棋): ~20M 参数
- MuZero (Atari): ~10M 参数
- 本项目: **51M 参数** (麻将,动作空间仅34)

**影响**:
- 前向传播慢 (每步需要推理 256 个环境)
- 反向传播慢 (batch_size=2048, 梯度计算量大)
- GPU 利用率可能不足 (如果在 CPU 上训练则更慢)

**预估影响**: **2-3x 速度损失**

---

### 🟡 瓶颈 #3: Rust 引擎超时保护机制

**位置**: [`blood_env.py:81-108`](blood-v2/python/blood/env/blood_env.py:81)

```python
def _run_with_timeout(fn, timeout=_RUST_TIMEOUT_SEC):
    # 每次调用创建新的 daemon 线程
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)  # 15秒超时
```

**影响**:
- 每次 `reset()` 和 `step()` 都创建新线程
- 线程创建/销毁开销 (256 envs × 每局 ~100 steps = 25,600 次/局)
- 15秒超时检测增加延迟

**预估影响**: **1.2-1.5x 速度损失**

---

### 🟢 瓶颈 #4: 频繁的 Arena 评估

**配置**: `blood_arena_eval_every: 200000` (Phase 1.5)

**影响**:
- 每 200K 步运行 50 局完整对局
- 评估在后台线程运行,但会占用 CPU 资源
- 当前速度下,200K 步需要 3 小时,评估频率合理

**预估影响**: **<1.1x** (次要)

---

### 🟢 瓶颈 #5: 过于频繁的 Checkpoint 保存

**配置**: `save_every_sec: 30` (每 30 秒保存一次)

**影响**:
- Checkpoint 大小 ~160MB (51M 参数 × 4 bytes)
- 每 30 秒写入 160MB 到磁盘
- 当前速度下,30秒只训练 ~560 步,保存频率过高

**预估影响**: **1.1-1.2x 速度损失**

---

## 优化建议 (按优先级排序)

### ⚡ 优先级 1: 解决 Serial Mode 瓶颈

**方案 A: 使用 SF2 的自定义损失扩展点 (推荐)**

Sample Factory 2 应该提供了自定义损失的官方方式。检查文档:
- `custom_loss_fn` 参数
- `LossComputer` 基类
- 或通过 `register_custom_component` 注册

**方案 B: 改用 Shared Memory 传递 Patch**

```python
# 主进程
import multiprocessing as mp
_patch_flag = mp.Value('i', 1)  # shared memory flag

# 子进程 (在模块导入时检查)
if _patch_flag.value == 1:
    _apply_patches()
```

**方案 C: 迁移到支持并行的框架**

- CleanRL (更轻量,易于自定义)
- RLlib (Ray 生态,天然并行)
- 或直接基于 PyTorch DDP 实现

**预期提升**: **10-20x** → 训练时间从 60天降到 3-6天

---

### ⚡ 优先级 2: 优化神经网络架构

**方案 A: 减少 ResBlock 数量**

```yaml
blood_num_res_blocks: 20  # 当前
→ blood_num_res_blocks: 12  # 建议 (减少 40%)
```

**方案 B: 减少 TileAttention 层数**

```yaml
blood_num_tile_attn_layers: 4  # 当前
→ blood_num_tile_attn_layers: 2  # 建议 (减少 50%)
```

**方案 C: 减小卷积通道数**

```yaml
blood_conv_channels: 256  # 当前
→ blood_conv_channels: 192  # 建议 (减少 25%)
```

**方案 D: 使用 Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(obs)
scaler.scale(loss).backward()
```

**预期提升**: **2-3x** → 配合方案A可达 **30-60x 总提升**

---

### ⚡ 优先级 3: 优化 Rust 引擎调用

**方案 A: 使用线程池代替临时线程**

```python
from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=4)

def _run_with_timeout(fn, timeout=15):
    future = _executor.submit(fn)
    return future.result(timeout=timeout)
```

**方案 B: 移除超时保护 (如果 Rust 引擎稳定)**

```python
# 直接调用,不创建线程
result = fn()
```

**方案 C: 批量调用 Rust 引擎**

```python
# 一次调用处理多个环境
results = engine.batch_step(actions)  # Rust 端实现
```

**预期提升**: **1.2-1.5x**

---

### 🔧 优先级 4: 调整训练超参数

**方案 A: 增大 Batch Size**

```yaml
batch_size: 2048  # 当前
→ batch_size: 4096  # 建议 (减少更新频率)
```

**方案 B: 减少 Checkpoint 保存频率**

```yaml
save_every_sec: 30  # 当前 (每 30 秒)
→ save_every_sec: 300  # 建议 (每 5 分钟)
```

**方案 C: 减少 Arena 评估频率**

```yaml
blood_arena_eval_every: 200000  # 当前
→ blood_arena_eval_every: 500000  # 建议
```

**预期提升**: **1.2-1.5x**

---

## 综合优化方案

### 🎯 短期方案 (1-2天实施)

1. **解决 Serial Mode** (方案 A 或 B)
2. **减少 ResBlock** (20 → 12)
3. **优化 Rust 调用** (使用线程池)
4. **调整保存频率** (30s → 300s)

**预期总提升**: **15-25x**  
**训练时间**: 60天 → **2-4天**

---

### 🚀 长期方案 (1-2周实施)

1. 短期方案全部
2. **减少 TileAttention** (4 → 2)
3. **减小通道数** (256 → 192)
4. **Mixed Precision Training**
5. **迁移到 CleanRL/RLlib**

**预期总提升**: **30-50x**  
**训练时间**: 60天 → **1-2天**

---

## 性能监控建议

### 添加性能指标

```python
# 在 callbacks.py 中添加
def extra_summaries(self, runner, policy_id, writer, env_steps):
    # 采样速度
    fps = runner.fps.get(policy_id, 0)
    writer.add_scalar("perf/fps", fps, env_steps)
    
    # GPU 利用率 (如果使用 GPU)
    if torch.cuda.is_available():
        gpu_util = torch.cuda.utilization()
        writer.add_scalar("perf/gpu_util", gpu_util, env_steps)
    
    # 前向传播时间
    if hasattr(learner, "_forward_time"):
        writer.add_scalar("perf/forward_ms", learner._forward_time * 1000, env_steps)
```

### 使用 Profiler

```python
# 在训练循环中
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# ... 训练代码 ...
profiler.disable()
profiler.dump_stats("training_profile.prof")

# 分析
# python -m pstats training_profile.prof
```

---

## 总结

当前训练速度慢的**根本原因**是 **Serial Mode 强制单进程训练**,丧失了 Sample Factory 2 的并行优势。

**立即行动项**:
1. 调研 SF2 的官方自定义损失扩展方式
2. 如果没有,考虑迁移到 CleanRL (更易自定义)
3. 同时优化神经网络架构 (减少 ResBlock 和 TileAttention)

**预期结果**:
- 短期优化: 训练时间从 60天 → **2-4天**
- 长期优化: 训练时间从 60天 → **1-2天**

---

生成时间: 2026-02-27 13:57 CST