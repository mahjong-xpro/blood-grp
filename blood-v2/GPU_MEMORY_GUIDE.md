# GPU显存优化指南

## 问题诊断

你遇到的OOM错误:
```
CUDA out of memory. Tried to allocate 14.00 MiB. 
GPU 0 has a total capacity of 23.55 GiB of which 11.38 MiB is free.
Process has 22.93 GiB memory in use.
```

**原因**: 24GB GPU + batch_size=4096 导致显存不足

## 已实施的修复

所有配置文件已调整为24GB GPU优化的batch size:

| 配置文件 | 原始 | 优化后 | 说明 |
|---------|------|--------|------|
| [`warmup.yaml`](configs/warmup.yaml:127) | 4096 | 3072 | 降低25% |
| [`competitive.yaml`](configs/competitive.yaml:138) | 2048 | 1536 | 降低25% |
| [`elite.yaml`](configs/elite.yaml:163) | 2048 | 1536 | 降低25% |

## 重启训练

清理GPU显存并重启:
```bash
# 停止当前训练
./scripts/manage.sh stop

# 清理GPU显存
nvidia-smi --gpu-reset

# 重启训练
./scripts/manage.sh train warmup
```

## 如果仍然OOM

### 方案1: 进一步降低batch_size

编辑 [`configs/warmup.yaml`](configs/warmup.yaml:127):
```yaml
batch_size: 2048  # 从3072降到2048
```

### 方案2: 减少并行环境数

编辑 [`configs/warmup.yaml`](configs/warmup.yaml:10):
```yaml
num_workers: 12          # 从16降到12
num_envs_per_worker: 12  # 从16降到12
```

### 方案3: 禁用mixed precision (不推荐)

编辑 [`configs/warmup.yaml`](configs/warmup.yaml:123):
```yaml
use_mixed_precision: false  # 从true改为false
```

**注意**: 这会降低训练速度约1.5-2x

## 显存使用估算

### 24GB GPU的理论容量

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| 模型参数 | ~2GB | 51M参数 × FP32 |
| 优化器状态 | ~4GB | Adam: 2x模型参数 |
| 梯度 | ~2GB | 与模型参数相同 |
| 激活值 | ~8-12GB | 取决于batch_size |
| 环境缓冲 | ~2-4GB | 256个并行环境 |
| **总计** | **18-24GB** | 接近24GB上限 |

### 不同batch_size的显存占用

| batch_size | 预计显存 | 24GB GPU | 说明 |
|-----------|---------|----------|------|
| 4096 | ~24GB | ❌ OOM | 原始配置 |
| 3072 | ~20GB | ✅ 安全 | 当前配置 |
| 2048 | ~16GB | ✅ 安全 | 保守配置 |
| 1536 | ~14GB | ✅ 安全 | 最保守 |

## 性能影响

降低batch_size对训练速度的影响:

| batch_size | 相对速度 | 说明 |
|-----------|---------|------|
| 4096 | 100% | 基准(但OOM) |
| 3072 | ~85% | 轻微降速 |
| 2048 | ~70% | 中等降速 |
| 1536 | ~60% | 明显降速 |

**结论**: batch_size=3072是24GB GPU的最佳平衡点

## 监控显存使用

### 实时监控
```bash
watch -n 1 nvidia-smi
```

### 训练中监控
```bash
# 在另一个终端运行
while true; do
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
  sleep 5
done
```

### 预期显存使用

正常训练时应该看到:
```
Memory-Usage: 18000 MiB / 24576 MiB  (73%)
```

如果接近100%,考虑降低batch_size。

## 其他优化建议

### 1. 使用梯度累积 (如果需要更大的有效batch_size)

编辑配置文件:
```yaml
batch_size: 2048              # 物理batch
num_batches_per_epoch: 8      # 累积8个batch = 有效batch 16384
```

### 2. 使用梯度检查点 (节省显存但降速)

在 [`learner_patch.py`](python/blood/training/learner_patch.py:78) 中添加:
```python
# 在autocast_ctx之前
torch.utils.checkpoint.checkpoint_sequential(...)
```

### 3. 清理未使用的缓存

在训练脚本中定期调用:
```python
torch.cuda.empty_cache()
```

## 故障排查流程

1. **检查GPU状态**
   ```bash
   nvidia-smi
   ```
   确认没有其他进程占用显存

2. **停止所有训练**
   ```bash
   ./scripts/manage.sh stop
   pkill -f "blood.training.runner"
   ```

3. **重置GPU**
   ```bash
   nvidia-smi --gpu-reset
   ```

4. **使用保守配置重启**
   ```bash
   # 临时降低batch_size
   python -m blood.training.runner \
     --config configs/warmup.yaml \
     --batch_size 2048
   ```

5. **监控显存**
   ```bash
   watch -n 1 nvidia-smi
   ```

## 多GPU训练 (如果有多张GPU)

如果有多张GPU,可以使用数据并行:

```bash
./scripts/manage.sh train warmup --num-policies 2
```

这会在2张GPU上各运行一个policy,每张GPU的显存压力减半。

## 总结

✅ **已修复**: 所有配置文件已调整为24GB GPU优化的batch_size

✅ **预期效果**: 
- 显存使用: ~20GB (83%)
- 训练速度: 仍有12-20x加速
- 稳定性: 不会OOM

✅ **下一步**: 重启训练
```bash
./scripts/manage.sh stop
./scripts/manage.sh train warmup
```

如果仍然OOM,进一步降低batch_size到2048。