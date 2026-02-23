# Blood-v2 部署教程

> 目标环境：Ubuntu 22.04 LTS · CPU 128核 · GPU 8× RTX 4090

---

## 一、系统基础环境

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential curl git pkg-config \
    libssl-dev libffi-dev \
    cmake ninja-build \
    htop nvtop tmux
```

---

## 二、NVIDIA 驱动 + CUDA 12.1

```bash
# 安装驱动（RTX 4090 需要 ≥ 525）
sudo apt install -y nvidia-driver-535
sudo reboot

# 验证驱动
nvidia-smi

# 安装 CUDA 12.1 Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# 写入环境变量
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

---

## 三、Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

rustup default stable
rustup update

# 验证（需要 ≥ 1.75）
rustc --version
cargo --version
```

---

## 四、安装 Conda + 创建环境

```bash
# 下载 Miniconda（推荐，比 Anaconda 轻量）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 验证
conda --version

# 创建 Python 3.11 环境
conda create -n blood python=3.11 -y
conda activate blood

# 安装 maturin（编译 Rust 扩展用）
pip install maturin
```

---

## 五、安装 PyTorch（CUDA 12.1）

```bash
conda activate blood

# 用 conda 安装 PyTorch（自动匹配 CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 验证 8 张 GPU 全部可见
python -c "
import torch
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
"
```

---

## 六、编译 Rust 引擎并安装 Python 包

```bash
conda activate blood
cd ~/Mahjong/blood/blood-v2

# release 模式编译（128 核并行，速度快）
maturin develop --release

# 验证引擎可导入
python -c "from blood._engine import BloodEnv; print('Engine OK')"
```

---

## 七、安装 Python 依赖

```bash
pip install "sample-factory>=2.0" "gymnasium>=0.29" "numpy>=1.24" "pyyaml>=6.0"

# 可选：开发工具
pip install pytest ruff tensorboard

# 完整导入验证
python -c "
from blood.training.runner import register_blood_components
register_blood_components()
print('All imports OK')
"
```

> 所有依赖都通过 pip 安装到 conda 环境中，不需要 `conda install`。
> conda 负责管理 Python 版本和 PyTorch/CUDA 的底层库，pip 负责其余包。

---

## 八、针对 8× 4090 调整训练配置

128 核 CPU + 8× 4090 可以大幅提升并行度。建议修改各阶段 yaml：

**configs/warmup.yaml**（当前值已合理，可选提升）
```yaml
num_workers: 16          # 128核 CPU 可支持更多 worker
num_envs_per_worker: 32  # 共 512 并行环境
batch_size: 8192         # warmup 阶段保守
```

**configs/competitive.yaml**
```yaml
num_workers: 16
num_envs_per_worker: 32
batch_size: 16384        # 4090 24GB 显存充足
num_batches_per_epoch: 8
```

**configs/elite.yaml**
```yaml
num_workers: 16
num_envs_per_worker: 32
batch_size: 16384
num_batches_per_epoch: 8
```

> SF2 默认单 GPU 训练。单张 4090（24GB）可容纳 batch_size=16384 的完整模型（~20M 参数）。
> 多 GPU 并行需要 SF2 分布式模式或手动 torchrun，当前配置单 GPU 已足够。

---

## 九、启动三阶段课程训练

```bash
conda activate blood
cd ~/Mahjong/blood/blood-v2

# 使用 tmux 防止 SSH 断线
tmux new -s blood_train

# Phase 1: Warmup（~2M 步，RuleBot 对手）
python -m blood.training.runner \
    --config configs/warmup.yaml \
    --device cuda

# Phase 2: Competitive（~5M 步，联赛自博弈）
# 等 warmup 完成后执行
python -m blood.training.runner \
    --config configs/competitive.yaml \
    --device cuda \
    --load_checkpoint_kind best

# Phase 3: Elite（~10M+ 步，精调）
python -m blood.training.runner \
    --config configs/elite.yaml \
    --device cuda \
    --load_checkpoint_kind best
```

tmux 操作：`Ctrl+B D` 脱离会话，`tmux attach -t blood_train` 重新连接。

---

## 十、监控训练

```bash
# TensorBoard（另开一个 tmux 窗口）
tensorboard --logdir=train_dir/ --port=6006 --bind_all
# 本地浏览器访问: http://<服务器IP>:6006

# GPU 实时监控
watch -n 1 nvidia-smi

# CPU/内存监控
htop
```

关键 TensorBoard 指标：
- `ppo_policy_loss`：纯 PPO 策略梯度（应稳定下降）
- `extra_loss_total`：辅助损失总和（aux + distill + oracle）
- `aux_loss` / `distill_loss`：辅助任务和蒸馏损失
- `reward`：每步平均奖励（应逐渐上升）

---

## 十一、运行测试

```bash
cd ~/Mahjong/blood/blood-v2
conda activate blood

# Rust 引擎单元测试
cargo test --release

# Python 模型测试
python -m pytest tests/ -v

# 快速冒烟测试
python -c "
import torch
from blood.training.runner import register_blood_components
register_blood_components()
print('GPU count:', torch.cuda.device_count())
print('System ready.')
"
```

---

## 十二、目录结构

```
~/Mahjong/blood/blood-v2/
├── checkpoints/
│   ├── blood_v2_warmup/        # warmup checkpoint
│   ├── blood_v2_competitive/   # competitive checkpoint
│   ├── blood_v2_elite/         # elite checkpoint
│   └── league/                 # 联赛池（最多 50 个历史模型）
├── train_dir/                  # TensorBoard logs
├── configs/
│   ├── warmup.yaml
│   ├── competitive.yaml
│   └── elite.yaml
└── ...
```

---

## 十三、常见问题

**`maturin develop` 报 linker error**
```bash
sudo apt install -y lld
export RUSTFLAGS="-C linker=lld"
maturin develop --release
```

**conda 环境每次登录需要手动激活**
```bash
# 设置默认激活（可选）
conda config --set auto_activate_base false
echo 'conda activate blood' >> ~/.bashrc
```

**`libcuda.so.1 not found`**
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**SF2 报 `CUDA out of memory`**

将 `batch_size` 减半（16384 → 8192），或减少 `num_envs_per_worker`（32 → 16）。

**GPU 利用率低（< 50%）**

增加 `num_workers`（建议 16~32），确保 CPU 环境生成速度跟得上 GPU 消费速度。
128 核 CPU 瓶颈通常不在 worker 数量，而在 Python GIL——可尝试 `num_workers: 32`。

**训练中断后恢复**

SF2 会自动保存 checkpoint，直接重新运行相同命令即可从最新 checkpoint 恢复。
