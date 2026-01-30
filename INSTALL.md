# Cluster Installation Guide

## 1. Prerequisites
- **OS**: Linux (Ubuntu 20.04/22.04 Recommended)
- **Driver**: NVIDIA Driver >= 550 (Support CUDA 12.6)
- **Conda**: Miniconda or Anaconda installed.

### 1.1 Install Miniconda (If not present)
Run these commands on each server if Conda is missing:
```bash
# Download and Install
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Initialize Shell (bash/zsh)
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Apply changes (or restart shell)
source ~/.bashrc  # or source ~/.zshrc
```

## 2. Setting up Conda Environment

### Option A: Fresh Install (Recommended)
Run this command on every node (Master + Workers):
```bash
conda env create -f environment.yml
```

### Option B: Update Existing Environment
If you modified `environment.yml` and want to update:
```bash
conda env update -f environment.yml --prune
```

### Option C: Force Re-create (Clean Slate)
If compilation is broken or packages conflict:
```bash
conda remove -n blood --all -y
conda env create -f environment.yml
```

## 3. Build Rust Extension
**Crucial**: You must rebuild `libblood` whenever Rust code changes or python version changes.
```bash
conda activate blood
maturin develop --release
```

## 4. Runtime Configuration (RTX 4090 Specific)
Add these to your `~/.bashrc` or `~/.zshrc` to prevent crashing:
```bash
# 1. Optimize for Ada Lovelace Architecture
export TORCH_CUDA_ARCH_LIST="8.9"

# 2. Disable P2P (Prevent 4090 Freeze)
export NCCL_P2P_DISABLE=1

# 3. Disable Infiniband (Unless you have it)
export NCCL_IB_DISABLE=1
```

## 5. Verification
Run this to check if PyTorch can see your GPUs:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"
```
