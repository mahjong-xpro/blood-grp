#!/usr/bin/env python3
"""
创建初始 GRP 模型的脚本

用于从零开始训练时，创建随机初始化的 GRP 模型。

使用方法：
    cd mortal
    python ../scripts/create-initial-grp.py
"""

import sys
import os
from datetime import datetime

# 添加 mortal 目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
mortal_dir = os.path.join(project_root, 'mortal')
sys.path.insert(0, mortal_dir)

# 加载 libblood 模块（macOS 上需要特殊处理）
try:
    import libblood_loader  # 这会自动加载 libblood
except ImportError:
    # 如果 libblood_loader 不存在，尝试直接导入（Linux/其他平台）
    pass

import torch
from model import GRP
from config import config

def main():
    print("=" * 60)
    print("创建初始 GRP 模型")
    print("=" * 60)
    
    # 检查配置
    if 'grp' not in config:
        print("错误: 配置文件中缺少 [grp] 部分")
        sys.exit(1)
    
    grp_cfg = config['grp']
    state_file = grp_cfg['state_file']
    network_cfg = grp_cfg['network']
    
    print(f"GRP 模型文件: {state_file}")
    print(f"网络配置: {network_cfg}")
    print()
    
    # 检查文件是否已存在
    if os.path.exists(state_file):
        response = input(f"GRP 模型文件已存在: {state_file}\n是否覆盖? (y/N): ")
        if response.lower() != 'y':
            print("取消操作")
            sys.exit(0)
        print("覆盖现有文件...")
    
    # 创建 GRP 模型
    print("创建随机初始化的 GRP 模型...")
    grp = GRP(**network_cfg)
    
    # 创建优化器（用于保存状态）
    optimizer = torch.optim.AdamW(grp.parameters())
    
    # 保存模型
    state = {
        'model': grp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps': 0,
        'timestamp': datetime.now().timestamp(),
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    torch.save(state, state_file)
    
    print()
    print("=" * 60)
    print("✓ GRP 模型创建成功！")
    print("=" * 60)
    print(f"文件路径: {state_file}")
    print()
    print("注意:")
    print("  - 这是一个随机初始化的模型，预测可能不准确")
    print("  - 建议先用少量数据训练 GRP，或使用自对局数据训练")
    print("  - 训练 GRP: cd mortal && python train_grp.py")
    print()
    print("下一步:")
    print("  1. 进行初始自对局生成数据（可选）")
    print("  2. 训练 GRP 模型（推荐，使用自对局数据）")
    print("  3. 开始主模型训练: ./scripts/blood-train.sh offline")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
