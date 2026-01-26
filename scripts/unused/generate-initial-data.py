#!/usr/bin/env python3
"""
生成初始自对局数据的脚本

用于从零开始训练时，生成第一批训练数据。

使用方法：
    cd mortal
    python ../scripts/generate-initial-data.py
"""

import sys
import os

# 添加 mortal 目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
mortal_dir = os.path.join(script_dir, '..', 'mortal')
sys.path.insert(0, mortal_dir)

import prelude
from model import Brain, DQN
from player import TrainPlayer
from config import config
import torch
import logging

logging.basicConfig(level=logging.INFO)

def main():
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    
    print("=" * 60)
    print("生成初始自对局数据")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"版本: {version}")
    print()
    
    # 检查 baseline 配置
    baseline_cfg = config['baseline']['train']
    baseline_file = baseline_cfg['state_file']
    
    # 创建或加载模型
    if os.path.exists(baseline_file):
        print(f"加载 baseline 模型: {baseline_file}")
        state = torch.load(baseline_file, weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', version)
        mortal = Brain(version=version, **cfg['resnet']).to(device).eval()
        dqn = DQN(version=version).to(device).eval()
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
    else:
        print("创建随机初始化的模型（baseline 不存在）")
        mortal = Brain(version=version, **config['resnet']).to(device).eval()
        dqn = DQN(version=version).to(device).eval()
    
    # 初始化训练玩家（会进行自对局）
    print("\n初始化训练玩家...")
    train_player = TrainPlayer()
    
    # 进行自对局生成数据
    print("\n开始自对局生成数据...")
    print(f"对局数: {config['train_play']['default']['games']}")
    print(f"数据保存目录: {config['train_play']['default']['log_dir']}")
    print()
    
    rankings, file_list = train_player.train_play(mortal, dqn, device)
    
    print("\n" + "=" * 60)
    print("自对局完成！")
    print("=" * 60)
    print(f"生成文件数: {len(file_list)}")
    print(f"排名分布: {rankings}")
    print(f"数据目录: {config['train_play']['default']['log_dir']}")
    print()
    print("下一步：")
    print("  1. 确保 config.toml 中 [dataset].globs 指向 train_play 目录")
    print("  2. 运行训练: ./scripts/blood-train.sh offline")
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
