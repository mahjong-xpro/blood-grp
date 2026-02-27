#!/usr/bin/env python3
"""诊断定缺偏差的深度分析脚本"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import torch
from blood.env.blood_env import BloodMahjongEnv

def analyze_initial_hands(n_samples=1000):
    """分析初始手牌的花色分布，看是否有系统性偏差"""
    env = BloodMahjongEnv()
    
    suit_counts = {0: [], 1: [], 2: []}  # Man, Pin, Sou
    
    for seed in range(n_samples):
        env.reset(seed=seed)
        if hasattr(env._env, 'get_agent_hand'):
            hand = env._env.get_agent_hand()
            for suit in range(3):
                count = sum(1 for t in hand if suit * 9 <= t < (suit + 1) * 9)
                suit_counts[suit].append(count)
    
    print("=" * 60)
    print("初始手牌花色分布分析 (1000局)")
    print("=" * 60)
    for suit, name in enumerate(['万', '筒', '索']):
        counts = suit_counts[suit]
        if counts:
            print(f"{name}: 平均 {np.mean(counts):.2f} 张 (标准差 {np.std(counts):.2f})")
            print(f"    最少 {min(counts)} 张的概率: {counts.count(min(counts))/len(counts)*100:.1f}%")
    print()

def analyze_observation_encoding(n_samples=100):
    """分析观测编码中Section 3的数据"""
    env = BloodMahjongEnv()
    
    section3_sums = []
    
    for seed in range(n_samples):
        obs_dict, _ = env.reset(seed=seed)
        obs = obs_dict['obs']
        obs_2d = obs.reshape(473, 27)
        
        # Section 3: channels 18-20
        section3 = obs_2d[18:21, :]
        section3_sums.append(section3.sum())
    
    print("=" * 60)
    print("观测编码Section 3分析 (100局)")
    print("=" * 60)
    print(f"Section 3总和: 平均 {np.mean(section3_sums):.2f} (标准差 {np.std(section3_sums):.2f})")
    print(f"全零样本数: {sum(1 for s in section3_sums if s == 0)}")
    if section3_sums[0] > 0:
        print("✓ 观测编码正常工作")
    else:
        print("✗ 观测编码仍然为空！")
    print()

def analyze_action_mask():
    """分析动作掩码是否对定缺动作有偏差"""
    env = BloodMahjongEnv()
    
    mask_counts = {31: 0, 32: 0, 33: 0}
    
    for seed in range(100):
        obs_dict, _ = env.reset(seed=seed)
        mask = obs_dict['action_mask']
        
        for action in [31, 32, 33]:
            if mask[action] > 0:
                mask_counts[action] += 1
    
    print("=" * 60)
    print("动作掩码分析 (100局)")
    print("=" * 60)
    for action, name in [(31, '万'), (32, '筒'), (33, '索')]:
        print(f"动作{action}({name}): {mask_counts[action]}/100 局可用")
    print()

def test_augmentation_symmetry():
    """测试数据增强是否保持对称性"""
    from blood.env.augment import SUIT_PERMUTATIONS, augment_action
    
    print("=" * 60)
    print("数据增强对称性测试")
    print("=" * 60)
    
    # 测试每个排列对定缺动作的映射
    for perm in SUIT_PERMUTATIONS:
        results = []
        for action in [31, 32, 33]:
            aug_action = augment_action(action, perm)
            results.append(aug_action - 31)
        print(f"排列 {perm}: 万→{results[0]}, 筒→{results[1]}, 索→{results[2]}")
    
    # 检查是否每个排列都是双射
    for perm in SUIT_PERMUTATIONS:
        mapped = [augment_action(31 + i, perm) - 31 for i in range(3)]
        if sorted(mapped) != [0, 1, 2]:
            print(f"✗ 排列 {perm} 不是双射: {mapped}")
        else:
            print(f"✓ 排列 {perm} 是双射")
    print()

if __name__ == '__main__':
    print("\n定缺偏差深度诊断\n")
    
    analyze_observation_encoding()
    analyze_action_mask()
    test_augmentation_symmetry()
    
    # 只有在get_agent_hand可用时才分析初始手牌
    env = BloodMahjongEnv()
    env.reset(seed=0)
    if hasattr(env._env, 'get_agent_hand'):
        analyze_initial_hands()
    else:
        print("=" * 60)
        print("注意: get_agent_hand() 不可用，跳过初始手牌分析")
        print("=" * 60)
    
    print("\n诊断完成。")
    print("\n如果观测编码正常但仍有偏差，可能原因:")
    print("1. 训练步数不够 - 模型还在学习中")
    print("2. 探索系数太低 - 增加exploration_loss_coeff")
    print("3. 需要更强的奖励塑形 - 实现get_agent_hand()并启用奖励塑形")