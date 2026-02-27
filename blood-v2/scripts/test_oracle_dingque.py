#!/usr/bin/env python3
"""测试Oracle在定缺阶段的策略是否有偏差"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import torch
from blood.env.blood_env import BloodMahjongEnv
from blood.model.oracle import OracleEncoder
from blood.consts import NUM_ORACLE_CHANNELS

def test_oracle_dingque_distribution(n_samples=1000):
    """测试Oracle模型在定缺阶段的动作分布"""
    env = BloodMahjongEnv()
    
    # 创建Oracle编码器
    oracle = OracleEncoder(
        obs_channels=NUM_ORACLE_CHANNELS,
        conv_ch=256,
        num_blocks=20,
        action_dim=34,
        num_tile_attn_layers=4,
        tile_attn_heads=4,
    )
    oracle.eval()
    
    action_counts = {31: 0, 32: 0, 33: 0}
    
    with torch.no_grad():
        for seed in range(n_samples):
            obs_dict, _ = env.reset(seed=seed)
            oracle_obs = torch.from_numpy(obs_dict['oracle_obs']).unsqueeze(0)
            action_mask = torch.from_numpy(obs_dict['action_mask']).unsqueeze(0)
            
            # 获取Oracle logits
            logits, _ = oracle(oracle_obs)
            
            # 应用mask
            masked_logits = logits.clone()
            masked_logits[~action_mask.bool()] = float('-inf')
            
            # 采样动作
            probs = torch.softmax(masked_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            
            if 31 <= action <= 33:
                action_counts[action] += 1
    
    print("=" * 60)
    print(f"Oracle定缺动作分布 ({n_samples}局)")
    print("=" * 60)
    total = sum(action_counts.values())
    for action, name in [(31, '万'), (32, '筒'), (33, '索')]:
        count = action_counts[action]
        pct = count / total * 100 if total > 0 else 0
        print(f"动作{action}({name}): {count:4d} ({pct:5.1f}%)")
    print()
    
    # 检查是否有显著偏差
    expected = total / 3
    chi_square = sum((count - expected)**2 / expected for count in action_counts.values())
    print(f"卡方统计量: {chi_square:.2f}")
    print(f"临界值(α=0.05, df=2): 5.99")
    if chi_square > 5.99:
        print("✗ 检测到显著偏差！")
    else:
        print("✓ 分布正常")
    print()

def test_oracle_logits_analysis(n_samples=100):
    """分析Oracle logits的数值特征"""
    env = BloodMahjongEnv()
    
    oracle = OracleEncoder(
        obs_channels=NUM_ORACLE_CHANNELS,
        conv_ch=256,
        num_blocks=20,
        action_dim=34,
        num_tile_attn_layers=4,
        tile_attn_heads=4,
    )
    oracle.eval()
    
    logits_31 = []
    logits_32 = []
    logits_33 = []
    
    with torch.no_grad():
        for seed in range(n_samples):
            obs_dict, _ = env.reset(seed=seed)
            oracle_obs = torch.from_numpy(obs_dict['oracle_obs']).unsqueeze(0)
            
            logits, _ = oracle(oracle_obs)
            logits_31.append(logits[0, 31].item())
            logits_32.append(logits[0, 32].item())
            logits_33.append(logits[0, 33].item())
    
    print("=" * 60)
    print(f"Oracle Logits分析 ({n_samples}局)")
    print("=" * 60)
    print(f"动作31(万): 均值={np.mean(logits_31):.4f}, 标准差={np.std(logits_31):.4f}")
    print(f"动作32(筒): 均值={np.mean(logits_32):.4f}, 标准差={np.std(logits_32):.4f}")
    print(f"动作33(索): 均值={np.mean(logits_33):.4f}, 标准差={np.std(logits_33):.4f}")
    print()
    
    # 检查均值差异
    mean_diff_31_33 = abs(np.mean(logits_31) - np.mean(logits_33))
    mean_diff_32_33 = abs(np.mean(logits_32) - np.mean(logits_33))
    print(f"均值差异 万-索: {mean_diff_31_33:.4f}")
    print(f"均值差异 筒-索: {mean_diff_32_33:.4f}")
    if mean_diff_31_33 > 0.1 or mean_diff_32_33 > 0.1:
        print("✗ 检测到logits偏差！")
    else:
        print("✓ Logits均衡")
    print()

if __name__ == '__main__':
    print("\nOracle定缺策略测试\n")
    
    print("注意: Oracle使用随机初始化权重，预期应该看到均匀分布")
    print("如果Oracle有偏差，说明模型架构或初始化有问题\n")
    
    test_oracle_dingque_distribution()
    test_oracle_logits_analysis()
    
    print("\n测试完成。")
    print("\n如果Oracle有偏差，可能原因:")
    print("1. 模型权重初始化不对称")
    print("2. 观测编码中有隐含偏差")
    print("3. 动作空间编码顺序导致的系统性偏差")
    print("\n建议: 如果Oracle有偏差，考虑禁用Oracle distillation")
    print("      在warmup.yaml中设置 oracle_enabled: false")