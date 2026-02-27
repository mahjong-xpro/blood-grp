#!/usr/bin/env python3
"""
调试脚本：检查训练过程中 dingque 阶段的 action mask 和 logits

用法:
    python scripts/debug_dingque_training.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blood.env.blood_env import BloodMahjongEnv


def debug_dingque_phase():
    """检查 dingque 阶段的 action mask 是否正确"""
    
    print("=" * 60)
    print("调试 DingQue 阶段的 Action Mask")
    print("=" * 60)
    
    # 创建环境
    env = BloodMahjongEnv()
    
    # 重置环境（应该进入 dingque 阶段）
    obs_dict, info = env.reset(seed=42)
    
    print(f"\n当前阶段: {env._env.get_phase()}")
    print(f"当前玩家: {env._env.get_current_player()}")
    
    # 检查 action mask
    mask = obs_dict["action_mask"]
    print(f"\nAction Mask 形状: {mask.shape}")
    print(f"Action Mask 类型: {mask.dtype}")
    
    # 找出所有合法动作
    legal_actions = np.where(mask > 0.5)[0]
    print(f"\n合法动作索引: {legal_actions}")
    print(f"合法动作数量: {len(legal_actions)}")
    
    # 检查 dingque 动作 (31=Man, 32=Pin, 33=Sou)
    dingque_mask = mask[31:34]
    print(f"\nDingQue Mask [31:34]: {dingque_mask}")
    print(f"  31 (Man): {mask[31]}")
    print(f"  32 (Pin): {mask[32]}")
    print(f"  33 (Sou): {mask[33]}")
    
    # 检查其他动作是否被正确屏蔽
    other_mask = mask[:31]
    print(f"\n其他动作 [0:31] 是否全为 0: {np.all(other_mask < 0.5)}")
    
    if not np.all(other_mask < 0.5):
        print(f"  警告: 发现非零的其他动作!")
        non_zero = np.where(other_mask > 0.5)[0]
        print(f"  非零索引: {non_zero}")
    
    # 测试多个种子
    print(f"\n{'=' * 60}")
    print("测试多个随机种子")
    print(f"{'=' * 60}")
    
    for seed in [42, 123, 456, 789, 1000]:
        obs_dict, _ = env.reset(seed=seed)
        mask = obs_dict["action_mask"]
        dingque_mask = mask[31:34]
        
        print(f"\nSeed {seed}:")
        print(f"  DingQue Mask: {dingque_mask}")
        print(f"  所有 dingque 动作都合法: {np.all(dingque_mask > 0.5)}")
        
        if not np.all(dingque_mask > 0.5):
            print(f"  ❌ 错误: 某些 dingque 动作被屏蔽!")
    
    env.close()
    
    print(f"\n{'=' * 60}")
    print("调试完成")
    print(f"{'=' * 60}")


def debug_model_logits():
    """检查模型初始化后的 logits 分布"""
    
    print(f"\n{'=' * 60}")
    print("调试模型 Logits 分布")
    print(f"{'=' * 60}")
    
    try:
        from blood.model.factory import make_blood_actor_critic
        from blood.consts import OBS_SIZE, ACTION_SPACE
        from sample_factory.utils.typing import Config
        import gymnasium as gym
        
        # 创建最小配置
        cfg = Config({
            'blood_obs_channels': 473,
            'blood_conv_channels': 256,
            'blood_num_res_blocks': 20,
            'blood_encoder_out_dim': 1024,
            'blood_enc_proj_layers': 3,
            'blood_num_tile_attn_layers': 4,
            'blood_tile_attn_heads': 4,
            'use_rnn': False,  # 简化测试
            'aux_shanten_weight': 0.0,
            'oracle_enabled': False,
            'opponent_predictor_enabled': False,
            'turn_attention_enabled': False,
        })
        
        obs_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(low=0, high=1, shape=(OBS_SIZE,), dtype=np.float32),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(ACTION_SPACE,), dtype=np.float32),
        })
        action_space = gym.spaces.Discrete(ACTION_SPACE)
        
        # 创建模型
        model = make_blood_actor_critic(cfg, obs_space, action_space)
        model.eval()
        
        # 创建测试输入
        batch_size = 100
        obs = torch.randn(batch_size, OBS_SIZE)
        
        # 创建 dingque mask (只有 31, 32, 33 合法)
        mask = torch.zeros(batch_size, ACTION_SPACE)
        mask[:, 31:34] = 1.0
        
        obs_dict = {'obs': obs, 'action_mask': mask}
        
        # 前向传播
        with torch.no_grad():
            result = model(obs_dict, None, values_only=False)
            logits = result['action_logits']
        
        # 检查 dingque logits
        dingque_logits = logits[:, 31:34]
        
        print(f"\nDingQue Logits 统计 (batch_size={batch_size}):")
        print(f"  形状: {dingque_logits.shape}")
        print(f"  均值: {dingque_logits.mean(dim=0).numpy()}")
        print(f"  标准差: {dingque_logits.std(dim=0).numpy()}")
        
        # 计算每个动作被选择的概率
        probs = torch.softmax(dingque_logits, dim=1)
        mean_probs = probs.mean(dim=0)
        
        print(f"\n平均选择概率:")
        print(f"  31 (Man): {mean_probs[0]:.4f}")
        print(f"  32 (Pin): {mean_probs[1]:.4f}")
        print(f"  33 (Sou): {mean_probs[2]:.4f}")
        
        # 检查是否有系统性偏差
        max_prob = mean_probs.max().item()
        min_prob = mean_probs.min().item()
        
        print(f"\n偏差检查:")
        print(f"  最大概率: {max_prob:.4f}")
        print(f"  最小概率: {min_prob:.4f}")
        print(f"  差异: {max_prob - min_prob:.4f}")
        
        if max_prob - min_prob > 0.1:
            print(f"  ⚠️  警告: 初始化存在偏差 (差异 > 0.1)")
        else:
            print(f"  ✅ 初始化均匀 (差异 < 0.1)")
        
    except Exception as e:
        print(f"\n模型测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_dingque_phase()
    debug_model_logits()