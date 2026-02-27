#!/usr/bin/env python3
"""
实时测试模型在dingque阶段的行为
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from blood.env.blood_env import BloodMahjongEnv


def test_dingque_mask_and_logits():
    """测试dingque阶段的mask和logits"""
    
    print("=" * 60)
    print("测试 DingQue 阶段的 Mask 和 Logits")
    print("=" * 60)
    
    # 创建环境
    env = BloodMahjongEnv()
    
    # 测试多个种子
    for seed in [42, 123, 456, 789, 1000]:
        obs_dict, _ = env.reset(seed=seed)
        
        mask = obs_dict["action_mask"]
        dingque_mask = mask[31:34]
        
        print(f"\nSeed {seed}:")
        print(f"  Phase: {env.get_phase()}")
        print(f"  DingQue Mask [31:34]: {dingque_mask}")
        print(f"  31 (Man): {mask[31]}")
        print(f"  32 (Pin): {mask[32]}")
        print(f"  33 (Sou): {mask[33]}")
        
        # 检查是否所有dingque动作都合法
        if not np.all(dingque_mask > 0.5):
            print(f"  ❌ ERROR: Some dingque actions are masked!")
            print(f"  Full mask: {mask}")
            return False
        
        # 检查其他动作是否被正确屏蔽
        other_mask = mask[:31]
        if not np.all(other_mask < 0.5):
            print(f"  ❌ ERROR: Some non-dingque actions are not masked!")
            non_zero = np.where(other_mask > 0.5)[0]
            print(f"  Non-zero indices: {non_zero}")
            return False
        
        print(f"  ✅ Mask is correct")
    
    env.close()
    return True


def test_model_dingque_behavior():
    """测试模型在dingque阶段的实际行为"""
    
    print(f"\n{'=' * 60}")
    print("测试模型的 DingQue 行为")
    print("=" * 60)
    
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
            'use_rnn': False,
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
        batch_size = 1000
        obs = torch.randn(batch_size, OBS_SIZE)
        
        # 创建dingque mask (只有31, 32, 33合法)
        mask = torch.zeros(batch_size, ACTION_SPACE)
        mask[:, 31:34] = 1.0
        
        obs_dict = {'obs': obs, 'action_mask': mask}
        
        # 前向传播
        with torch.no_grad():
            result = model(obs_dict, None, values_only=False)
            logits = result['action_logits']
            actions = result.get('actions')
        
        # 检查logits
        dingque_logits = logits[:, 31:34]
        other_logits = logits[:, :31]
        
        print(f"\nLogits 统计 (batch_size={batch_size}):")
        print(f"  DingQue logits [31:34]:")
        print(f"    均值: {dingque_logits.mean(dim=0).numpy()}")
        print(f"    标准差: {dingque_logits.std(dim=0).numpy()}")
        print(f"    最小值: {dingque_logits.min(dim=0).values.numpy()}")
        print(f"    最大值: {dingque_logits.max(dim=0).values.numpy()}")
        
        print(f"\n  Other logits [0:31]:")
        print(f"    均值: {other_logits.mean().item():.2f}")
        print(f"    最大值: {other_logits.max().item():.2f}")
        print(f"    (应该是 -inf 或非常小的负数)")
        
        # 检查采样的actions
        if actions is not None:
            action_counts = torch.bincount(actions.flatten(), minlength=ACTION_SPACE)
            dingque_counts = action_counts[31:34]
            
            print(f"\n采样的 Actions 分布:")
            print(f"  31 (Man): {dingque_counts[0].item()} ({dingque_counts[0].item()/batch_size*100:.1f}%)")
            print(f"  32 (Pin): {dingque_counts[1].item()} ({dingque_counts[1].item()/batch_size*100:.1f}%)")
            print(f"  33 (Sou): {dingque_counts[2].item()} ({dingque_counts[2].item()/batch_size*100:.1f}%)")
            
            # 检查是否有action 31
            if dingque_counts[0].item() == 0:
                print(f"\n  ❌ ERROR: Action 31 (Man) never sampled!")
                print(f"  这说明存在系统性偏差")
                return False
            
            # 检查分布是否合理
            max_count = dingque_counts.max().item()
            min_count = dingque_counts.min().item()
            if max_count > batch_size * 0.5:
                print(f"\n  ⚠️  WARNING: 分布不均匀 (max={max_count}, min={min_count})")
            else:
                print(f"\n  ✅ 分布相对均匀")
        
        return True
        
    except Exception as e:
        print(f"\n模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    # 测试1: Mask正确性
    if not test_dingque_mask_and_logits():
        success = False
    
    # 测试2: 模型行为
    if not test_model_dingque_behavior():
        success = False
    
    print(f"\n{'=' * 60}")
    if success:
        print("✅ 所有测试通过")
    else:
        print("❌ 发现问题")
    print("=" * 60)
    
    sys.exit(0 if success else 1)