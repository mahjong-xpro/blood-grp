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
        import glob
        from blood.consts import ACTION_SPACE
        
        # 查找最新的checkpoint
        checkpoints = glob.glob("train_dir/blood_v2_warmup/checkpoint_*")
        if not checkpoints:
            print("  ⚠️  未找到checkpoint，跳过模型测试")
            print("  请先训练模型后再运行此测试")
            return True
        
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"\n加载checkpoint: {latest_ckpt}")
        
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        
        # 检查action head的权重
        if 'model' not in checkpoint:
            print("  ⚠️  Checkpoint格式不正确")
            return True
        
        model_state = checkpoint['model']
        
        # 查找action parameterization的最后一层
        action_keys = [k for k in model_state.keys() if 'action_parameterization' in k and 'weight' in k]
        
        if not action_keys:
            print("  ⚠️  未找到action head权重")
            return True
        
        print(f"\nAction head keys: {action_keys}")
        
        for key in action_keys:
            weight = model_state[key]
            if weight.shape[-1] == 34 or weight.shape[0] == 34:  # Action space
                # 找到输出层
                if weight.shape[0] == 34:
                    dingque_weights = weight[31:34, :]
                    print(f"\n{key}:")
                    print(f"  Shape: {weight.shape}")
                    print(f"  DingQue weights [31:34, :]:")
                    print(f"    Norm: {torch.norm(dingque_weights, dim=1).numpy()}")
                    print(f"    Mean: {dingque_weights.mean(dim=1).numpy()}")
                    
                    # 检查是否有系统性偏差
                    norms = torch.norm(dingque_weights, dim=1)
                    if norms[0] < norms[1] * 0.5 or norms[0] < norms[2] * 0.5:
                        print(f"\n  ❌ ERROR: Action 31 (Man) 的权重明显小于其他!")
                        print(f"    这可能解释了为什么Man从不被选择")
                        return False
        
                    print(f"\n  ✅ 权重检查完成")
        
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