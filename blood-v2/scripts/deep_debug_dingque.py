#!/usr/bin/env python3
"""
深度调试脚本：检查 dingque 决策的完整流程

分析点：
1. Rust engine 的 action mask 是否正确
2. Python 端的 augmentation 是否正确应用
3. 模型的 logits 分布
4. 采样过程是否有偏差
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch


def test_rust_engine_mask():
    """测试 Rust engine 生成的 action mask"""
    print("=" * 60)
    print("测试 1: Rust Engine Action Mask")
    print("=" * 60)
    
    try:
        from blood._engine import RustMahjongEnv
        
        # 测试多个种子
        for seed in [42, 123, 456]:
            env = RustMahjongEnv(seed, "rulebot", 100000)
            obs_dict = env.reset(seed)
            
            mask = obs_dict["action_mask"]
            dingque_mask = mask[31:34]
            
            print(f"\nSeed {seed}:")
            print(f"  Phase: {env.get_phase()}")
            print(f"  DingQue Mask [31:34]: {dingque_mask}")
            print(f"  All legal: {all(m > 0.5 for m in dingque_mask)}")
            
            if not all(m > 0.5 for m in dingque_mask):
                print(f"  ❌ ERROR: Some dingque actions are masked!")
                print(f"  Full mask: {mask}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_augmentation_logic():
    """测试 augmentation 逻辑"""
    print(f"\n{'=' * 60}")
    print("测试 2: Augmentation Logic")
    print("=" * 60)
    
    from blood.env.augment import augment_action, SUIT_PERMUTATIONS
    
    # 测试每个 permutation
    for perm in SUIT_PERMUTATIONS:
        print(f"\nPermutation: {perm}")
        
        # 测试 dingque actions
        for action in [31, 32, 33]:
            augmented = augment_action(action, perm)
            print(f"  {action} -> {augmented}")
        
        # 验证逆变换
        inv_perm = tuple(perm.index(i) for i in range(3))
        print(f"  Inverse: {inv_perm}")
        
        for action in [31, 32, 33]:
            augmented = augment_action(action, perm)
            recovered = augment_action(augmented, inv_perm)
            if recovered != action:
                print(f"  ❌ ERROR: {action} -> {augmented} -> {recovered}")


def test_model_sampling():
    """测试模型采样过程"""
    print(f"\n{'=' * 60}")
    print("测试 3: Model Sampling")
    print("=" * 60)
    
    # 模拟均匀的 logits
    logits = torch.zeros(3)  # [31, 32, 33]
    
    # 测试采样分布
    samples = []
    for _ in range(1000):
        probs = torch.softmax(logits, dim=0)
        action = torch.multinomial(probs, 1).item()
        samples.append(action)
    
    counts = [samples.count(i) for i in range(3)]
    print(f"\n均匀 logits 采样 1000 次:")
    print(f"  Action 0 (Man): {counts[0]} ({counts[0]/10:.1f}%)")
    print(f"  Action 1 (Pin): {counts[1]} ({counts[1]/10:.1f}%)")
    print(f"  Action 2 (Sou): {counts[2]} ({counts[2]/10:.1f}%)")
    
    # 测试有偏差的 logits
    biased_logits = torch.tensor([0.0, 0.0, 2.0])  # Sou 偏高
    samples = []
    for _ in range(1000):
        probs = torch.softmax(biased_logits, dim=0)
        action = torch.multinomial(probs, 1).item()
        samples.append(action)
    
    counts = [samples.count(i) for i in range(3)]
    print(f"\n有偏 logits [0, 0, 2] 采样 1000 次:")
    print(f"  Action 0 (Man): {counts[0]} ({counts[0]/10:.1f}%)")
    print(f"  Action 1 (Pin): {counts[1]} ({counts[1]/10:.1f}%)")
    print(f"  Action 2 (Sou): {counts[2]} ({counts[2]/10:.1f}%)")


def test_checkpoint_logits():
    """测试 checkpoint 的 logits 分布"""
    print(f"\n{'=' * 60}")
    print("测试 4: Checkpoint Logits Distribution")
    print("=" * 60)
    
    import glob
    checkpoints = glob.glob("train_dir/blood_v2_warmup/checkpoint_*")
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    # 使用最新的 checkpoint
    latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"\nLoading: {latest_ckpt}")
    
    try:
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        
        # 检查 action head 的权重
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            
            # 查找 action head 的最后一层
            action_head_keys = [k for k in model_state.keys() if 'action' in k.lower() and 'weight' in k]
            print(f"\nAction head keys: {action_head_keys}")
            
            for key in action_head_keys:
                weight = model_state[key]
                if weight.shape[-1] == 34:  # Action space
                    dingque_weights = weight[:, 31:34]
                    print(f"\n{key}:")
                    print(f"  Shape: {weight.shape}")
                    print(f"  DingQue weights [31:34]:")
                    print(f"    Mean: {dingque_weights.mean(dim=0)}")
                    print(f"    Std: {dingque_weights.std(dim=0)}")
                    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rust_engine_mask()
    test_augmentation_logic()
    test_model_sampling()
    test_checkpoint_logits()
    
    print(f"\n{'=' * 60}")
    print("调试完成")
    print("=" * 60)