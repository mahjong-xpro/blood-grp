#!/usr/bin/env python3
"""
测试模型对dingque决策的logits输出
直接检查是否observation encoding导致action 31的logit系统性偏低
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import numpy as np
from pathlib import Path
from blood.env.blood_env import BloodEnv
from sample_factory.algo.utils.torch_utils import to_torch_dtype
from sample_factory.cfg.arguments import load_from_path
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.typing import Config


def load_model(checkpoint_path: str):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载配置
    train_dir = Path(checkpoint_path).parent.parent
    cfg = load_from_path(train_dir)
    
    # 创建模型
    actor_critic = create_actor_critic(cfg, env=None)
    actor_critic.load_state_dict(checkpoint['model'])
    actor_critic.eval()
    
    return actor_critic, cfg


def test_dingque_logits():
    """测试dingque阶段的logits输出"""
    
    print("=" * 60)
    print("测试DingQue Logits输出")
    print("=" * 60)
    
    # 找到最新的checkpoint
    train_dir = Path("train_dir/blood_v2_warmup")
    checkpoint_dir = train_dir / "checkpoint_p0"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
    
    if not checkpoints:
        print("❌ 未找到checkpoint")
        return
    
    latest_checkpoint = checkpoints[-1]
    print(f"\n加载checkpoint: {latest_checkpoint}")
    
    # 加载模型
    actor_critic, cfg = load_model(str(latest_checkpoint))
    
    # 创建环境
    env = BloodEnv(full_cfg=cfg)
    
    # 收集多个dingque决策点的logits
    num_samples = 100
    all_logits = []
    all_hands = []
    
    print(f"\n收集 {num_samples} 个dingque决策点...")
    
    for i in range(num_samples):
        obs = env.reset()
        
        # 等待dingque阶段
        done = False
        while not done:
            # 检查是否是dingque阶段
            if hasattr(env, '_env'):
                phase = env._env.get_phase()
                if phase == 1:  # DingQue phase
                    # 获取observation
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                    
                    # 前向传播
                    with torch.no_grad():
                        result = actor_critic(obs_tensor)
                        logits = result['action_logits'][0]  # [34]
                        
                        # 提取dingque的logits
                        dingque_logits = logits[31:34].numpy()
                        all_logits.append(dingque_logits)
                        
                        # 记录手牌
                        hand = env._env.get_agent_hand()
                        all_hands.append(hand)
                    
                    break
            
            # 随机动作
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        
        if (i + 1) % 20 == 0:
            print(f"  已收集 {i + 1}/{num_samples}")
    
    # 分析logits
    all_logits = np.array(all_logits)  # [num_samples, 3]
    
    print("\n" + "=" * 60)
    print("Logits统计分析")
    print("=" * 60)
    
    print("\n各action的logits统计:")
    for i, action_name in enumerate(['Man (31)', 'Pin (32)', 'Sou (33)']):
        logits_i = all_logits[:, i]
        print(f"\n{action_name}:")
        print(f"  Mean:   {logits_i.mean():.4f}")
        print(f"  Std:    {logits_i.std():.4f}")
        print(f"  Min:    {logits_i.min():.4f}")
        print(f"  Max:    {logits_i.max():.4f}")
        print(f"  Median: {np.median(logits_i):.4f}")
    
    # 检查是否有系统性偏差
    print("\n" + "=" * 60)
    print("偏差检测")
    print("=" * 60)
    
    mean_logits = all_logits.mean(axis=0)
    print(f"\n平均logits: Man={mean_logits[0]:.4f}, Pin={mean_logits[1]:.4f}, Sou={mean_logits[2]:.4f}")
    
    # 计算相对差异
    max_logit = mean_logits.max()
    min_logit = mean_logits.min()
    diff = max_logit - min_logit
    
    print(f"最大差异: {diff:.4f}")
    
    if diff > 0.5:
        print(f"\n⚠️  检测到显著偏差！")
        print(f"   最高: {['Man', 'Pin', 'Sou'][mean_logits.argmax()]}")
        print(f"   最低: {['Man', 'Pin', 'Sou'][mean_logits.argmin()]}")
    else:
        print(f"\n✅ logits相对均匀")
    
    # 分析softmax后的概率
    print("\n" + "=" * 60)
    print("Softmax概率分析")
    print("=" * 60)
    
    probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
    mean_probs = probs.mean(axis=0)
    
    print(f"\n平均概率:")
    print(f"  Man: {mean_probs[0]*100:.1f}%")
    print(f"  Pin: {mean_probs[1]*100:.1f}%")
    print(f"  Sou: {mean_probs[2]*100:.1f}%")
    
    # 检查手牌分布
    print("\n" + "=" * 60)
    print("手牌花色分布")
    print("=" * 60)
    
    suit_counts = {'man': [], 'pin': [], 'sou': []}
    for hand in all_hands:
        man_count = sum(1 for t in hand if 0 <= t < 9)
        pin_count = sum(1 for t in hand if 9 <= t < 18)
        sou_count = sum(1 for t in hand if 18 <= t < 27)
        
        suit_counts['man'].append(man_count)
        suit_counts['pin'].append(pin_count)
        suit_counts['sou'].append(sou_count)
    
    print(f"\n平均手牌数量:")
    print(f"  Man: {np.mean(suit_counts['man']):.1f}")
    print(f"  Pin: {np.mean(suit_counts['pin']):.1f}")
    print(f"  Sou: {np.mean(suit_counts['sou']):.1f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_dingque_logits()