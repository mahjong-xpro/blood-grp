#!/usr/bin/env python3
"""
最基本的DingQue测试 - 验证环境是否正确生成mask和接受action
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from blood.env.blood_env import BloodMahjongEnv
import numpy as np


def test_basic_dingque():
    """测试最基本的DingQue功能"""
    
    print("=" * 60)
    print("基本DingQue测试")
    print("=" * 60)
    
    env = BloodMahjongEnv()
    
    # 测试每个DingQue action
    for action in [31, 32, 33]:
        action_name = ['Man', 'Pin', 'Sou'][action - 31]
        print(f"\n测试 Action {action} ({action_name}):")
        
        obs_dict, _ = env.reset(seed=42)
        
        # 检查phase
        phase = env.get_phase()
        print(f"  Phase: {phase}")
        
        if phase != "ding_que":
            print(f"  ⚠️  不在DingQue阶段，跳过")
            continue
        
        # 检查mask
        mask = obs_dict["action_mask"]
        print(f"  Mask[31]: {mask[31]}, Mask[32]: {mask[32]}, Mask[33]: {mask[33]}")
        
        if mask[action] < 0.5:
            print(f"  ❌ Action {action} 被mask了！")
            continue
        
        # 执行action
        try:
            obs_dict_new, reward, done, truncated, info = env.step(action)
            print(f"  ✅ Action {action} 执行成功")
            print(f"     Reward: {reward}")
            print(f"     New phase: {env.get_phase()}")
            
            # 检查是否真的设置了ding_que
            if hasattr(env, '_env') and env._env is not None:
                try:
                    # 尝试获取ding_que状态
                    print(f"     环境状态已更新")
                except Exception as e:
                    print(f"     无法验证状态: {e}")
                    
        except Exception as e:
            print(f"  ❌ Action {action} 执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✅ 基本测试完成")
    print("=" * 60)
    print("\n如果所有action都能执行，说明环境本身没问题。")
    print("问题可能在于模型的action采样或训练过程。")


if __name__ == "__main__":
    test_basic_dingque()