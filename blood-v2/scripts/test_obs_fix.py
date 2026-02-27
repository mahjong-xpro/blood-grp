#!/usr/bin/env python3
"""
测试observation encoding修复
验证Section 3在DingQue阶段是否正确提供花色统计信息
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from blood.env.blood_env import BloodEnv
from sample_factory.cfg.arguments import load_from_path
from pathlib import Path


def test_obs_encoding():
    """测试observation encoding"""
    
    print("=" * 60)
    print("测试Observation Encoding修复")
    print("=" * 60)
    
    # 加载配置
    train_dir = Path("train_dir/blood_v2_warmup")
    cfg = load_from_path(train_dir)
    
    # 创建环境
    env = BloodEnv(full_cfg=cfg)
    
    # 测试多个初始状态
    num_tests = 10
    print(f"\n测试 {num_tests} 个DingQue决策点...")
    
    for test_id in range(num_tests):
        obs = env.reset()
        
        # 等待DingQue阶段
        done = False
        step_count = 0
        while not done and step_count < 100:
            # 检查是否是DingQue阶段
            if hasattr(env, '_env'):
                phase = env._env.get_phase()
                if phase == 1:  # DingQue phase
                    # 获取手牌
                    hand = env._env.get_agent_hand()
                    
                    # 计算实际花色数量
                    man_count = sum(1 for t in hand if 0 <= t < 9)
                    pin_count = sum(1 for t in hand if 9 <= t < 18)
                    sou_count = sum(1 for t in hand if 18 <= t < 27)
                    
                    # 提取Section 3的前3个通道（通道5-7）
                    # Section 1: 0-4 (5 ch)
                    # Section 2: 5-17 (13 ch)
                    # Section 3: 18-34 (17 ch) <- 我们要检查的
                    section3_start = 5 + 13  # = 18
                    
                    # 每个通道是27个tile的值
                    obs_reshaped = obs.reshape(473, 27)
                    
                    # 提取前3个通道的值（应该是花色统计）
                    man_channel = obs_reshaped[section3_start]
                    pin_channel = obs_reshaped[section3_start + 1]
                    sou_channel = obs_reshaped[section3_start + 2]
                    
                    # 检查是否所有值都相同（fill_ch的效果）
                    man_val = man_channel[0]
                    pin_val = pin_channel[0]
                    sou_val = sou_channel[0]
                    
                    # 验证所有tile位置的值都相同
                    man_uniform = np.allclose(man_channel, man_val)
                    pin_uniform = np.allclose(pin_channel, pin_val)
                    sou_uniform = np.allclose(sou_channel, sou_val)
                    
                    print(f"\n测试 {test_id + 1}:")
                    print(f"  手牌: Man={man_count}, Pin={pin_count}, Sou={sou_count}")
                    print(f"  Obs:  Man={man_val:.3f}, Pin={pin_val:.3f}, Sou={sou_val:.3f}")
                    print(f"  期望: Man={man_count/13:.3f}, Pin={pin_count/13:.3f}, Sou={sou_count/13:.3f}")
                    
                    # 验证
                    expected_man = man_count / 13.0
                    expected_pin = pin_count / 13.0
                    expected_sou = sou_count / 13.0
                    
                    man_ok = abs(man_val - expected_man) < 0.01
                    pin_ok = abs(pin_val - expected_pin) < 0.01
                    sou_ok = abs(sou_val - expected_sou) < 0.01
                    
                    if man_ok and pin_ok and sou_ok and man_uniform and pin_uniform and sou_uniform:
                        print(f"  ✅ 通过")
                    else:
                        print(f"  ❌ 失败")
                        if not man_uniform:
                            print(f"     Man通道值不统一")
                        if not pin_uniform:
                            print(f"     Pin通道值不统一")
                        if not sou_uniform:
                            print(f"     Sou通道值不统一")
                        if not man_ok:
                            print(f"     Man值不匹配: {man_val:.3f} vs {expected_man:.3f}")
                        if not pin_ok:
                            print(f"     Pin值不匹配: {pin_val:.3f} vs {expected_pin:.3f}")
                        if not sou_ok:
                            print(f"     Sou值不匹配: {sou_val:.3f} vs {expected_sou:.3f}")
                    
                    break
            
            # 随机动作
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            step_count += 1
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
    print("\n如果所有测试都通过，说明observation encoding已正确修复。")
    print("现在需要重新编译Rust代码并重新训练模型。")


if __name__ == "__main__":
    test_obs_encoding()