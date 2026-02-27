#!/usr/bin/env python3
"""
测试observation encoding修复
验证Section 3在DingQue阶段是否正确提供花色统计信息
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from blood.env.blood_env import BloodMahjongEnv


def test_obs_encoding():
    """测试observation encoding"""
    
    print("=" * 60)
    print("测试Observation Encoding修复")
    print("=" * 60)
    
    # 创建环境（不使用augmentation以便直接观察）
    class SimpleConfig:
        suit_augment_prob = 0.0  # 禁用augmentation
        opponent_mode = "rulebot"
        initial_score = 100000
    
    env = BloodMahjongEnv(cfg=SimpleConfig())
    
    # 测试多个初始状态
    num_tests = 10
    print(f"\n测试 {num_tests} 个DingQue决策点...")
    
    all_passed = True
    
    for test_id in range(num_tests):
        obs_dict, _ = env.reset(seed=42 + test_id)
        
        # 检查是否是DingQue阶段
        phase = env.get_phase()
        if phase != "ding_que":
            print(f"\n测试 {test_id + 1}: ⚠️  跳过（不在DingQue阶段）")
            continue
        
        obs = obs_dict["obs"]
        
        # 提取Section 3的前3个通道（通道18-20）
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
        print(f"  Phase: {phase}")
        print(f"  Section 3 通道值:")
        print(f"    Man (ch 18): {man_val:.3f}")
        print(f"    Pin (ch 19): {pin_val:.3f}")
        print(f"    Sou (ch 20): {sou_val:.3f}")
        
        # 检查是否非零（修复后应该有值）
        has_values = (man_val > 0.01 or pin_val > 0.01 or sou_val > 0.01)
        
        # 检查是否合理（应该在0-1之间，且总和接近1.0）
        total = man_val + pin_val + sou_val
        reasonable = (0.8 <= total <= 1.2)  # 允许一些浮点误差
        
        if man_uniform and pin_uniform and sou_uniform and has_values and reasonable:
            print(f"  ✅ 通过 (总和: {total:.3f})")
        else:
            print(f"  ❌ 失败")
            all_passed = False
            if not man_uniform:
                print(f"     Man通道值不统一")
            if not pin_uniform:
                print(f"     Pin通道值不统一")
            if not sou_uniform:
                print(f"     Sou通道值不统一")
            if not has_values:
                print(f"     所有值都是0（未修复）")
            if not reasonable:
                print(f"     总和不合理: {total:.3f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 测试完成 - Observation encoding已正确修复")
        print("\n下一步:")
        print("1. 重新编译: cd blood-v2 && maturin develop --release")
        print("2. 清理旧数据: rm -rf train_dir/blood_v2_warmup")
        print("3. 重新训练: python3 -m blood.train --config configs/warmup.yaml")
    else:
        print("❌ 测试失败 - 需要检查修复")
        print("\n可能的原因:")
        print("1. Rust代码未重新编译")
        print("2. 修复代码有误")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_obs_encoding()
    sys.exit(0 if success else 1)