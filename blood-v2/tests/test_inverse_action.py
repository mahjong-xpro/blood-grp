#!/usr/bin/env python3
"""测试_inverse_action的正确性"""

import sys
sys.path.insert(0, 'python')

from blood.env.augment import augment_action, SUIT_PERMUTATIONS

def test_inverse_action_logic():
    """验证反向映射的正确性"""
    
    print("=" * 60)
    print("测试反向映射逻辑")
    print("=" * 60)
    
    # 测试所有排列
    for perm in SUIT_PERMUTATIONS[1:]:  # 跳过identity
        print(f"\n排列: {perm}")
        
        # 测试定缺动作
        for original_action in [31, 32, 33]:
            # 前向: 原始 -> 增强
            augmented_action = augment_action(original_action, perm)
            
            # 方法1: 使用反向排列
            inv_perm_method1 = tuple(perm.index(i) for i in range(3))
            recovered_action_method1 = augment_action(augmented_action, inv_perm_method1)
            
            # 方法2: 直接使用相同排列
            recovered_action_method2 = augment_action(augmented_action, perm)
            
            print(f"  原始动作{original_action} -> 增强动作{augmented_action}")
            print(f"    方法1(反向排列{inv_perm_method1}): 恢复为{recovered_action_method1} {'✓' if recovered_action_method1 == original_action else '✗'}")
            print(f"    方法2(相同排列{perm}): 恢复为{recovered_action_method2} {'✓' if recovered_action_method2 == original_action else '✗'}")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    
    # 验证哪个方法正确
    all_correct_method1 = True
    all_correct_method2 = True
    
    for perm in SUIT_PERMUTATIONS[1:]:
        inv_perm = tuple(perm.index(i) for i in range(3))
        for original_action in [31, 32, 33]:
            augmented = augment_action(original_action, perm)
            recovered1 = augment_action(augmented, inv_perm)
            recovered2 = augment_action(augmented, perm)
            
            if recovered1 != original_action:
                all_correct_method1 = False
            if recovered2 != original_action:
                all_correct_method2 = False
    
    if all_correct_method1:
        print("✓ 方法1(反向排列)是正确的")
    else:
        print("✗ 方法1(反向排列)是错误的")
    
    if all_correct_method2:
        print("✓ 方法2(相同排列)是正确的")
    else:
        print("✗ 方法2(相同排列)是错误的")

if __name__ == '__main__':
    test_inverse_action_logic()