#!/usr/bin/env python3
"""
测试DingQue reward shaping是否生效
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from blood.env.selfplay_env import SelfPlayEnv


class TestConfig:
    suit_augment_prob = 0.0
    opponent_mode = "rulebot"
    initial_score = 100000
    warmup_reward_shaping = True
    warmup_win_bonus = 0.1
    warmup_deal_in_penalty = 0.0
    warmup_dangerous_discard_penalty = 0.03
    reward_tsumo_bonus = 0.0
    reward_deal_in_penalty = 0.0
    reward_shanten_progress = 0.003
    reward_shanten_regress = 0.001
    reward_safe_discard = 0.0
    reward_rank_bonus = 0.0
    shanten_reward_decay_steps = 0
    shanten_fan_bonus_scale = 0.0
    league_enabled = False
    opponent_refresh_every = 999999


def test_reward_shaping():
    """测试reward shaping是否对DingQue决策产生影响"""
    
    print("=" * 60)
    print("测试DingQue Reward Shaping")
    print("=" * 60)
    
    env = SelfPlayEnv(cfg=TestConfig())
    
    # 测试多个场景
    num_tests = 10
    rewards_collected = []
    
    print(f"\n测试 {num_tests} 个DingQue决策...")
    
    for test_id in range(num_tests):
        obs_dict, _ = env.reset(seed=42 + test_id)
        
        # 检查是否是DingQue阶段
        if not hasattr(env, '_env') or env._env is None:
            print(f"\n测试 {test_id + 1}: ⚠️  环境未初始化")
            continue
            
        phase = env._env.get_phase()
        if phase != "ding_que":
            print(f"\n测试 {test_id + 1}: ⚠️  不在DingQue阶段")
            continue
        
        # 获取手牌
        try:
            hand = env._env.get_agent_hand()
            suit_counts = [
                sum(1 for t in hand if 0 <= t < 9),   # Man
                sum(1 for t in hand if 9 <= t < 18),  # Pin
                sum(1 for t in hand if 18 <= t < 27), # Sou
            ]
            
            print(f"\n测试 {test_id + 1}:")
            print(f"  手牌花色: Man={suit_counts[0]}, Pin={suit_counts[1]}, Sou={suit_counts[2]}")
            
            # 测试每个DingQue动作的reward
            test_rewards = {}
            for action in [31, 32, 33]:
                obs_dict_copy, _ = env.reset(seed=42 + test_id)
                obs_dict_step, reward, done, truncated, info = env.step(action)
                test_rewards[action] = reward
                print(f"  Action {action} ({['Man', 'Pin', 'Sou'][action-31]}): reward={reward:.4f}")
            
            rewards_collected.append((suit_counts, test_rewards))
            
            # 检查reward是否符合预期
            min_suit = suit_counts.index(min(suit_counts))
            max_suit = suit_counts.index(max(suit_counts))
            
            expected_best = 31 + min_suit
            expected_worst = 31 + max_suit
            
            actual_best = max(test_rewards, key=test_rewards.get)
            actual_worst = min(test_rewards, key=test_rewards.get)
            
            if expected_best == actual_best and expected_worst == actual_worst:
                print(f"  ✅ Reward shaping正确")
            else:
                print(f"  ❌ Reward shaping不符合预期")
                print(f"     期望最佳: {expected_best}, 实际最佳: {actual_best}")
                print(f"     期望最差: {expected_worst}, 实际最差: {actual_worst}")
                
        except Exception as e:
            print(f"\n测试 {test_id + 1}: ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()
    
    print("\n" + "=" * 60)
    if len(rewards_collected) > 0:
        print("✅ 测试完成")
        print("\n如果reward shaping正确，选择最少花色应该获得最高reward")
    else:
        print("❌ 没有收集到任何数据")
    print("=" * 60)


if __name__ == "__main__":
    test_reward_shaping()