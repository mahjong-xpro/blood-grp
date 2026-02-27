#!/usr/bin/env python3
"""
分析定缺逻辑是否正确

用法:
    python scripts/analyze_dingque.py replays/
"""

import json
import gzip
import sys
from pathlib import Path
from collections import Counter

def analyze_replay(replay_path):
    """分析单个回放文件的定缺情况"""
    try:
        if replay_path.suffix == '.gz':
            with gzip.open(replay_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(replay_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        dingque_events = []
        initial_hands = {}
        
        for line in lines:
            event = json.loads(line.strip())
            
            # 记录初始手牌
            if event.get('type') == 'deal':
                player = event['player']
                tiles = event['tiles']
                initial_hands[player] = tiles
            
            # 记录定缺事件
            if event.get('type') == 'ding_que':
                player = event['player']
                suit = event['suit']
                dingque_events.append({
                    'player': player,
                    'suit': suit,
                    'hand': initial_hands.get(player, [])
                })
        
        return dingque_events
    
    except Exception as e:
        print(f"Error analyzing {replay_path}: {e}")
        return []


def count_tiles_by_suit(tiles):
    """统计每个花色的牌数"""
    # 万: 0-8, 筒: 9-17, 条: 18-26
    man_count = sum(1 for t in tiles if 0 <= t <= 8)
    pin_count = sum(1 for t in tiles if 9 <= t <= 17)
    sou_count = sum(1 for t in tiles if 18 <= t <= 26)
    return {'man': man_count, 'pin': pin_count, 'sou': sou_count}


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/analyze_dingque.py <replay_dir>")
        sys.exit(1)
    
    replay_dir = Path(sys.argv[1])
    if not replay_dir.exists():
        print(f"目录不存在: {replay_dir}")
        sys.exit(1)
    
    # 收集所有回放文件
    replay_files = list(replay_dir.glob("*.json")) + list(replay_dir.glob("*.json.gz"))
    
    if not replay_files:
        print(f"未找到回放文件: {replay_dir}")
        sys.exit(1)
    
    print(f"找到 {len(replay_files)} 个回放文件\n")
    
    # 统计定缺选择
    suit_counter = Counter()
    player_suit_counter = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}
    
    # 分析每个回放
    total_games = 0
    total_dingque = 0
    
    for replay_file in replay_files:
        dingque_events = analyze_replay(replay_file)
        if dingque_events:
            total_games += 1
            
            for event in dingque_events:
                player = event['player']
                suit = event['suit']
                hand = event['hand']
                
                # 统计定缺选择
                suit_counter[suit] += 1
                player_suit_counter[player][suit] += 1
                total_dingque += 1
                
                # 分析手牌
                counts = count_tiles_by_suit(hand)
                min_suit = min(counts, key=counts.get)
                min_count = counts[min_suit]
                
                # 检查是否选择了最少的花色
                if suit != min_suit:
                    print(f"⚠️  玩家 {player} 定缺 {suit}, 但 {min_suit} 更少!")
                    print(f"   手牌: {hand}")
                    print(f"   万:{counts['man']} 筒:{counts['pin']} 条:{counts['sou']}")
                    print(f"   选择: {suit}, 最少: {min_suit} ({min_count}张)\n")
    
    # 打印统计结果
    print(f"\n{'='*60}")
    print(f"分析了 {total_games} 局游戏, {total_dingque} 次定缺")
    print(f"{'='*60}\n")
    
    print("总体定缺分布:")
    for suit in ['man', 'pin', 'sou']:
        count = suit_counter[suit]
        pct = count / total_dingque * 100 if total_dingque > 0 else 0
        print(f"  {suit}: {count:4d} ({pct:5.1f}%)")
    
    print("\n各玩家定缺分布:")
    for player in range(4):
        print(f"\n玩家 {player}:")
        for suit in ['man', 'pin', 'sou']:
            count = player_suit_counter[player][suit]
            total = sum(player_suit_counter[player].values())
            pct = count / total * 100 if total > 0 else 0
            print(f"  {suit}: {count:4d} ({pct:5.1f}%)")
    
    # 判断是否有问题
    print(f"\n{'='*60}")
    if suit_counter['man'] > total_dingque * 0.5:
        print("❌ 发现问题: 万子定缺比例过高 (>50%)")
        print("   可能原因:")
        print("   1. 初始手牌分配有偏差")
        print("   2. 定缺逻辑有bug")
        print("   3. 随机数种子问题")
    elif max(suit_counter.values()) > total_dingque * 0.4:
        print("⚠️  警告: 某个花色定缺比例偏高 (>40%)")
        print("   理论上应该接近33.3%")
    else:
        print("✅ 定缺分布正常 (接近33.3%)")


if __name__ == "__main__":
    main()