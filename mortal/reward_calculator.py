import numpy as np

class RewardCalculator:
    def __init__(self, config=None):
        self.config = config or {}
        # 排名奖励配置
        reward_config = self.config.get('reward_shaping', {})
        self.rank_bonus_enabled = reward_config.get('rank_bonus_enabled', True)
        self.rank_bonuses = reward_config.get('rank_bonuses', [0.3, 0.1, -0.1, -0.3])
        
        # 动作级奖励配置 (和牌/放铳)
        self.action_bonus_enabled = reward_config.get('action_bonus_enabled', False)
        self.agari_bonus = reward_config.get('agari_bonus', 0.1)  # 和牌奖励
        self.houjuu_penalty = reward_config.get('houjuu_penalty', -0.1)  # 放铳惩罚

    def calc_delta_points(self, player_id, scores_history, final_scores):
        # scores_history is raw scores [kyoku_count, 4]
        # final_scores is [4]
        
        # We construct the sequence of scores for this player throughout the game
        # scores_history contains the scores at the start of each kyoku
        # final_scores contains the scores at the end of the game
        
        # Extract player's score column
        player_scores = scores_history[:, player_id]
        
        seq = np.concatenate((player_scores, [final_scores[player_id]]))
        
        # Calculate the change in score (delta) for each step
        delta_points = seq[1:] - seq[:-1]
        return delta_points

    def calc_rank_bonus(self, player_id, final_scores):
        """计算排名奖励，仅在游戏结束时加到最后一个 kyoku
        
        Args:
            player_id: 玩家ID (0-3)
            final_scores: 最终分数列表 [4]
        
        Returns:
            float: 排名奖励值 (1st=+0.3, 2nd=+0.1, 3rd=-0.1, 4th=-0.3)
        """
        if not self.rank_bonus_enabled:
            return 0.0
        # 计算排名 (0=1st, 3=4th)
        # FIX: 使用 stable 排序，与 dataloader.py 中 player_ranks 的计算方式一致。
        # 默认 quicksort 不稳定，同分时排名可能与辅助任务目标不一致。
        ranks = (-np.array(final_scores)).argsort(kind='stable').argsort(kind='stable')
        player_rank = ranks[player_id]
        return self.rank_bonuses[player_rank]
    
    def calc_action_bonus(self, agari_count, houjuu_count):
        """计算动作级奖励 (和牌奖励 + 放铳惩罚)
        
        Args:
            agari_count: 该 kyoku 中和牌次数
            houjuu_count: 该 kyoku 中放铳次数
        
        Returns:
            float: 动作级奖励值
        """
        if not self.action_bonus_enabled:
            return 0.0
        return agari_count * self.agari_bonus + houjuu_count * self.houjuu_penalty

