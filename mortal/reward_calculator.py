import numpy as np

class RewardCalculator:
    def __init__(self, config=None):
        self.config = config or {}
        # 排名奖励配置
        reward_config = self.config.get('reward_shaping', {})
        self.rank_bonus_enabled = reward_config.get('rank_bonus_enabled', True)
        self.rank_bonuses = reward_config.get('rank_bonuses', [0.3, 0.1, -0.1, -0.3])

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
        ranks = (-np.array(final_scores)).argsort().argsort()
        player_rank = ranks[player_id]
        return self.rank_bonuses[player_rank]
