import torch
import numpy as np

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.uniform_init = uniform_init

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    def calc_grp(self, grp_feature):
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)

        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        return reward.cpu().numpy()

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        # GRP feature is [kyoku, score[0], score[1], score[2], score[3], agari[0], agari[1], agari[2], agari[3], ding_que[0], ding_que[1], ding_que[2], ding_que[3]]
        # So score indices are 1, 2, 3, 4
        # agari indices are 5, 6, 7, 8
        # ding_que indices are 9, 10, 11, 12
        seq = np.concatenate((grp_feature[:, 1 + player_id] * 1e4, [final_scores[player_id]]))
        delta_points = seq[1:] - seq[:-1]
        return delta_points
