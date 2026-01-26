import numpy as np

class RewardCalculator:
    def __init__(self):
        pass

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
