"""Runtime Policy Adaptation (RTPA).

Dynamically adjusts the policy temperature based on game state:
- When tenpai → lower temperature (aggressive, exploit winning tiles)
- When opponents are likely tenpai → higher temperature (defensive, diverse discards)
- When behind on score → slightly more aggressive
- When ahead → slightly more conservative
"""

import numpy as np

NUM_TILE_TYPES = 27
ACTION_SPACE = 34

NUM_STUDENT_CHANNELS = 464
# Channel offsets derived from crates/engine/src/obs/student.rs (0-indexed)
# Verified by tracing every ch += N in student.rs:
#   Sections 1-3 consume 5+13+17=35 ch → Section 4 starts at ch=35
# Section 4, ch=35: wall_remaining / 55.0
CH_WALL_REMAINING = 35
# Section 10: wait_tiles at ch=340 (1ch), shanten one-hot at ch=341..345 (5ch)
CH_SHANTEN_BASE = 341
# Section 9: opponent meld counts at ch=333, 334, 335 (3 opponents)
CH_OPP_MELD_BASE = 333


class RTPA:
    """Runtime Policy Adaptation for inference-time play."""

    def __init__(
        self,
        base_temp: float = 1.0,
        attack_temp: float = 0.8,
        defend_temp: float = 1.5,
        score_sensitivity: float = 0.1,
    ):
        self.base_temp = base_temp
        self.attack_temp = attack_temp
        self.defend_temp = defend_temp
        self.score_sensitivity = score_sensitivity

    def compute_temperature(
        self,
        is_tenpai: bool,
        opponents_likely_tenpai: int,
        my_score: int,
        avg_opponent_score: float,
        wall_remaining: int,
    ) -> float:
        """Compute adaptive temperature based on game context."""
        temp = self.base_temp

        if is_tenpai:
            temp = self.attack_temp
        elif opponents_likely_tenpai > 0:
            # Scale defense with number of dangerous opponents
            defense_factor = min(opponents_likely_tenpai, 3) / 3.0
            temp = self.base_temp + defense_factor * (self.defend_temp - self.base_temp)

        # When ahead (score_diff > 0): increase temperature → conservative
        # When behind (score_diff < 0): decrease temperature → aggressive
        score_diff = my_score - avg_opponent_score
        score_adjust = self.score_sensitivity * np.sign(score_diff) * min(abs(score_diff) / 32000.0, 0.2)
        temp += score_adjust

        if wall_remaining < 10:
            temp *= 1.2

        return max(0.3, min(temp, 3.0))

    def adapt_logits(
        self,
        logits: np.ndarray,
        mask: np.ndarray,
        is_tenpai: bool = False,
        opponents_likely_tenpai: int = 0,
        my_score: int = 100000,
        avg_opponent_score: float = 100000.0,
        wall_remaining: int = 50,
        danger_scores: np.ndarray = None,
    ) -> np.ndarray:
        """Apply RTPA to policy logits.

        Returns adjusted logits with temperature and optional danger penalty.
        """
        temp = self.compute_temperature(
            is_tenpai, opponents_likely_tenpai,
            my_score, avg_opponent_score, wall_remaining,
        )

        adjusted = logits.copy()
        adjusted[mask < 0.5] = -1e9

        if danger_scores is not None and not is_tenpai and opponents_likely_tenpai > 0:
            for i in range(NUM_TILE_TYPES):
                if mask[i] > 0.5:
                    adjusted[i] -= danger_scores[i] * 2.0

        adjusted /= max(temp, 1e-8)
        return adjusted


class GameStateTracker:
    """Tracks game state features needed for RTPA decisions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.my_tenpai = False
        self.opponents_tenpai_count = 0
        self.my_score = 100000
        self.opponent_scores = [100000, 100000, 100000]
        self.wall_remaining = 108 - 13 * 4

    def update_from_obs(self, obs: np.ndarray, scores: list = None):
        """Extract game state features from the observation tensor.

        Parses known channel offsets from the 464×27 student observation
        (derived from crates/engine/src/obs/student.rs):
        - Channel 35 (Section 4, ch0): wall_remaining / 55.0
        - Channels 341-345 (Section 10, shanten one-hot): ch341=tenpai
        - Channels 333-335 (Section 9, opponent meld counts): high → likely tenpai
        """
        if scores is not None and len(scores) >= 4:
            self.my_score = scores[0]
            self.opponent_scores = list(scores[1:4])

        if obs is not None and obs.shape[0] >= NUM_STUDENT_CHANNELS * NUM_TILE_TYPES:
            obs_2d = obs.reshape(-1, NUM_TILE_TYPES)
            if obs_2d.shape[0] > CH_WALL_REMAINING:
                wall_val = float(obs_2d[CH_WALL_REMAINING].mean())
                self.wall_remaining = max(0, int(wall_val * 55.0 + 0.5))

            if obs_2d.shape[0] > CH_SHANTEN_BASE + 4:
                shanten_channels = [float(obs_2d[CH_SHANTEN_BASE + i].mean()) for i in range(5)]
                self.my_tenpai = shanten_channels[0] > 0.5

            self.opponents_tenpai_count = 0
            if obs_2d.shape[0] > CH_OPP_MELD_BASE + 2:
                for i in range(3):
                    meld_ratio = float(obs_2d[CH_OPP_MELD_BASE + i].mean())
                    if meld_ratio >= 0.5:
                        self.opponents_tenpai_count += 1

    @property
    def avg_opponent_score(self) -> float:
        return sum(self.opponent_scores) / max(len(self.opponent_scores), 1)
