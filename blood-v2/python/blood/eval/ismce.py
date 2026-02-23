"""Python wrapper for ISMCE inference-time search.

Combines the Rust ISMCE evaluation with the policy network
to produce refined action selections at test time.
"""

import logging
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

ACTION_SPACE = 34
NUM_TILE_TYPES = 27


class ISMCESearcher:
    """Combines neural policy with ISMCE evaluation for refined action selection.

    At inference time:
        1. Get policy logits from the neural network
        2. Run ISMCE evaluation on discard candidates
        3. Blend the two signals: final = policy_weight * policy + ismce_weight * ismce
    """

    def __init__(
        self,
        policy_weight: float = 0.7,
        ismce_weight: float = 0.3,
        num_worlds: int = 64,
        rollout_depth: int = 4,
    ):
        self.policy_weight = policy_weight
        self.ismce_weight = ismce_weight
        self.num_worlds = num_worlds
        self.rollout_depth = rollout_depth

        self._ismce_available = False
        try:
            from blood._engine import ismce_evaluate, ismce_danger
            self._ismce_evaluate = ismce_evaluate
            self._ismce_danger = ismce_danger
            self._ismce_available = True
        except ImportError:
            log.warning("ISMCE Rust module not available; falling back to policy-only")

    def select_action(
        self,
        policy_logits: np.ndarray,
        action_mask: np.ndarray,
        hand: Optional[np.ndarray] = None,
        melds_count: int = 0,
        ding_que: int = -1,
        tiles_seen: Optional[np.ndarray] = None,
        wall_remaining: int = 50,
        temperature: float = 1.0,
    ) -> int:
        """Select action by blending policy with ISMCE scores."""
        logits = policy_logits.copy()
        logits[action_mask < 0.5] = -1e9

        if not self._ismce_available or hand is None or tiles_seen is None:
            return self._sample_from_logits(logits, temperature)

        discard_candidates = [i for i in range(NUM_TILE_TYPES) if action_mask[i] > 0.5]
        if not discard_candidates:
            return self._sample_from_logits(logits, temperature)

        try:
            base_seed = int(time.monotonic_ns()) & 0xFFFFFFFFFFFFFFFF
            results = self._ismce_evaluate(
                hand.astype(np.uint8),
                melds_count,
                ding_que,
                tiles_seen.astype(np.uint8),
                discard_candidates,
                wall_remaining,
                self.num_worlds,
                self.rollout_depth,
                base_seed,
            )
        except Exception:
            return self._sample_from_logits(logits, temperature)

        ismce_scores = np.full(ACTION_SPACE, 0.0)
        for tile_idx, win_rate, tenpai_rate, improvement in results:
            combined = 2.0 * win_rate + tenpai_rate + 0.5 * improvement
            ismce_scores[tile_idx] = combined

        # Blend in log-space: final_logits = policy_logits/T + ismce_weight * ismce_logits
        # This is more principled than probability-space blending because:
        # 1. Avoids probability collapse when one distribution is very peaked
        # 2. Preserves the relative ordering from both signals
        # 3. ismce_scores are already in a comparable scale (win_rate ∈ [0,1])
        policy_logits_t = logits / max(temperature, 1e-8)
        # Normalize ISMCE scores to zero-mean over candidates to avoid scale mismatch
        candidate_scores = ismce_scores[discard_candidates]
        ismce_scores_norm = ismce_scores.copy()
        ismce_scores_norm[discard_candidates] = candidate_scores - candidate_scores.mean()

        blended_logits = policy_logits_t.copy()
        for i in discard_candidates:
            blended_logits[i] = (self.policy_weight * policy_logits_t[i]
                                 + self.ismce_weight * ismce_scores_norm[i])
        blended_logits[action_mask < 0.5] = -1e9

        from blood.utils import softmax
        blended = softmax(blended_logits)
        total = blended.sum()
        if total < 1e-8:
            return self._sample_from_logits(logits, temperature)

        return int(np.random.choice(ACTION_SPACE, p=blended))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        from blood.utils import softmax
        return softmax(x)

    @staticmethod
    def _sample_from_logits(logits: np.ndarray, temperature: float = 1.0) -> int:
        from blood.utils import softmax
        probs = softmax(logits / max(temperature, 1e-8))
        return int(np.random.choice(len(probs), p=probs))
