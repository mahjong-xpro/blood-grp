"""Python wrapper for ISMCE inference-time search.

Combines the Rust ISMCE evaluation with the policy network
to produce refined action selections at test time.
"""

import logging
import time
from typing import Optional

import numpy as np

from blood.utils import softmax as _softmax  # Issue #53: moved from hot loop to module level

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
        self._ismce_full_available = False
        try:
            from blood._engine import ismce_evaluate, ismce_danger
            self._ismce_evaluate = ismce_evaluate
            self._ismce_danger = ismce_danger
            self._ismce_available = True
            try:
                from blood._engine import ismce_evaluate_full
                self._ismce_evaluate_full = ismce_evaluate_full
                self._ismce_full_available = True
            except ImportError:
                log.debug("ismce_evaluate_full not available; defense path disabled")
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
        opponent_ding_que: Optional[list] = None,
        opponent_meld_counts: Optional[list] = None,
        opponent_discard_counts: Optional[list] = None,
        opponent_discards: Optional[list] = None,
    ) -> int:
        """Select action by blending policy with ISMCE scores.

        When opponent state is provided and the full evaluator is available,
        uses constrained sampling + defense-aware rollouts for stronger play.
        """
        logits = policy_logits.copy()
        logits[action_mask < 0.5] = -1e9

        if not self._ismce_available or hand is None or tiles_seen is None:
            return self._sample_from_logits(logits, temperature)

        discard_candidates = [i for i in range(NUM_TILE_TYPES) if action_mask[i] > 0.5]
        if not discard_candidates:
            return self._sample_from_logits(logits, temperature)

        try:
            base_seed = int(time.monotonic_ns()) & 0xFFFFFFFFFFFFFFFF

            # Use full evaluation if opponent info available and full evaluator loaded
            if (self._ismce_full_available
                    and opponent_ding_que is not None
                    and opponent_meld_counts is not None
                    and opponent_discard_counts is not None
                    and opponent_discards is not None):
                results = self._ismce_evaluate_full(
                    hand.astype(np.uint8),
                    melds_count,
                    ding_que,
                    tiles_seen.astype(np.uint8),
                    discard_candidates,
                    wall_remaining,
                    opponent_ding_que,
                    opponent_meld_counts,
                    opponent_discard_counts,
                    opponent_discards,
                    self.num_worlds,
                    self.rollout_depth,
                    base_seed,
                )
            else:
                # Fallback to basic evaluation
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

        # --- log 空间尺度归一化 ---
        # 对 policy logits 和 ISMCE 分数都在候选牌上做标准化（零均值 + 单位方差），
        # 使得混合权重能准确反映两个信号的相对重要性，不受原始尺度影响。
        policy_logits_t = logits / max(temperature, 1e-8)

        # 提取候选牌上的 policy logits，做标准化
        cand_idx = np.array(discard_candidates)
        policy_cand = policy_logits_t[cand_idx]
        p_mean = policy_cand.mean()
        p_std = policy_cand.std()
        if p_std < 1e-8:
            # 所有候选牌 logit 几乎相同，标准化后全为 0
            policy_cand_norm = np.zeros_like(policy_cand)
        else:
            policy_cand_norm = (policy_cand - p_mean) / p_std

        # 提取候选牌上的 ISMCE 分数，做标准化
        ismce_cand = ismce_scores[cand_idx]
        i_mean = ismce_cand.mean()
        i_std = ismce_cand.std()
        if i_std < 1e-8:
            # 所有候选牌 ISMCE 分数几乎相同，标准化后全为 0
            ismce_cand_norm = np.zeros_like(ismce_cand)
        else:
            ismce_cand_norm = (ismce_cand - i_mean) / i_std

        # 按权重混合标准化后的信号
        blended_logits = np.full(ACTION_SPACE, -1e9)
        for j, i in enumerate(discard_candidates):
            blended_logits[i] = (self.policy_weight * policy_cand_norm[j]
                                 + self.ismce_weight * ismce_cand_norm[j])

        blended = _softmax(blended_logits)
        total = blended.sum()
        if total < 1e-8:
            return self._sample_from_logits(logits, temperature)

        return int(np.random.choice(ACTION_SPACE, p=blended))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        return _softmax(x)

    @staticmethod
    def _sample_from_logits(logits: np.ndarray, temperature: float = 1.0) -> int:
        probs = _softmax(logits / max(temperature, 1e-8))
        return int(np.random.choice(len(probs), p=probs))
