"""Self-play Gymnasium environment for Bloody Battle Mahjong.

Extends BloodMahjongEnv by replacing the Rust-side RuleBot with
a neural network opponent loaded from league checkpoints.

The Rust engine runs in "external" mode — opponent decisions are
handled entirely on the Python side using cached PolicyModel inference.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .blood_env import (
    BloodMahjongEnv, NUM_TILE_TYPES, ACTION_SPACE,
    OBS_SIZE, NUM_STUDENT_CHANNELS,
)
from blood.consts import REWARD_NORM
from blood.model.inference import OpponentModelPool
from blood.training.league import LeagueManager

log = logging.getLogger(__name__)

NUM_PLAYERS = 4
MAX_LOOP_GUARD = 500


def _score_delta_to_fan(delta: float) -> int:
    """Infer fan count from a score delta using the inverse of calc_score.

    calc_score(fan) = 1000 * 2^(fan-1), capped at 6 fan = 32000.
    Tsumo: delta = score_per_player * N_payers (3, 2, or 1 depending on win_count).
    Ron:   delta = score_per_player * 1 (1 payer).
    Returns 0 if unclear.
    """
    if delta <= 0:
        return 0
    # Try all possible payer counts: 3 (normal tsumo), 2 (one player already won),
    # 1 (ron or two players already won).
    for divisor in (3, 2, 1):
        per_payer = delta / divisor
        for fan in range(1, 7):
            expected = 1000 * (1 << (fan - 1))
            if abs(per_payer - expected) < 50:
                return fan
    return 0


class SelfPlayEnv(BloodMahjongEnv):
    """Env where opponents are controlled by a neural-network policy.

    On each game step the flow is:
        1. Apply the agent's action (seat 0)
        2. Loop: advance through all opponent decisions using the
           cached PolicyModel until it is the agent's turn again
           or the game ends.
        3. Return the observation for seat 0.
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)

        self._opponent_mode = "external"

        self._opp_pool = OpponentModelPool(device="cpu")
        self._league: Optional[LeagueManager] = None
        self._refresh_every = 20
        self._episodes_since_refresh = 0

        if cfg is not None:
            pool_dir = getattr(cfg, "league_pool_dir", "checkpoints/league")
            newest_w = getattr(cfg, "league_newest_weight", 2.0)
            uniform_floor = getattr(cfg, "league_uniform_floor", 0.1)
            self_play_prob = getattr(cfg, "league_self_play_prob", 0.2)
            # Fix R10-H3: pass frozen_window and Elo params (were silently ignored)
            frozen_window = getattr(cfg, "league_frozen_window", 0)
            self._league = LeagueManager(
                pool_dir,
                newest_weight=newest_w,
                max_pool_size=getattr(cfg, "league_max_pool_size", 200),  # Fix R12-M5
                uniform_floor=uniform_floor,
                self_play_prob=self_play_prob,
                frozen_window=frozen_window,
            )
            self._refresh_every = getattr(cfg, "opponent_refresh_every", 20)

        self._warmup_reward_shaping = False
        self._warmup_dq_bonus = 0.0
        self._warmup_win_bonus = 0.0
        self._warmup_deal_in_penalty = 0.0
        self._warmup_dangerous_discard_penalty = 0.0
        if cfg is not None:
            self._warmup_reward_shaping = getattr(cfg, "warmup_reward_shaping", False)
            self._warmup_dq_bonus = getattr(cfg, "warmup_dq_bonus", 0.0)
            self._warmup_win_bonus = getattr(cfg, "warmup_win_bonus", 0.0)
            self._warmup_deal_in_penalty = getattr(cfg, "warmup_deal_in_penalty", 0.0)
            self._warmup_dangerous_discard_penalty = getattr(cfg, "warmup_dangerous_discard_penalty", 0.03)

        # Structured reward shaping (all phases)
        self._reward_tsumo_bonus = 0.0
        self._reward_deal_in_penalty = 0.0
        self._reward_shanten_progress = 0.0
        self._reward_shanten_regress = 0.0
        self._reward_safe_discard = 0.0
        self._reward_rank_bonus = 0.0
        if cfg is not None:
            self._reward_tsumo_bonus = getattr(cfg, "reward_tsumo_bonus", 0.1)
            self._reward_deal_in_penalty = getattr(cfg, "reward_deal_in_penalty", 0.05)
            self._reward_shanten_progress = getattr(cfg, "reward_shanten_progress", 0.003)
            self._reward_shanten_regress = getattr(cfg, "reward_shanten_regress", 0.001)
            self._reward_safe_discard = getattr(cfg, "reward_safe_discard", 0.0)
            self._reward_rank_bonus = getattr(cfg, "reward_rank_bonus", 0.0)

        # 向听奖励衰减调度：随训练进度线性衰减向听奖励，避免贪心追求听牌而忽略番数
        self._shanten_decay_steps = 0
        self._shanten_min_ratio = 0.3
        if cfg is not None:
            raw_decay_steps = getattr(cfg, "shanten_reward_decay_steps", 0)
            self._shanten_min_ratio = getattr(cfg, "shanten_reward_min_ratio", 0.3)
            # Adjust decay steps for parallel envs so per-env counter reaches
            # the target at the correct wall-clock point (Issue #40).
            num_envs = getattr(cfg, 'num_workers', 1) * getattr(cfg, 'num_envs_per_worker', 1)
            self._shanten_decay_steps = max(1, raw_decay_steps // max(num_envs, 1)) if raw_decay_steps > 0 else 0
        # 全局环境步数计数器，用于衰减调度
        self._global_env_steps = 0

        # 向听奖励番数加权：向听改善时乘以 (1 + scale * estimated_fan / max_fan)
        # 引导模型在向听数相同时倾向于选择番数更高的手牌方向
        self._shanten_fan_bonus_scale = 0.0
        self._shanten_fan_max = 8.0
        if cfg is not None:
            self._shanten_fan_bonus_scale = getattr(cfg, "shanten_fan_bonus_scale", 0.0)
            self._shanten_fan_max = getattr(cfg, "shanten_fan_max", 8.0)

        self._max_steps = 500
        self._step_count = 0
        self._prev_scores = np.zeros(4, dtype=np.float32)
        self._prev_agent_shanten: Optional[int] = None

    def _get_shanten_decay_factor(self) -> float:
        """计算当前向听奖励的衰减系数。

        衰减公式: factor = max(min_ratio, 1.0 - progress)
        其中 progress = global_env_steps / decay_steps

        返回值范围 [min_ratio, 1.0]，乘以基础向听奖励得到有效奖励。
        当 decay_steps=0 时不衰减，始终返回 1.0。
        """
        if self._shanten_decay_steps <= 0:
            return 1.0
        progress = self._global_env_steps / self._shanten_decay_steps
        return max(self._shanten_min_ratio, 1.0 - progress)

    def _get_fan_bonus_factor(self) -> float:
        """计算基于预估番数的向听奖励加权系数。

        加权公式: factor = 1.0 + fan_bonus_scale * estimated_fan / max_fan
        当 fan_bonus_scale=0 时返回 1.0（无加权）。

        优先使用 Rust 引擎的 get_agent_estimated_fan()（精确计算听牌番数 /
        启发式估计非听牌番数潜力）。若引擎不支持则返回 1.0。
        """
        if self._shanten_fan_bonus_scale <= 0 or self._env is None:
            return 1.0
        try:
            if hasattr(self._env, "get_agent_estimated_fan"):
                estimated_fan = float(self._env.get_agent_estimated_fan())
            else:
                return 1.0
        except Exception:
            return 1.0
        # 归一化并计算加权系数
        normalized = min(estimated_fan / self._shanten_fan_max, 1.0)
        return 1.0 + self._shanten_fan_bonus_scale * normalized

    def _try_refresh_opponent(self):
        self._episodes_since_refresh += 1
        if self._episodes_since_refresh < self._refresh_every:
            return

        self._episodes_since_refresh = 0
        if self._league is None:
            return

        path = self._league.sample_opponent(
            current_elo=getattr(self, "_current_elo", None),
        )
        if path is not None:
            try:
                ok = self._opp_pool.load(str(path))
                if not ok:
                    self._league.remove_checkpoint(path)
            except Exception as e:
                log.warning("Failed to load opponent checkpoint from %s: %s", path, e)

    def _opp_action(self, player_id: int) -> int:
        """Get an opponent's action using the neural model or fallback."""
        obs_dict = self._env.get_player_obs(player_id)
        obs = torch.as_tensor(np.array(obs_dict["obs"], dtype=np.float32))
        mask = torch.as_tensor(np.array(obs_dict["action_mask"], dtype=np.float32))
        return self._opp_pool.get_action(obs, mask, opponent_id=player_id)

    def _advance_external_opponents(self):
        """Drive the game forward through all opponent decision points."""
        agent = 0
        prev_state = None
        for _ in range(MAX_LOOP_GUARD):
            if self._env.is_done():
                break

            phase = self._env.get_phase()

            if phase in ("scoring", "done"):
                break

            # Stall detection: if game state hasn't changed since last iteration,
            # an action was a NO-OP and we'd loop forever. Break to avoid that.
            cp = self._env.get_current_player()
            pending = tuple(self._env.get_reaction_pending())
            cur_state = (phase, cp, pending)
            if cur_state == prev_state:
                log.warning(
                    "_advance_external_opponents: stall at phase=%s cp=%d; forcing scoring",
                    phase, cp,
                )
                self._env.finalize_scoring()
                break
            prev_state = cur_state

            if phase == "ding_que":
                dq_done = self._env.get_ding_que_done()
                if not dq_done[agent]:
                    break
                any_applied = False
                for pid in range(NUM_PLAYERS):
                    if pid == agent or dq_done[pid]:
                        continue
                    action = self._opp_action(pid)
                    self._env.apply_ext_action(pid, action)
                    any_applied = True
                if any_applied:
                    continue
                else:
                    break

            elif phase == "self_check":
                cp = self._env.get_current_player()
                if cp == agent:
                    if self._env.has_decision(agent):
                        break
                    self._env.apply_ext_action(agent, 30)  # Pass
                    continue
                # If opponent has no decision (mask all-zeros), apply Pass directly.
                # Calling the model with an all-zeros mask causes uniform sampling,
                # which can output Agari on a non-complete hand, triggering a
                # process_win early-return that leaves phase=SelfCheck and deadlocks.
                if not self._env.has_decision(cp):
                    self._env.apply_ext_action(cp, 30)  # Pass → Phase::Discard
                    continue
                action = self._opp_action(cp)
                self._env.apply_ext_action(cp, action)
                # After a Kan the engine stays in SelfCheck with the same cp
                # (rinshan draw). Reset prev_state so the stall detector doesn't
                # misfire on the next iteration when (phase, cp, pending) repeats.
                prev_state = None

            elif phase == "kan_select":
                cp = self._env.get_current_player()
                if cp == agent:
                    break
                action = self._opp_action(cp)
                self._env.apply_ext_action(cp, action)

            elif phase == "discard":
                cp = self._env.get_current_player()
                if cp == agent:
                    break
                action = self._opp_action(cp)
                self._env.apply_ext_action(cp, action)

            elif phase == "reaction":
                pending = self._env.get_reaction_pending()
                if pending[agent]:
                    break
                for pid in range(NUM_PLAYERS):
                    if pid == agent or not pending[pid]:
                        continue
                    # Re-fetch pending before each action: resolve_reactions() may
                    # have been triggered by a previous apply_ext_action, advancing
                    # the phase. Applying more reactions after that would call
                    # resolve_reactions() again with stale data (double-advance bug).
                    live_pending = self._env.get_reaction_pending()
                    if not live_pending[pid]:
                        continue
                    if self._env.get_phase() != "reaction":
                        break
                    action = self._opp_action(pid)
                    self._env.apply_ext_action(pid, action)
                pending = self._env.get_reaction_pending()
                if pending[agent]:
                    break

            else:
                break

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        game_seed = int(self._rng.integers(0, 2**32))
        self._maybe_pick_augmentation()
        self._try_refresh_opponent()
        self._opp_pool.reset_hidden_states()  # reset LSTM state for new episode

        self._step_count = 0
        if self._engine_cls is not None:
            self._env = self._engine_cls(game_seed, "external", self._initial_score)
            self._env.reset(game_seed)
            self._advance_external_opponents()

            obs_dict = self._env.get_player_obs(0)
            obs = np.array(obs_dict["obs"], dtype=np.float32)
            mask = np.array(obs_dict["action_mask"], dtype=np.float32)
            oracle_obs = np.array(self._env.get_oracle_obs(), dtype=np.float32)
            shanten, ow = self._compute_labels()
            self._prev_scores = np.array(self._env.get_scores(), dtype=np.float32)
            if hasattr(self._env, "get_agent_shanten"):
                self._prev_agent_shanten = self._env.get_agent_shanten()
            else:
                self._prev_agent_shanten = None
        else:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            mask = np.zeros(ACTION_SPACE, dtype=np.float32)
            mask[31:34] = 1.0
            oracle_obs = np.zeros(self.observation_space["oracle_obs"].shape[0], dtype=np.float32)
            shanten = np.zeros(15, dtype=np.float32)
            ow = np.zeros(81, dtype=np.float32)
            self._prev_scores = np.full(4, float(self._initial_score), dtype=np.float32)
            self._prev_agent_shanten = None

        self._episode_count += 1
        return {
            "obs": self._apply_augment_obs(obs),
            "oracle_obs": self._apply_augment_oracle_obs(oracle_obs),
            "action_mask": self._apply_augment_mask(mask),
            "shanten_labels": self._apply_augment_shanten(shanten),
            "ow_labels": self._apply_augment_ow(ow),
        }, {}

    def step(self, action):
        if self._env is None:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            oracle = np.zeros(self.observation_space["oracle_obs"].shape[0], dtype=np.float32)
            mask = np.zeros(ACTION_SPACE, dtype=np.float32)
            shanten = np.zeros(15, dtype=np.float32)
            ow = np.zeros(81, dtype=np.float32)
            return {
                "obs": obs, "oracle_obs": oracle, "action_mask": mask,
                "shanten_labels": shanten, "ow_labels": ow,
            }, 0.0, True, False, {}

        engine_action = self._inverse_action(int(action))

        # Capture oracle ow_labels BEFORE applying action (for defense rewards)
        ow_before = None
        _need_ow = (
            (self._warmup_reward_shaping and self._warmup_dangerous_discard_penalty > 0)
            or self._reward_safe_discard > 0
        )
        if _need_ow:
            _, ow_before = self._compute_labels()

        self._env.apply_ext_action(0, engine_action)

        self._advance_external_opponents()

        if self._env.get_phase() in ("scoring", "done"):
            self._env.finalize_scoring()

        scores = np.array(self._env.get_scores(), dtype=np.float32)
        agent_delta = scores[0] - self._prev_scores[0]
        opp_deltas = scores[1:] - self._prev_scores[1:]

        # 在 finalize_scoring() 之后计算 terminated，确保排名奖励等逻辑
        # 能正确感知游戏结束状态。之前此处在 finalize_scoring() 之前计算，
        # 导致 finalize_scoring() 将游戏从未结束变为结束时，terminated 仍为
        # False，排名奖励的 guard 条件不满足，奖励被跳过。
        terminated = self._env.is_done() or self._env.player_has_won(0)

        # Base reward: sqrt-compressed normalized score delta.
        # REWARD_NORM = 32000 = max single-player payment per hand (6-fan ron cap).
        # agent_delta can exceed 32000: 6-fan tsumo = 32000 × 3 payers = 96000.
        # Linear range: 6-fan ron → +1.0, 6-fan tsumo → +3.0.
        # But 1000×2^(fan-1) creates a 32:1 ratio between 1-fan and 6-fan,
        # causing high reward variance. Sqrt compression reduces this to ~5.6:1:
        #   1-fan ron  → 0.177,  6-fan ron  → 1.000
        #   1-fan tsumo→ 0.306,  6-fan tsumo→ 1.732
        _r = float(agent_delta) / float(REWARD_NORM)
        reward = float(np.sign(_r) * np.sqrt(abs(_r)))

        # --- Structured reward shaping (all phases) ---
        # When warmup shaping is active, skip structured tsumo/deal-in bonuses
        # to prevent double-counting with warmup win/deal-in rewards (Issue #47).
        #
        # Score-weighted intensity: shaping bonuses scale with sqrt(|Δ|/32000)
        # so low-fan events get proportionally smaller shaping signals.
        # Same philosophy as rank_bonus: config value is the maximum.
        if not self._warmup_reward_shaping:
            # Tsumo bonus: agent won and multiple opponents paid.
            # Score-weighted: 1-fan tsumo(Δ3000)→0.031, 6-fan tsumo(Δ96000)→0.100
            if self._reward_tsumo_bonus > 0 and self._env.player_has_won(0) and agent_delta > 0:
                if int(np.sum(opp_deltas < -100)) >= 2:
                    t_intensity = min(1.0, float(np.sqrt(agent_delta / float(REWARD_NORM))))
                    t_intensity = max(0.25, t_intensity)
                    reward += self._reward_tsumo_bonus * t_intensity

            # Deal-in penalty: agent score down, at least one opponent up.
            # Score-weighted: 1-fan deal-in(Δ1000)→0.014, 6-fan deal-in(Δ32000)→0.050
            elif self._reward_deal_in_penalty > 0 and agent_delta < -100:
                if int(np.sum(opp_deltas > 100)) >= 1 and int(np.sum(opp_deltas < -100)) == 0:
                    d_intensity = min(1.0, float(np.sqrt(abs(agent_delta) / float(REWARD_NORM))))
                    d_intensity = max(0.25, d_intensity)
                    reward -= self._reward_deal_in_penalty * d_intensity

        # 向听进退奖励（带衰减调度 + 番数加权）
        # Guard against game-end: when terminated, shanten may be -1 (complete hand)
        # which would produce a spurious large progress reward on top of the win reward.
        if (not terminated
                and self._prev_agent_shanten is not None
                and hasattr(self._env, "get_agent_shanten")):
            current_shanten = self._env.get_agent_shanten()
            shanten_delta = current_shanten - self._prev_agent_shanten
            # 计算衰减系数：随 global_env_steps 线性衰减到 min_ratio
            decay_factor = self._get_shanten_decay_factor()
            # 番数加权：向听改善时乘以 (1 + scale * estimated_fan / max_fan)
            # 引导模型在向听数相同时倾向于选择番数更高的手牌方向
            fan_bonus = self._get_fan_bonus_factor()
            if shanten_delta < 0 and self._reward_shanten_progress > 0:
                reward += self._reward_shanten_progress * (-shanten_delta) * decay_factor * fan_bonus
            elif shanten_delta > 0 and self._reward_shanten_regress > 0:
                reward -= self._reward_shanten_regress * shanten_delta * decay_factor
            self._prev_agent_shanten = current_shanten

        # Safe discard reward: gradient-based danger penalty.
        # Instead of binary safe/dangerous, reward scales with (1 - max_danger):
        #   safe tile (danger=0) → full reward, risky tile (danger=0.8) → 0.2× reward.
        # Only applies to discard actions (0-26) and when ow_before is available.
        if (self._reward_safe_discard > 0
                and ow_before is not None
                and 0 <= engine_action < 27):
            ow_3d = ow_before.reshape(3, 27)
            any_tenpai = bool(np.any(ow_3d.sum(axis=1) > 0.01))
            if any_tenpai:
                max_danger = float(np.max(ow_3d[:, engine_action]))
                reward += self._reward_safe_discard * (1.0 - max_danger)

        # Rank bonus at game end: encourages maximizing relative ranking, not just
        # absolute score. Applied when the game fully terminates OR when the agent
        # wins (guaranteed rank 1). Without the early-win case, agents that win first
        # never receive a positive rank bonus, creating an asymmetry where only
        # losing agents get rank penalties.
        #
        # Score-weighted intensity: rank bonus scales with sqrt(|score_delta|/REWARD_NORM)
        # so that rank matters more in high-stakes games and less in quiet ones.
        if self._reward_rank_bonus > 0 and terminated:
            final_scores = self._env.get_scores()
            player_score = final_scores[0]

            if self._env.is_done():
                # Game fully ended: compute actual rank
                above = sum(1 for s in final_scores if s > player_score)
                equal = sum(1 for s in final_scores if s == player_score)
                avg_rank = above + (equal + 1) / 2.0
            elif self._env.player_has_won(0):
                # Agent won but game not done: guaranteed rank 1
                avg_rank = 1.0
            else:
                avg_rank = 2.5  # fallback: neutral

            rank_idx = min(int(avg_rank - 0.5), 3)
            rank_multipliers = [1.0, 0.3, -0.3, -1.0]
            score_delta = abs(float(player_score - self._initial_score))
            intensity = min(1.0, float(np.sqrt(score_delta / float(REWARD_NORM))))
            intensity = max(0.25, intensity)  # floor: rank always has some weight
            reward += self._reward_rank_bonus * rank_multipliers[rank_idx] * intensity

        # Warmup shaping (phase 1 only)
        if self._warmup_reward_shaping:
            # Pass engine_action (original tile space) not action (augmented space).
            # ow_before is from the Rust engine (unaugmented), so the tile index must
            # match the original space. Using the augmented action would check the wrong
            # tile when suit permutation is active (50% of warmup episodes).
            reward += self._compute_shaping_reward(float(self._prev_scores[0]), engine_action, ow_before)

        self._prev_scores = scores

        self._step_count += 1
        # 累加全局环境步数，用于向听奖励衰减调度
        self._global_env_steps += 1

        # terminated 已在 finalize_scoring() 之后统一计算，此处无需重复赋值
        truncated = self._step_count >= self._max_steps and not terminated

        obs_dict = self._env.get_player_obs(0)
        obs = np.array(obs_dict["obs"], dtype=np.float32)
        mask = np.array(obs_dict["action_mask"], dtype=np.float32)
        oracle_obs = np.array(self._env.get_oracle_obs(), dtype=np.float32)
        shanten, ow = self._compute_labels()

        info = {
            "win_count": self._env.get_win_count(),
            "player_won": self._env.player_has_won(0),
            "fan_count": _score_delta_to_fan(agent_delta) if self._env.player_has_won(0) and agent_delta > 0 else 0,
        }

        return {
            "obs": self._apply_augment_obs(obs),
            "oracle_obs": self._apply_augment_oracle_obs(oracle_obs),
            "action_mask": self._apply_augment_mask(mask),
            "shanten_labels": self._apply_augment_shanten(shanten),
            "ow_labels": self._apply_augment_ow(ow),
        }, float(reward), terminated, truncated, info

    def _compute_labels(self):
        """Compute auxiliary labels from the environment's Rust API.

        Returns shanten_labels (3x5 one-hot, shape [15]) and ow_labels ([81]).
        Falls back to zero labels if the API is unavailable.
        """
        try:
            if self._env is not None and hasattr(self._env, "get_aux_labels"):
                labels = self._env.get_aux_labels(0)
                shanten = np.array(labels["shanten_labels"], dtype=np.float32)
                ow = np.array(labels["ow_labels"], dtype=np.float32)
                return shanten, ow
        except Exception:
            pass
        shanten = np.zeros(15, dtype=np.float32)
        ow = np.zeros(81, dtype=np.float32)
        return shanten, ow

    def _compute_shaping_reward(self, prev_score: float, action: int, ow_before) -> float:
        """Extra reward signals during warmup phase."""
        bonus = 0.0

        # DingQue bonus: reward choosing the suit with fewest tiles in hand.
        # get_agent_suit_counts() reads hand *after* apply_ding_que(), but that
        # method only sets the ding_que flag — it never modifies the hand array,
        # so the counts still reflect the pre-dingque hand.
        if self._warmup_dq_bonus > 0 and 31 <= action <= 33:
            chosen_suit = action - 31  # 0=Man, 1=Pin, 2=Sou
            counts = self._env.get_agent_suit_counts()  # [man, pin, sou]
            if counts[chosen_suit] == min(counts):
                bonus += self._warmup_dq_bonus

        if self._env.player_has_won(0) and self._warmup_win_bonus > 0:
            bonus += self._warmup_win_bonus
        # Deal-in penalty: only apply when agent discards (action 0-26) and score drops.
        # Kan payment (action 28) also reduces score but is NOT a deal-in — guard with
        # action < 27 to avoid penalizing legitimate kan operations.
        current_score = float(self._env.get_scores()[0])
        if (current_score < prev_score
                and self._warmup_deal_in_penalty > 0
                and action < 27):
            bonus -= self._warmup_deal_in_penalty
        # Oracle-guided dangerous discard penalty:
        # penalize discarding a tile that any opponent is waiting for.
        if (ow_before is not None
                and self._warmup_dangerous_discard_penalty > 0
                and 0 <= action < 27):
            for opp in range(3):
                if ow_before[opp * 27 + action] > 0.5:
                    bonus -= self._warmup_dangerous_discard_penalty
                    break  # penalize once even if multiple opponents wait on it
        return bonus
