"""Gymnasium wrapper for Bloody Battle Mahjong."""

import logging
import os
import signal
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .augment import SUIT_PERMUTATIONS, augment_obs, augment_action

log = logging.getLogger(__name__)

# Per-process flag: set to False if Rust engine hangs/fails in this worker
_rust_engine_ok = True
_RUST_TIMEOUT_SEC = 15


def _rust_alarm_handler(signum, frame):
    raise TimeoutError(f"RustMahjongEnv timed out after {_RUST_TIMEOUT_SEC}s (pid={os.getpid()})")

NUM_TILE_TYPES = 27
ACTION_SPACE = 34
NUM_STUDENT_CHANNELS = 464
NUM_ORACLE_CHANNELS = 516
OBS_SIZE = NUM_STUDENT_CHANNELS * NUM_TILE_TYPES
ORACLE_OBS_SIZE = NUM_ORACLE_CHANNELS * NUM_TILE_TYPES


class BloodMahjongEnv(gym.Env):
    """Single-agent Bloody Battle Mahjong environment.

    The agent controls seat 0. Opponents use rule-based or neural policies
    managed by the Rust engine.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg=None, **kwargs):
        super().__init__()

        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(ACTION_SPACE,), dtype=np.float32),
            "shanten_labels": spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32),
            "ow_labels": spaces.Box(low=0.0, high=1.0, shape=(81,), dtype=np.float32),
            "obs": spaces.Box(low=0.0, high=1.0, shape=(464 * 27,), dtype=np.float32),
            "oracle_obs": spaces.Box(low=0.0, high=1.0, shape=((464 + 52) * 27,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(ACTION_SPACE)

        self._opponent_mode = "rulebot"
        self._augment_prob = 0.5
        self._episode_count = 0
        self._current_perm = None

        if cfg is not None:
            self._opponent_mode = getattr(cfg, "opponent_mode", "rulebot")
            self._augment_prob = getattr(cfg, "suit_augment_prob", 0.5)

        log.debug("BloodMahjongEnv.__init__ pid=%d", os.getpid())
        try:
            from blood._engine import RustMahjongEnv
            self._engine_cls = RustMahjongEnv
            log.debug("blood._engine imported OK pid=%d", os.getpid())
        except ImportError as e:
            log.warning("blood._engine not available: %s", e)
            self._engine_cls = None

        self._env = None
        self._rng = np.random.default_rng()

    def _maybe_pick_augmentation(self):
        """Choose a suit permutation for this episode."""
        if self._rng.random() < self._augment_prob:
            idx = int(self._rng.integers(1, 6))
            self._current_perm = SUIT_PERMUTATIONS[idx]
        else:
            self._current_perm = None

    def _apply_augment_obs(self, obs_flat):
        if self._current_perm is None:
            return obs_flat
        obs_2d = obs_flat.reshape(NUM_STUDENT_CHANNELS, NUM_TILE_TYPES)
        aug = augment_obs(obs_2d, self._current_perm)
        return aug.reshape(-1)

    def _apply_augment_oracle_obs(self, obs_flat):
        if self._current_perm is None:
            return obs_flat
        obs_2d = obs_flat.reshape(NUM_ORACLE_CHANNELS, NUM_TILE_TYPES)
        aug = augment_obs(obs_2d, self._current_perm)
        return aug.reshape(-1)

    def _apply_augment_mask(self, mask):
        if self._current_perm is None:
            return mask
        if not hasattr(self, '_augment_action_map') or self._augment_action_perm != self._current_perm:
            self._augment_action_perm = self._current_perm
            self._augment_action_map = np.array(
                [augment_action(i, self._current_perm) for i in range(ACTION_SPACE)],
                dtype=np.intp,
            )
        new_mask = np.zeros_like(mask)
        new_mask[self._augment_action_map] = mask
        return new_mask

    def _apply_augment_shanten(self, shanten):
        # Shanten labels are suit-invariant; no permutation needed.
        return shanten

    def _apply_augment_ow(self, ow):
        if self._current_perm is None:
            return ow
        perm = self._current_perm
        ow_2d = ow.reshape(3, NUM_TILE_TYPES)
        new_ow = np.zeros_like(ow_2d)
        for new_suit, old_suit in enumerate(perm):
            new_ow[:, new_suit * 9:(new_suit + 1) * 9] = ow_2d[:, old_suit * 9:(old_suit + 1) * 9]
        return new_ow.reshape(-1)

    def _inverse_action(self, action):
        """Convert agent's augmented action back to engine's original space."""
        if self._current_perm is None:
            return action
        inv_perm = tuple(self._current_perm.index(i) for i in range(3))
        return augment_action(action, inv_perm)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        game_seed = int(self._rng.integers(0, 2**32))
        self._maybe_pick_augmentation()

        global _rust_engine_ok
        if self._engine_cls is not None and _rust_engine_ok:
            old_handler = signal.signal(signal.SIGALRM, _rust_alarm_handler)
            signal.alarm(_RUST_TIMEOUT_SEC)
            try:
                log.debug("RustMahjongEnv(seed=%d, mode=%s) pid=%d", game_seed, self._opponent_mode, os.getpid())
                self._env = self._engine_cls(game_seed, self._opponent_mode)
                obs_dict = self._env.reset(game_seed)
                signal.alarm(0)
            except TimeoutError as e:
                log.error("FATAL: %s — Rust engine is hanging. "
                          "Check blood._engine build. Disabling for this worker.", e)
                _rust_engine_ok = False
                self._env = None
                signal.alarm(0)
            except Exception as e:
                log.error("RustMahjongEnv error pid=%d: %s", os.getpid(), e, exc_info=True)
                self._env = None
                signal.alarm(0)
            finally:
                signal.signal(signal.SIGALRM, old_handler)

        if self._env is not None:
            obs = np.array(obs_dict["obs"], dtype=np.float32)
            oracle = np.array(obs_dict["oracle_obs"], dtype=np.float32)
            mask = np.array(obs_dict["action_mask"], dtype=np.float32)
            shanten = np.array(obs_dict["shanten_labels"], dtype=np.float32)
            ow = np.array(obs_dict["ow_labels"], dtype=np.float32)
        else:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            oracle = np.zeros(ORACLE_OBS_SIZE, dtype=np.float32)
            mask = np.zeros(ACTION_SPACE, dtype=np.float32)
            mask[31:34] = 1.0
            shanten = np.zeros(15, dtype=np.float32)
            ow = np.zeros(81, dtype=np.float32)

        self._episode_count += 1
        return {
            "obs": self._apply_augment_obs(obs),
            "oracle_obs": self._apply_augment_oracle_obs(oracle),
            "action_mask": self._apply_augment_mask(mask),
            "shanten_labels": self._apply_augment_shanten(shanten),
            "ow_labels": self._apply_augment_ow(ow),
        }, {}

    def step(self, action):
        if self._env is None:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            oracle = np.zeros(ORACLE_OBS_SIZE, dtype=np.float32)
            mask = np.zeros(ACTION_SPACE, dtype=np.float32)
            shanten = np.zeros(15, dtype=np.float32)
            ow = np.zeros(81, dtype=np.float32)
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}

        engine_action = self._inverse_action(int(action))
        obs_dict, reward, terminated, truncated, info = self._env.step(engine_action)
        obs = np.array(obs_dict["obs"], dtype=np.float32)
        oracle = np.array(obs_dict["oracle_obs"], dtype=np.float32)
        mask = np.array(obs_dict["action_mask"], dtype=np.float32)
        shanten = np.array(obs_dict["shanten_labels"], dtype=np.float32)
        ow = np.array(obs_dict["ow_labels"], dtype=np.float32)

        return {
            "obs": self._apply_augment_obs(obs),
            "oracle_obs": self._apply_augment_oracle_obs(oracle),
            "action_mask": self._apply_augment_mask(mask),
            "shanten_labels": self._apply_augment_shanten(shanten),
            "ow_labels": self._apply_augment_ow(ow),
        }, float(reward), terminated, truncated, info
