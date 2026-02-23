"""Gymnasium wrapper for Bloody Battle Mahjong."""

import logging
import os
import threading
import gymnasium as gym
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from gymnasium import spaces

from .augment import SUIT_PERMUTATIONS, augment_obs, augment_action

log = logging.getLogger(__name__)

# Per-process flag: set to False if Rust engine hangs/fails in this worker
_rust_engine_ok = True
_RUST_TIMEOUT_SEC = 15

# One thread per process for running Rust engine calls with timeout.
# Using a single-thread executor avoids spawning a new thread per call
# while still allowing future.result(timeout=) from any calling thread.
_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rust_engine")
    return _executor


def _run_with_timeout(fn, timeout=_RUST_TIMEOUT_SEC):
    """Run fn() in a dedicated thread; raise TimeoutError if it takes too long.

    Works from any thread (including SF2 event-loop threads) unlike signal.alarm
    which requires the main thread.
    """
    future = _get_executor().submit(fn)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        raise TimeoutError(f"Rust engine call timed out after {timeout}s (pid={os.getpid()})")


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

    def _dummy_obs(self):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        oracle = np.zeros(ORACLE_OBS_SIZE, dtype=np.float32)
        mask = np.zeros(ACTION_SPACE, dtype=np.float32)
        shanten = np.zeros(15, dtype=np.float32)
        ow = np.zeros(81, dtype=np.float32)
        return obs, oracle, mask, shanten, ow

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        game_seed = int(self._rng.integers(0, 2**32))
        self._maybe_pick_augmentation()

        global _rust_engine_ok
        obs_dict = None
        if self._engine_cls is not None and _rust_engine_ok:
            engine_cls = self._engine_cls
            opp_mode = self._opponent_mode
            try:
                self._env, obs_dict = _run_with_timeout(
                    lambda: _reset_engine(engine_cls, game_seed, opp_mode),
                    timeout=_RUST_TIMEOUT_SEC,
                )
            except TimeoutError as e:
                log.error("FATAL: %s — Rust engine is hanging. Disabling for this worker.", e)
                _rust_engine_ok = False
                self._env = None
                obs_dict = None
            except Exception as e:
                log.error("RustMahjongEnv error pid=%d: %s", os.getpid(), e, exc_info=True)
                self._env = None
                obs_dict = None

        if obs_dict is not None:
            obs = np.array(obs_dict["obs"], dtype=np.float32)
            oracle = np.array(obs_dict["oracle_obs"], dtype=np.float32)
            mask = np.array(obs_dict["action_mask"], dtype=np.float32)
            # "shanten_labels" is the current key; "dq_labels" was used in older builds
            shanten_raw = obs_dict.get("shanten_labels", obs_dict.get("dq_labels"))
            shanten = np.array(shanten_raw, dtype=np.float32) if shanten_raw is not None else np.zeros(15, dtype=np.float32)
            ow = np.array(obs_dict["ow_labels"], dtype=np.float32)
        else:
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            mask[31:34] = 1.0

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
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}

        global _rust_engine_ok
        engine_action = self._inverse_action(int(action))
        env = self._env
        try:
            result = _run_with_timeout(
                lambda: env.step(engine_action),
                timeout=_RUST_TIMEOUT_SEC,
            )
            obs_dict, reward, terminated, truncated, info = result
        except TimeoutError as e:
            log.error("FATAL: %s — Rust engine hung in step(). Disabling for this worker.", e)
            _rust_engine_ok = False
            self._env = None
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}
        except Exception as e:
            log.error("step() error pid=%d: %s", os.getpid(), e, exc_info=True)
            self._env = None
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}

        obs = np.array(obs_dict["obs"], dtype=np.float32)
        oracle = np.array(obs_dict["oracle_obs"], dtype=np.float32)
        mask = np.array(obs_dict["action_mask"], dtype=np.float32)
        shanten_raw = obs_dict.get("shanten_labels", obs_dict.get("dq_labels"))
        shanten = np.array(shanten_raw, dtype=np.float32) if shanten_raw is not None else np.zeros(15, dtype=np.float32)
        ow = np.array(obs_dict["ow_labels"], dtype=np.float32)

        return {
            "obs": self._apply_augment_obs(obs),
            "oracle_obs": self._apply_augment_oracle_obs(oracle),
            "action_mask": self._apply_augment_mask(mask),
            "shanten_labels": self._apply_augment_shanten(shanten),
            "ow_labels": self._apply_augment_ow(ow),
        }, float(reward), terminated, truncated, info


def _reset_engine(engine_cls, game_seed, opp_mode):
    """Create and reset a RustMahjongEnv; returns (env, obs_dict) tuple."""
    env = engine_cls(game_seed, opp_mode)
    obs_dict = env.reset(game_seed)
    return env, obs_dict
