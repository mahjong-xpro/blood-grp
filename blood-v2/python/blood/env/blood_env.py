"""Gymnasium wrapper for Bloody Battle Mahjong."""

import logging
import os
import threading
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .augment import SUIT_PERMUTATIONS, augment_obs, augment_action

log = logging.getLogger(__name__)

# --- Rust 引擎冷却恢复机制 ---
# 替代原来的布尔开关，支持超时后冷却恢复：
#   - 每次超时后进入冷却期（_COOLDOWN_STEPS 次 step 调用）
#   - 冷却期结束后重新尝试使用 Rust 引擎
#   - 连续超时达到 _MAX_CONSECUTIVE_TIMEOUTS 次则永久禁用
_rust_engine_ok = True              # 当前是否可用
_rust_cooldown_remaining = 0        # 冷却剩余步数，>0 时暂时禁用
_rust_consecutive_timeouts = 0      # 连续超时计数
_rust_permanently_disabled = False   # 永久禁用标志

_RUST_TIMEOUT_SEC = 15
_COOLDOWN_STEPS = 100               # 单次超时后的冷却步数
_MAX_CONSECUTIVE_TIMEOUTS = 3       # 连续超时多少次后永久禁用


def _rust_engine_available() -> bool:
    """检查 Rust 引擎当前是否可用（考虑冷却和永久禁用状态）。"""
    global _rust_engine_ok, _rust_cooldown_remaining, _rust_permanently_disabled
    # 永久禁用后不再恢复
    if _rust_permanently_disabled:
        return False
    # 冷却期中，不可用
    if _rust_cooldown_remaining > 0:
        return False
    return _rust_engine_ok


def _rust_engine_tick():
    """每次 step 调用时递减冷却计数器，冷却结束后恢复引擎可用状态。"""
    global _rust_cooldown_remaining, _rust_engine_ok
    if _rust_cooldown_remaining > 0:
        _rust_cooldown_remaining -= 1
        if _rust_cooldown_remaining == 0:
            # 冷却期结束，重新尝试使用 Rust 引擎
            _rust_engine_ok = True
            log.info("Rust 引擎冷却期结束，重新启用 (pid=%d)", os.getpid())


def _rust_engine_on_timeout():
    """超时时调用：进入冷却期或永久禁用。"""
    global _rust_engine_ok, _rust_cooldown_remaining
    global _rust_consecutive_timeouts, _rust_permanently_disabled
    _rust_engine_ok = False
    _rust_consecutive_timeouts += 1
    if _rust_consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
        # 连续超时次数达到上限，永久禁用
        _rust_permanently_disabled = True
        log.error(
            "Rust 引擎连续超时 %d 次，永久禁用 (pid=%d)",
            _rust_consecutive_timeouts, os.getpid(),
        )
    else:
        # 进入冷却期
        _rust_cooldown_remaining = _COOLDOWN_STEPS
        log.warning(
            "Rust 引擎超时（第 %d 次），进入 %d 步冷却期 (pid=%d)",
            _rust_consecutive_timeouts, _COOLDOWN_STEPS, os.getpid(),
        )


def _rust_engine_on_success():
    """Rust 引擎调用成功时重置连续超时计数。"""
    global _rust_consecutive_timeouts
    if _rust_consecutive_timeouts > 0:
        _rust_consecutive_timeouts = 0


def _run_with_timeout(fn, timeout=_RUST_TIMEOUT_SEC):
    """Run fn() in a fresh daemon thread; raise TimeoutError if it takes too long.

    Uses a per-call daemon thread instead of a shared ThreadPoolExecutor so that
    a hung Rust call never blocks subsequent calls.  When the timeout fires the
    caller gives up; the daemon thread is abandoned and will be reaped when the
    worker process exits (or when Rust eventually unblocks).

    Works from any thread (including SF2 event-loop threads) unlike signal.alarm
    which requires the main thread.
    """
    result: list = [None]
    exc: list = [None]

    def _target():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True, name=f"rust_engine_{os.getpid()}")
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"Rust engine call timed out after {timeout}s (pid={os.getpid()})")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


from blood.consts import (
    NUM_TILE_TYPES, ACTION_SPACE,
    NUM_STUDENT_CHANNELS, NUM_ORACLE_CHANNELS,
    OBS_SIZE, ORACLE_OBS_SIZE,
    INITIAL_SCORE,
    sqrt_compress_reward,
)


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
            "obs": spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32),
            "oracle_obs": spaces.Box(low=0.0, high=1.0, shape=(ORACLE_OBS_SIZE,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(ACTION_SPACE)

        self._opponent_mode = "rulebot"
        self._augment_prob = 0.5
        self._episode_count = 0
        self._current_perm = None
        self._initial_score = INITIAL_SCORE

        if cfg is not None:
            self._opponent_mode = getattr(cfg, "opponent_mode", "rulebot")
            self._augment_prob = getattr(cfg, "suit_augment_prob", 0.5)
            self._initial_score = getattr(cfg, "initial_score", INITIAL_SCORE)

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
        """Convert agent's augmented action back to engine's original space.

        augment_obs uses pull semantics: perm[new_suit] = old_suit.
        augment_action maps old_suit → new_suit = perm.index(old_suit).

        Inverse: given new_suit, recover old_suit = perm[new_suit].
        This is a direct lookup, not perm.index().
        """
        if self._current_perm is None:
            return action
        perm = self._current_perm
        if action >= 27:
            if 31 <= action <= 33:
                new_suit = action - 31
                old_suit = perm[new_suit]
                return 31 + old_suit
            return action
        new_suit = action // 9
        rank = action % 9
        old_suit = perm[new_suit]
        return old_suit * 9 + rank

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

        obs_dict = None
        if self._engine_cls is not None and _rust_engine_available():
            engine_cls = self._engine_cls
            opp_mode = self._opponent_mode
            init_score = self._initial_score
            try:
                self._env, obs_dict = _run_with_timeout(
                    lambda: _reset_engine(engine_cls, game_seed, opp_mode, init_score),
                    timeout=_RUST_TIMEOUT_SEC,
                )
                # 成功调用，重置连续超时计数
                _rust_engine_on_success()
            except TimeoutError as e:
                log.error("FATAL: %s — Rust engine is hanging. Entering cooldown.", e)
                _rust_engine_on_timeout()
                self._env = None
                obs_dict = None
            except Exception as e:
                log.error("RustMahjongEnv error pid=%d: %s", os.getpid(), e, exc_info=True)
                self._env = None
                obs_dict = None

        if obs_dict is not None:
            obs = np.asarray(obs_dict["obs"], dtype=np.float32)
            oracle = np.asarray(obs_dict["oracle_obs"], dtype=np.float32)
            mask = np.asarray(obs_dict["action_mask"], dtype=np.float32)
            shanten = np.asarray(obs_dict["shanten_labels"], dtype=np.float32)
            ow = np.asarray(obs_dict["ow_labels"], dtype=np.float32)
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
        # 每次 step 递减冷却计数器
        _rust_engine_tick()

        if self._env is None:
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}

        engine_action = self._inverse_action(int(action))
        env = self._env
        try:
            result = _run_with_timeout(
                lambda: env.step(engine_action),
                timeout=_RUST_TIMEOUT_SEC,
            )
            obs_dict, reward, terminated, truncated, info = result
            # 成功调用，重置连续超时计数
            _rust_engine_on_success()

            # Apply sqrt compression to match SelfPlayEnv reward scale (Issue #R9-C3).
            # Rust returns linear: delta / REWARD_NORM. Sqrt compression reduces the
            # 32:1 ratio between 1-fan and 6-fan to ~5.6:1, stabilizing value learning.
            reward = sqrt_compress_reward(float(reward))
        except TimeoutError as e:
            log.error("FATAL: %s — Rust engine hung in step(). Entering cooldown.", e)
            _rust_engine_on_timeout()
            self._env = None
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}
        except Exception as e:
            log.error("step() error pid=%d: %s", os.getpid(), e, exc_info=True)
            self._env = None
            obs, oracle, mask, shanten, ow = self._dummy_obs()
            return {"obs": obs, "oracle_obs": oracle, "action_mask": mask, "shanten_labels": shanten, "ow_labels": ow}, 0.0, True, False, {}

        obs = np.asarray(obs_dict["obs"], dtype=np.float32)
        oracle = np.asarray(obs_dict["oracle_obs"], dtype=np.float32)
        mask = np.asarray(obs_dict["action_mask"], dtype=np.float32)
        shanten = np.asarray(obs_dict["shanten_labels"], dtype=np.float32)
        ow = np.asarray(obs_dict["ow_labels"], dtype=np.float32)

        return {
            "obs": self._apply_augment_obs(obs),
            "oracle_obs": self._apply_augment_oracle_obs(oracle),
            "action_mask": self._apply_augment_mask(mask),
            "shanten_labels": self._apply_augment_shanten(shanten),
            "ow_labels": self._apply_augment_ow(ow),
        }, float(reward), terminated, truncated, info

    def get_scores(self):
        """Public API to retrieve current scores for all 4 players."""
        if self._env is not None:
            return list(self._env.get_scores())
        return [self._initial_score] * 4

    def get_events_jsonl(self) -> str:
        """Public API to retrieve game events as JSONL."""
        if self._env is not None:
            return self._env.get_events_jsonl()
        return ""

    def get_game_header_json(self, names: list) -> str:
        """Public API to retrieve game header JSON."""
        if self._env is not None:
            return self._env.get_game_header_json(names)
        return ""

    def get_final_scores(self):
        """Public API to retrieve final scores (alias for get_scores)."""
        return self.get_scores()

    def get_phase(self) -> str:
        """Public API to retrieve current game phase."""
        if self._env is not None:
            return self._env.get_phase()
        return "done"

    @property
    def has_engine(self) -> bool:
        """Whether the Rust engine is active for this episode."""
        return self._env is not None


def _reset_engine(engine_cls, game_seed, opp_mode, initial_score=100_000):
    """Create and reset a RustMahjongEnv; returns (env, obs_dict) tuple."""
    env = engine_cls(game_seed, opp_mode, initial_score)
    obs_dict = env.reset(game_seed)
    return env, obs_dict
