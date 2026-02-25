"""Centralized constants for Blood Mahjong.

All game constants are imported from the Rust engine (`blood._engine`) as the
single source of truth.  Fallback values are provided so that Python-only code
(tests, linting, type-checking) works even when the native extension is not
compiled.

Every Python module that needs these values should import from here instead of
hardcoding magic numbers.
"""

import logging

log = logging.getLogger(__name__)

# ── Defaults (must match crates/engine/src/consts.rs) ────────────────────────
# These are ONLY used when the Rust engine is not available (e.g. pure-Python
# tests, CI without a Rust toolchain).  In production the Rust values always
# take precedence.

_DEFAULTS = {
    "NUM_TILE_TYPES": 27,
    "NUM_PLAYERS": 4,
    "ACTION_SPACE": 34,
    "NUM_STUDENT_CHANNELS": 470,  # 464 + 6 (Section 14: 对手手牌数 + 副露来源)
    "NUM_ORACLE_EXTRA_CHANNELS": 52,
    "NUM_ORACLE_CHANNELS": 522,  # 470 + 52
    "INITIAL_SCORE": 100_000,
    "REWARD_NORM": 32_000,
    "MAX_FAN": 6,
    "MAX_TURNS": 28,
    # Observation channel offsets (must match crates/engine/src/obs/student.rs)
    "CH_WALL_REMAINING": 35,
    "CH_OPP_MELD_BASE": 333,
    "CH_SHANTEN_BASE": 341,
    # 对手听牌推断所需的额外通道偏移量
    "CH_TURN_PROGRESS": 17,          # Section 2: 回合进度 (turn_count / MAX_TURNS)
    "CH_OPP_DING_QUE_BASE": 23,     # Section 3: 对手定缺 (3×3 ch)
    "CH_OPP_AGARI_BASE": 32,        # Section 3: 对手和牌状态 (3 ch)
    "CH_OPP_KAWA_BASE": 98,         # Section 6: 对手牌河起始
    "CH_OPP_SUIT_RATIO_BASE": 320,  # Section 8: 对手花色打牌比例 (3×3 ch)
    "CH_OPP_TERMINAL_RATIO_BASE": 336,  # Section 9: 对手幺九打牌比例 (3 ch)
    "CH_SELF_DISCARD_COUNT": 339,    # Section 9: 自家打牌数
}

# ── Import from Rust engine ──────────────────────────────────────────────────

try:
    from blood._engine import (
        NUM_TILE_TYPES,
        NUM_PLAYERS,
        ACTION_SPACE,
        NUM_STUDENT_CHANNELS,
        NUM_ORACLE_CHANNELS,
        NUM_ORACLE_EXTRA_CHANNELS,
        INITIAL_SCORE,
        REWARD_NORM,
        MAX_FAN,
        MAX_TURNS,
        CH_WALL_REMAINING,
        CH_OPP_MELD_BASE,
        CH_SHANTEN_BASE,
    )
except ImportError:
    log.warning(
        "blood._engine not available; using Python fallback constants. "
        "Build the Rust engine (`maturin develop`) for production use."
    )
    NUM_TILE_TYPES = _DEFAULTS["NUM_TILE_TYPES"]
    NUM_PLAYERS = _DEFAULTS["NUM_PLAYERS"]
    ACTION_SPACE = _DEFAULTS["ACTION_SPACE"]
    NUM_STUDENT_CHANNELS = _DEFAULTS["NUM_STUDENT_CHANNELS"]
    NUM_ORACLE_EXTRA_CHANNELS = _DEFAULTS["NUM_ORACLE_EXTRA_CHANNELS"]
    NUM_ORACLE_CHANNELS = _DEFAULTS["NUM_ORACLE_CHANNELS"]
    INITIAL_SCORE = _DEFAULTS["INITIAL_SCORE"]
    REWARD_NORM = _DEFAULTS["REWARD_NORM"]
    MAX_FAN = _DEFAULTS["MAX_FAN"]
    MAX_TURNS = _DEFAULTS["MAX_TURNS"]
    CH_WALL_REMAINING = _DEFAULTS["CH_WALL_REMAINING"]
    CH_OPP_MELD_BASE = _DEFAULTS["CH_OPP_MELD_BASE"]
    CH_SHANTEN_BASE = _DEFAULTS["CH_SHANTEN_BASE"]

# ── Python-only 通道偏移量（从 student.rs 布局推导，不由 Rust 导出）──────────
CH_TURN_PROGRESS = _DEFAULTS["CH_TURN_PROGRESS"]
CH_OPP_DING_QUE_BASE = _DEFAULTS["CH_OPP_DING_QUE_BASE"]
CH_OPP_AGARI_BASE = _DEFAULTS["CH_OPP_AGARI_BASE"]
CH_OPP_KAWA_BASE = _DEFAULTS["CH_OPP_KAWA_BASE"]
CH_OPP_SUIT_RATIO_BASE = _DEFAULTS["CH_OPP_SUIT_RATIO_BASE"]
CH_OPP_TERMINAL_RATIO_BASE = _DEFAULTS["CH_OPP_TERMINAL_RATIO_BASE"]
CH_SELF_DISCARD_COUNT = _DEFAULTS["CH_SELF_DISCARD_COUNT"]

# ── Derived constants ────────────────────────────────────────────────────────

TILES_PER_SUIT = NUM_TILE_TYPES // 3  # 9
OBS_SIZE = NUM_STUDENT_CHANNELS * NUM_TILE_TYPES
ORACLE_OBS_SIZE = NUM_ORACLE_CHANNELS * NUM_TILE_TYPES
