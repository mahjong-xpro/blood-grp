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
    "NUM_STUDENT_CHANNELS": 473,  # 470 + 3 (Section 15: 現物/genbutsu safe tiles)
    "NUM_ORACLE_EXTRA_CHANNELS": 52,
    "NUM_ORACLE_CHANNELS": 525,  # 473 + 52
    "INITIAL_SCORE": 100_000,
    "REWARD_NORM": 32_000,
    "MAX_FAN": 6,
    "MAX_TURNS": 28,
    # Observation channel offsets (must match crates/engine/src/consts.rs)
    "CH_HAND_BASE": 0,
    "CH_HAND_COUNT": 5,
    "CH_GAME_CONTEXT_BASE": 5,
    "CH_TURN_PROGRESS": 17,
    "CH_DING_QUE_BASE": 18,
    "CH_OPP_DING_QUE_BASE": 23,
    "CH_OPP_AGARI_BASE": 32,
    "CH_WALL_REMAINING": 35,
    "CH_SELF_KAWA_BASE": 40,
    "CH_OPP_KAWA_BASE": 98,
    "CH_OPP_KAWA_STRIDE": 58,
    "CH_VISIBLE_TILES_BASE": 272,
    "CH_OPP_KAWA_OVERVIEW_BASE": 272,
    "CH_OPP_SUIT_RATIO_BASE": 320,
    "CH_TILES_REMAINING": 329,
    "CH_SELF_MENZEN": 330,
    "CH_SELF_MELDS": 331,
    "CH_OPP_MELD_BASE": 333,
    "CH_OPP_TERMINAL_RATIO_BASE": 336,
    "CH_SELF_DISCARD_COUNT": 339,
    "CH_HAND_ANALYSIS_BASE": 340,
    "CH_SHANTEN_BASE": 341,
    "CH_ACTION_CONTEXT_BASE": 346,
    "CH_SP_TABLE_BASE": 358,
    "CH_FAN_CONFIG_BASE": 457,
    "CH_OPP_HAND_INFO_BASE": 464,
    "CH_GENBUTSU_BASE": 470,
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
        CH_HAND_BASE,
        CH_HAND_COUNT,
        CH_GAME_CONTEXT_BASE,
        CH_TURN_PROGRESS,
        CH_DING_QUE_BASE,
        CH_OPP_DING_QUE_BASE,
        CH_OPP_AGARI_BASE,
        CH_WALL_REMAINING,
        CH_SELF_KAWA_BASE,
        CH_OPP_KAWA_BASE,
        CH_OPP_KAWA_STRIDE,
        CH_VISIBLE_TILES_BASE,
        CH_OPP_KAWA_OVERVIEW_BASE,
        CH_OPP_SUIT_RATIO_BASE,
        CH_TILES_REMAINING,
        CH_SELF_MENZEN,
        CH_SELF_MELDS,
        CH_OPP_MELD_BASE,
        CH_OPP_TERMINAL_RATIO_BASE,
        CH_SELF_DISCARD_COUNT,
        CH_HAND_ANALYSIS_BASE,
        CH_SHANTEN_BASE,
        CH_ACTION_CONTEXT_BASE,
        CH_SP_TABLE_BASE,
        CH_FAN_CONFIG_BASE,
        CH_OPP_HAND_INFO_BASE,
        CH_GENBUTSU_BASE,
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
    CH_HAND_BASE = _DEFAULTS["CH_HAND_BASE"]
    CH_HAND_COUNT = _DEFAULTS["CH_HAND_COUNT"]
    CH_GAME_CONTEXT_BASE = _DEFAULTS["CH_GAME_CONTEXT_BASE"]
    CH_TURN_PROGRESS = _DEFAULTS["CH_TURN_PROGRESS"]
    CH_DING_QUE_BASE = _DEFAULTS["CH_DING_QUE_BASE"]
    CH_OPP_DING_QUE_BASE = _DEFAULTS["CH_OPP_DING_QUE_BASE"]
    CH_OPP_AGARI_BASE = _DEFAULTS["CH_OPP_AGARI_BASE"]
    CH_WALL_REMAINING = _DEFAULTS["CH_WALL_REMAINING"]
    CH_SELF_KAWA_BASE = _DEFAULTS["CH_SELF_KAWA_BASE"]
    CH_OPP_KAWA_BASE = _DEFAULTS["CH_OPP_KAWA_BASE"]
    CH_OPP_KAWA_STRIDE = _DEFAULTS["CH_OPP_KAWA_STRIDE"]
    CH_VISIBLE_TILES_BASE = _DEFAULTS["CH_VISIBLE_TILES_BASE"]
    CH_OPP_KAWA_OVERVIEW_BASE = _DEFAULTS["CH_OPP_KAWA_OVERVIEW_BASE"]
    CH_OPP_SUIT_RATIO_BASE = _DEFAULTS["CH_OPP_SUIT_RATIO_BASE"]
    CH_TILES_REMAINING = _DEFAULTS["CH_TILES_REMAINING"]
    CH_SELF_MENZEN = _DEFAULTS["CH_SELF_MENZEN"]
    CH_SELF_MELDS = _DEFAULTS["CH_SELF_MELDS"]
    CH_OPP_MELD_BASE = _DEFAULTS["CH_OPP_MELD_BASE"]
    CH_OPP_TERMINAL_RATIO_BASE = _DEFAULTS["CH_OPP_TERMINAL_RATIO_BASE"]
    CH_SELF_DISCARD_COUNT = _DEFAULTS["CH_SELF_DISCARD_COUNT"]
    CH_HAND_ANALYSIS_BASE = _DEFAULTS["CH_HAND_ANALYSIS_BASE"]
    CH_SHANTEN_BASE = _DEFAULTS["CH_SHANTEN_BASE"]
    CH_ACTION_CONTEXT_BASE = _DEFAULTS["CH_ACTION_CONTEXT_BASE"]
    CH_SP_TABLE_BASE = _DEFAULTS["CH_SP_TABLE_BASE"]
    CH_FAN_CONFIG_BASE = _DEFAULTS["CH_FAN_CONFIG_BASE"]
    CH_OPP_HAND_INFO_BASE = _DEFAULTS["CH_OPP_HAND_INFO_BASE"]
    CH_GENBUTSU_BASE = _DEFAULTS["CH_GENBUTSU_BASE"]

# ── Derived constants ────────────────────────────────────────────────────────

TILES_PER_SUIT = NUM_TILE_TYPES // 3  # 9
OBS_SIZE = NUM_STUDENT_CHANNELS * NUM_TILE_TYPES
ORACLE_OBS_SIZE = NUM_ORACLE_CHANNELS * NUM_TILE_TYPES
