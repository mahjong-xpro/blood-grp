pub const NUM_TILE_TYPES: usize = 27;
pub const TILES_PER_SUIT: usize = 9;
pub const NUM_SUITS: usize = 3;
pub const COPIES_PER_TILE: usize = 4;
pub const TOTAL_TILES: usize = NUM_TILE_TYPES * COPIES_PER_TILE; // 108
pub const NUM_PLAYERS: usize = 4;
pub const INITIAL_SCORE: i32 = 100_000;
pub const HAND_SIZE: usize = 13;
pub const MAX_MELDS: usize = 4;

pub const ACTION_SPACE: usize = 34;
pub const MAX_FAN: u8 = 6;
pub const MAX_TURNS: usize = 28;

/// Normalization constant for rewards.
///
/// Derivation (6-fan cap, score = 1000 × 2^(fan-1)):
///   Max single-player payment = 32000 (6-fan hand)
///
/// Tsumo payment (all active opponents pay):
///   1-fan tsumo (3 pay): agent +3000,  each opp -1000
///   3-fan tsumo (3 pay): agent +12000, each opp -4000
///   6-fan tsumo (3 pay): agent +96000, each opp -32000
///
/// Used with sqrt compression in Python: reward = sign(Δ/32000) × sqrt(|Δ/32000|)
///   1-fan ron  (1000)  → 0.177    6-fan ron  (32000) → 1.000
///   1-fan tsumo(3000)  → 0.306    6-fan tsumo(96000) → 1.732
///   1-fan deal-in      → -0.177   6-fan deal-in      → -1.000
///
/// Sqrt compression reduces the 32:1 linear ratio to ~5.6:1, lowering
/// reward variance while preserving ordering. Centered at 0 for PPO.
pub const REWARD_NORM: i32 = 32_000;

/// 注意：通道数变更需要重新训练模型。
/// 464 → 470: 新增 Section 14（对手手牌数 3ch + 副露来源 3ch）
/// 470 → 473: 新增 Section 15（现物标记 3ch — 对手弃过的牌 100% 安全）
pub const NUM_STUDENT_CHANNELS: usize = 473;
pub const NUM_ORACLE_EXTRA_CHANNELS: usize = 52;
pub const NUM_ORACLE_CHANNELS: usize = NUM_STUDENT_CHANNELS + NUM_ORACLE_EXTRA_CHANNELS;

// ── Observation channel offsets (0-indexed) ─────────────────────────────────
// These must stay in sync with crates/engine/src/obs/student.rs.
// Exported via PyO3 so Python code (RTPA, etc.) never hardcodes offsets.

// === Section 1: Hand ===
pub const CH_HAND_BASE: usize = 0;
pub const CH_HAND_COUNT: usize = 5;

// === Section 2: Game Context ===
pub const CH_GAME_CONTEXT_BASE: usize = 5;
// turn_progress is at ch=14: 5(hand) + 4(scores) + 4(ranks) + 1(is_dealer) = 14
pub const CH_TURN_PROGRESS: usize = 14;

// === Section 3: Ding Que ===
pub const CH_DING_QUE_BASE: usize = 18;
pub const CH_OPP_DING_QUE_BASE: usize = 23;
pub const CH_OPP_AGARI_BASE: usize = 32;

// === Section 4: Game State ===
/// Section 4, ch 0: wall_remaining / 55.0
pub const CH_WALL_REMAINING: usize = 35;

// === Section 5: Self Kawa ===
pub const CH_SELF_KAWA_BASE: usize = 40;

// === Section 6: Opponent Kawa ===
pub const CH_OPP_KAWA_BASE: usize = 98;
pub const CH_OPP_KAWA_STRIDE: usize = 58; // MAX_TURNS * 2 + 2

// === Section 7: Visible Tiles ===
pub const CH_VISIBLE_TILES_BASE: usize = 272;
pub const CH_OPP_KAWA_OVERVIEW_BASE: usize = 272;

// === Section 8: Defense ===
pub const CH_OPP_SUIT_RATIO_BASE: usize = 320;

// === Section 9: Derived ===
pub const CH_TILES_REMAINING: usize = 329;
pub const CH_SELF_MENZEN: usize = 330;
pub const CH_SELF_MELDS: usize = 331;
/// Section 9: opponent meld counts (3 opponents at ch+0, ch+1, ch+2)
pub const CH_OPP_MELD_BASE: usize = 333;
pub const CH_OPP_TERMINAL_RATIO_BASE: usize = 336;
pub const CH_SELF_DISCARD_COUNT: usize = 339;

// === Section 10: Hand Analysis ===
pub const CH_HAND_ANALYSIS_BASE: usize = 340;
/// Section 10: wait_tiles at ch=340, shanten one-hot at ch=341..345
pub const CH_SHANTEN_BASE: usize = 341;

// === Section 11: Action Context ===
pub const CH_ACTION_CONTEXT_BASE: usize = 346;

// === Section 12: SP Table ===
pub const CH_SP_TABLE_BASE: usize = 358;

// === Section 13: Fan Config ===
pub const CH_FAN_CONFIG_BASE: usize = 457;

// === Section 14: Opponent Hand Info ===
pub const CH_OPP_HAND_INFO_BASE: usize = 464;

// === Section 15: Genbutsu (Safe Tiles) ===
pub const CH_GENBUTSU_BASE: usize = 470;
