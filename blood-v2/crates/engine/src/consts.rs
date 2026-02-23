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

pub const NUM_STUDENT_CHANNELS: usize = 464;
pub const NUM_ORACLE_EXTRA_CHANNELS: usize = 52;
pub const NUM_ORACLE_CHANNELS: usize = NUM_STUDENT_CHANNELS + NUM_ORACLE_EXTRA_CHANNELS;
