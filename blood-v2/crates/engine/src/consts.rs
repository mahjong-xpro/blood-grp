pub const NUM_TILE_TYPES: usize = 27;
pub const TILES_PER_SUIT: usize = 9;
pub const NUM_SUITS: usize = 3;
pub const COPIES_PER_TILE: usize = 4;
pub const TOTAL_TILES: usize = NUM_TILE_TYPES * COPIES_PER_TILE; // 108
pub const NUM_PLAYERS: usize = 4;
pub const INITIAL_SCORE: i32 = 60_000;
pub const HAND_SIZE: usize = 13;
pub const MAX_MELDS: usize = 4;

pub const ACTION_SPACE: usize = 34;
pub const MAX_FAN: u8 = 5;
pub const MAX_TURNS: usize = 28;

pub const NUM_STUDENT_CHANNELS: usize = 384;
pub const NUM_ORACLE_EXTRA_CHANNELS: usize = 46;
pub const NUM_ORACLE_CHANNELS: usize = NUM_STUDENT_CHANNELS + NUM_ORACLE_EXTRA_CHANNELS;
