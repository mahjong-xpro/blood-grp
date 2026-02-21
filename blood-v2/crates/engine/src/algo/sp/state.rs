use crate::consts::*;
use crate::tile::{Tile, Suit};
use crate::hand::HandCounts;

/// Initial state for SP calculation
#[derive(Debug, Clone)]
pub struct SPInitState {
    pub tehai: HandCounts,
    pub tiles_seen: [u8; NUM_TILE_TYPES],
    pub tiles_left: u8,
    pub num_melds: usize,
    pub ding_que: Option<Suit>,
}

/// Mutable state during SP calculation
#[derive(Debug, Clone)]
pub struct SPState {
    pub tehai: HandCounts,
    pub tiles_seen: [u8; NUM_TILE_TYPES],
    pub remaining: u8,
}

impl SPState {
    pub fn from_init(init: &SPInitState) -> Self {
        Self {
            tehai: init.tehai,
            tiles_seen: init.tiles_seen,
            remaining: init.tiles_left,
        }
    }

    pub fn available_count(&self, t: Tile) -> u8 {
        (COPIES_PER_TILE as u8).saturating_sub(self.tiles_seen[t as usize])
    }
}
