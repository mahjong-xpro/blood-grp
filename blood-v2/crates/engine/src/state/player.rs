use crate::consts::*;
use crate::tile::{Tile, Suit};
use crate::hand::*;

/// Per-player game state
#[derive(Debug, Clone)]
pub struct PlayerState {
    pub hand: HandCounts,
    pub melds: Vec<MeldType>,
    pub discards: Vec<Tile>,
    pub tsumogiri: Vec<bool>,
    /// Full discard history for furiten checks (not modified by pon/kan)
    pub discard_history: Vec<Tile>,
    pub score: i32,
    pub ding_que: Option<Suit>,
    pub has_won: bool,
    pub last_drawn_tile: Option<Tile>,
    pub is_rinshan: bool,

    // Furiten tracking
    pub temporary_furiten: bool,
    pub furiten_passed_ron_fan: Option<u8>,

    // Derived
    pub tiles_seen: [u8; NUM_TILE_TYPES],
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            hand: [0; NUM_TILE_TYPES],
            melds: Vec::new(),
            discards: Vec::new(),
            tsumogiri: Vec::new(),
            discard_history: Vec::new(),
            score: INITIAL_SCORE,
            ding_que: None,
            has_won: false,
            last_drawn_tile: None,
            is_rinshan: false,
            temporary_furiten: false,
            furiten_passed_ron_fan: None,
            tiles_seen: [0; NUM_TILE_TYPES],
        }
    }

    pub fn hand_count(&self) -> u8 {
        total_tiles(&self.hand)
    }

    pub fn is_menzen(&self) -> bool {
        self.melds.iter().all(|m| !m.is_open())
    }

    pub fn ding_que_completed(&self) -> bool {
        match self.ding_que {
            Some(suit) => !has_suit_tiles(&self.hand, suit),
            None => false,
        }
    }

    pub fn ding_que_remaining(&self) -> u8 {
        match self.ding_que {
            Some(suit) => suit_tile_count(&self.hand, suit),
            None => 0,
        }
    }

    /// Compute legal discard tiles (must discard ding que suit first if present)
    pub fn discard_candidates(&self) -> Vec<Tile> {
        let mut candidates = Vec::new();
        let must_discard_dq = self.ding_que.map_or(false, |s| has_suit_tiles(&self.hand, s));

        for t in 0..NUM_TILE_TYPES as u8 {
            if self.hand[t as usize] == 0 { continue; }
            if must_discard_dq {
                if let Some(suit) = self.ding_que {
                    if Suit::from_tile(t) != suit { continue; }
                }
            }
            candidates.push(t);
        }
        candidates
    }

    /// Check if this player is in permanent furiten (past discards contain wait tile)
    pub fn is_permanent_furiten(&self, waits: &[Tile]) -> bool {
        for &w in waits {
            if self.discard_history.contains(&w) {
                return true;
            }
        }
        false
    }

    pub fn can_ankan_tiles(&self) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for t in 0..NUM_TILE_TYPES as u8 {
            if self.hand[t as usize] >= 4 {
                if let Some(suit) = self.ding_que {
                    if Suit::from_tile(t) == suit { continue; }
                }
                tiles.push(t);
            }
        }
        tiles
    }

    pub fn can_kakan_tiles(&self) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for m in &self.melds {
            if let MeldType::Pon(t) = m {
                if self.hand[*t as usize] > 0 {
                    tiles.push(*t);
                }
            }
        }
        tiles
    }

    pub fn see_tile(&mut self, t: Tile) {
        self.tiles_seen[t as usize] = self.tiles_seen[t as usize].saturating_add(1);
    }

    pub fn see_tiles(&mut self, tiles: &[Tile]) {
        for &t in tiles {
            self.see_tile(t);
        }
    }
}
