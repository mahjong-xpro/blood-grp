use crate::consts::*;
use crate::tile::{Tile, Suit};
use crate::hand::*;
use super::ding_que;

/// Per-player game state
#[derive(Debug, Clone)]
pub struct PlayerState {
    pub hand: HandCounts,
    pub melds: Vec<MeldType>,
    /// 每个副露的来源玩家（绝对座位号）。
    /// AnKan 为 None（暗杠无来源），Pon/MinKan 为 Some(from)，
    /// KaKan 继承原 Pon 的来源。与 melds 一一对应。
    pub meld_from: Vec<Option<usize>>,
    pub discards: Vec<Tile>,
    pub tsumogiri: Vec<bool>,
    pub score: i32,
    pub ding_que: Option<Suit>,
    pub has_won: bool,
    pub last_drawn_tile: Option<Tile>,
    pub is_rinshan: bool,

    // 过手加番: set when player passes on a winning tile; cleared on next draw/meld
    pub furiten_passed_ron_fan: Option<u8>,

    // Derived
    pub tiles_seen: [u8; NUM_TILE_TYPES],
}

impl PlayerState {
    pub fn new() -> Self {
        Self::with_score(INITIAL_SCORE)
    }

    pub fn with_score(initial_score: i32) -> Self {
        Self {
            hand: [0; NUM_TILE_TYPES],
            melds: Vec::new(),
            meld_from: Vec::new(),
            discards: Vec::new(),
            tsumogiri: Vec::new(),
            score: initial_score,
            ding_que: None,
            has_won: false,
            last_drawn_tile: None,
            is_rinshan: false,
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
        ding_que::is_completed(&self.hand, self.ding_que)
    }

    pub fn ding_que_remaining(&self) -> u8 {
        ding_que::remaining_count(&self.hand, self.ding_que)
    }

    /// Compute legal discard tiles (must discard ding que suit first if present)
    pub fn discard_candidates(&self) -> Vec<Tile> {
        let mut candidates = Vec::new();
        let must_dq = ding_que::must_discard_ding_que(&self.hand, self.ding_que);

        for t in 0..NUM_TILE_TYPES as u8 {
            if self.hand[t as usize] == 0 { continue; }
            if must_dq && !ding_que::is_ding_que_tile(self.ding_que, t) { continue; }
            candidates.push(t);
        }
        candidates
    }

    pub fn can_ankan_tiles(&self) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for t in 0..NUM_TILE_TYPES as u8 {
            if self.hand[t as usize] >= 4 {
                if ding_que::is_ding_que_tile(self.ding_que, t) { continue; }
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
                    if ding_que::is_ding_que_tile(self.ding_que, *t) { continue; }
                    tiles.push(*t);
                }
            }
        }
        tiles
    }

    pub fn see_tile(&mut self, t: Tile) {
        self.tiles_seen[t as usize] = self.tiles_seen[t as usize].saturating_add(1);
    }

    /// Fix R11-M4: reverse a see_tile call (used when chankan reverts kakan).
    pub fn unsee_tile(&mut self, t: Tile) {
        self.tiles_seen[t as usize] = self.tiles_seen[t as usize].saturating_sub(1);
    }

    pub fn see_tile_n(&mut self, t: Tile, n: u8) {
        self.tiles_seen[t as usize] = self.tiles_seen[t as usize].saturating_add(n);
    }

    pub fn see_tiles(&mut self, tiles: &[Tile]) {
        for &t in tiles {
            self.see_tile(t);
        }
    }
}
