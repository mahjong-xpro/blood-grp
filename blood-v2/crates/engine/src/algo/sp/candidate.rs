use crate::consts::MAX_TURNS;
use crate::tile::Tile;

/// A discard candidate with SP table values
#[derive(Debug, Clone)]
pub struct Candidate {
    pub tile: Tile,
    pub shanten_diff: i8,
    pub tenpai_probs: [f32; MAX_TURNS],
    pub win_probs: [f32; MAX_TURNS],
    pub exp_values: [f32; MAX_TURNS],
}

impl Candidate {
    pub fn new(tile: Tile, shanten_diff: i8) -> Self {
        Self {
            tile,
            shanten_diff,
            tenpai_probs: [0.0; MAX_TURNS],
            win_probs: [0.0; MAX_TURNS],
            exp_values: [0.0; MAX_TURNS],
        }
    }

    pub fn total_ev(&self) -> f32 {
        self.exp_values.iter().sum()
    }

    pub fn max_ev(&self) -> f32 {
        self.exp_values.iter().cloned().fold(0.0f32, f32::max)
    }

    pub fn total_win_prob(&self) -> f32 {
        self.win_probs.iter().sum()
    }

    pub fn max_win_prob(&self) -> f32 {
        self.win_probs.iter().cloned().fold(0.0f32, f32::max)
    }
}
