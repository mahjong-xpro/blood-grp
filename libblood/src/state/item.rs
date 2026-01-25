use crate::tile::Tile;
use std::fmt;

use serde::Serialize;
use tinyvec::ArrayVec;

#[derive(Debug, Clone, Serialize)]
pub(super) struct KawaItem {
    pub(super) kan: ArrayVec<[Tile; 4]>,
    pub(super) sutehai: Sutehai,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct Sutehai {
    pub(super) tile: Tile,
    /// Whether the tile was discarded from hand (手出) vs tsumogiri (摸切)
    /// Note: In Bloody Battle Mahjong, this distinction is not used in game logic,
    /// but kept for observation space compatibility (used in obs encoding)
    pub(super) is_tedashi: bool,
}

impl fmt::Display for Sutehai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            self.tile,
            if self.is_tedashi { "" } else { "^" },
        )
    }
}


impl fmt::Display for KawaItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.kan.is_empty() {
            f.write_str("{")?;
            for kan in self.kan {
                write!(f, "{kan}")?;
            }
            f.write_str("}")?;
        }

        write!(f, "{}", self.sutehai)
    }
}
