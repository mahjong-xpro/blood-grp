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
}

impl fmt::Display for Sutehai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tile)
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
