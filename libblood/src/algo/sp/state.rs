use super::CALC_SHANTEN_FN;
use super::tile::{DiscardTile, DrawTile, RequiredTile};
use crate::tile::Tile;
use crate::{must_tile, t, tu8};

use tinyvec::ArrayVec;

/// Mutable state of both the hand and the board.
/// Bloody Battle: 27 tile kinds (no jihai, no red 5s)
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct State {
    // hand
    pub(super) tehai: [u8; 27],
    // Bloody Battle: No akas
    pub(super) akas_in_hand: [bool; 3], // Kept for compatibility, but will be all false

    // global
    pub(super) tiles_in_wall: [u8; 27],
    // Bloody Battle: No akas
    pub(super) akas_in_wall: [bool; 3], // Kept for compatibility, but will be all false
    pub(super) n_extra_tsumo: u8,
}

/// Mutable state of both the hand and the board.
/// Bloody Battle: 27 tile kinds (no jihai, no red 5s)
#[derive(Clone)]
pub struct InitState {
    // hand
    pub tehai: [u8; 27],
    // Bloody Battle: No akas
    pub akas_in_hand: [bool; 3], // Kept for compatibility, but will be all false

    // global
    pub tiles_seen: [u8; 27],
    // Bloody Battle: No akas
    pub akas_seen: [bool; 3], // Kept for compatibility, but will be all false
}

impl From<InitState> for State {
    fn from(
        InitState {
            tehai,
            akas_in_hand,
            tiles_seen,
            akas_seen,
        }: InitState,
    ) -> Self {
        let mut tiles_in_wall = tiles_seen;
        let mut akas_in_wall = akas_seen;
        tiles_in_wall.iter_mut().for_each(|v| *v = 4 - *v);
        akas_in_wall.iter_mut().for_each(|v| *v = !*v);
        Self {
            tehai,
            akas_in_hand,
            tiles_in_wall,
            akas_in_wall,
            n_extra_tsumo: 0,
        }
    }
}

impl State {
    pub(super) const fn discard(&mut self, tile: Tile) {
        self.tehai[tile.deaka().as_usize()] -= 1;
        // Bloody Battle: No akas
    }

    pub(super) const fn undo_discard(&mut self, tile: Tile) {
        self.tehai[tile.deaka().as_usize()] += 1;
        // Bloody Battle: No akas
    }

    pub(super) const fn deal(&mut self, tile: Tile) {
        self.tiles_in_wall[tile.deaka().as_usize()] -= 1;
        // Bloody Battle: No akas
        self.undo_discard(tile);
    }

    pub(super) const fn undo_deal(&mut self, tile: Tile) {
        self.discard(tile);
        self.tiles_in_wall[tile.deaka().as_usize()] += 1;
        // Bloody Battle: No akas
    }

    pub(super) fn get_discard_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DiscardTile; 14]> {
        let mut discard_tiles = ArrayVec::default();

        let mut tehai = self.tehai;
        // Bloody Battle: 27 tile kinds
        for tid in 0..27 {
            if tehai[tid] == 0 {
                continue;
            }

            tehai[tid] -= 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] += 1;

            let shanten_diff = shanten_after - shanten;

            // Bloody Battle: No akas
            let tile = must_tile!(tid);

            discard_tiles.push(DiscardTile { tile, shanten_diff });
        }

        discard_tiles
    }

    pub(super) fn get_draw_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DrawTile; 27]> {
        let mut draw_tiles = ArrayVec::default();

        let mut tehai = self.tehai;
        // Bloody Battle: 27 tile kinds, no akas
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] -= 1;

            let shanten_diff = shanten_after - shanten;

            let tile = must_tile!(tid);
            // Bloody Battle: No akas, just add the tile directly
            draw_tiles.push(DrawTile {
                tile,
                count,
                shanten_diff,
            });
        }

        draw_tiles
    }

    pub(super) fn get_required_tiles(&self, tehai_len_div3: u8) -> ArrayVec<[RequiredTile; 27]> {
        let mut tehai = self.tehai;

        let shanten = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
        let mut required_tiles = ArrayVec::default();

        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] -= 1;

            if shanten_after < shanten {
                required_tiles.push(RequiredTile {
                    tile: must_tile!(tid),
                    count,
                });
            }
        }

        required_tiles
    }

    pub(super) fn sum_left_tiles(&self) -> u8 {
        self.tiles_in_wall.iter().sum()
    }
}
