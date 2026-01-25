use crate::arena::Board;
use crate::array::Simple2DArray;
use crate::consts::oracle_obs_shape;
use crate::mjai::Event;
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::must_tile;
use std::iter;
use std::mem;

use ndarray::prelude::*;
use rand::prelude::*;
use rand::rng;

/// All fields are sorted early -> late.
/// Bloody Battle Mahjong: Only yama is needed (no rinshan, dora_indicators, ura_indicators)
#[derive(Default)]
pub struct Invisible {
    pub yama: Vec<Tile>,
    // Bloody Battle: No rinshan, dora_indicators, or ura_indicators
    pub rinshan: Vec<Tile>, // Kept for compatibility, but will be empty
    pub dora_indicators: Vec<Tile>, // Kept for compatibility, but will be empty
    pub ura_indicators: Vec<Tile>, // Kept for compatibility, but will be empty
}

impl Invisible {
    pub fn new(game: &[Event], trust_seed: bool) -> Vec<Self> {
        let mut ret = vec![];
        let mut cur = Self::default();
        let mut seed = None;
        let mut from_rinshan = false;
        let mut ura_is_recorded = false;
        let mut unknown_tiles = new_unknown_tiles();

        for event in game {
            match event {
                // If the game was emulated by our lib, then use the seed directly
                Event::StartGame {
                    seed: Some(game_seed),
                    ..
                } if trust_seed => {
                    seed = Some(*game_seed);
                }

                Event::StartKyoku {
                    kyoku,
                    tehais,
                    ..
                } => {
                    if let Some(seed) = seed {
                        // Bloody Battle: No bakaze, honba, or dora_marker
                        let mut board = Board {
                            kyoku: kyoku - 1,
                            ..Default::default()
                        };
                        board.init_from_seed(seed);

                        cur.yama = board.yama;
                        // Bloody Battle: No rinshan, dora_indicators, or ura_indicators in Board
                        cur.rinshan.clear();
                        cur.dora_indicators.clear();
                        cur.ura_indicators.clear();

                        // reverse because of the way Board pops tiles
                        cur.yama.reverse();

                        ret.push(mem::take(&mut cur));
                        continue;
                    }
                    // Bloody Battle: No dora_marker
                    tehais
                        .iter()
                        .flatten()
                        .for_each(|tile| unknown_tiles[tile.as_usize()] -= 1);
                }
                _ => (),
            };

            if seed.is_some() {
                continue;
            }

            match event {
                Event::Tsumo { pai, .. } => {
                    if from_rinshan {
                        cur.rinshan.push(*pai);
                        from_rinshan = false;
                    } else {
                        cur.yama.push(*pai);
                        assert!(cur.yama.len() <= 56, "yama size overflow"); // Bloody Battle: 56 tiles in yama
                    }
                    unknown_tiles[pai.as_usize()] -= 1;
                }
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                    // Bloody Battle: No rinshan, tiles come from yama directly
                    from_rinshan = false;
                }
                // Event::Dora removed - Bloody Battle Mahjong does not have dora
                Event::Hora { .. } => {
                    // Bloody Battle: No ura_markers
                }
                Event::EndKyoku => {
                    let mut filler: Vec<_> = unknown_tiles
                        .into_iter()
                        .enumerate()
                        .filter(|&(_, count)| count > 0)
                        .flat_map(|(tid, count)| iter::repeat_n(must_tile!(tid), count as usize))
                        .collect();
                    filler.shuffle(&mut rng());

                    // Bloody Battle: 56 tiles in yama (108 - 52 = 56)
                    while cur.yama.len() < 56 {
                        cur.yama.push(filler.pop().unwrap());
                    }
                    // Bloody Battle: No rinshan, dora_indicators, or ura_indicators
                    // Keep them empty for compatibility
                    assert!(filler.is_empty());

                    ret.push(mem::take(&mut cur));
                    from_rinshan = false;
                    ura_is_recorded = false;
                    unknown_tiles = new_unknown_tiles();
                }

                _ => (),
            };
        }

        ret
    }

    // TODO: merge this this arena::board::BoardState::encode_oracle_obs; they
    // should be identical.
    pub fn encode(
        &self,
        opponent_states: &[PlayerState; 3],
        yama_idx: usize,
        _rinshan_idx: usize,
        version: u32,
    ) -> Array2<f32> {
        let shape = oracle_obs_shape(version);
        // Bloody Battle: 27 tile kinds (no jihai)
        let mut arr = Simple2DArray::<27, f32>::new(shape.0);
        let mut idx = 0;

        for state in opponent_states {
            state
                .tehai()
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .for_each(|(tile_id, &count)| arr.assign_rows(idx, tile_id, count as usize, 1.));
            idx += 4;

            // Bloody Battle: No akas_in_hand
            idx += 3; // Keep same offset for compatibility

            let n = state.shanten() as usize;
            match version {
                1 => {
                    arr.fill_rows(idx, n, 1.);
                    idx += 6;
                }
                2 | 3 | 4 => {
                    arr.fill(idx + n, 1.);
                    idx += 7;

                    let v = n as f32 / 6.;
                    arr.fill(idx, v);
                    idx += 1;
                }
                _ => unreachable!(),
            }

            state
                .waits()
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| arr.assign(idx, t, 1.));
            idx += 1;

            // Bloody Battle: No furiten
            idx += 1; // Keep same offset for compatibility
        }

        let mut encode_tile = |idx: usize, tile: Tile| -> usize {
            let tile_id = tile.deaka().as_usize();
            arr.assign(idx, tile_id, 1.);
            // Bloody Battle: No akas, so only use 1 dimension per tile
            idx + 1
        };

        // Bloody Battle: yama encoding - encode remaining tiles
        // Each tile uses 1 dimension (no aka encoding)
        for &tile in &self.yama[yama_idx..] {
            idx = encode_tile(idx, tile);
        }
        // Skip remaining yama slots to maintain fixed size
        // Original used 69 max tiles, keep same for compatibility
        // Bloody Battle has 56 tiles max, but we keep 69 slots for compatibility
        let max_yama_tiles: usize = 69;
        let encoded_tiles = self.yama.len().saturating_sub(yama_idx);
        let remaining_yama = max_yama_tiles.saturating_sub(encoded_tiles);
        idx += remaining_yama;

        // Bloody Battle: No rinshan, skip encoding (was 4 * 2 = 8)
        idx += 4 * 1; // Keep offset but use 1 dimension

        // Bloody Battle: No dora_indicators, skip encoding (was 5 * 2 = 10)
        idx += 5 * 1; // Keep offset but use 1 dimension

        // Bloody Battle: No ura_indicators, skip encoding (was 5 * 2 = 10)
        idx += 5 * 1; // Keep offset but use 1 dimension

        assert_eq!(idx, shape.0);
        arr.build()
    }
}

// Bloody Battle: 27 tile kinds (no jihai, no red 5s)
const fn new_unknown_tiles() -> [u8; 27] {
    [4; 27] // All tiles have 4 copies
}
