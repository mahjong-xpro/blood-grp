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
#[derive(Default)]
pub struct Invisible {
    pub yama: Vec<Tile>,
}

impl Invisible {
    pub fn new(game: &[Event], trust_seed: bool) -> Vec<Self> {
        let mut ret = vec![];
        let mut cur = Self::default();
        let mut seed = None;
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
                        let mut board = Board {
                            kyoku: kyoku - 1,
                            ..Default::default()
                        };
                        board.init_from_seed(seed);

                        cur.yama = board.yama;

                        // reverse because of the way Board pops tiles
                        cur.yama.reverse();

                        ret.push(mem::take(&mut cur));
                        continue;
                    }
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
                    cur.yama.push(*pai);
                    unknown_tiles[pai.as_usize()] -= 1;
                }
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                }
                Event::Hora { .. } => {
                }
                Event::EndKyoku => {
                    let mut filler: Vec<_> = unknown_tiles
                        .into_iter()
                        .enumerate()
                        .filter(|&(_, count)| count > 0)
                        .flat_map(|(tid, count)| iter::repeat_n(must_tile!(tid), count as usize))
                        .collect();
                    filler.shuffle(&mut rng());

                    while cur.yama.len() < 56 {
                        cur.yama.push(filler.pop().unwrap());
                    }
                    assert!(filler.is_empty());

                    ret.push(mem::take(&mut cur));
                    unknown_tiles = new_unknown_tiles();
                }

                _ => (),
            };
        }

        ret
    }

    // TODO: merge this with arena::board::BoardState::encode_oracle_obs; they
    // should be identical. This is a code quality improvement to reduce duplication.
    pub fn encode(
        &self,
        opponent_states: &[PlayerState; 3],
        yama_idx: usize,
        _unused: usize, // Reserved for future use (previously rinshan_idx, not used in Bloody Battle)
        version: u32,
    ) -> Array2<f32> {
        let shape = oracle_obs_shape(version);
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

            idx += 3;

            // FIX: shanten 可能因定缺惩罚超过 6，但 one-hot 只有 7 通道（v2+）/6（v1）。
            // 不裁剪会写入后续通道（rescale / waits），污染 oracle 特征。
            // 与 obs_repr.rs 保持一致：.min(6) 裁剪。
            let raw_shanten = state.shanten().max(0) as usize;
            match version {
                1 => {
                    let n = raw_shanten.min(5);
                    arr.fill_rows(idx, n, 1.);
                    idx += 6;
                }
                2 | 3 | 4 => {
                    let n = raw_shanten.min(6);
                    arr.fill(idx + n, 1.);
                    idx += 7;

                    let v = raw_shanten as f32 / 6.;
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

            idx += 1;
        }

        let mut encode_tile = |idx: usize, tile: Tile| -> usize {
            let tile_id = tile.as_usize();
            arr.assign(idx, tile_id, 1.);
            idx + 1
        };

        // Each tile uses 1 dimension (no aka encoding)
        for &tile in &self.yama[yama_idx..] {
            idx = encode_tile(idx, tile);
        }
        // Skip remaining yama slots to maintain fixed size
        // Original used 69 max tiles. For Bloody Battle (v2+), use 56.
        let max_yama_tiles: usize = if version >= 2 { 56 } else { 69 };
        let encoded_tiles = self.yama.len().saturating_sub(yama_idx);
        let remaining_yama = max_yama_tiles.saturating_sub(encoded_tiles);
        idx += remaining_yama;

        idx += 4 * 1;

        idx += 5 * 1;

        idx += 5 * 1;

        assert_eq!(idx, shape.0);
        arr.build()
    }
}

const fn new_unknown_tiles() -> [u8; 27] {
    [4; 27] // All tiles have 4 copies
}
