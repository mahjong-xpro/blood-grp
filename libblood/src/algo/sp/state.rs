use super::CALC_SHANTEN_FN;
use super::tile::{DiscardTile, DrawTile, RequiredTile};
use crate::tile::Tile;
use crate::must_tile;

use tinyvec::ArrayVec;

/// Mutable state of both the hand and the board.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct State {
    // hand
    pub(super) tehai: [u8; 27],

    // global
    pub(super) tiles_in_wall: [u8; 27],
    pub(super) n_extra_tsumo: u8,
}

/// Mutable state of both the hand and the board.
#[derive(Clone)]
pub struct InitState {
    // hand
    pub tehai: [u8; 27],

    // global
    pub tiles_seen: [u8; 27],
    
    // tiles_left is the authoritative count of tiles remaining in the wall
    // Used to validate and correct tiles_in_wall calculation
    pub tiles_left: u8,
}

impl From<InitState> for State {
    fn from(
        InitState {
            tehai,
            tiles_seen,
            tiles_left,
        }: InitState,
    ) -> Self {
        // 基础规则验证：每种 tile 最多只有 4 张，所以 tiles_seen 的每个值不应该超过 4
        // 如果超过，说明计算有误（可能是重复计算），需要修正
        let mut corrected_tiles_seen = tiles_seen;
        for count in corrected_tiles_seen.iter_mut() {
            *count = (*count).min(4);
        }
        
        // 计算 tiles_in_wall = 4 - tiles_seen
        // 注意：如果 tiles_seen 不完整（缺少其他玩家的手牌），tiles_in_wall 会偏高
        let mut tiles_in_wall = corrected_tiles_seen;
        tiles_in_wall.iter_mut().for_each(|v| *v = 4u8.saturating_sub(*v));
        
        // 基础规则验证：tiles_in_wall 的总和必须等于 tiles_left
        // 如果 tiles_seen 不完整（缺少其他玩家的手牌），计算出的 tiles_in_wall 总和会超过 tiles_left
        // 此时需要按比例缩放 tiles_in_wall 以匹配 tiles_left
        let calculated_sum: u8 = tiles_in_wall.iter().sum();
        if calculated_sum != tiles_left {
            // tiles_seen 不完整，需要修正 tiles_in_wall
            // 按比例缩放每个 tile 类型的 tiles_in_wall，使总和等于 tiles_left
            // 同时确保每个值不超过 4（每种 tile 最多 4 张）
            if calculated_sum > 0 && calculated_sum > tiles_left {
                // tiles_in_wall 总和过高，需要缩小
                let scale_factor = tiles_left as f32 / calculated_sum as f32;
                for count in tiles_in_wall.iter_mut() {
                    *count = ((*count as f32 * scale_factor).round() as u8).min(4);
                }
                // 由于浮点舍入和 min(4) 限制，总和可能不完全等于 tiles_left，需要微调
                let adjusted_sum: u8 = tiles_in_wall.iter().sum();
                if adjusted_sum != tiles_left {
                    let diff = tiles_left as i16 - adjusted_sum as i16;
                    // 将差值分配到 tiles_in_wall 中值最大的几个位置（但不超过 4）
                    let mut remaining_diff = diff;
                    let mut indices: Vec<usize> = (0..27).collect();
                    indices.sort_by_key(|&i| std::cmp::Reverse(tiles_in_wall[i]));
                    for &i in indices.iter() {
                        if remaining_diff == 0 {
                            break;
                        }
                        if remaining_diff > 0 && tiles_in_wall[i] < 4 {
                            tiles_in_wall[i] += 1;
                            remaining_diff -= 1;
                        } else if remaining_diff < 0 && tiles_in_wall[i] > 0 {
                            tiles_in_wall[i] -= 1;
                            remaining_diff += 1;
                        }
                    }
                }
            } else if calculated_sum < tiles_left {
                // tiles_in_wall 总和过低，需要增加
                // 这种情况不应该发生（因为 tiles_seen 不完整会导致 tiles_in_wall 偏高）
                // 但如果发生了，按比例增加（但不超过 4）
                let scale_factor = tiles_left as f32 / calculated_sum as f32;
                for count in tiles_in_wall.iter_mut() {
                    *count = ((*count as f32 * scale_factor).round() as u8).min(4);
                }
                // 微调：确保总和完全匹配 tiles_left
                let adjusted_sum: u8 = tiles_in_wall.iter().sum();
                if adjusted_sum != tiles_left {
                    let mut diff = tiles_left as i16 - adjusted_sum as i16;
                    // 循环调整直到完全匹配
                    let mut max_iterations = 100; // 防止无限循环
                    while diff != 0 && max_iterations > 0 {
                        max_iterations -= 1;
                        let mut indices: Vec<usize> = (0..27).collect();
                        if diff > 0 {
                            // 需要增加：优先从值小的位置开始（更容易增加）
                            indices.sort_by_key(|&i| tiles_in_wall[i]);
                            for &i in indices.iter() {
                                if diff <= 0 {
                                    break;
                                }
                                if tiles_in_wall[i] < 4 {
                                    let can_add = (4 - tiles_in_wall[i]).min(diff as u8);
                                    tiles_in_wall[i] += can_add;
                                    diff -= can_add as i16;
                                }
                            }
                        } else if diff < 0 {
                            // 需要减少：优先从值大的位置开始
                            indices.sort_by_key(|&i| std::cmp::Reverse(tiles_in_wall[i]));
                            for &i in indices.iter() {
                                if diff >= 0 {
                                    break;
                                }
                                if tiles_in_wall[i] > 0 {
                                    let can_sub = tiles_in_wall[i].min((-diff) as u8);
                                    tiles_in_wall[i] -= can_sub;
                                    diff += can_sub as i16;
                                }
                            }
                        }
                        // 检查是否真的改变了
                        let new_sum: u8 = tiles_in_wall.iter().sum();
                        let new_diff = tiles_left as i16 - new_sum as i16;
                        if new_diff == diff {
                            // 无法再调整，强制调整一个位置
                            if diff > 0 {
                                // 找到值最小的位置强制增加（可能超过4，后面会修正）
                                let min_idx = (0..27)
                                    .min_by_key(|&i| tiles_in_wall[i])
                                    .unwrap_or(0);
                                tiles_in_wall[min_idx] = (tiles_in_wall[min_idx] + diff as u8).min(255);
                                diff = 0;
                            } else if diff < 0 {
                                // 找到值最大的位置强制减少
                                let max_idx = (0..27)
                                    .max_by_key(|&i| tiles_in_wall[i])
                                    .unwrap_or(0);
                                let can_sub = tiles_in_wall[max_idx].min((-diff) as u8);
                                tiles_in_wall[max_idx] -= can_sub;
                                diff += can_sub as i16;
                            }
                        } else {
                            diff = new_diff;
                        }
                    }
                    // 修正任何超过4的值
                    for count in tiles_in_wall.iter_mut() {
                        *count = (*count).min(4);
                    }
                }
            }
        }
        
        // 最终验证：每个 tiles_in_wall 值不应该超过 4，总和必须等于 tiles_left
        for (i, &count) in tiles_in_wall.iter().enumerate() {
            assert!(
                count <= 4,
                "tiles_in_wall[{}] = {} exceeds maximum 4. This indicates a fundamental bug. tiles_in_wall: {:?}, tiles_seen: {:?}, corrected_tiles_seen: {:?}",
                i,
                count,
                tiles_in_wall,
                tiles_seen,
                corrected_tiles_seen
            );
        }
        let final_sum: u8 = tiles_in_wall.iter().sum();
        assert!(
            final_sum == tiles_left,
            "After correction, sum_left_tiles() = {} != tiles_left = {}. This indicates a fundamental bug in tiles_in_wall calculation. tiles_in_wall: {:?}, tiles_seen: {:?}, corrected_tiles_seen: {:?}",
            final_sum,
            tiles_left,
            tiles_in_wall,
            tiles_seen,
            corrected_tiles_seen
        );
        
        Self {
            tehai,
            tiles_in_wall,
            n_extra_tsumo: 0,
        }
    }
}

impl State {
    pub(super) const fn discard(&mut self, tile: Tile) {
        self.tehai[tile.as_usize()] -= 1;
    }

    pub(super) const fn undo_discard(&mut self, tile: Tile) {
        self.tehai[tile.as_usize()] += 1;
    }

    pub(super) const fn deal(&mut self, tile: Tile) {
        self.tiles_in_wall[tile.as_usize()] -= 1;
        self.undo_discard(tile);
    }

    pub(super) const fn undo_deal(&mut self, tile: Tile) {
        self.discard(tile);
        self.tiles_in_wall[tile.as_usize()] += 1;
    }

    pub(super) fn get_discard_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DiscardTile; 14]> {
        let mut discard_tiles = ArrayVec::default();

        let mut tehai = self.tehai;
        for tid in 0..27 {
            if tehai[tid] == 0 {
                continue;
            }

            tehai[tid] -= 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] += 1;

            let shanten_diff = shanten_after - shanten;

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
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] -= 1;

            let shanten_diff = shanten_after - shanten;

            let tile = must_tile!(tid);
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
        let sum: u8 = self.tiles_in_wall.iter().sum();
        // 血战到底基础规则：初始108张牌，发牌后剩余56张
        // 如果计算出的值超过56，说明 tiles_in_wall 的计算有严重错误，必须panic
        // 注意：这个检查在 From<InitState> 中已经通过 tiles_left 验证和修正，这里只是双重检查
        assert!(
            sum <= 56,
            "sum_left_tiles() = {} exceeds maximum 56. This indicates a fundamental bug in tiles_in_wall calculation. tiles_in_wall: {:?}",
            sum,
            self.tiles_in_wall
        );
        sum
    }
}
