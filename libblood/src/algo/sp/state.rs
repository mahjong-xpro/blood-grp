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
        // 1. tiles_in_wall = 4 - clamp(tiles_seen, 0, 4)
        let mut tiles_in_wall = [0u8; 27];
        for i in 0..27 {
            tiles_in_wall[i] = 4u8.saturating_sub(tiles_seen[i].min(4));
        }

        // 2. 修正总和使其等于 tiles_left
        adjust_wall_sum(&mut tiles_in_wall, tiles_left);

        // 3. 不变量检查
        debug_assert!(
            tiles_in_wall.iter().all(|&c| c <= 4),
            "tiles_in_wall has value > 4: {:?}",
            tiles_in_wall,
        );
        debug_assert!(
            tiles_in_wall.iter().map(|&c| c as u16).sum::<u16>() == tiles_left as u16,
            "tiles_in_wall sum {} != tiles_left {}",
            tiles_in_wall.iter().map(|&c| c as u16).sum::<u16>(),
            tiles_left,
        );

        Self { tehai, tiles_in_wall, n_extra_tsumo: 0 }
    }
}

/// tiles_in_wall 总和修正为 `target`。
///
/// tiles_seen 不完整时（缺少其他玩家手牌），`sum(tiles_in_wall)` 会偏高。
/// 贪心策略：差值 > 0 时从最大值的位置减 1，差值 < 0 时从最小非 4 位置加 1。
/// 每步保证 0 ≤ tiles_in_wall[i] ≤ 4，单遍即可收敛。
fn adjust_wall_sum(wall: &mut [u8; 27], target: u8) {
    let current: u16 = wall.iter().map(|&c| c as u16).sum();
    let target16 = target as u16;
    if current == target16 {
        return;
    }

    // 按 wall[i] 降序排列索引（稳定排序，保证确定性）
    let mut idx: [usize; 27] = core::array::from_fn(|i| i);
    idx.sort_by(|&a, &b| wall[b].cmp(&wall[a]).then(a.cmp(&b)));

    if current > target16 {
        // 需要减少：从最大值的位置逐 1 减
        let mut excess = (current - target16) as u8;
        // 多遍循环，每遍最多减 1/位，保证均匀分散
        while excess > 0 {
            for &i in &idx {
                if excess == 0 { break; }
                if wall[i] > 0 {
                    wall[i] -= 1;
                    excess -= 1;
                }
            }
        }
    } else {
        // 需要增加（罕见）：从最小非 4 位置逐 1 加
        let mut deficit = (target16 - current) as u8;
        idx.reverse(); // 升序
        while deficit > 0 {
            for &i in &idx {
                if deficit == 0 { break; }
                if wall[i] < 4 {
                    wall[i] += 1;
                    deficit -= 1;
                }
            }
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
        ding_que: Option<crate::mjai::Suit>,
    ) -> ArrayVec<[DiscardTile; 14]> {
        let mut discard_tiles = ArrayVec::default();

        let mut tehai = self.tehai;

        let mut ding_que_filter = false;
        let mut ding_que_range = 0..0;

        if let Some((start, end)) = crate::ding_que::ding_que_forced_range(&self.tehai, ding_que) {
            ding_que_filter = true;
            ding_que_range = start..end;
        }

        for tid in 0..27 {
            if tehai[tid] == 0 {
                continue;
            }

            // Enforce Ding Que rule
            if ding_que_filter && !ding_que_range.contains(&tid) {
                continue;
            }

            tehai[tid] -= 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3, ding_que);
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
        ding_que: Option<crate::mjai::Suit>,
    ) -> ArrayVec<[DrawTile; 27]> {
        let mut draw_tiles = ArrayVec::default();

        let mut tehai = self.tehai;
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3, ding_que);
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

    pub(super) fn get_required_tiles(&self, tehai_len_div3: u8, ding_que: Option<crate::mjai::Suit>) -> ArrayVec<[RequiredTile; 27]> {
        let mut tehai = self.tehai;

        let shanten = CALC_SHANTEN_FN(&tehai, tehai_len_div3, ding_que);
        let mut required_tiles = ArrayVec::default();

        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3, ding_que);
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
