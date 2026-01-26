use super::{PlayerState, SinglePlayerTables};
use crate::algo::agari::AgariCalculator;
use crate::algo::point::Point;
use crate::algo::shanten;
use crate::algo::sp::{InitState, SPCalculator};
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::tuz; // Used in yaokyuu_kind_count

use anyhow::{Context, Result, ensure};

/// Compute global tiles_seen from all players' states.
/// This counts ALL tiles that are out of the wall:
/// - All players' private hands (tehai)
/// - All discarded tiles (kawa_overview)
/// - All melded tiles (fuuro_overview)
/// - All concealed kans (ankan_overview)
///
/// This is the accurate global count needed for SPCalculator's tiles_in_wall calculation.
pub fn compute_global_tiles_seen(player_states: &[PlayerState; 4]) -> [u8; 27] {
    let mut global_tiles_seen = [0u8; 27];
    
    // Count all players' private hands
    for player_state in player_states.iter() {
        for (tid, &count) in player_state.tehai.iter().enumerate() {
            global_tiles_seen[tid] += count;
        }
    }
    
    // Count all discarded tiles (from kawa_overview)
    for player_state in player_states.iter() {
        for &tile in player_state.kawa_overview.iter().flatten() {
            global_tiles_seen[tile.as_usize()] += 1;
        }
    }
    
    // Count all melded tiles (from fuuro_overview)
    for player_state in player_states.iter() {
        for meld in player_state.fuuro_overview.iter().flatten() {
            for &tile in meld.iter() {
                global_tiles_seen[tile.as_usize()] += 1;
            }
        }
    }
    
    // Count all concealed kans (from ankan_overview)
    for player_state in player_states.iter() {
        for &tile in player_state.ankan_overview.iter().flatten() {
            // Ankan uses 4 tiles of the same type
            global_tiles_seen[tile.as_usize()] += 4;
        }
    }
    
    global_tiles_seen
}

impl PlayerState {
    /// Compute a partial global tiles_seen from this player's perspective.
    /// This includes:
    /// - This player's private hand (tehai)
    /// - All discarded tiles (from kawa_overview - all players)
    /// - All melded tiles (from fuuro_overview - all players)
    /// - All concealed kans (from ankan_overview - all players)
    /// 
    /// Note: This is incomplete because it doesn't include other players' private hands.
    /// For accurate calculations, use `compute_global_tiles_seen` with all PlayerStates.
    /// 
    /// Also note: This function counts tiles from scratch, so it should not exceed 4 per tile type.
    /// If it does, it indicates a bug in the game state (e.g., duplicate tiles in kawa_overview).
    fn compute_partial_global_tiles_seen(&self) -> [u8; 27] {
        let mut global_tiles_seen = [0u8; 27];
        
        // Count this player's private hand
        for (tid, &count) in self.tehai.iter().enumerate() {
            global_tiles_seen[tid] += count;
        }
        
        // Count all discarded tiles (from kawa_overview - all players)
        for kawa in self.kawa_overview.iter() {
            for &tile in kawa.iter() {
                let tid = tile.as_usize();
                global_tiles_seen[tid] += 1;
                // 基础规则验证：每种 tile 最多只有 4 张
                // 如果超过，说明游戏状态有误（可能是重复计算或数据损坏）
                if global_tiles_seen[tid] > 4 {
                    // 限制为 4，避免后续计算错误
                    global_tiles_seen[tid] = 4;
                }
            }
        }
        
        // Count all melded tiles (from fuuro_overview - all players)
        for meld_group in self.fuuro_overview.iter() {
            for meld in meld_group.iter() {
                for &tile in meld.iter() {
                    let tid = tile.as_usize();
                    global_tiles_seen[tid] += 1;
                    if global_tiles_seen[tid] > 4 {
                        global_tiles_seen[tid] = 4;
                    }
                }
            }
        }
        
        // Count all concealed kans (from ankan_overview - all players)
        for ankan_group in self.ankan_overview.iter() {
            for &tile in ankan_group.iter() {
                let tid = tile.as_usize();
                // Ankan uses 4 tiles of the same type
                global_tiles_seen[tid] += 4;
                if global_tiles_seen[tid] > 4 {
                    global_tiles_seen[tid] = 4;
                }
            }
        }
        
        global_tiles_seen
    }

    /// Used by `BoardState` to check if a player is making 4 kans on his own.
    #[inline]
    #[must_use]
    pub fn kans_count(&self) -> usize {
        self.minkans.len() + self.ankans.len()
    }

    /// Used by `Agent` impls, must be called at 3n+2.
    #[must_use]
    pub fn discard_candidates(&self) -> [bool; 27] {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        let mut ret = [false; 27];

        if let Some(ding_que_suit) = self.ding_que {
            let ding_que_suit_id = match ding_que_suit {
                crate::mjai::Suit::Man => 0,
                crate::mjai::Suit::Pin => 1,
                crate::mjai::Suit::Sou => 2,
            };
            let ding_que_start = ding_que_suit_id * 9;
            let ding_que_end = ding_que_start + 9;
            
            // Check if hand still has any ding_que suit tiles
            let has_ding_que_tiles = (ding_que_start..ding_que_end)
                .any(|i| self.tehai[i] > 0);
            
            if has_ding_que_tiles {
                // Must discard ding_que suit tiles first - only allow ding_que suit tiles
                for i in ding_que_start..ding_que_end {
                    if self.tehai[i] > 0 && !self.forbidden_tiles[i] {
                        ret[i] = true;
                    }
                }
                // 如果所有定缺花色牌都被标记为forbidden_tiles，那么允许丢弃所有非forbidden_tiles的牌
                // 这是为了避免游戏卡死（虽然这种情况理论上不应该发生）
                if ret.iter().all(|&x| !x) {
                    // 所有定缺花色牌都被标记为forbidden_tiles，允许丢弃所有非forbidden_tiles的牌
                    for (i, count) in self.tehai.iter().copied().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        ret[i] = !self.forbidden_tiles[i];
                    }
                }
                return ret;
            }
        }

        for (i, count) in self.tehai.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }

            ret[i] = !self.forbidden_tiles[i];
        }

        ret
    }

    /// Must be called at 3n+2.
    ///
    /// The return value indicates the tiles which can make the hand tenpai for
    /// real after being discarded, with the number of future tenpai tiles left
    /// and forbidden win conditions considered, without depending on any incidental yaku, and is
    #[must_use]
    pub fn discard_candidates_with_unconditional_tenpai(&self) -> [bool; 27] {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        let mut ret = [false; 27];

        if self.tiles_left == 0 // last tile
            || self.shanten > 1 // impossible to discard-to-tenpai
            || self.shanten == 1 && !self.has_next_shanten_discard
        {
            return ret;
        }

        if let Some(last_self_tsumo) = self.last_self_tsumo {
            if self.waits[last_self_tsumo.as_usize()] {
                // already agari and any discard will result in forbidden win
                return ret;
            }
            // All valid waits can agari
            ret[last_self_tsumo.as_usize()] = true;
            return ret;
        } else if shanten::calc_all(&self.tehai, self.tehai_len_div3) == -1 {
            // Ditto but for discard after pon (Bloody Battle Mahjong has no chi)
            return ret;
        }

        let tenpai_discards = if self.shanten == 1 {
            self.next_shanten_discards
        } else {
            self.keep_shanten_discards
        };

        // Replace and test
        tenpai_discards
            .iter()
            .copied()
            .enumerate()
            .filter(|&(tid, b)| b && !self.forbidden_tiles[tid])
            .for_each(|(discard, _)| {
                let mut tehai_3n1 = self.tehai;
                tehai_3n1[discard] -= 1;

                for (tsumo, seen) in self.tiles_seen.iter().copied().enumerate() {
                    if tsumo == discard || tehai_3n1[tsumo] == 4 {
                        continue;
                    }

                    let mut tehai_3n2 = tehai_3n1;
                    tehai_3n2[tsumo] += 1;
                    if shanten::calc_all(&tehai_3n2, self.tehai_len_div3) > -1 {
                        continue;
                    }

                    if seen == 4 || ret[discard] {
                        continue;
                    }

                    // Validate tehai total: should be 14 - 3 * fuuro_count for agari
                    // AGARI_TABLE only contains valid agari structures with exactly 14 tiles in tehai
                    let fuuro_count = self.pons.len() + self.minkans.len() + self.ankans.len();
                    let expected_tehai_total = 14 - (fuuro_count as u8 * 3);
                    let tehai_total: u8 = tehai_3n2.iter().sum();
                    if tehai_total != expected_tehai_total {
                        // This is not a valid agari state - skip it
                        // This can happen when the hand structure is invalid or when
                        // the calculation is in an intermediate state
                        continue;
                    }

                    let agari_calc = AgariCalculator {
                        tehai: &tehai_3n2,

                        exclude_gen_tile: None,
                        pons: &self.pons,
                        minkans: &self.minkans,
                        ankans: &self.ankans,
                        winning_tile: tsumo as u8,
                        is_ron: true,
                        ding_que: self.ding_que,
                        is_after_kan: false,
                        is_kan_discard: false,
                        is_chankan: false,
                        is_haidi: false,
                        is_tianhu: false,
                        is_dihu: false,
                    };
                    ret[discard] = agari_calc.has_yaku();
                }
            });


        ret
    }

    #[inline]
    #[must_use]
    pub fn yaokyuu_kind_count(&self) -> u8 {
        tuz![1m, 9m, 1p, 9p, 1s, 9s]
            .iter()
            .map(|&i| self.tehai[i].min(1))
            .sum()
    }

    #[inline]
    #[must_use]
    pub fn rule_based_ryukyoku(&self) -> bool {
        return false;
        // self.rule_based_ryukyoku_slow()
    }

    fn rule_based_ryukyoku_slow(&self) -> bool {
        // Do not ryukyoku if the hand is already <= 2 shanten.
        if shanten::calc_all(&self.tehai, self.tehai_len_div3) <= 2 {
            return false;
        }

        // (This logic may need adjustment based on actual game flow)

        // Simplified logic: allow ryukyoku if we are oya or we are not the last
        // Ryukyoku if we are oya or we are not the last,
        // because it is hard to decide whether it is appropriate to not
        // ryukyoku.
        if self.oya == 0 || self.rank < 3 {
            return true;
        }

        // At all-last, we are the last and we are not oya. If even a
        // haneman tsumo cannot let us avoid the last, then do not ryukyoku.
        let mut scores = [-3000; 4];
        scores[0] = 12000;
        scores[self.oya as usize] = -6000;
        vec_add_assign(&mut scores, &self.scores);
        self.get_rank(scores) < 3
    }

    #[inline]
    #[must_use]
    pub fn rule_based_agari(&self) -> bool {
        if !self.last_cans.can_agari() {
            return false;
        }
        self.rule_based_agari_slow(
            self.last_cans.can_ron_agari,
            self.rel(self.last_cans.target_actor),
        )
    }

    fn rule_based_agari_slow(&self, is_ron: bool, target_rel: usize) -> bool {
        // Agari if we are oya ourselves, or we are not the last place at all.
        if self.oya == 0 || self.rank < 3 {
            return true;
        }

        // (This logic may need adjustment based on actual game flow)
        if self.scores.iter().all(|&s| s < 30000) {
            // Simplified agari condition
            return true;
        }

        // Calculate the max theoretical score we can achieve through this agari.
        let max_win_point = {
            let mut tehai_full = self.tehai;
            for t in &self.ankan_overview[0] {
                tehai_full[t.as_usize()] += 4;
            }

            // Just calculate agari points directly
            // TODO: This is a simplified calculation, may need improvement
            // Note: This is used for SPCalculator's expected value calculation
            // The simplified version should be sufficient for decision-making
            // We pass is_haidi = false, is_tianhu = false, is_dihu = false here for simulation estimation
            self.agari_points(is_ron, false, false, false, &[]).unwrap()
        };

        // Calculate the best post-hora situation for us.
        let mut exp_scores = self.scores;
        if is_ron {
            exp_scores[0] += max_win_point.ron;
            exp_scores[target_rel] -= max_win_point.ron;
        } else {
            let tsumo_total = max_win_point.tsumo_total(false);
            exp_scores[0] += tsumo_total;
            exp_scores
                .iter_mut()
                .enumerate()
                .skip(1)
                .for_each(|(_idx, s)| {
                    *s -= max_win_point.tsumo_ko;
                });
        }

        //
        // Agari if 西入 or keeping 西入 is possible. This condition is sound
        // and complete.
        if exp_scores.iter().all(|&s| s < 30000) {
            return true;
        }

        // Agari if the best post-hora situation in theory will make us avoid
        // taking the last place.
        self.get_rank(exp_scores) < 3
    }

    /// Err is returned if the hand cannot agari, or cannot retrieve the winning
    /// tile.
    ///
    /// This function should be called immediately, otherwise the state may
    /// change.
    /// 
    pub fn agari_points(&self, is_ron: bool, is_haidi: bool, is_tianhu: bool, is_dihu: bool, _ura_indicators: &[Tile]) -> Result<Point> {
        ensure!(
            is_ron && self.last_cans.can_ron_agari || self.last_cans.can_tsumo_agari,
            "cannot agari"
        );

        let winning_tile = if is_ron {
            self.last_kawa_tile
        } else {
            self.last_self_tsumo
        }
        .context("cannot find the winning tile")?;

        // Add winning tile to tehai for agari calculation
        let mut tehai = self.tehai;
        if is_ron {
            let tid = winning_tile.as_usize();
            tehai[tid] += 1;
        }

        // is_after_kan: true if tsumo and at_rinshan (杠上花)
        // is_kan_discard: true if ron and the discarded tile was after a kan (杠上炮)
        // is_chankan: true if this is chankan (抢杠) - ron on kakan
        // Note: 
        //   - 抢杠、杠上花、杠上炮是不同的：
        //     * 抢杠：在别人加杠时抢杠和牌，+1番（平胡1番 + 抢杠1番 = 2番）
        //     * 杠上花：杠牌后摸牌自摸，+1番（自摸1番 + 平胡1番 + 杠上花1番 = 3番）
        //     * 杠上炮：杠牌后打出的牌和牌，+1番（平胡1番 + 杠上炮1番 = 2番）
        //   - chankan is detected via chankan_chance (set in update.rs::kakan())
        //   - kan_discard for dahai is detected via last_discard_was_after_kan (set in update.rs::dahai())
        //   - 抢杠时，被抢杠的玩家的根不应该计算（因为加杠的牌被抢走了）
        let is_chankan = is_ron && self.chankan_chance.is_some();
        let is_kan_discard_from_dahai = is_ron && self.last_discard_was_after_kan && !is_chankan;
        // For chankan, exclude the kakan tile from gen count for the kakan player
        // But this is for the winning player, so exclude_gen_tile is None
        // The kakan player's gen exclusion is handled separately in handle_hora
        let agari_calc = AgariCalculator {
            tehai: &tehai,

            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,
            winning_tile: winning_tile.as_u8(),
            is_ron,
            ding_que: self.ding_que,
            is_after_kan: !is_ron && self.at_rinshan, // 杠上花：自摸且从岭上牌摸的
            is_kan_discard: is_kan_discard_from_dahai, // 杠上炮：杠后打出的牌（不包括抢杠）
            is_chankan, // 抢杠：在别人加杠时抢杠和牌
            exclude_gen_tile: None, // For winning player, no exclusion needed
            is_haidi,
            is_tianhu,
            is_dihu,
        };
        let agari = agari_calc
            .agari()
            .context("not a hora hand")?;

        Ok(agari.point(false))
    }

    /// Calculate agari points excluding gen for a specific tile (for chankan)
    /// This is used when calculating the payment amount for the kakan player in chankan
    pub fn agari_points_exclude_gen(&self, is_ron: bool, exclude_tile: u8, is_haidi: bool, _ura_indicators: &[Tile]) -> Result<Point> {
        ensure!(
            is_ron && self.last_cans.can_ron_agari || self.last_cans.can_tsumo_agari,
            "cannot agari"
        );

        let winning_tile = if is_ron {
            self.last_kawa_tile
        } else {
            self.last_self_tsumo
        }
        .context("cannot find the winning tile")?;

        // Add winning tile to tehai for agari calculation
        let mut tehai = self.tehai;
        if is_ron {
            let tid = winning_tile.as_usize();
            tehai[tid] += 1;
        }
        
        // Validate tehai total (should be 14 tiles for agari)
        let tehai_total: u8 = tehai.iter().sum();
        ensure!(
            tehai_total == 14,
            "tehai total should be 14 for agari, but got {} (is_ron: {}, tehai_len_div3: {})",
            tehai_total,
            is_ron,
            self.tehai_len_div3
        );

        let is_chankan = is_ron && self.chankan_chance.is_some();
        let is_kan_discard_from_dahai = is_ron && self.last_discard_was_after_kan && !is_chankan;
        let agari_calc = AgariCalculator {
            tehai: &tehai,

            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,
            winning_tile: winning_tile.as_u8(),
            is_ron,
            ding_que: self.ding_que,
            is_after_kan: !is_ron && self.at_rinshan,
            is_kan_discard: is_kan_discard_from_dahai,
            is_chankan,
            exclude_gen_tile: Some(exclude_tile), // Exclude this tile from gen count
            is_haidi,
            is_tianhu: false, // Chankan cannot be TianHu
            is_dihu: false, // Chankan cannot be DiHu (DiHu is on first discard, Chankan is on Kan)
        };
        let agari = agari_calc
            .agari()
            .context("not a hora hand")?;

        Ok(agari.point(false))
    }

    /// Calculate the actual shanten at this point. Unlike `self.shanten`, this
    /// function properly calculates the shanten at 3n+2, which follows the
    /// definition of shanten most people acknowledge.
    pub fn real_time_shanten(&self) -> i8 {
        if !self.last_cans.can_discard {
            // 3n+1, `self.shanten` is accurate.
            return self.shanten;
        }

        if self.shanten > 0 {
            // 3n+2, not tenpai, shanten is `self.shanten - 1` if there is any
            // discard that can decrease the shanten number.
            return if self.has_next_shanten_discard {
                self.shanten - 1
            } else {
                self.shanten
            };
        }

        if let Some(tile) = self.last_self_tsumo {
            // 3n+2, tenpai after tsumo.
            return if self.waits[tile.as_usize()] {
                -1
            } else {
                0
            };
        }

        // 3n+2, tenpai after pon. `self.shanten` is 0, but the actual
        // shanten could be 0 or -1.
        //
        // At 223m 55p 45s, `self.shanten` is 1. After pon, `self.shanten`
        // becomes 0 because `update_shanten` is always called after a pon
        // event. The actual shanten is 0 as well.
        //
        // At 123m 55p 45s, `self.shanten` is 0. After pon, `self.shanten`
        // becomes 0 because `update_shanten` clamps the value to be >= 0. The
        // actual shanten is -1.
        // Note: Bloody Battle Mahjong has no chi (吃牌)
        shanten::calc_all(&self.tehai, self.tehai_len_div3)
    }

    /// Can be called at both 3n+1 and 3n+2, but `self.real_time_shanten` must
    /// be >= 0 and `self.tiles_left` must be >= 4.
    ///
    /// This function is currently highly internal.
    ///
    /// # Arguments
    /// * `global_tiles_seen` - A global count of all tiles that are out of the wall.
    ///   This should include all players' hands, all discards, all melds, and all kans.
    ///   If `None`, falls back to `self.tiles_seen` (per-player perspective, which is
    ///   incomplete and may cause incorrect calculations).
    pub(super) fn single_player_tables(&self, global_tiles_seen: Option<[u8; 27]>) -> Result<SinglePlayerTables> {
        ensure!(self.tiles_left >= 4, "need at least one more tsumo");

        let cur_shanten = self.real_time_shanten();
        ensure!(cur_shanten >= 0, "can't calculate an agari hand");

        let can_discard = self.last_cans.can_discard;
        let tsumos_left = if can_discard {
            self.tiles_left / 4
        } else {
            let target = self.rel(self.last_cans.target_actor) as u8;
            // Let's just ignore chankan here.
            let tiles_left_at_next_tsumo = self.tiles_left.saturating_sub(4 - target);
            tiles_left_at_next_tsumo / 4
        };
        ensure!(tsumos_left >= 1, "need at least one more tsumo");

        let tehai = self.tehai;

        // Use global_tiles_seen if provided, otherwise compute partial global tiles_seen
        // from this player's perspective (includes all public info but missing other players' private hands)
        // Note: Partial global tiles_seen is still incomplete but better than per-player tiles_seen
        let tiles_seen = global_tiles_seen.unwrap_or_else(|| self.compute_partial_global_tiles_seen());

        let init_state = InitState {
            tehai,
            tiles_seen,
            tiles_left: self.tiles_left,
        };
        let sp_calc = SPCalculator {
            tehai_len_div3: self.tehai_len_div3,
            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,

            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: false,
            calc_shanten_down: false,
            ding_que: self.ding_que,
        };

        let max_ev_table = sp_calc.calc(init_state, can_discard, tsumos_left, cur_shanten)?;

        Ok(SinglePlayerTables { max_ev_table })
    }
}
