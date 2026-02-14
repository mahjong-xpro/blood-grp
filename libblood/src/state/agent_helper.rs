use super::{PlayerState, SinglePlayerTables};
use crate::algo::agari::AgariCalculator;
use crate::algo::point::Point;
use crate::algo::shanten;
use crate::algo::sp::{InitState, SPCalculator};
use crate::tile::Tile;
use crate::tuz; // Used in yaokyuu_kind_count

use anyhow::{Context, Result, ensure};

impl PlayerState {
    /// Compute a partial global tiles_seen from this player's perspective.
    /// This includes:
    /// - This player's private hand (tehai)
    /// - All discarded tiles (from kawa_overview - all players)
    /// - All melded tiles (from fuuro_overview - all players, excluding the called tile
    ///   which is already counted in kawa_overview of the discarding player)
    /// - All concealed kans (from ankan_overview - all players)
    /// 
    /// Note: This is incomplete because it doesn't include other players' private hands.
    fn compute_partial_global_tiles_seen(&self) -> [u8; 27] {
        let mut global_tiles_seen = [0u8; 27];
        
        // Count this player's private hand
        for (tid, &count) in self.tehai.iter().enumerate() {
            global_tiles_seen[tid] += count;
        }
        
        // Count all discarded tiles (from kawa_overview - all players)
        // Note: kawa_overview retains tiles even after they've been called by pon/daiminkan
        for kawa in self.kawa_overview.iter() {
            for &tile in kawa.iter() {
                let tid = tile.as_usize();
                global_tiles_seen[tid] = global_tiles_seen[tid].saturating_add(1).min(4);
            }
        }
        
        // Count melded tiles (from fuuro_overview - all players)
        // 碰(pon)/大明杠(daiminkan)中，被叫的那张牌已在 kawa_overview 中计数过。
        // fuuro_overview 包含完整的副露（碰 = [消耗1, 消耗2, 叫牌]，大明杠 = [消耗1, 消耗2, 消耗3, 叫牌]）。
        // 由于碰/杠内所有牌都是同一种（同 tile_id），跳过任意一张即可避免与 kawa 重复计数。
        // 加杠(kakan)会在碰的基础上 push 第4张（变为4张同种牌），同理跳过1张。
        for meld_group in self.fuuro_overview.iter() {
            for meld in meld_group.iter() {
                // 碰/杠中所有牌 tile_id 相同，skip(1) 跳过1张以抵消 kawa_overview 中的重复
                for &tile in meld.iter().skip(1) {
                    let tid = tile.as_usize();
                    global_tiles_seen[tid] = global_tiles_seen[tid].saturating_add(1).min(4);
                }
            }
        }
        
        // Count all concealed kans (from ankan_overview - all players)
        for ankan_group in self.ankan_overview.iter() {
            for &tile in ankan_group.iter() {
                let tid = tile.as_usize();
                // Ankan uses 4 tiles of the same type
                global_tiles_seen[tid] = global_tiles_seen[tid].saturating_add(4).min(4);
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

        if let Some((start, end)) = crate::ding_que::ding_que_forced_range(&self.tehai, self.ding_que) {
            // Must discard ding_que suit tiles first - only allow ding_que suit tiles
            for i in start..end {
                if self.tehai[i] > 0 && !self.forbidden_tiles[i] {
                    ret[i] = true;
                }
            }
            // 安全网 第一层：如果所有定缺花色牌都被 forbidden_tiles 阻止，
            // 回退到允许丢弃任何非 forbidden 的牌。
            // 触发场景：碰后禁打恰好覆盖了唯一的定缺牌（极罕见）。
            if ret.iter().all(|&x| !x) {
                for (i, &count) in self.tehai.iter().enumerate() {
                    if count > 0 {
                        ret[i] = !self.forbidden_tiles[i];
                    }
                }
            }
            // 安全网 第二层：如果 forbidden_tiles 阻止了手中所有牌
            // （碰后禁打 + 定缺约束完全覆盖），强制允许丢弃任何手中有的牌，
            // 避免 mask 全 false 导致 panic / 游戏死锁。
            if ret.iter().all(|&x| !x) {
                for (i, &count) in self.tehai.iter().enumerate() {
                    if count > 0 {
                        ret[i] = true;
                    }
                }
            }
            return ret;
        }

        for (i, &count) in self.tehai.iter().enumerate() {
            if count > 0 {
                ret[i] = !self.forbidden_tiles[i];
            }
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

        // shanten == 0 特有的快速路径：
        // - 自摸后：手牌已是听牌 (3n+2)，摸到的牌若在 waits 中则已和牌，
        //   否则摸切（打回摸到的牌）一定能恢复听牌。
        // - 碰后：直接检查是否已成牌。
        //
        // shanten == 1 时此快速路径不适用：摸切不一定能改善向听，
        // 需要走下面的完整计算（next_shanten_discards 逐牌验证）。
        if self.shanten == 0 {
            if let Some(last_self_tsumo) = self.last_self_tsumo {
                if self.waits[last_self_tsumo.as_usize()] {
                    // already agari and any discard will result in forbidden win
                    return ret;
                }
                // All valid waits can agari
                ret[last_self_tsumo.as_usize()] = true;
                return ret;
            } else {
                // `tehai_len_div3` can desync after kan/pon flows; derive it from tehai shape instead.
                let len_div3 = (self.tehai.iter().sum::<u8>() / 3) as u8;
                if shanten::calc_all(&self.tehai, len_div3, self.ding_que) == -1 {
                    // Ditto but for discard after pon (Bloody Battle Mahjong has no chi)
                    return ret;
                }
            }
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
                    // `tehai_len_div3` can desync after kan/pon flows; derive it from tehai shape instead.
                    let len_div3_3n2 = (tehai_3n2.iter().sum::<u8>() / 3) as u8;
                    if shanten::calc_all(&tehai_3n2, len_div3_3n2, self.ding_que) > -1 {
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
                        fan_config: self.fan_config,
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
        // In Bloody Battle, ryukyoku (流局) logic is simplified
        false
    }

    #[inline]
    #[must_use]
    pub fn rule_based_agari(&self) -> bool {
        // In Bloody Battle, we almost always want to agari if possible.
        self.last_cans.can_agari()
    }

    /// Err is returned if the hand cannot agari, or cannot retrieve the winning
    /// tile.
    ///
    /// This function should be called immediately, otherwise the state may
    /// change.
    /// 
    pub fn agari_points(&self, is_ron: bool, is_haidi: bool, is_tianhu: bool, is_dihu: bool, _ura_indicators: &[Tile]) -> Result<Point> {
        ensure!(
            if is_ron { self.last_cans.can_ron_agari } else { self.last_cans.can_tsumo_agari },
            "cannot agari: is_ron={}, can_ron={}, can_tsumo={}",
            is_ron, self.last_cans.can_ron_agari, self.last_cans.can_tsumo_agari
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
            fan_config: self.fan_config,
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
            if is_ron { self.last_cans.can_ron_agari } else { self.last_cans.can_tsumo_agari },
            "cannot agari: is_ron={}, can_ron={}, can_tsumo={}",
            is_ron, self.last_cans.can_ron_agari, self.last_cans.can_tsumo_agari
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
        
        // Validate tehai total: 14 - 3 × 副露数（碰/杠每组消耗3张手牌）
        let fuuro_count = (self.pons.len() + self.minkans.len() + self.ankans.len()) as u8;
        let expected_tehai_total = 14u8.saturating_sub(fuuro_count * 3);
        let tehai_total: u8 = tehai.iter().sum();
        ensure!(
            tehai_total == expected_tehai_total,
            "tehai total should be {} for agari (fuuro_count={}), but got {} (is_ron: {})",
            expected_tehai_total,
            fuuro_count,
            tehai_total,
            is_ron
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
            fan_config: self.fan_config,
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
        // `tehai_len_div3` can desync after kan/pon flows; derive it from tehai shape instead.
        let len_div3 = (self.tehai.iter().sum::<u8>() / 3) as u8;
        shanten::calc_all(&self.tehai, len_div3, self.ding_que)
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
        // Critical invariants:
        //
        // Hand count must match the decision window:
        // - can_discard=true  => 3n+2 (e.g. 14, 11, 8, 5, 2) — our turn, we have drawn.
        // - can_discard=false =>
        //   - Reaction (pon/daiminkan/ron): hand includes the tile we may call, so 3n+2 (14, 11, 8, 5, 2).
        //   - Waiting for draw: 3n+1 (13, 10, 7, 4, 1).
        //
        // If violated, downstream SP logic can panic (e.g. ArrayVec overflow).
        let tehai_sum: u8 = self.tehai.iter().sum();
        ensure!(
            tehai_sum <= 14,
            "SP invariant violation: concealed hand too large (tehai_sum={} > 14). \
             kyoku={}, turn={}, tiles_left={}, ding_que={:?}, cans={:?}",
            tehai_sum,
            self.kyoku,
            self.at_turn,
            self.tiles_left,
            self.ding_que,
            self.last_cans,
        );
        let reaction = self.last_cans.can_pon
            || self.last_cans.can_daiminkan
            || self.last_cans.can_ron_agari;
        if can_discard {
            ensure!(
                tehai_sum % 3 == 2,
                "SP invariant violation: can_discard=true but tehai_sum%3 != 2 (tehai_sum={}). \
                 kyoku={}, turn={}, tiles_left={}, ding_que={:?}, cans={:?}",
                tehai_sum,
                self.kyoku,
                self.at_turn,
                self.tiles_left,
                self.ding_que,
                self.last_cans,
            );
        } else if reaction {
            // Reaction phase: tehai may be 3n+1 (tile not yet in hand) or 3n+2 (tile already in hand).
            // Only reject impossible counts (3n, or >14).
            ensure!(
                tehai_sum % 3 != 0,
                "SP invariant violation: can_discard=false (reaction) but tehai_sum%3 == 0 (tehai_sum={}). \
                 kyoku={}, turn={}, tiles_left={}, ding_que={:?}, cans={:?}",
                tehai_sum,
                self.kyoku,
                self.at_turn,
                self.tiles_left,
                self.ding_que,
                self.last_cans,
            );
        } else {
            // Waiting for draw: 3n+1, at most 13.
            ensure!(
                tehai_sum % 3 == 1,
                "SP invariant violation: can_discard=false (waiting) but tehai_sum%3 != 1 (tehai_sum={}). \
                 kyoku={}, turn={}, tiles_left={}, ding_que={:?}, cans={:?}",
                tehai_sum,
                self.kyoku,
                self.at_turn,
                self.tiles_left,
                self.ding_que,
                self.last_cans,
            );
            ensure!(
                tehai_sum <= 13,
                "SP invariant violation: can_discard=false (waiting) but tehai_sum={} > 13. \
                 kyoku={}, turn={}, tiles_left={}, ding_que={:?}, cans={:?}",
                tehai_sum,
                self.kyoku,
                self.at_turn,
                self.tiles_left,
                self.ding_que,
                self.last_cans,
            );
        }
        // FIX: 血战到底中已和牌玩家不再摸牌，活跃摸牌者 = 自己 + 未和牌对手。
        // 之前始终 / 4 估算每人剩余摸牌次数，当 1-2 人已和牌时严重低估，
        // 导致 SP 计算认为获胜概率过低，AI 过于保守。
        let n_active_payers = (1..4).filter(|&i| !self.players_agari[i]).count() as u8;
        let active_players = (n_active_payers + 1).max(1); // self + active opponents

        // 近似值：tiles_left / active_players，误差 ≤1 次摸牌。
        // can_discard 和 reaction 两种情况下结果相同，无需区分。
        let tsumos_left = self.tiles_left / active_players;
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
        // `tehai_len_div3` can desync after kan/pon flows; derive it from tehai shape instead.
        let tehai_len_div3 = (self.tehai.iter().sum::<u8>() / 3) as u8;
        let sp_calc = SPCalculator {
            tehai_len_div3,
            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,

            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: false,
            calc_shanten_down: false,
            ding_que: self.ding_que,
            n_active_payers,
            fan_config: self.fan_config,
        };

        let max_ev_table = sp_calc.calc(init_state, can_discard, tsumos_left, cur_shanten)?;

        Ok(SinglePlayerTables { max_ev_table })
    }
}
