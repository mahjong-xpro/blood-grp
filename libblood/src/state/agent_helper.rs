use super::{PlayerState, SinglePlayerTables};
use crate::algo::agari::AgariCalculator;
use crate::algo::point::Point;
use crate::algo::shanten;
use crate::algo::sp::{InitState, SPCalculator};
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::tuz; // Used in yaokyuu_kind_count

use anyhow::{Context, Result, ensure};
// Bloody Battle: array_vec not used

impl PlayerState {
    /// Used by `BoardState` to check if a player is making 4 kans on his own.
    #[inline]
    #[must_use]
    pub fn kans_count(&self) -> usize {
        self.minkans.len() + self.ankans.len()
    }

    /// Used by `Agent` impls, must be called at 3n+2.
    #[must_use]
    pub fn discard_candidates(&self) -> [bool; 27] {
        // Bloody Battle: No akas, just return the aka version directly
        self.discard_candidates_aka()
    }

    /// Aka dora covered version of `discard_candidates`.
    #[must_use]
    pub fn discard_candidates_aka(&self) -> [bool; 27] {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        // Bloody Battle: 27 tile kinds (no jihai, no red 5s)
        let mut ret = [false; 27];

        // Bloody Battle: Check Ding Que rule - must discard ding_que suit tiles first if any remain
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
                // No other tiles can be discarded
                return ret;
            }
        }

        // Bloody Battle: No riichi (立直)
        for (i, count) in self.tehai.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }

            ret[i] = !self.forbidden_tiles[i];
        }

        // Bloody Battle: No akas
        ret
    }

    /// Must be called at 3n+2.
    ///
    /// The return value indicates the tiles which can make the hand tenpai for
    /// real after being discarded, with the number of future tenpai tiles left
    /// and furiten considered, without depending on any incidental yaku, and is
    /// not affected by the riichi (立直) status of the player (Bloody Battle has no riichi).
    #[must_use]
    pub fn discard_candidates_with_unconditional_tenpai(&self) -> [bool; 27] {
        // Bloody Battle: No akas, just return the aka version directly
        self.discard_candidates_with_unconditional_tenpai_aka()
    }

    /// Aka dora covered version of `discard_candidates_with_unconditional_tenpai`.
    #[must_use]
    pub fn discard_candidates_with_unconditional_tenpai_aka(&self) -> [bool; 27] {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        // Bloody Battle: 27 tile kinds (no jihai, no red 5s)
        let mut ret = [false; 27];

        if self.tiles_left == 0 // haitei
            || self.shanten > 1 // impossible to discard-to-tenpai
            || self.shanten == 1 && !self.has_next_shanten_discard
        {
            return ret;
        }

        if let Some(last_self_tsumo) = self.last_self_tsumo {
            if self.waits[last_self_tsumo.deaka().as_usize()] {
                // already agari and any discard will result in furiten
                return ret;
            }
            // Bloody Battle: No riichi (立直) or furiten (振听)
            // All valid waits can agari
            ret[last_self_tsumo.as_usize()] = true;
            return ret;
        } else if shanten::calc_all(&self.tehai, self.tehai_len_div3) == -1 {
            // Ditto but for discard after chi/pon
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

                    // Furiten
                    if self.discarded_tiles[tsumo] {
                        ret[discard] = false;
                        break;
                    }

                    // Must be placed after the furiten check above
                    if seen == 4 || ret[discard] {
                        continue;
                    }

                    let agari_calc = AgariCalculator {
                        tehai: &tehai_3n2,
                        is_menzen: self.is_menzen,
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
                    };
                    ret[discard] = agari_calc.has_yaku();
                }
            });

        // Bloody Battle: No akas

        ret
    }

    #[inline]
    #[must_use]
    pub fn yaokyuu_kind_count(&self) -> u8 {
        // Bloody Battle: Only 1m, 9m, 1p, 9p, 1s, 9s (no jihai)
        tuz![1m, 9m, 1p, 9p, 1s, 9s]
            .iter()
            .map(|&i| self.tehai[i].min(1))
            .sum()
    }

    #[inline]
    #[must_use]
    pub fn rule_based_ryukyoku(&self) -> bool {
        if !self.last_cans.can_ryukyoku {
            return false;
        }
        self.rule_based_ryukyoku_slow()
    }

    fn rule_based_ryukyoku_slow(&self) -> bool {
        // Do not ryukyoku if the hand is already <= 2 shanten.
        if shanten::calc_all(&self.tehai, self.tehai_len_div3) <= 2 {
            return false;
        }

        // Bloody Battle: No bakaze, simplified ryukyoku logic
        // (This logic may need adjustment based on actual game flow)

        // Bloody Battle: No all_last concept (game ends when 3 players win or draw)
        // Simplified logic: allow ryukyoku if we are oya or we are not the last
        {
            // Ryukyoku if we are oya or we are not the last,
            // because it is hard to decide whether it is appropriate to not
            // ryukyoku.
            if self.oya == 0 || self.rank < 3 {
                return true;
            }

            // At all-last, we are the last and we are not oya. If even a
            // haneman tsumo cannot let us avoid the last, then do not ryukyoku.
            // Bloody Battle: No honba or kyotaku
            let mut scores = [-3000; 4];
            scores[0] = 12000;
            scores[self.oya as usize] = -6000;
            vec_add_assign(&mut scores, &self.scores);
            return self.get_rank(scores) < 3;
        }

        // Do not ryukyoku if we have >= 10 yaokyuu tiles.
        if self.yaokyuu_kind_count() >= 10 {
            return false;
        }

        // Bloody Battle: No jihai (字牌), so skip jihai check
        // Original check: if we have all the jihai kinds, do not ryukyoku
        // This doesn't apply to Bloody Battle Mahjong

        // Ryukyoku otherwise.
        true
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
        // Bloody Battle: No all_last concept
        // Agari if we are oya ourselves, or we are not the last place at all.
        if self.oya == 0 || self.rank < 3 {
            return true;
        }

        // Bloody Battle: No bakaze, simplified agari logic
        // (This logic may need adjustment based on actual game flow)
        if self.scores.iter().all(|&s| s < 30000) {
            // Simplified agari condition
            return true;
        }

        // Calculate the max theoretical score we can achieve through this agari.
        // Bloody Battle: No riichi (立直), calculate max win point directly
        let max_win_point = {
            let mut tehai_full = self.tehai;
            for t in &self.ankan_overview[0] {
                tehai_full[t.as_usize()] += 4;
            }

            // Bloody Battle: No uradora calculation
            // Just calculate agari points directly
            // TODO: This is a simplified calculation, may need improvement
            self.agari_points(is_ron, &[]).unwrap()
        };

        // Calculate the best post-hora situation for us.
        let mut exp_scores = self.scores;
        if is_ron {
            // Bloody Battle: No kyotaku or honba
            exp_scores[0] += max_win_point.ron;
            exp_scores[target_rel] -= max_win_point.ron;
        } else {
            // Bloody Battle: No kyotaku or honba, no oya advantage
            let tsumo_total = max_win_point.tsumo_total(false);
            exp_scores[0] += tsumo_total;
            exp_scores
                .iter_mut()
                .enumerate()
                .skip(1)
                .for_each(|(_idx, s)| {
                    // Bloody Battle: All players pay the same (no oya advantage)
                    *s -= max_win_point.tsumo_ko;
                });
        }

        // Bloody Battle: No bakaze, simplified logic
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
    /// Calculate agari points for Bloody Battle Mahjong
    /// 
    /// Bloody Battle: No ura_indicators (里宝牌指示牌), no riichi (立直), no dora (宝牌)
    pub fn agari_points(&self, is_ron: bool, _ura_indicators: &[Tile]) -> Result<Point> {
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
            let tid = winning_tile.deaka().as_usize();
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
            is_menzen: self.is_menzen,
            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,
            winning_tile: winning_tile.deaka().as_u8(),
            is_ron,
            ding_que: self.ding_que,
            is_after_kan: !is_ron && self.at_rinshan, // 杠上花：自摸且从岭上牌摸的
            is_kan_discard: is_kan_discard_from_dahai, // 杠上炮：杠后打出的牌（不包括抢杠）
            is_chankan, // 抢杠：在别人加杠时抢杠和牌
            exclude_gen_tile: None, // For winning player, no exclusion needed
        };
        let agari = agari_calc
            .agari()
            .context("not a hora hand")?;

        // Bloody Battle: No oya advantage
        Ok(agari.point(false))
    }

    /// Calculate agari points excluding gen for a specific tile (for chankan)
    /// This is used when calculating the payment amount for the kakan player in chankan
    pub fn agari_points_exclude_gen(&self, is_ron: bool, exclude_tile: u8, _ura_indicators: &[Tile]) -> Result<Point> {
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
            let tid = winning_tile.deaka().as_usize();
            tehai[tid] += 1;
        }

        let is_chankan = is_ron && self.chankan_chance.is_some();
        let is_kan_discard_from_dahai = is_ron && self.last_discard_was_after_kan && !is_chankan;
        let agari_calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: self.is_menzen,
            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,
            winning_tile: winning_tile.deaka().as_u8(),
            is_ron,
            ding_que: self.ding_que,
            is_after_kan: !is_ron && self.at_rinshan,
            is_kan_discard: is_kan_discard_from_dahai,
            is_chankan,
            exclude_gen_tile: Some(exclude_tile), // Exclude this tile from gen count
        };
        let agari = agari_calc
            .agari()
            .context("not a hora hand")?;

        // Bloody Battle: No oya advantage
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
            return if self.waits[tile.deaka().as_usize()] {
                -1
            } else {
                0
            };
        }

        // 3n+2, tenpai after chi or pon. `self.shanten` is 0, but the actual
        // shanten could be 0 or -1.
        //
        // At 223m 55p 45s, `self.shanten` is 1. After 6s chi, `self.shanten`
        // becomes 0 because `update_shanten` is always called after a chi/pon
        // event. The actual shanten is 0 as well.
        //
        // At 123m 55p 45s, `self.shanten` is 0. After 6s chi, `self.shanten`
        // becomes 0 because `update_shanten` clamps the value to be >= 0. The
        // actual shanten is -1.
        shanten::calc_all(&self.tehai, self.tehai_len_div3)
    }

    /// Can be called at both 3n+1 and 3n+2, but `self.real_time_shanten` must
    /// be >= 0 and `self.tiles_left` must be >= 4.
    ///
    /// This function is currently highly internal.
    pub(super) fn single_player_tables(&self) -> Result<SinglePlayerTables> {
        ensure!(self.tiles_left >= 4, "need at least one more tsumo");

        let cur_shanten = self.real_time_shanten();
        ensure!(cur_shanten >= 0, "can't calculate an agari hand");

        let mut can_discard = self.last_cans.can_discard;
        let (tsumos_left, _calc_haitei) = if can_discard {
            (self.tiles_left / 4, self.tiles_left.is_multiple_of(4))
        } else {
            let target = self.rel(self.last_cans.target_actor) as u8;
            // Let's just ignore chankan here.
            let tiles_left_at_next_tsumo = self.tiles_left.saturating_sub(4 - target);
            (
                tiles_left_at_next_tsumo / 4,
                tiles_left_at_next_tsumo.is_multiple_of(4),
            )
        };
        ensure!(tsumos_left >= 1, "need at least one more tsumo");

        // Bloody Battle: No dora (宝牌), riichi (立直), or akas (红5)
        let num_doras_in_fuuro = 0;
        let prefer_riichi = false;
        let calc_double_riichi = false;

        // Bloody Battle: No riichi (立直), so no special discard handling
        let tehai = self.tehai;

        // Bloody Battle: SPCalculator is updated for Bloody Battle rules
        // - No riichi (立直), dora (宝牌), haitei (海底) calculations (fields set to false/empty)
        // - get_score() method uses Bloody Battle fan-based scoring
        // - All Japanese Mahjong-specific calculations are disabled
        let init_state = InitState {
            tehai,
            akas_in_hand: [false; 3], // Bloody Battle: No akas
            tiles_seen: self.tiles_seen,
            akas_seen: [false; 3], // Bloody Battle: No akas
        };
        let sp_calc = SPCalculator {
            tehai_len_div3: self.tehai_len_div3,
            is_menzen: self.is_menzen,
            chis: &[], // Bloody Battle: No chis
            pons: &self.pons,
            minkans: &self.minkans,
            ankans: &self.ankans,
            bakaze: 0, // Bloody Battle: No bakaze
            jikaze: 0, // Bloody Battle: No jikaze
            num_doras_in_fuuro,
            prefer_riichi,
            dora_indicators: &[], // Bloody Battle: No dora
            calc_double_riichi,
            calc_haitei: false, // Bloody Battle: No haitei
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: false,
            calc_shanten_down: false,
        };

        let mut max_ev_table = sp_calc.calc(init_state, can_discard, tsumos_left, cur_shanten)?;
        // Bloody Battle: No riichi (立直) discard handling

        Ok(SinglePlayerTables { max_ev_table })
    }
}
