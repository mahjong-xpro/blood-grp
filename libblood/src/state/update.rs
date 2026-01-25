use super::PlayerState;
use super::action::ActionCandidate;
use super::item::{KawaItem, Sutehai};
use crate::algo::agari::AgariCalculator;
use crate::algo::shanten;
use crate::mjai::Event;
use crate::rankings::Rankings;
use crate::tile::Tile;
use crate::must_tile;
use std::cmp::Ordering;
use std::{iter, mem};

use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy)]
pub(super) enum MoveType {
    Tsumo,
    Discard,
    FuuroConsume,
}

impl PlayerState {
    #[inline]
    pub fn update(&mut self, event: &Event) -> Result<ActionCandidate> {
        self.update_with_keep_cans(event, false)
    }

    /// If `keep_cans_on_announce` is true, then ReachAccepted, Dora and Hora
    /// events will keep `self.last_cans`, `self.ankan_candidates` and
    /// `self.kakan_candidates` unchanged from the last update. Currently
    /// setting it to true is only useful in validate_logs.
    pub fn update_with_keep_cans(
        &mut self,
        event: &Event,
        keep_cans_on_announce: bool,
    ) -> Result<ActionCandidate> {
        self.update_inner(event, keep_cans_on_announce)
            .with_context(|| format!("on event {event:?}"))
    }

    fn update_inner(
        &mut self,
        event: &Event,
        keep_cans_on_announce: bool,
    ) -> Result<ActionCandidate> {
        if !keep_cans_on_announce || !event.is_in_game_announce() {
            self.last_cans = ActionCandidate {
                target_actor: event.actor().unwrap_or(self.player_id),
                ..Default::default()
            };
            self.ankan_candidates.clear();
            self.kakan_candidates.clear();
        }


        match *event {
            Event::StartKyoku {
                kyoku,
                oya,
                scores,
                tehais,
            } => self.start_kyoku(
                kyoku,
                oya,
                scores,
                tehais,
            )?,
            
            Event::DingQue { actor, suit } => self.ding_que(actor, suit)?,

            Event::Tsumo { actor, pai } => self.tsumo(actor, pai)?,
            Event::Dahai {
                actor,
                pai,
                tsumogiri,
            } => self.dahai(actor, pai, tsumogiri)?,


            Event::Pon {
                actor,
                target,
                pai,
                consumed,
            } => self.pon(actor, target, pai, consumed)?,

            Event::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => self.daiminkan(actor, target, pai, consumed)?,

            Event::Kakan { actor, pai, .. } => self.kakan(actor, pai)?,
            Event::Ankan { actor, consumed } => self.ankan(actor, consumed)?,

            _ => (),
        };

        Ok(self.last_cans)
    }

    fn start_kyoku(
        &mut self,
        kyoku: u8,
        oya: u8,
        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
    ) -> Result<()> {
        self.tehai.fill(0);
        self.waits.fill(false);
        self.tiles_seen.fill(0);
        self.keep_shanten_discards.fill(false);
        self.next_shanten_discards.fill(false);
        self.forbidden_tiles.fill(false);
        self.discarded_tiles.fill(false);

        self.oya = self.rel(oya) as u8;
        self.kyoku = kyoku - 1;

        self.scores = scores;
        self.scores.rotate_left(self.player_id as usize);

        self.ding_que = None;
        self.other_ding_que.fill(None);
        self.has_agari = false;

        self.ankan_candidates.clear();
        self.kakan_candidates.clear();
        self.chankan_chance = None;
        self.chankan_kakan_actor = None;
        self.chankan_kakan_tile = None;
        self.last_discard_was_after_kan = false;
        
        self.tiles_left = 56;
        self.at_turn = 0;
        
        // Initialize tehai from tehais
        for &tile in &tehais[self.player_id as usize] {
            let tid = tile.as_usize();
            self.tehai[tid] += 1;
        }
        
        self.tehai_len_div3 = (self.tehai.iter().sum::<u8>() % 3) as u8;
        self.update_shanten();
        
        Ok(())
    }
    
    /// Handle DingQue event (定缺)
    fn ding_que(&mut self, actor: u8, suit: crate::mjai::Suit) -> Result<()> {
        if actor == self.player_id {
            self.ding_que = Some(suit);
        } else {
            let actor_rel = self.rel(actor);
            if actor_rel < 3 {
                self.other_ding_que[actor_rel] = Some(suit);
            }
        }
        Ok(())
    }

    fn tsumo(&mut self, actor: u8, pai: Tile) -> Result<()> {
        // Allow tsumo if tiles_left is 0 but this is the last tile (haitei)
        // This handles the case where tiles_left might be slightly out of sync
        if self.tiles_left == 0 {
            // This is the last tile, allow it but don't decrement further
            // The game should end after this tsumo
        } else {
            self.tiles_left -= 1;
        }
        if actor != self.player_id {
            return Ok(());
        }
        self.at_turn += 1;

        self.last_cans.can_discard = true;
        self.last_self_tsumo = Some(pai);
        self.witness_tile(pai)?;
        self.move_tile(pai, MoveType::Tsumo)?;

        self.update_shanten_discards();

        if self.waits[pai.as_usize()] {
            // Even for is_menzen, tiles_left == 0, or at_rinshan cases
            let agari_calc = AgariCalculator {
                tehai: &self.tehai,
                is_menzen: self.is_menzen,
                pons: &self.pons,
                minkans: &self.minkans,
                ankans: &self.ankans,
                winning_tile: pai.as_u8(),
                is_ron: false,
                ding_que: self.ding_que,
                is_after_kan: self.at_rinshan, // 杠上花：从岭上牌摸的
                is_kan_discard: false,
                is_chankan: false,
                exclude_gen_tile: None,
            };
            self.last_cans.can_tsumo_agari = agari_calc.has_yaku();
        }

        // haitei tile cannot be used for kakan or ankan
        if self.tiles_left == 0 {
            return Ok(());
        }


        if self.kans_on_board < 4 {
            self.tehai
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .for_each(|(tid, &count)| {
                    let tile = must_tile!(tid);
                    if count == 4 {
                        self.last_cans.can_ankan = true;
                        self.ankan_candidates.push(tile);
                    } else if self.pons.contains(&(tid as u8)) {
                        self.last_cans.can_kakan = true;
                        self.kakan_candidates.push(tile);
                    }
                });
        }

        Ok(())
    }

    fn dahai(&mut self, actor: u8, pai: Tile, tsumogiri: bool) -> Result<()> {
        let actor_rel = self.rel(actor);
        if actor_rel == 0 {
            self.move_tile(pai, MoveType::Discard)?;
        } else {
            self.witness_tile(pai)?;
        }

        // Check if there was a kan before this discard (for 杠上炮)
        let was_kan_before_discard = !self.intermediate_kan.is_empty();
        // Store this info for agari_points() to use later
        self.last_discard_was_after_kan = was_kan_before_discard;
        
        let sutehai = Sutehai {
            tile: pai,
            is_tedashi: !tsumogiri,
        };
        let kawa_item = KawaItem {
            chi_pon: None,
            kan: mem::take(&mut self.intermediate_kan),
            sutehai,
        };
        self.kawa[actor_rel].push(Some(kawa_item));
        self.kawa_overview[actor_rel].push(pai);
        self.last_kawa_tile = Some(pai);

        if !tsumogiri {
            self.last_tedashis[actor_rel] = Some(sutehai);
        }

        if actor_rel == 0 {
            self.forbidden_tiles.fill(false);
            self.at_rinshan = false;
            self.discarded_tiles[pai.as_usize()] = true;

            if self.next_shanten_discards[pai.as_usize()] {
                self.shanten -= 1;
            } else if !self.keep_shanten_discards[pai.as_usize()] {
                self.update_shanten();
            }
            self.update_waits_and_furiten();

            return Ok(());
        }

        if self.waits[pai.as_usize()] {
            // Always check has_yaku() to ensure ding_que rule is checked
            // Even for tiles_left == 0 case
            let mut tehai_with_winning_tile = self.tehai;
            tehai_with_winning_tile[pai.as_usize()] += 1;

            let agari_calc = AgariCalculator {
                tehai: &tehai_with_winning_tile,
                is_menzen: self.is_menzen,
                pons: &self.pons,
                minkans: &self.minkans,
                ankans: &self.ankans,
                winning_tile: pai.as_u8(),
                is_ron: true,
                ding_que: self.ding_que,
                is_after_kan: false, // 荣和不是从岭上牌摸的
                is_kan_discard: was_kan_before_discard, // 杠上炮：刚有人杠后打出的牌
                is_chankan: false, // dahai()中的荣和不是抢杠
                exclude_gen_tile: None,
            };
            self.last_cans.can_ron_agari = agari_calc.has_yaku();

        }

        if self.tiles_left == 0 {
            return Ok(());
        }

        self.last_cans.can_pon = self.tehai[pai.as_usize()] >= 2;
        self.last_cans.can_daiminkan =
            self.kans_on_board < 4 && self.tehai[pai.as_usize()] == 3;

        Ok(())
    }


    fn pon(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 2]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let full_set = consumed.into_iter().chain(iter::once(pai)).collect();
        self.fuuro_overview[actor_rel].push(full_set);
        // Chi/pon info is stored in fuuro_overview, not intermediate_chi_pon
        self.pad_kawa_for_pon_or_daiminkan(actor, target);

        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            for _t in full_set {
            }
            return Ok(());
        }

        self.last_cans.can_discard = true;
        self.is_menzen = false;
        self.tehai_len_div3 = self.tehai_len_div3.saturating_sub(1);
        // Marked explicitly as `None` to let `Agent` impls set
        // `tsumogiri` to false in the Dahai after Pon
        self.last_self_tsumo = None;

        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        self.pons.push(pai.as_u8());

        if self.tehai[pai.as_usize()] > 0 {
            self.forbidden_tiles[pai.as_usize()] = true;
        }

        // NOTES: this is 3n+2
        // The shanten can change after pon, for example 122334789 pon 2.
        self.update_shanten();
        self.update_shanten_discards();

        Ok(())
    }

    fn daiminkan(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 3]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let full_set = consumed.into_iter().chain(iter::once(pai)).collect();
        self.fuuro_overview[actor_rel].push(full_set);
        self.intermediate_kan.push(pai);
        self.pad_kawa_for_pon_or_daiminkan(actor, target);
        self.kans_on_board += 1;

        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            for _t in full_set {
            }
            return Ok(());
        }

        self.at_rinshan = true;
        self.is_menzen = false;
        self.tehai_len_div3 = self.tehai_len_div3.saturating_sub(1);

        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        self.minkans.push(pai.as_u8());

        // The shanten number and the shape of tenpai (if any) may be
        // changed after a daiminkan.
        //
        // For example: 12223m 456p 12378s + 2m
        self.update_shanten();
        self.update_waits_and_furiten();

        Ok(())
    }

    fn kakan(&mut self, actor: u8, pai: Tile) -> Result<()> {
        let actor_rel = self.rel(actor);
        for fuuro in &mut self.fuuro_overview[actor_rel] {
            if fuuro[0] == pai {
                fuuro.push(pai);
                break;
            }
        }
        self.intermediate_kan.push(pai);
        self.kans_on_board += 1;

        if actor_rel != 0 {
            self.witness_tile(pai)?;
            self.last_kawa_tile = Some(pai); // for getting winning tile in self.agari

            // 槍槓 (抢杠)
            // 当其他玩家加杠时，如果听的牌正好是加杠的牌，可以抢杠和牌
            // 抢杠时，加杠的玩家的根不应该计算
            if !self.at_furiten && self.waits[pai.as_usize()] {
                self.last_cans.can_ron_agari = true;
                self.to_mark_same_cycle_furiten = Some(());
                self.chankan_chance = Some(());
                self.chankan_kakan_actor = Some(actor); // 记录加杠的玩家，用于排除其根
                self.chankan_kakan_tile = Some(pai.as_u8()); // 记录加杠的牌，用于排除其根
            } else {
            }

            return Ok(());
        }

        self.at_rinshan = true;
        self.move_tile(pai, MoveType::FuuroConsume)?;
        self.pons.retain(|&t| t != pai.as_u8());
        self.minkans.push(pai.as_u8());

        // The shanten number and the shape of tenpai (if any) may
        // be changed after an kakan, because the kan'd tile may
        // come from the existing hand.
        if self.next_shanten_discards[pai.as_usize()] {
            self.shanten -= 1;
        } else if !self.keep_shanten_discards[pai.as_usize()] {
            self.update_shanten();
        }
        self.update_waits_and_furiten();

        Ok(())
    }

    fn ankan(&mut self, actor: u8, consumed: [Tile; 4]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let tile = consumed[0];
        self.ankan_overview[actor_rel].push(tile);
        self.intermediate_kan.push(tile);
        self.kans_on_board += 1;


        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            return Ok(());
        }

        self.at_rinshan = true;
        self.tehai_len_div3 = self.tehai_len_div3.saturating_sub(1);
        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        self.ankans.push(tile.as_u8());

        // The shanten number and the shape of tenpai (if any) may
        // be changed after an ankan. See the example in daiminkan.
        self.update_shanten();
        self.update_waits_and_furiten();

        Ok(())
    }


    pub(super) const fn rel(&self, actor: u8) -> usize {
        ((actor + 4 - self.player_id) % 4) as usize
    }

    ///
    /// Returns an error if we have already witnessed 4 such tiles.
    pub(super) fn witness_tile(&mut self, tile: Tile) -> Result<()> {
        ensure!(
            !tile.is_unknown(),
            "rule violation: attempt to witness an unknown tile",
        );
        let tile_id = tile.as_usize();

        let seen = &mut self.tiles_seen[tile_id];
        ensure!(
            *seen < 4,
            "rule violation: attempt to witness the fifth {tile}",
        );
        *seen += 1;

        Ok(())
    }

    ///
    /// Returns an error when trying to discard or consume a tile that the
    /// player doesn't own.
    pub(super) fn move_tile(&mut self, tile: Tile, move_type: MoveType) -> Result<()> {
        let tile_id = tile.as_usize();
        let tehai_tile = &mut self.tehai[tile_id];
        match move_type {
            MoveType::Tsumo => {
                *tehai_tile += 1;
            }
            MoveType::Discard => {
                ensure!(
                    *tehai_tile > 0,
                    "rule violation: attempt to discard {tile} from void",
                );
                *tehai_tile -= 1;
            }
            MoveType::FuuroConsume => {
                ensure!(
                    *tehai_tile > 0,
                    "rule violation: attempt to consume {tile} from void",
                );
                *tehai_tile -= 1;
            }
        }

        Ok(())
    }

    /// Updates `dora_indicators`, witness the dora indicator itself and
    /// recounts doras (`doras_seen` and `doras_owned`) based on all the seen
    /// tiles.

    pub(super) fn pad_kawa_for_pon_or_daiminkan(&mut self, abs_actor: u8, abs_target: u8) {
        let mut i = (abs_target + 1) % 4;
        while i != abs_actor {
            let rel = self.rel(i);
            self.kawa[rel].push(None);
            i = (i + 1) % 4;
        }
    }

    #[allow(dead_code)] // Kept for compatibility, may be used in future
    pub(super) fn pad_kawa_at_start(&mut self) {
        self.kawa
            .iter_mut()
            .take(self.oya as usize)
            .for_each(|kawa| kawa.push(None));
    }


    /// Can be called at either 3n+1 or 3n+2.
    ///
    /// For 3n+2, the return value of `shanten::calc_all` may be `-1`. We don't
    /// allow `-1` and it will be written as `0` in order for
    /// `_shanten_discards` to be calculated properly.
    pub(super) fn update_shanten(&mut self) {
        self.shanten = shanten::calc_all(&self.tehai, self.tehai_len_div3).max(0);
        debug_assert!(matches!(self.shanten, 0..=6));
    }

    /// Must be called at 3n+2.
    pub(super) fn update_shanten_discards(&mut self) {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        self.next_shanten_discards.fill(false);
        self.keep_shanten_discards.fill(false);
        self.has_next_shanten_discard = false;

        let mut tehai = self.tehai;
        for (tid, &count) in self.tehai.iter().enumerate() {
            // `self.forbidden_tiles[tid]` is not checked here, but it is
            // acceptable because forbidden tiles are always keep-shanten
            // discards, so it won't affect the result of
            // `has_next_shanten_discard`. We will take forbidden_tiles into
            // account when generating discard candidates.
            if count == 0 {
                continue;
            }
            tehai[tid] -= 1;
            let shanten_after = shanten::calc_all(&tehai, self.tehai_len_div3);
            tehai[tid] += 1;
            match shanten_after.cmp(&self.shanten) {
                Ordering::Less => {
                    self.next_shanten_discards[tid] = true;
                    self.has_next_shanten_discard = true;
                }
                Ordering::Equal => {
                    self.keep_shanten_discards[tid] = true;
                }
                _ => (),
            };
        }
    }

    /// Caller must assure current tehai is 3n+1, and `self.shanten` must be up
    /// to date and correct.
    pub(super) fn update_waits_and_furiten(&mut self) {
        assert!(!self.last_cans.can_discard, "tehai is not 3n+1");

        // Reset the furiten flag here for:
        // 1. Clearing same-cycle furiten.
        // 2. The fact that furiten doesn't make sense if we are no longer
        //    tenpai.
        self.at_furiten = false;
        self.waits.fill(false);

        if self.shanten > 0 {
            return;
        }

        for (t, is_wait) in self.waits.iter_mut().enumerate() {
            if self.tehai[t] == 4 {
                // Cannot wait, not even furiten for the 5th tile.
                //
                // However waiting for the 5th tile when all 4 of them are
                // already in the kawa or fuuro is a valid furiten.
                //
                // Note that although [karaten] is not considered as a wait and
                // thus will not be written to the `waits` in this impl anyways,
                // it is still a valid ryukyoku tenpai in our rule spec.
                continue;
            }
            let mut tehai_after = self.tehai;
            tehai_after[t] += 1;

            if shanten::calc_all(&tehai_after, self.tehai_len_div3) == -1 {
                // furiten is not affected by `tiles_seen`
                if self.discarded_tiles[t] {
                    self.at_furiten = true;
                }
                *is_wait = self.tiles_seen[t] < 4;
            }
        }
    }


    #[allow(dead_code)] // May be used in future
    pub(super) fn update_rank(&mut self) {
        self.rank = self.get_rank(self.scores);
    }

    pub(super) fn get_rank(&self, mut scores_rel: [i32; 4]) -> u8 {
        let scores_abs = {
            scores_rel.rotate_right(self.player_id as usize);
            scores_rel
        };
        Rankings::new(scores_abs).rank_by_player[self.player_id as usize]
    }
}
