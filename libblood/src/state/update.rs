use super::PlayerState;
use super::action::ActionCandidate;
use super::item::{KawaItem, Sutehai};
use crate::algo::agari::AgariCalculator;
use crate::algo::shanten;
use crate::mjai::Event;
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
        self.update_inner(event)
            .with_context(|| format!("on event {event:?}"))
    }

    fn update_inner(
        &mut self,
        event: &Event,
    ) -> Result<ActionCandidate> {
        if !event.is_in_game_announce() {
            // Guo Shou Hu (Temporary Furiten) Detection
            // If we could Ron previously, but didn't (and the new event is not our own Win),
            // OR if we could Tsumo previously, but didn't (pass Tsumo implies Furiten until next turn),
            // then we missed it. Set temporary_furiten.
            if self.last_cans.can_ron_agari || self.last_cans.can_tsumo_agari {
                let passed = match event {
                    Event::Hora { actor, .. } => *actor != self.player_id,
                    _ => true,
                };
                if passed {
                    self.temporary_furiten = true;
                }
            }

            self.last_cans = ActionCandidate {
                target_actor: event.actor().unwrap_or(self.player_id),
                ..Default::default()
            };
            self.ankan_candidates.clear();
            self.kakan_candidates.clear();

            // DingQue availability must persist until chosen, regardless of turn/actor.
            // This is critical because DingQue selection can happen after the dealer's initial tsumo.
            if self.ding_que.is_none() {
                self.last_cans.can_ding_que = true;
            }
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
                deltas,
            } => self.daiminkan(actor, target, pai, consumed, deltas)?,

            Event::Kakan { actor, pai, deltas, .. } => self.kakan(actor, pai, deltas)?,
            Event::Ankan { actor, consumed, deltas } => self.ankan(actor, consumed, deltas)?,
            Event::Hora { actor, target, deltas, .. } => self.hora(actor, target, deltas)?,
            Event::Ryukyoku { deltas, .. } => self.ryukyoku(deltas)?,

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
        self.temporary_furiten = false;
        self.tiles_seen.fill(0);
        self.keep_shanten_discards.fill(false);
        self.next_shanten_discards.fill(false);
        self.forbidden_tiles.fill(false);
        self.players_agari.fill(false);
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
        self.pending_kakan_tile = None;
        self.last_discard_was_after_kan = false;
        self.intermediate_kan.clear(); // 新局开始时清空 intermediate_kan
        
        // 新局开始时清空所有副露相关的状态
        self.pons.clear();
        self.minkans.clear();
        self.ankans.clear();

        self.kans_on_board = 0;
        self.at_rinshan = false;
        
        // 新局开始时清空 kawa 和 kawa_overview（业务规则：每局开始时打牌记录应该重置）
        for kawa in &mut self.kawa {
            kawa.clear();
        }
        for kawa_overview in &mut self.kawa_overview {
            kawa_overview.clear();
        }
        // 清空 fuuro_overview 和 ankan_overview
        for fuuro in &mut self.fuuro_overview {
            fuuro.clear();
        }
        for ankan in &mut self.ankan_overview {
            ankan.clear();
        }
        
        self.tiles_left = 56;
        self.at_turn = 0;
        
        // Initialize this player's tehai and witness only this player's initial hand tiles
        // tiles_seen should only include tiles that are known to this player:
        // - This player's own hand (private, but known to this player)
        // - Discarded tiles (public)
        // - Fuuro tiles (public)
        // Other players' private hands should NOT be included in tiles_seen
        for &tile in &tehais[self.player_id as usize] {
            let tid = tile.as_usize();
            self.tehai[tid] += 1;
            // Witness this player's own initial hand tiles
            // These are private but known to this player, so they should be counted
            self.witness_tile(tile)?;
        }
        
        self.tehai_len_div3 = (self.tehai.iter().sum::<u8>() / 3) as u8;
        self.update_shanten();
        
        // At start of kyoku, player must choose Ding Que
        self.last_cans.can_ding_que = true;
        
        Ok(())
    }
    
    /// Handle DingQue event (定缺)
    fn ding_que(&mut self, actor: u8, suit: crate::mjai::Suit) -> Result<()> {
        if actor == self.player_id {
            self.ding_que = Some(suit);
            self.last_cans.can_ding_que = false;
        } else {
            let actor_rel = self.rel(actor);
            if actor_rel > 0 {
                // map relative 1, 2, 3 to index 0, 1, 2
                self.other_ding_que[actor_rel as usize - 1] = Some(suit);
            }
        }

        // If all players have selected DingQue, enable the dealer's first discard (14 tiles).
        // This is required because DingQue selection can happen after the dealer's initial tsumo.
        let all_selected = self.ding_que.is_some() && self.other_ding_que.iter().all(|s| s.is_some());
        if all_selected && self.is_oya() {
            let tehai_sum: u8 = self.tehai.iter().sum();
            if tehai_sum % 3 == 2 {
                // From this POV, it's now the dealer's discard turn.
                self.last_cans.target_actor = self.player_id;
                self.last_cans.can_discard = true;
                self.update_ding_que_forbidden_tiles();
                self.update_shanten();
                self.update_shanten_discards();
            }
        }
        Ok(())
    }

    fn tsumo(&mut self, actor: u8, pai: Tile) -> Result<()> {
        // Allow tsumo if tiles_left is 0 but this is the last tile
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
        
        // Safety check: specific to Blood Battle. A player who has won cannot act.
        if self.has_agari {
            return Ok(());
        }

        self.forbidden_tiles.fill(false);
        self.temporary_furiten = false;
        self.at_turn += 1;

        self.last_self_tsumo = Some(pai);
        // If we successfully kakan'd/ankan'd before this draw, it is no longer pending.
        self.pending_kakan_tile = None;
        self.witness_tile(pai)?;
        self.move_tile(pai, MoveType::Tsumo)?;

        // DingQue selection must happen before any play actions (discard/agari/kan).
        // During DingQue phase, the player can only select DingQue.
        if self.ding_que.is_none() {
            self.last_cans.can_ding_que = true;
            self.last_cans.can_discard = false;
            // Keep shanten roughly up-to-date for observation features.
            self.update_shanten();
            return Ok(());
        }

        self.last_cans.can_discard = true;
        self.update_shanten_discards();

        if self.waits[pai.as_usize()] {
            // Even for is_menzen, tiles_left == 0, or at_rinshan cases
            let agari_calc = AgariCalculator {
                tehai: &self.tehai,

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
                is_haidi: self.tiles_left == 0,
                is_tianhu: false,
                is_dihu: false,
            };
            self.last_cans.can_tsumo_agari = agari_calc.has_yaku();
        }

        // Last tile cannot be used for kakan or ankan
        if self.tiles_left == 0 {
            return Ok(());
        }

        // 在计算 ankan_candidates 之前，确保已经清空（虽然在 update_inner 中已经清空，但这里再次确保）
        // 然后重新计算 ankan_candidates 和 kakan_candidates
        self.ankan_candidates.clear();
        self.kakan_candidates.clear();
        if self.kans_on_board < 4 {
            self.tehai
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .for_each(|(tid, &count)| {
                    let tile = must_tile!(tid);
                    // 基础规则：定缺花色不能暗杠或加杠
                    if crate::ding_que::is_ding_que_tile(tile.as_usize(), self.ding_que) {
                        return;
                    }

                    if count == 4 {
                        self.last_cans.can_ankan = true;
                        let ankan_len = self.ankan_candidates.len();
                        assert!(
                            ankan_len < 3,
                            "ankan_candidates capacity overflow: player {} has {} ankan candidates, attempting to add one more. Maximum is 3. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
                            self.player_id,
                            ankan_len,
                            self.kyoku,
                            self.at_turn,
                            self.tiles_left
                        );
                        self.ankan_candidates.push(tile);
                    } else if self.pons.contains(&(tid as u8)) {
                        self.last_cans.can_kakan = true;
                        let kakan_len = self.kakan_candidates.len();
                        assert!(
                            kakan_len < 3,
                            "kakan_candidates capacity overflow: player {} has {} kakan candidates, attempting to add one more. Maximum is 3. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
                            self.player_id,
                            kakan_len,
                            self.kyoku,
                            self.at_turn,
                            self.tiles_left
                        );
                        self.kakan_candidates.push(tile);
                    }
                });
        }

        // Check Ding Que constraints and update forbidden_tiles
        if self.player_id == actor {
            self.update_ding_que_forbidden_tiles();
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
        // 基础规则：intermediate_kan 应该在杠操作后被设置，在打牌时被清空
        // 如果 intermediate_kan 有多个元素，说明有多个杠操作没有及时打牌，这是不正常的
        assert!(
            self.intermediate_kan.len() <= 1,
            "intermediate_kan has {} elements, but should have at most 1. This indicates a fundamental bug: multiple kan operations without discard.",
            self.intermediate_kan.len()
        );
        
        let was_kan_before_discard = !self.intermediate_kan.is_empty();
        // Store this info for agari_points() to use later
        self.last_discard_was_after_kan = was_kan_before_discard;
        
        let sutehai = Sutehai {
            tile: pai,
            is_tsumogiri: tsumogiri,
        };
        let kawa_item = KawaItem {
            kan: mem::take(&mut self.intermediate_kan),
            sutehai,
        };
        
        // 基础规则：打牌后 intermediate_kan 应该被清空
        assert!(
            self.intermediate_kan.is_empty(),
            "intermediate_kan should be empty after discard, but has {} elements. This indicates a fundamental bug in kan tracking.",
            self.intermediate_kan.len()
        );
        // kawa capacity is 55, which is the theoretical maximum (108 total tiles - 52 initial hands - 1 last draw)
        // If this fails, it indicates invalid game log data or a bug in game logic
        // 业务规则：一个玩家最多只能打55张牌，这是理论最大值
        // 如果超过55，说明游戏状态不一致或日志数据有问题，必须panic
        let kawa_len = self.kawa[actor_rel].len();
        assert!(
            kawa_len < 55,
            "kawa capacity overflow: player {} (relative {}) has {} discards, attempting to add one more. Maximum is 55. This indicates a fundamental bug in game logic or invalid game log data. Current tile: {:?}, kyoku: {}, at_turn: {}, tiles_left: {}, tehai_sum: {}",
            actor,
            actor_rel,
            kawa_len,
            pai,
            self.kyoku,
            self.at_turn,
            self.tiles_left,
            self.tehai.iter().sum::<u8>()
        );
        // Push to kawa (will panic if capacity exceeded, but we've already checked)
        self.kawa[actor_rel].push(Some(kawa_item));
        // Also check kawa_overview capacity before pushing
        let kawa_overview_len = self.kawa_overview[actor_rel].len();
        assert!(
            kawa_overview_len < 55,
            "kawa_overview capacity overflow: player {} (relative {}) has {} discards in overview, attempting to add one more. Maximum is 55. This indicates a fundamental bug in game logic or invalid game log data. Current tile: {:?}, kyoku: {}, at_turn: {}, tiles_left: {}, tehai_sum: {}",
            actor,
            actor_rel,
            kawa_overview_len,
            pai,
            self.kyoku,
            self.at_turn,
            self.tiles_left,
            self.tehai.iter().sum::<u8>()
        );
        self.kawa_overview[actor_rel].push(pai);
        self.last_kawa_tile = Some(pai);

        if actor_rel == 0 {
            self.forbidden_tiles.fill(false);
            self.at_rinshan = false;
            self.discarded_tiles[pai.as_usize()] = true;

            if self.next_shanten_discards[pai.as_usize()] {
                self.shanten -= 1;
            } else if !self.keep_shanten_discards[pai.as_usize()] {
                self.update_shanten();
            }
            self.update_waits();

            return Ok(());
        }

        // If the player has already won, they cannot take any actions
        if self.has_agari {
            return Ok(());
        }

        // Ron check: 点炮时以 has_yaku 为准，不再依赖 waits[]，避免 waits 漏判导致无法胡牌
        let tehai_sum: u8 = self.tehai.iter().sum();
        let hand_total: u8 = tehai_sum
            + 3 * self.pons.len() as u8
            + 3 * self.minkans.len() as u8
            + 3 * self.ankans.len() as u8;
        let pai_idx = pai.as_usize();
        let can_add_tile = hand_total == 13
            && self.tehai[pai_idx] < 4
            && self.tiles_seen[pai_idx] < 4;

        self.last_cans.can_ron_agari = if self.temporary_furiten || !can_add_tile {
            false
        } else {
            let mut tehai_with_winning_tile = self.tehai;
            tehai_with_winning_tile[pai_idx] += 1;

            let agari_calc = AgariCalculator {
                tehai: &tehai_with_winning_tile,
                pons: &self.pons,
                minkans: &self.minkans,
                ankans: &self.ankans,
                winning_tile: pai.as_u8(),
                is_ron: true,
                ding_que: self.ding_que,
                is_after_kan: false,
                is_kan_discard: was_kan_before_discard,
                is_chankan: false,
                exclude_gen_tile: None,
                is_haidi: self.tiles_left == 0,
                is_tianhu: false,
                is_dihu: false,
            };
            agari_calc.has_yaku()
        };

        if self.tiles_left == 0 {
            return Ok(());
        }

        // Check if the discarded tile is the ding_que suit
        let is_ding_que_tile = crate::ding_que::is_ding_que_tile(pai.as_usize(), self.ding_que);

        // 基础规则：定缺花色不能碰或明杠
        if !is_ding_que_tile {
            self.last_cans.can_pon = self.tehai[pai.as_usize()] >= 2;
            self.last_cans.can_daiminkan =
                self.kans_on_board < 4 && self.tehai[pai.as_usize()] == 3;
        } else {
            self.last_cans.can_pon = false;
            self.last_cans.can_daiminkan = false;
        }

        Ok(())
    }


    fn pon(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 2]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let full_set = consumed.into_iter().chain(iter::once(pai)).collect();
        let fuuro_len = self.fuuro_overview[actor_rel].len();
        assert!(
            fuuro_len < 4,
            "fuuro_overview capacity overflow: player {} (relative {}) has {} fuuro, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
            actor,
            actor_rel,
            fuuro_len,
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.fuuro_overview[actor_rel].push(full_set);
        // Pon info is stored in fuuro_overview (Bloody Battle Mahjong has no chi)
        // Only pad kawa from the actor's perspective to avoid duplicate pushes when broadcast
        if actor_rel == 0 {
            self.pad_kawa_for_pon_or_daiminkan(actor, target)?;
        }

        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            for _t in full_set {
            }
            return Ok(());
        }

        self.forbidden_tiles.fill(false);
        self.temporary_furiten = false;
        self.last_cans.can_discard = true;
        self.tehai_len_div3 = self.tehai_len_div3.saturating_sub(1);
        // Marked explicitly as `None` to let `Agent` impls set
        // `tsumogiri` to false in the Dahai after Pon
        self.last_self_tsumo = None;

        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        let pons_len = self.pons.len();
        assert!(
            pons_len < 4,
            "pons capacity overflow: player {} has {} pons, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
            self.player_id,
            pons_len,
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.pons.push(pai.as_u8());

        if self.tehai[pai.as_usize()] > 0 {
            self.forbidden_tiles[pai.as_usize()] = true;
        }

        // Enforce Ding Que rule: if holding Ding Que tiles, must discard them
        self.update_ding_que_forbidden_tiles();


        // NOTES: this is 3n+2
        // The shanten can change after pon, for example 122334789 pon 2.
        self.update_shanten();
        self.update_shanten_discards();

        self.update_shanten();
        self.update_shanten_discards();

        Ok(())
    }

    fn hora(&mut self, actor: u8, target: u8, deltas: Option<[i32; 4]>) -> Result<()> {
        let actor_rel = self.rel(actor);
        self.players_agari[actor_rel] = true;

        // Chankan (抢杠) replay fix:
        // If we attempted kakan and then got ronned immediately on that kakan tile (target == self),
        // the kong should be cancelled and our meld must revert from (min)kan back to pon.
        //
        // The arena engine currently does a similar state correction; we must mirror it here so that
        // logs are replayable and downstream dataset generation stays consistent.
        if target == self.player_id && actor != self.player_id {
            if let Some(tile) = self.pending_kakan_tile.take() {
                // Revert minkans -> pons for this tile (do not change tehai: the 4th tile is robbed).
                if let Some(pos) = self.minkans.iter().position(|&t| t == tile) {
                    self.minkans.remove(pos);
                    // Strong rule: reverting a robbed kong must not create invalid meld state.
                    // If we end up with duplicate/overflow, crash early with a clear message.
                    assert!(
                        !self.pons.iter().any(|&t| t == tile),
                        "chankan revert: duplicate pon tile {} for player {} (already in pons). This indicates invalid state/log.",
                        tile,
                        self.player_id
                    );
                    assert!(
                        self.pons.len() < 4,
                        "chankan revert: pons capacity overflow (len=4) for player {}. This indicates invalid state/log.",
                        self.player_id
                    );
                    self.pons.push(tile);
                }

                // Revert fuuro_overview from 4 tiles back to 3 tiles if present.
                // (fuuro_overview is stored per relative seat; for self it's always 0.)
                for fuuro in &mut self.fuuro_overview[0] {
                    if fuuro.first().is_some_and(|t0| t0.as_u8() == tile) && fuuro.len() == 4 {
                        fuuro.pop();
                        break;
                    }
                }

                // After a robbed kong, we are not at rinshan anymore from our perspective.
                self.at_rinshan = false;

                // FIX: Clear intermediate_kan because the kong was robbed and invalidated.
                // Otherwise, the next discard will be incorrectly flagged as is_kan_discard.
                self.intermediate_kan.clear();

                // State changed, update caches to keep subsequent legality checks stable.
                self.update_shanten();
                if self.last_cans.can_discard {
                    // 3n+2
                    self.update_shanten_discards();
                } else {
                    // 3n+1
                    self.update_waits();
                }
            }
        }
        
        if let Some(d) = deltas {
             // deltas is [i32; 4] absolute.
             // self.scores is [i32; 4], relative to self.player_id.
             // We need to rotate deltas to match self.scores (which is rotated left by self.player_id)
             let mut d_rel = d;
             d_rel.rotate_left(self.player_id as usize);
             
             for i in 0..4 {
                 self.scores[i] += d_rel[i];
             }
        }
        
        Ok(())
    }
    
    fn ryukyoku(&mut self, deltas: Option<[i32; 4]>) -> Result<()> {
        if let Some(d) = deltas {
             let mut d_rel = d;
             d_rel.rotate_left(self.player_id as usize);
             
             for i in 0..4 {
                 self.scores[i] += d_rel[i];
             }
        }
        Ok(())
    }

    fn daiminkan(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 3], deltas: Option<[i32; 4]>) -> Result<()> {
        if let Some(d) = deltas {
             let mut d_rel = d;
             d_rel.rotate_left(self.player_id as usize);
             for i in 0..4 {
                 self.scores[i] += d_rel[i];
             }
        }

        let actor_rel = self.rel(actor);
        let full_set = consumed.into_iter().chain(iter::once(pai)).collect();
        let fuuro_len = self.fuuro_overview[actor_rel].len();
        assert!(
            fuuro_len < 4,
            "fuuro_overview capacity overflow: player {} (relative {}) has {} fuuro, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
            actor,
            actor_rel,
            fuuro_len,
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.fuuro_overview[actor_rel].push(full_set);
        // Clear any previous kan before adding new one (only track most recent kan before discard)
        self.intermediate_kan.clear();
        // intermediate_kan should be empty after clear(), so we can safely push
        // But add a check just in case clear() didn't work as expected
        assert!(
            self.intermediate_kan.len() < 4,
            "intermediate_kan capacity overflow: player {} has {} tiles in intermediate_kan after clear(), attempting to add one more. Maximum is 4. This indicates a bug in intermediate_kan management. kyoku: {}, at_turn: {}, tiles_left: {}",
            self.player_id,
            self.intermediate_kan.len(),
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.intermediate_kan.push(pai);
        // Only pad kawa from the actor's perspective to avoid duplicate pushes when broadcast
        if actor_rel == 0 {
            self.pad_kawa_for_pon_or_daiminkan(actor, target)?;
        }
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
        self.temporary_furiten = false;

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
        self.update_waits();

        Ok(())
    }

    fn kakan(&mut self, actor: u8, pai: Tile, deltas: Option<[i32; 4]>) -> Result<()> {
        if let Some(d) = deltas {
             let mut d_rel = d;
             d_rel.rotate_left(self.player_id as usize);
             for i in 0..4 {
                 self.scores[i] += d_rel[i];
             }
        }

        let actor_rel = self.rel(actor);
        for fuuro in &mut self.fuuro_overview[actor_rel] {
            if fuuro[0] == pai {
                let fuuro_len = fuuro.len();
                assert!(
                    fuuro_len < 4,
                    "fuuro capacity overflow: player {} (relative {}) has {} tiles in fuuro, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
                    actor,
                    actor_rel,
                    fuuro_len,
                    self.kyoku,
                    self.at_turn,
                    self.tiles_left
                );
                fuuro.push(pai);
                break;
            }
        }
        // Clear any previous kan before adding new one (only track most recent kan before discard)
        self.intermediate_kan.clear();
        // intermediate_kan should be empty after clear(), so we can safely push
        // But add a check just in case clear() didn't work as expected
        assert!(
            self.intermediate_kan.len() < 4,
            "intermediate_kan capacity overflow: player {} has {} tiles in intermediate_kan after clear(), attempting to add one more. Maximum is 4. This indicates a bug in intermediate_kan management. kyoku: {}, at_turn: {}, tiles_left: {}",
            self.player_id,
            self.intermediate_kan.len(),
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.intermediate_kan.push(pai);
        self.kans_on_board += 1;

        if actor_rel != 0 {
            self.witness_tile(pai)?;
            self.last_kawa_tile = Some(pai); // for getting winning tile in self.agari

            // 槍槓 (抢杠)：与点炮 Ron 一致，以 has_yaku 为准，不依赖 waits[]
            // 抢杠时不检查 tiles_seen：加杠的牌必定可用（正在被加杠）
            let hand_total: u8 = self.tehai.iter().sum::<u8>()
                + 3 * self.pons.len() as u8
                + 3 * self.minkans.len() as u8
                + 3 * self.ankans.len() as u8;
            let pai_idx = pai.as_usize();
            let can_chankan = !self.temporary_furiten
                && hand_total == 13
                && self.tehai[pai_idx] < 4;

            if can_chankan {
                let mut tehai_with_winning_tile = self.tehai;
                tehai_with_winning_tile[pai_idx] += 1;

                let agari_calc = AgariCalculator {
                    tehai: &tehai_with_winning_tile,
                    pons: &self.pons,
                    minkans: &self.minkans,
                    ankans: &self.ankans,
                    winning_tile: pai.as_u8(),
                    is_ron: true,
                    ding_que: self.ding_que,
                    is_after_kan: false,
                    is_kan_discard: false,
                    is_chankan: true,
                    exclude_gen_tile: None,
                    is_haidi: self.tiles_left == 0,
                    is_tianhu: false,
                    is_dihu: false,
                };

                if agari_calc.has_yaku() {
                    self.last_cans.can_ron_agari = true;
                    self.chankan_chance = Some(());
                    self.chankan_kakan_actor = Some(actor);
                    self.chankan_kakan_tile = Some(pai.as_u8());
                }
            }

            return Ok(());
        }

        self.at_rinshan = true;
        self.temporary_furiten = false;
        // Mark this kakan as pending until we either draw from rinshan (success)
        // or get ronned immediately (chankan, handled in hora()).
        self.pending_kakan_tile = Some(pai.as_u8());
        self.move_tile(pai, MoveType::FuuroConsume)?;
        self.pons.retain(|&t| t != pai.as_u8());
        let minkans_len = self.minkans.len();
        assert!(
            minkans_len < 4,
            "minkans capacity overflow: player {} has {} minkans, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
            self.player_id,
            minkans_len,
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.minkans.push(pai.as_u8());

        // The shanten number and the shape of tenpai (if any) may
        // be changed after an kakan, because the kan'd tile may
        // come from the existing hand.
        if self.next_shanten_discards[pai.as_usize()] {
            self.shanten -= 1;
        } else if !self.keep_shanten_discards[pai.as_usize()] {
            self.update_shanten();
        }
        self.update_waits();

        Ok(())
    }

    fn ankan(&mut self, actor: u8, consumed: [Tile; 4], deltas: Option<[i32; 4]>) -> Result<()> {
        if let Some(d) = deltas {
             let mut d_rel = d;
             d_rel.rotate_left(self.player_id as usize);
             for i in 0..4 {
                 self.scores[i] += d_rel[i];
             }
        }

        let actor_rel = self.rel(actor);
        let tile = consumed[0];
        let ankan_len = self.ankan_overview[actor_rel].len();
        assert!(
            ankan_len < 4,
            "ankan_overview capacity overflow: player {} (relative {}) has {} ankans, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
            actor,
            actor_rel,
            ankan_len,
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.ankan_overview[actor_rel].push(tile);
        // Clear any previous kan before adding new one (only track most recent kan before discard)
        self.intermediate_kan.clear();
        self.intermediate_kan.push(tile);
        self.kans_on_board += 1;


        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            return Ok(());
        }

        self.at_rinshan = true;
        self.temporary_furiten = false;
        self.tehai_len_div3 = self.tehai_len_div3.saturating_sub(1);
        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        let ankans_len = self.ankans.len();
        assert!(
            ankans_len < 4,
            "ankans capacity overflow: player {} has {} ankans, attempting to add one more. Maximum is 4. This indicates invalid game log data or a bug in game logic. kyoku: {}, at_turn: {}, tiles_left: {}",
            self.player_id,
            ankans_len,
            self.kyoku,
            self.at_turn,
            self.tiles_left
        );
        self.ankans.push(tile.as_u8());

        // The shanten number and the shape of tenpai (if any) may
        // be changed after an ankan. See the example in daiminkan.
        self.update_shanten();
        self.update_waits();

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

    /// Pads the kawa (discard pile) for pon or daiminkan actions.
    /// This ensures the discard pile has the correct structure when a player
    /// calls pon or daiminkan from another player's discard.
    pub(super) fn pad_kawa_for_pon_or_daiminkan(&mut self, abs_actor: u8, abs_target: u8) -> Result<()> {
        let mut i = (abs_target + 1) % 4;
        while i != abs_actor {
            let rel = self.rel(i);
            // kawa capacity is 55, which is the theoretical maximum
            // If this fails, it indicates invalid game log data or a bug in game logic
            // 业务规则：一个玩家最多只能打55张牌，这是理论最大值
            // 如果超过55，说明游戏状态不一致或日志数据有问题，必须panic
            let kawa_len = self.kawa[rel].len();
            assert!(
                kawa_len < 55,
                "kawa capacity overflow in pad_kawa_for_pon_or_daiminkan: player {} (relative {}) has {} discards, attempting to pad. Maximum is 55. This indicates a fundamental bug in game logic or invalid game log data. abs_actor: {}, abs_target: {}, kyoku: {}, at_turn: {}, tiles_left: {}",
                i,
                rel,
                kawa_len,
                abs_actor,
                abs_target,
                self.kyoku,
                self.at_turn,
                self.tiles_left
            );
            // Push to kawa (will panic if capacity exceeded, but we've already checked)
            self.kawa[rel].push(None);
            i = (i + 1) % 4;
        }
        Ok(())
    }



    /// Can be called at either 3n+1 or 3n+2.
    ///
    /// For 3n+2, the return value of `shanten::calc_all` may be `-1`. We don't
    /// allow `-1` and it will be written as `0` in order for
    /// `_shanten_discards` to be calculated properly.
    pub(crate) fn update_shanten(&mut self) {
        // Check ding_que rule first: if holding DingQue suit tiles, cannot agari/tenpai.
        // Set shanten to 8 (infinity/invalid). Normal max shanten is 6.
        if crate::ding_que::has_ding_que_tiles(&self.tehai, self.ding_que) {
            self.shanten = 8;
            return;
        }

        // Use dynamic calculation instead of fragile state variable
        // This fixes the bug where Kan (Gang) operations caused tehai_len_div3 to desync
        let current_len_div3 = (self.tehai.iter().sum::<u8>() / 3) as u8;
        self.shanten = shanten::calc_all(&self.tehai, current_len_div3, self.ding_que).max(0);
        
        debug_assert!(matches!(self.shanten, 0..=8));
    }

    /// Must be called at 3n+2.
    pub(crate) fn update_shanten_discards(&mut self) {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        self.next_shanten_discards.fill(false);
        self.keep_shanten_discards.fill(false);
        self.has_next_shanten_discard = false;

        // `tehai_len_div3` can desync after kan/pon flows; derive it from tehai shape instead.
        // At 3n+2, every candidate discard produces a 3n+1 hand, so the divisor is constant.
        let tehai_sum: u8 = self.tehai.iter().sum();
        debug_assert!(tehai_sum >= 1, "tehai sum must be >= 1 at 3n+2");
        let len_div3_after_discard: u8 = ((tehai_sum.saturating_sub(1)) / 3) as u8;

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
            let shanten_after = shanten::calc_all(&tehai, len_div3_after_discard, self.ding_que);
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

    // Caller must assure current tehai is 3n+1, and `self.shanten` must be up
    // to date and correct.
    pub(crate) fn update_waits(&mut self) {
        assert!(!self.last_cans.can_discard, "tehai is not 3n+1");

        self.waits.fill(false);

        if self.shanten > 0 {
            return;
        }

        // `tehai_len_div3` can desync after kan/pon flows; derive it from tehai shape instead.
        // At 3n+1, every candidate wait adds one tile to make a 3n+2 hand, so the divisor is constant.
        let tehai_sum: u8 = self.tehai.iter().sum();
        let len_div3_after_draw: u8 = ((tehai_sum.saturating_add(1)) / 3) as u8;

        for (t, is_wait) in self.waits.iter_mut().enumerate() {
            if self.tehai[t] == 4 {
                continue;
            }
            let mut tehai_after = self.tehai;
            tehai_after[t] += 1;

            if shanten::calc_all(&tehai_after, len_div3_after_draw, self.ding_que) == -1 {
                *is_wait = self.tiles_seen[t] < 4;
            }
        }
    }

    /// Update forbidden_tiles based on Ding Que rule.
    /// If the player has tiles of the Ding Que suit, they MUST discard them first.
    /// In this case, all cards of other suits become forbidden.
    fn update_ding_que_forbidden_tiles(&mut self) {
        if let Some(ding_que_suit) = self.ding_que {
            let (ding_que_start, ding_que_end) = crate::ding_que::suit_range(ding_que_suit);
            
            // Check if hand still has any ding_que suit tiles
            let has_ding_que_tiles = crate::ding_que::has_suit_tiles(&self.tehai, ding_que_suit);
                
            if has_ding_que_tiles {
                // Determine which tiles are forbidden (all non-DingQue tiles)
                for i in 0..27 {
                    if i < ding_que_start || i >= ding_que_end {
                        self.forbidden_tiles[i] = true;
                    }
                }
            }
        }
    }
}

