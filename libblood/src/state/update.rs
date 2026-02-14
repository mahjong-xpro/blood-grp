use super::PlayerState;
use super::action::ActionCandidate;
use super::item::{KawaItem, Sutehai};
use crate::algo::agari::{Agari, AgariCalculator};
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
        // 过手胡（Temporary Furiten）检测
        // 仅在放弃了 **荣和** 机会时触发：如果上一步能 ron 但未 ron，
        // 则在下次自摸之前不能 ron 同一张牌。
        //
        // NOTE:
        // 这里不能放在 `!event.is_in_game_announce()` 分支内。
        // 若他家先和（Event::Hora）而自己选择了不和，此事件是 announce，
        // 仍应视为“过手胡”并置 temporary_furiten=true。
        if self.last_cans.can_ron_agari {
            let passed = !matches!(event, Event::Hora { actor, .. } if *actor == self.player_id);
            if passed {
                self.temporary_furiten = true;
                // FIX: 多次放弃荣和时，取历史最大番数而非直接覆盖。
                // 旧逻辑：放弃 3 番 → 放弃 1 番 → furiten_passed_ron_fan = 1，
                // 此时 2 番荣和会通过 (2 > 1)，但应被阻止 (2 < 3)。
                self.furiten_passed_ron_fan = match (self.furiten_passed_ron_fan, self.current_ron_fan) {
                    (Some(prev), Some(cur)) => Some(prev.max(cur)),
                    (prev, cur) => prev.or(cur),
                };
            }
        }

        if !event.is_in_game_announce() {

            self.last_cans = ActionCandidate {
                target_actor: event.actor().unwrap_or(self.player_id),
                ..Default::default()
            };
            self.ankan_candidates.clear();
            self.kakan_candidates.clear();
            self.current_ron_fan = None;

            // 抢杠机会必须在事件流转时清除，否则后续荣和会被错误标记为抢杠
            self.chankan_chance = None;
            self.chankan_kakan_actor = None;
            self.chankan_kakan_tile = None;

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
            Event::Hora { actor, target, deltas, .. } => {
                self.hora(actor, target, deltas)?;
                // NOTE: 不在此处清除 last_cans。
                // Hora 是 announce 事件，可能连续出现（多家荣和）。
                // 如果在首个 Hora 广播时清除所有玩家的 last_cans，
                // 后续玩家调用 agari_points() 时会因 can_ron=false 而失败。
                //
                // 数据集回放中可能因 stale last_cans 产生虚假训练样本的问题，
                // 由 gameplay.rs 的 Hora 窗口跳过逻辑处理（而非在状态机层面清除）。
            }
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
        self.furiten_passed_ron_fan = None;
        self.current_ron_fan = None;
        self.tiles_seen.fill(0);
        self.keep_shanten_discards.fill(false);
        self.next_shanten_discards.fill(false);
        self.forbidden_tiles.fill(false);
        self.players_agari.fill(false);
        self.oya = self.rel(oya) as u8;
        self.kyoku = kyoku - 1;

        self.scores = scores;
        self.scores.rotate_left(self.player_id as usize);

        self.ding_que = None;
        self.other_ding_que.fill(None);
        self.has_agari = false;

        self.last_self_tsumo = None;
        self.last_kawa_tile = None;

        self.ankan_candidates.clear();
        self.kakan_candidates.clear();
        self.chankan_chance = None;
        self.chankan_kakan_actor = None;
        self.chankan_kakan_tile = None;
        self.pending_kakan_tile = None;
        self.last_discard_was_after_kan = false;
        self.intermediate_kan.clear();
        
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

                // 天胡检查：定缺完成后，如果庄家初始手牌（14张）已成和牌型，
                // 应当允许宣告自摸和牌（天胡）。
                if let Some(winning_tile) = self.last_self_tsumo {
                    let agari_calc = AgariCalculator {
                        tehai: &self.tehai,
                        pons: &self.pons,
                        minkans: &self.minkans,
                        ankans: &self.ankans,
                        winning_tile: winning_tile.as_u8(),
                        is_ron: false,
                        ding_que: self.ding_que,
                        is_after_kan: false,
                        is_kan_discard: false,
                        is_chankan: false,
                        exclude_gen_tile: None,
                        is_haidi: false,
                        is_tianhu: false, // board.rs 会在 handle_hora 时根据 tiles_left 判断天胡
                        is_dihu: false,
                        fan_config: self.fan_config,
                    };
                    self.last_cans.can_tsumo_agari = agari_calc.has_yaku();
                }

                // FIX: 庄家 14 张初始手牌中可能存在暗杠/加杠候选，
                // 与 tsumo() 中 lines 320-364 相同的逻辑。
                // 血战到底无「四杠散了」规则，不使用 kans_on_board 做限制。
                // 此处 tiles_left > 0 恒成立（定缺刚结束，tiles_left == 55），仅作安全守卫。
                if self.tiles_left > 0 {
                    self.tehai
                        .iter()
                        .enumerate()
                        .filter(|&(_, &count)| count > 0)
                        .for_each(|(tid, &count)| {
                            let tile = must_tile!(tid);
                            if crate::ding_que::is_ding_que_tile(tile.as_usize(), self.ding_que) {
                                return;
                            }
                            if count == 4 {
                                self.last_cans.can_ankan = true;
                                self.ankan_candidates.push(tile);
                            } else if self.pons.contains(&(tid as u8)) {
                                self.last_cans.can_kakan = true;
                                self.kakan_candidates.push(tile);
                            }
                        });
                }
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
        self.furiten_passed_ron_fan = None;
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
        // FIX: 非庄家首次摸牌时 self.shanten 可能是 StartKyoku 阶段（ding_que=None）
        // 计算的值，定缺选择后未被更新。update_shanten_discards() 内部用 self.ding_que
        // 计算 shanten_after 但与过时的 self.shanten 比较，会产生错误的分类。
        // 全量重算确保 self.shanten 反映当前 ding_que 状态。
        self.update_shanten();
        self.update_shanten_discards();

        // Tsumo check: 自摸时以 has_yaku 为准，不再依赖 waits[]，避免 waits 漏判导致无法胡牌
        // 摸牌后 self.tehai 已包含摸到的牌（14 张），直接检查是否成牌
        {
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
                fan_config: self.fan_config,
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
        // 血战到底无「四杠散了」规则：不再用 kans_on_board < 4 限制杠操作。
        // tiles_left > 0 已在上方 early-return 中保证，此处直接计算候选。
        {
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
                            kakan_len < 4,
                            "kakan_candidates capacity overflow: player {} has {} kakan candidates, attempting to add one more. Maximum is 4. kyoku: {}, at_turn: {}, tiles_left: {}",
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
        // (non-self actors already returned at line 308, so this is always self)
        self.update_ding_que_forbidden_tiles();

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

            // 定缺修正：当手牌中存在定缺牌时（void_count > 0），clean_len_div3
            // 可能因打牌而跳档（clean_sum 跨越 3 的倍数边界），
            // 导致向听改善量可达 2，而 `shanten -= 1` 仅减 1。
            //
            // 触发条件：
            // (1) 打出定缺牌 → void_count 减少 + clean tile count 增加，双重变化
            // (2) 打出非定缺牌但 void_count > 0 → clean_sum 跨越 3k 边界时
            //     clean_len_div3 跳档，structural shanten 可下降 2
            //
            // 仅当 void_count == 0 时 clean_len_div3 = len_div3 不变，
            // 标准增量 `-= 1` 才安全。
            if self.next_shanten_discards[pai.as_usize()] {
                if crate::ding_que::is_ding_que_tile(pai.as_usize(), self.ding_que)
                    || crate::ding_que::ding_que_forced_range(&self.tehai, self.ding_que).is_some()
                {
                    // 手中仍有定缺牌 或 打出的就是定缺牌 → 全量重算
                    self.update_shanten();
                } else {
                    self.shanten -= 1;
                }
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
        // 注意：不检查 tiles_seen < 4，因为 witness_tile 已在上方执行（tiles_seen 已含本次弃牌）。
        // 当对手打出某牌的第 4 张时，witness 后 tiles_seen=4，但该牌仍可用于荣和。
        // tehai < 4 足以防止手牌溢出，hand14_for_division 内部也有 > 4 检查。
        let can_add_tile = hand_total == 13
            && self.tehai[pai_idx] < 4;

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
                fan_config: self.fan_config,
            };
            if let Some(Agari::Fan(fan)) = agari_calc.agari() {
                self.current_ron_fan = Some(fan);
                true
            } else {
                false
            }
        };

        // Rule extension: 过手加番可胡
        // If the player is in temporary furiten, ron is allowed only when
        // current fan is strictly greater than the fan of the passed ron.
        if self.temporary_furiten && can_add_tile {
            self.last_cans.can_ron_agari = if let Some(Agari::Fan(fan)) = {
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
                    fan_config: self.fan_config,
                };
                agari_calc.agari()
            } {
                self.current_ron_fan = Some(fan);
                self.furiten_passed_ron_fan.is_some_and(|passed_fan| fan > passed_fan)
            } else {
                false
            };
        }

        if self.tiles_left == 0 {
            return Ok(());
        }

        // Check if the discarded tile is the ding_que suit
        let is_ding_que_tile = crate::ding_que::is_ding_que_tile(pai.as_usize(), self.ding_que);

        // 基础规则：定缺花色不能碰或明杠
        // 血战到底无「四杠散了」规则：不再用 kans_on_board < 4 限制大明杠。
        // tiles_left > 0 已由上方 early-return (line 530) 保证。
        if !is_ding_que_tile {
            self.last_cans.can_pon = self.tehai[pai.as_usize()] >= 2;
            self.last_cans.can_daiminkan = self.tehai[pai.as_usize()] == 3;
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
        self.furiten_passed_ron_fan = None;
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

        Ok(())
    }

    fn hora(&mut self, actor: u8, target: u8, deltas: Option<[i32; 4]>) -> Result<()> {
        let actor_rel = self.rel(actor);
        self.players_agari[actor_rel] = true;
        // PERF-01: players_agari 变化影响 SP 的 active_players 计算
        self.invalidate_sp_cache();
        // 自身和牌时设置 has_agari，确保独立使用 PlayerState（日志回放、数据集生成）时
        // 后续事件（tsumo/dahai/kakan）正确跳过已和牌的玩家
        if actor_rel == 0 {
            self.has_agari = true;
        }

        // Chankan (抢杠) PlayerState 回退：
        // 加杠后被抢杠（target == self），杠被撤销，副露必须从 minkan 恢复为 pon。
        //
        // 职责划分：board.rs 只负责棋盘级状态（gang_history、last_kan_actor、kans 计数器），
        // 所有 PlayerState 级回退（minkans→pons、fuuro_overview、intermediate_kan、
        // kans_on_board、shanten、waits）由此处统一处理。
        // 这保证了 arena 对局和 log replay/dataset 生成共用同一套回退逻辑。
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

                // FIX: 抢杠后杠被撤销，kans_on_board 必须递减，否则后续杠操作会被错误阻止
                self.kans_on_board = self.kans_on_board.saturating_sub(1);

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

        // FIX BUG B/C/D: 抢杠后非受害者玩家的状态清理。
        //
        // kakan() 为 ALL 玩家更新了 intermediate_kan / kans_on_board / fuuro_overview。
        // 上面的 victim 路径（target == self.player_id）已处理受害者的回退。
        // 此处处理和牌者和旁观者：
        //   - actor != target  → 这是荣和（不是自摸），排除杠上开花
        //   - intermediate_kan 非空 → 有未消费的杠记录（chankan 前未经过 dahai 清除）
        //   - target != self.player_id → 不是受害者（受害者已在上面处理）
        if target != self.player_id && actor != target && !self.intermediate_kan.is_empty() {
            let chankan_tile = self.intermediate_kan[0];
            self.intermediate_kan.clear();
            self.kans_on_board = self.kans_on_board.saturating_sub(1);

            // 回退 fuuro_overview：kakan 在受害者的副露中添加了第 4 张牌
            let victim_rel = self.rel(target);
            for fuuro in &mut self.fuuro_overview[victim_rel] {
                if fuuro.first().is_some_and(|t0| *t0 == chankan_tile) && fuuro.len() == 4 {
                    fuuro.pop();
                    break;
                }
            }
        }

        // FIX: 自摸和牌（含杠上开花）后清除 intermediate_kan。
        // 正常流程：杠→岭上摸→打牌（dahai 清除 intermediate_kan）。
        // 杠上开花：杠→岭上摸→自摸和（没有打牌！intermediate_kan 残留）。
        // 残留的 intermediate_kan 会导致下一个玩家的打牌被错误标记为
        // is_kan_discard（杠上炮），产生虚假 +1 番。
        // 注意：杠上开花本身的番数由 is_after_kan (at_rinshan) 判定，不依赖 intermediate_kan。
        if actor == target && !self.intermediate_kan.is_empty() {
            self.intermediate_kan.clear();
        }

        self.apply_score_deltas(deltas);
        
        Ok(())
    }
    
    fn ryukyoku(&mut self, deltas: Option<[i32; 4]>) -> Result<()> {
        self.apply_score_deltas(deltas);
        Ok(())
    }

    fn daiminkan(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 3], deltas: Option<[i32; 4]>) -> Result<()> {
        self.apply_score_deltas(deltas);

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
        // 只跟踪最近一次杠（打牌前重置）
        self.intermediate_kan.clear();
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
            return Ok(());
        }

        self.at_rinshan = true;
        self.temporary_furiten = false;
        self.furiten_passed_ron_fan = None;

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
        self.apply_score_deltas(deltas);

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
        // 只跟踪最近一次杠（打牌前重置）
        self.intermediate_kan.clear();
        self.intermediate_kan.push(pai);
        self.kans_on_board += 1;

        if actor_rel != 0 {
            // 已和牌的玩家不再响应任何事件（与 tsumo/dahai 守卫一致）
            if self.has_agari {
                return Ok(());
            }

            self.witness_tile(pai)?;
            self.last_kawa_tile = Some(pai); // for getting winning tile in self.agari

            // 槍槓 (抢杠)：与点炮 Ron 一致，以 has_yaku 为准，不依赖 waits[]
            // 抢杠时不检查 tiles_seen：加杠的牌必定可用（正在被加杠）
            let hand_total: u8 = self.tehai.iter().sum::<u8>()
                + 3 * self.pons.len() as u8
                + 3 * self.minkans.len() as u8
                + 3 * self.ankans.len() as u8;
            let pai_idx = pai.as_usize();
            let can_chankan = hand_total == 13
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
                    fan_config: self.fan_config,
                };

                if let Some(Agari::Fan(fan)) = agari_calc.agari() {
                    self.current_ron_fan = Some(fan);
                    self.last_cans.can_ron_agari = if self.temporary_furiten {
                        self.furiten_passed_ron_fan.is_some_and(|passed_fan| fan > passed_fan)
                    } else {
                        true
                    };
                    self.chankan_chance = Some(());
                    self.chankan_kakan_actor = Some(actor);
                    self.chankan_kakan_tile = Some(pai.as_u8());
                }
            }

            return Ok(());
        }

        self.at_rinshan = true;
        self.temporary_furiten = false;
        self.furiten_passed_ron_fan = None;
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
        //
        // 定缺修正：加杠牌必是非缺门（tsumo 阶段已过滤），但当手中仍有缺门牌
        // （void > 0）且 clean tile count 从 C → C-1 跨越 3 的整数倍时，
        // clean_len_div3 跳档，结构向听可能改善 2 档。此时 `-= 1` 不够。
        // 用 `ding_que_forced_range` 判断是否仍有缺门牌；若有，走全量重算。
        if self.next_shanten_discards[pai.as_usize()] {
            if crate::ding_que::ding_que_forced_range(&self.tehai, self.ding_que).is_some() {
                self.update_shanten();
            } else {
                self.shanten -= 1;
            }
        } else if !self.keep_shanten_discards[pai.as_usize()] {
            self.update_shanten();
        }
        self.update_waits();

        Ok(())
    }

    fn ankan(&mut self, actor: u8, consumed: [Tile; 4], deltas: Option<[i32; 4]>) -> Result<()> {
        self.apply_score_deltas(deltas);

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
        self.furiten_passed_ron_fan = None;
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

    /// 将绝对 deltas 旋转到自身视角后累加到 self.scores。
    #[inline]
    fn apply_score_deltas(&mut self, deltas: Option<[i32; 4]>) {
        if let Some(d) = deltas {
            let mut d_rel = d;
            d_rel.rotate_left(self.player_id as usize);
            for i in 0..4 {
                self.scores[i] += d_rel[i];
            }
        }
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

        // PERF-01: 手牌变化 → SP 缓存失效
        self.invalidate_sp_cache();

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
        // calc_all 已正确处理定缺：将定缺牌置零后计算结构向听，再加上 void_count 惩罚。
        // 例如：结构向听 1 + 定缺牌 2 张 = 向听 3。
        //
        // 之前此处硬编码 shanten = 8（当有定缺牌时），但 update_shanten_discards()
        // 使用 calc_all 计算 shanten_after（返回 2-6），导致几乎所有牌都被标记为
        // next_shanten_discard（因为 2 < 8），然后 dahai() 的 shanten -= 1 使
        // shanten 从 8 逐步漂移到 7→6→5...，严重偏离真实值。
        //
        // 移除硬编码后，shanten 与 shanten_after 使用同一公式（calc_all），
        // 增量路径（shanten -= 1）和全量路径保持一致。
        let current_len_div3 = (self.tehai.iter().sum::<u8>() / 3) as u8;
        self.shanten = shanten::calc_all(&self.tehai, current_len_div3, self.ding_que).max(0);
        
        // 定缺惩罚 = structural_shanten + void_count。极端情况下
        // （如手中大量缺门牌）shanten 可达 ~15，放宽上限。
        debug_assert!(matches!(self.shanten, 0..=20));
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
    /// 定缺出牌约束：手牌中仍有缺门花色时，非缺门牌标记为 forbidden。
    ///
    /// **仅做 additive 标记（置 true），绝不清除其它 restriction（如碰后禁打）。**
    /// 调用者在适当时机已执行 `forbidden_tiles.fill(false)` 并可能设置碰后禁打，
    /// 本函数只在此基础上叠加定缺约束。
    fn update_ding_que_forbidden_tiles(&mut self) {
        if let Some((start, end)) = crate::ding_que::ding_que_forced_range(&self.tehai, self.ding_que) {
            self.forbidden_tiles[..start].fill(true);
            self.forbidden_tiles[end..].fill(true);
        }
        // 缺门牌已清空时不做任何操作：
        // - 之前的 fill(false) 已清除上一轮的定缺 forbidden 标记
        // - 碰后禁打等其它 restriction 必须保留，不可在此处清除
    }
}

