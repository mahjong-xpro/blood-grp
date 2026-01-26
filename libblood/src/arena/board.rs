use super::result::KyokuResult;
use crate::array::Simple2DArray;
use crate::consts::oracle_obs_shape;
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::t;
use std::convert::TryInto;
use std::{array, mem};

use anyhow::{Context, Result, bail, ensure};
use derivative::Derivative;
use ndarray::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use sha3::{Digest, Sha3_256};
use serde::{Serialize, Deserialize};
use tinyvec::ArrayVec;

///
/// Game ends when 3 players have agari (和牌) or when tiles are exhausted (流局).
#[derive(Debug, Default)]
pub struct Board {
    /// Counts from 0 (for recording only, no game flow impact)
    pub kyoku: u8,
    /// [25000; 4]
    pub scores: [i32; 4],

    pub haipai: [[Tile; 13]; 4],
    /// Goes backward (pop)
    pub yama: Vec<Tile>,
}

#[derive(Derivative)]
#[derivative(Default)]
pub struct BoardState {
    board: Board,
    // Absolute seat, with the oya always being 0
    oya: u8,
    player_states: [PlayerState; 4],

    #[derivative(Default(value = "[false; 4]"))]
    players_agari: [bool; 4],
    agari_count: u8,
    kyoku_deltas: [i32; 4],

    #[derivative(Default(value = "56"))]
    tiles_left: u8,
    tsumo_actor: u8,
    kans: u8,
    // check_four_kan removed


    // For Score Transfer (Hujiaozhuanyi)
    #[derivative(Default(value = "0"))]
    last_kan_revenue: i32,
    #[derivative(Default(value = "None"))]
    last_kan_actor: Option<u8>,
    
    // Gang History for Tui Shui (Refund Gangs)
    #[derivative(Default(value = "Vec::new()"))]
    gang_history: Vec<GangRecord>,

    // 定缺选择阶段状态
    #[derivative(Default(value = "false"))]
    ding_que_phase: bool,
    #[derivative(Default(value = "[false; 4]"))]
    ding_que_selected: [bool; 4],

    log: Vec<EventExt>,
}

pub struct AgentContext<'a> {
    pub player_states: &'a [PlayerState; 4],
    pub log: &'a [EventExt],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GangRecord {
    pub actor: u8,
    pub deltas: [i32; 4], // The exact points transfer that occurred (to be reversed if needed)
}

#[derive(Clone, Copy)]
pub enum Poll {
    InGame,
    End,
}

impl Board {
    pub fn init_from_seed(&mut self, game_seed: (u64, u64)) {
        let (nonce, key) = game_seed;
        let kyoku_seed = Sha3_256::new()
            .chain_update(nonce.to_le_bytes())
            .chain_update(key.to_le_bytes())
            .finalize()
            .into();
        let mut rng = ChaCha12Rng::from_seed(kyoku_seed);
        let mut seq = UNSHUFFLED;
        seq.shuffle(&mut rng);

        // Deal 13 tiles to each of 4 players
        self.haipai = array::from_fn(|i| seq[i * 13..(i + 1) * 13].try_into().unwrap());
        let idx = 13 * 4;

        // Remaining tiles go to yama (108 - 52 = 56 tiles)
        self.yama = seq[idx..].to_vec();
        assert_eq!(self.yama.len(), 56);
    }

    pub fn into_state(self) -> BoardState {
        let oya = self.kyoku % 4;

        BoardState {
            board: self,
            oya,
            player_states: array::from_fn(|i| PlayerState::new(i as u8)),
            agari_count: 0,
            ..Default::default()
        }
    }
}

impl BoardState {
    /// Returns iff any player on the board can act or the kyoku has ended.
    pub fn poll(&mut self, reactions: [EventExt; 4]) -> Result<Poll> {
        let mut loop_count = 0;
        const MAX_LOOP_COUNT: usize = 1000; // 防止无限循环
        
        // Use a mutable local copy of reactions that can be cleared after each step
        let mut current_reactions = reactions;

        loop {
            loop_count += 1;
            if loop_count > MAX_LOOP_COUNT {
                bail!(
                    "poll() loop exceeded maximum iterations ({}). This indicates a deadlock bug. \
                    Current state: ding_que_phase={}, tiles_left={}, can_act={:?}",
                    MAX_LOOP_COUNT,
                    self.ding_que_phase,
                    self.tiles_left,
                    self.player_states.iter().map(|s| s.last_cans().can_act()).collect::<Vec<_>>()
                );
            }
            
            let poll = self.step(&current_reactions)?;
            match poll {
                Poll::InGame => {
                    // 在定缺选择阶段，即使can_act()返回false，也应该返回InGame
                    // 因为我们需要等待Agent返回反应（或自动选择定缺）
                    if self.ding_que_phase {
                        return Ok(poll);
                    }
                    // 正常游戏阶段，检查是否有玩家可以行动
                    if self.player_states.iter().any(|c| c.last_cans().can_act()) {
                        return Ok(poll);
                    }
                    // 如果没有玩家可以行动，但是step()返回了InGame，说明是内部状态流转（如发牌后立即判断及流局等）
                    // 继续循环 step()直到有玩家可以行动或游戏结束
                    // 避免不必要的空步返回给Python层
                }
                Poll::End => {
                    self.add_log_no_meta(Event::EndKyoku);
                    vec_add_assign(&mut self.board.scores, &self.kyoku_deltas);
                    return Ok(poll);
                }
            };
            current_reactions = Default::default();
        }
    }

    #[inline]
    pub fn agent_context(&self) -> AgentContext<'_> {
        AgentContext {
            player_states: &self.player_states,
            log: &self.log,
        }
    }

    #[inline]
    pub const fn is_ding_que_phase(&self) -> bool {
        self.ding_que_phase
    }

    #[inline]
    pub const fn ding_que_selected(&self, player_id: usize) -> bool {
        self.ding_que_selected[player_id]
    }

    #[inline]
    pub const fn end(&self) -> KyokuResult {
        KyokuResult {
            kyoku: self.board.kyoku,
            can_renchan: false,
            scores: self.board.scores,
        }
    }

    #[inline]
    pub fn take_log(&mut self) -> Vec<EventExt> {
        mem::take(&mut self.log)
    }

    #[inline]
    fn add_log(&mut self, ev: EventExt) {
        self.log.push(ev);
    }

    #[inline]
    fn add_log_no_meta(&mut self, ev: Event) {
        self.log.push(EventExt::no_meta(ev));
    }

    #[inline]
    fn broadcast(&mut self, ev: &Event) {
        for s in &mut self.player_states {
            s.update(ev).expect("fatal internal bug in BoardState");
        }
    }

    fn haipai(&mut self) -> Result<()> {
        let start_kyoku = Event::StartKyoku {
            kyoku: self.oya + 1,
            oya: self.oya,
            scores: self.board.scores,
            tehais: self.board.haipai,
        };
        self.broadcast(&start_kyoku);
        self.add_log_no_meta(start_kyoku);

        // 进入定缺选择阶段（基础规则：血战到底必须在打牌前选择定缺）
        self.ding_que_phase = true;
        self.ding_que_selected = [false; 4];

        Ok(())
    }

    pub(crate) fn exhaustive_ryukyoku(&mut self) {
        // Flow: 1. Check 查花猪 (huazhu), 2. Check 查大叫 (tenpai)
        let mut final_deltas = [0; 4];

        // Step 1: 查花猪 (Check Huazhu - players with ding_que suit tiles remaining)
        // 花猪的定义：选择了定缺，但手牌中还有定缺花色的牌
        // 如果玩家没有选择定缺（ding_que == None），不应该被认为是花猪
        let huazhu_actors: ArrayVec<[_; 4]> = self
            .player_states
            .iter()
            .enumerate()
            .filter(|&(_, s)| {
                // 改进：只有选择了定缺但还有定缺花色牌的玩家才是花猪
                if let Some(_) = s.ding_que {
                    !s.check_ding_que_complete() // 选择了定缺但还有定缺花色牌
                } else {
                    false // 没有选择定缺，不是花猪
                }
            })
            .map(|(i, _)| i)
            .collect();

        // 检查是否有玩家没有选择定缺（这应该是游戏状态错误）
        let players_with_ding_que: usize = self
            .player_states
            .iter()
            .filter(|s| s.ding_que.is_some())
            .count();
        
        if players_with_ding_que == 0 {
            // 所有玩家都没有选择定缺，这是游戏状态错误
            // 基础规则：血战到底必须在打牌前选择定缺
            // 如果所有玩家都没有选择定缺，说明游戏流程有问题
            // 这里我们记录警告，但不panic，因为可能是旧日志或测试数据
            log::warn!(
                "All players have no ding_que selected in exhaustive_ryukyoku. This indicates a bug in game flow. \
                In normal gameplay, all players should have selected ding_que before playing."
            );
        }

        if !huazhu_actors.is_empty() {
            // Count eligible targets (Non-Huazhu AND Non-Agari)
            let non_huazhu_targets: Vec<usize> = (0..4)
                .filter(|&i| !huazhu_actors.contains(&i) && !self.players_agari[i])
                .collect();
            let target_count = non_huazhu_targets.len();

            if target_count > 0 {
                // 花猪罚分：花猪向每个非花猪支付极刑（封顶分）
                // 四川麻将规则：查花猪赔给非花猪每家满分（通常是极刑，这里定为16000）
                // Pay 16000 to EACH non-huazhu player.
                let penalty_per_target = 16000;
                
                let mut huazhu_deltas = [0; 4];
                
                // Each Huazhu pays penalty_per_target * target_count
                for &huazhu in &huazhu_actors {
                    huazhu_deltas[huazhu] = -(penalty_per_target * target_count as i32);
                }
                
                // Each Non-Huazhu receives penalty_per_target * huazhu_count
                // Each eligible target receives penalty_per_target * huazhu_count
                for &target in &non_huazhu_targets {
                     huazhu_deltas[target] += penalty_per_target as i32 * huazhu_actors.len() as i32;
                }

                
                // Wait, if I filter Agari players out of reception, I must also ensure
                // Huazhu calculation didn't assume they exist.
                // `huazhu_actors` iteration was correct.
                // But `non_huazhu_count` above used `4 - len`. This counted Agari players.
                // We should recount `non_huazhu_alive_count`.
                
                // Let's refine the logic inside the loop manually.
                
                vec_add_assign(&mut final_deltas, &huazhu_deltas);
            } else {
                // 如果所有玩家都是花猪，不计算花猪罚分（因为没有人可以接收罚分）
                // 这是边界情况，在正常游戏中不应该发生
                log::warn!(
                    "All players are huazhu in exhaustive_ryukyoku. No penalty applied. \
                    This is an edge case that should not occur in normal gameplay."
                );
            }
        }

        // Step 2: 查大叫 (Check Tenpai - exclude huazhu players)
        // 排除花猪玩家后，未听牌玩家向听牌玩家赔付
        // 改进：只检查选择了定缺的玩家（基础规则：只有选择了定缺的玩家才参与查大叫）
        let _tenpai_actors: ArrayVec<[_; 4]> = self
            .player_states
            .iter()
            .enumerate()
            .filter(|&(i, s)| {
                // 排除花猪玩家
                !huazhu_actors.contains(&i)
                // 改进：只检查选择了定缺的玩家
                && s.ding_que.is_some()
                && s.shanten() == 0
            })
            .map(|(i, _)| i)
            .collect();

        // Calculate Cha Da Jiao (No-Ten Penalty)
        // Standard Sichuan Rule:
        // No-Ten players pay Tenpai players based on the Tenpai player's max possible hand value.
        // Agari players and Huazhu players are excluded.
        
        // 1. Identify Tenpai players (who are alive, not Huazhu)
        // 2. Calculate Max Fan for each Tenpai player
        // 3. Alive No-Ten players pay that amount to the Tenpai player.
        
        let mut tenpai_details = Vec::new();
        
        for i in 0..4 {
             if self.players_agari[i] || huazhu_actors.contains(&i) {
                 continue;
             }
             
             let state = &self.player_states[i];
             if state.shanten() == 0 && state.ding_que.is_some() {
                 // Is Tenpai. Calculate Max Point.
                 // Heuristic: Check all waiting tiles. Assume Tsumo. Calculate Points. Take Max.
                 let mut max_points = 0;
                 // state.waits is [bool; 27]
                 for (tid, &is_wait) in state.waits().iter().enumerate() {
                     if is_wait {
                         // Mock tsumo event for calculation
                         // Clone state is expensive? No, AgariCalculator takes references.
                         // But we need to add winning tile to tehai temporarily?
                         // AgariCalculator expects `tehai` to include winning tile? 
                         // Yes: "Must include the winning tile"
                         
                         let mut temp_tehai = state.tehai;
                         temp_tehai[tid] += 1;
                         
                         let agari_calc = crate::algo::agari::AgariCalculator {
                             tehai: &temp_tehai,

                             pons: &state.pons,
                             minkans: &state.minkans,
                             ankans: &state.ankans,
                             winning_tile: tid as u8,
                             is_ron: false, // Assume Tsumo for max value? 
                             // Usually "Retreat" based on what? 
                             // Cha Da Jiao usually pays according to Tsumo-like scores or Ron-like?
                             // Rule: "Pei (Pay)". Effectively Ron payment X players?
                             // Sichuan scoring is consistent (Ron = Tsumo_Ko).
                             // So just calc Point.
                             ding_que: state.ding_que,
                             is_after_kan: false, // Assuming normal win
                             is_kan_discard: false,
                             is_chankan: false,
                             exclude_gen_tile: None,

                             is_haidi: false, // Do not give Haidi bonus for Cha Da Jiao theoretical hand
                             // But usually Cha Da Jiao doesn't include Haidi unless strictly specified.
                             // Let's be conservative: No Haidi bonus for "Theoretical" hand unless they WON on Haidi.
                             // Here they did NOT win. So No Haidi.
                             is_tianhu: false,
                             is_dihu: false,
                         };
                         
                         if let Some(agari) = agari_calc.agari() {
                             let p = agari.point(false).ron; // Use ron value (base points)
                             if p > max_points {
                                 max_points = p;
                             }
                         }
                     }
                 }
                 
                 // If max_points is 0 (can't agari even if tenpai? e.g. no yaku?), treat as No-Ten?
                 // We'll trust `max_points`.
                 if max_points > 0 {
                     tenpai_details.push((i, max_points));
                 }
             }
        }
        
        if !tenpai_details.is_empty() {
            let mut chadajiao_deltas = [0; 4];
            
            // Calculate penalty for No-Ten players
            // Who are No-Ten?
            // Alive, Non-Huazhu, and NOT in tenpai_details
            
            let no_ten_actors: Vec<usize> = (0..4)
                .filter(|&i| !self.players_agari[i] && !huazhu_actors.contains(&i) && !tenpai_details.iter().any(|(t, _)| *t == i))
                .collect();
                
            // Execution: Each No-Ten pays Each Tenpai (points of that Tenpai)
            for &no_ten in &no_ten_actors {
                for &(tenpai, points) in &tenpai_details {
                    chadajiao_deltas[no_ten] -= points;
                    chadajiao_deltas[tenpai] += points;
                }
            }
             vec_add_assign(&mut final_deltas, &chadajiao_deltas);
        }

        // Step 3: 退税 (Tui Shui / Refund Gangs)
        // Rule: Any player who is Huazhu (Pig) OR No-Ten (Wei Ting) must REFUND all Gang income.
        // "Hua Zhu Wu Gang" (Pig has no Gang).
        // "Wei Ting Tui Shui" (No-Ten refunds Gangs).
        // Note: Agari players keep their Gangs. Tenpai players keep their Gangs.
         
        let refund_actors: Vec<usize> = (0..4)
             .filter(|&i| !self.players_agari[i] && !tenpai_details.iter().any(|(t, _)| *t == i))
             .collect();
             
        // Iterate gang history
        for record in &self.gang_history {
             if refund_actors.contains(&(record.actor as usize)) {
                 // Loop over the recorded payment deltas and reverse them
                 // record.deltas[i] is positive if i received money (actor), negative if i paid.
                 // Actually record.deltas[actor] is total revenue (positive).
                 // We need to reverse the transaction.
                 
                 for i in 0..4 {
                     // Check if this player was involved (non-zero delta)
                     if record.deltas[i] != 0 {
                         final_deltas[i] -= record.deltas[i];
                     }
                 }
                 log::info!("Tui Shui: Player {} refunds Gang (Revenue: {})", record.actor, record.deltas[record.actor as usize]);
             }
        }

        vec_add_assign(&mut self.kyoku_deltas, &final_deltas);
        let ryukyoku = Event::Ryukyoku {
            deltas: Some(final_deltas),
        };
        self.add_log_no_meta(ryukyoku);
        // no need to broadcast
    }

    // These functions are removed

    fn handle_hora(
        &mut self,
        single_actor: u8,
        single_target: u8,
        _reactions: &[EventExt; 4],
        _is_multi_ron: bool,
    ) -> Result<()> {
        let is_ron = single_actor != single_target;
        


        // Phase 13: Chankan Refund Logic
        // Strict Rule: If Chankan (Robbing the Kong) occurs, the Kong is invalid.
        // The "Instant Payment" (Gua Feng) that happened in the previous step must be REFUNDED.
        // It should NOT be transferred to the winner (that is for Valid Kongs).
        if is_ron {
            // Check if this is Chankan
            // We check the player state directly to avoid moving `self`
            let is_chankan = self.player_states[single_actor as usize].chankan_kakan_actor.is_some();
            
            if is_chankan {
                if let Some(kan_actor) = self.last_kan_actor {
                    // Assuming Chankan only happens on Kakan (Add Kong), where payment is 1000.
                    // If Kan Actor matches Target (which it must for Chankan), and we have revenue to refund.
                    if kan_actor == single_target && self.last_kan_revenue > 0 {
                        let refund_per_person = 1000;
                        let mut refund_deltas = [0; 4];
                        
                        // Identify who paid and needs refund.
                        // Payers are: Everyone except Kan Actor AND except those who were ALREADY Agari (before this turn).
                        // Note: We check `!self.players_agari[i]` here.
                        // Crucial: This block runs BEFORE we mark current winners as Agari (lines below).
                        // So current winners (who paid the tax) will correctly be identified as Non-Agari here, and get refund.
                        for i in 0..4 {
                            if i != kan_actor as usize && !self.players_agari[i] {
                                refund_deltas[i] += refund_per_person;
                                refund_deltas[kan_actor as usize] -= refund_per_person;
                            }
                        }
                        
                        vec_add_assign(&mut self.kyoku_deltas, &refund_deltas);
                        
                        // Clear revenue so it's not refunded again (Multi-Ron) or transferred later
                        self.last_kan_revenue = 0; 
                    }
                }
            }
        }

        if !self.players_agari[single_actor as usize] {
            self.players_agari[single_actor as usize] = true;
            self.agari_count += 1;
            self.player_states[single_actor as usize].has_agari = true;
        }

        // Check if this is chankan (抢杠)
        // 抢杠：在别人加杠时抢杠和牌，被抢杠的玩家的根不应该计算
        // Note: chankan_chance is private, so we check via chankan_kakan_actor instead
        let is_chankan = is_ron && self.player_states[single_actor as usize].chankan_kakan_actor.is_some();
        let chankan_kakan_actor = if is_chankan {
            self.player_states[single_actor as usize].chankan_kakan_actor
        } else {
            None
        };
        let chankan_kakan_tile = if is_chankan {
            self.player_states[single_actor as usize].chankan_kakan_tile
        } else {
            None
        };

        // Check for Haidi (Last Tile)
        // Haidi Lao Yue (Tsumo) or Haidi Pao (Ron)
        // Check if tiles are exhausted
        let is_haidi = self.tiles_left == 0 || self.board.yama.is_empty();

        // Check for TianHu / DiHu
        // Total tiles: 108. 4 players * 13 = 52. Remaining: 56.
        // First tsumo: 55 left.
        // TianHu: Tsumo, Oya, First Turn (55 left, 0 kans)
        // DiHu: Ron, Target is Oya, First Turn (55 left, 0 kans)
        // Note: For DiHu, single_target is the one who discarded (Oya).
        //       And it must be the FIRST discard of the game.
        //       Usually tiles_left == 55 check implies no other draws happened.
        //       And we need to check if it's the first discard.
        //       Normally if tiles_left == 55, only 1 tile has been drawn (by Oya) and discarded.
        let is_first_turn = self.tiles_left == 55 && self.kans == 0;
        let is_tianhu = is_first_turn && !is_ron && single_actor == self.oya;
        let is_dihu = is_first_turn && is_ron && single_target == self.oya;

        // This uses the actual fan calculation from AgariCalculator
        let point = self.player_states[single_actor as usize]
            .agari_points(is_ron, is_haidi, is_tianhu, is_dihu, &[])
            .context("failed to calculate agari points")?;
        let mut deltas = [0; 4];
        
        if is_ron {
            // Ron: target pays full amount
            // 抢杠时，被抢杠的玩家的根不应该计算
            if is_chankan && chankan_kakan_actor.is_some() && chankan_kakan_tile.is_some() {
                // For chankan, recalculate the kakan player's payment excluding gen
                // The kakan player's hand should be calculated without the kakan tile as gen
                let _kakan_player_state = &self.player_states[chankan_kakan_actor.unwrap() as usize];
                // Note: The kakan player is not agari, so we need to calculate what they would pay
                // But actually, in chankan, the kakan player is the target (single_target)
                // So we need to recalculate their payment amount excluding the gen
                // However, the kakan player is not agari, so we can't use agari_points_exclude_gen directly
                // Instead, we need to calculate the payment based on the winning player's fan
                // but adjust for the kakan player's gen exclusion
                // Actually, the payment is based on the winning player's fan, not the kakan player's
                // So we don't need to recalculate the kakan player's agari points
                // The gen exclusion only affects the kakan player's own hand evaluation, not the payment
                // Wait, let me re-read the user's explanation...
                // "抢杠时，加杠的玩家的根不应该计算" - this means the kakan player's gen should not be counted
                // But the payment is from the kakan player to the winning player
                // So we need to know: does the payment amount depend on the kakan player's gen?
                // So the gen exclusion for the kakan player doesn't affect the payment amount
                // The gen exclusion only affects the kakan player's own hand evaluation (if they were to agari)
                // So we don't need to do anything special here - the payment is correct as is
                deltas[single_target as usize] = -point.ron;
                deltas[single_actor as usize] = point.ron;
            } else {
                deltas[single_target as usize] = -point.ron;
                deltas[single_actor as usize] = point.ron;
            }

            // Chankan Refund (Gua Feng Chexiao)
            // If this is Chankan, we must invalid the previous Kakan's instant payment (Gua Feng)
            if is_chankan {
                 if let Some(kan_actor) = self.last_kan_actor {
                     if kan_actor == single_target && self.last_kan_revenue > 0 {
                         // Recalculate who paid to refund them
                         let refund_per_person = 1000;
                         let mut total_refund = 0;
                         for i in 0..4 {
                             if i != single_target as usize && !self.players_agari[i] {
                                 deltas[i] += refund_per_person;
                                 total_refund += refund_per_person;
                             }
                         }
                         deltas[single_target as usize] -= total_refund;
                         
                         // Clear revenue so subsequent winners (Multi-Ron) don't double refund
                         self.last_kan_revenue = 0;
                         
                         // Remove the invalidated Kakan from gang history
                         // Checks for safety: ensure the last record matches our expectation
                         if let Some(last_gang) = self.gang_history.last() {
                             if last_gang.actor == kan_actor {
                                 self.gang_history.pop();
                             }
                         }
                     }
                 }
            }
            
            // Score Transfer (Hujiaozhuanyi)
            if is_ron {
                 // Check if the target (discarder) was the last kan actor
                 // and if we have revenue to transfer
                 if let Some(kan_actor) = self.last_kan_actor {
                     if kan_actor == single_target && self.last_kan_revenue > 0 {
                         // Transfer: Subtract from discarder, Add to winner
                         deltas[single_target as usize] -= self.last_kan_revenue;
                         deltas[single_actor as usize] += self.last_kan_revenue;
                         
                         // Clear revenue so subsequent winners (Multi-Ron) don't get double transfer
                         // (assuming 'Heads' rule: first winner takes the transfer)
                         // Since we process winners in loop, this works.
                         self.last_kan_revenue = 0;
                     }
                 }
            }
            
        } else {
            // Tsumo: active players pay (no oya advantage)
            let mut real_tsumo_total = 0;
            for i in 0..4 {
                if i != single_actor as usize && !self.players_agari[i] {
                    deltas[i] = -point.tsumo_ko;
                    real_tsumo_total += point.tsumo_ko;
                }
            }
            deltas[single_actor as usize] = real_tsumo_total;
        }

        vec_add_assign(&mut self.kyoku_deltas, &deltas);

        let hora = Event::Hora {
            actor: single_actor,
            target: single_target,
            deltas: Some(deltas),
        };
        self.add_log_no_meta(hora);
        // No need to broadcast
        Ok(())
    }

    fn step(&mut self, reactions: &[EventExt; 4]) -> Result<Poll> {
        if self.agari_count >= 3 {
            return Ok(Poll::End);
        }

        // 只有在tiles_left==56且还没有进入定缺阶段时才调用haipai()
        // 如果已经进入定缺阶段，说明haipai()已经被调用过了
        if self.tiles_left == 56 && !self.ding_que_phase {
            self.haipai()?;
            return Ok(Poll::InGame);
        }

        // 处理定缺选择阶段（基础规则：血战到底必须在打牌前选择定缺）
        if self.ding_que_phase {
            // 处理所有玩家的定缺选择
            for (actor, ev) in reactions.iter().enumerate() {
                if !self.ding_que_selected[actor] {
                    // 如果玩家还没有选择定缺
                    let ding_que_event = if let Event::DingQue { actor: ev_actor, suit } = ev.event {
                        // Agent返回了DingQue事件，验证actor匹配
                        ensure!(
                            ev_actor == actor as u8,
                            "DingQue event actor mismatch: expected {}, got {}",
                            actor,
                            ev_actor
                        );
                        Event::DingQue { actor: actor as u8, suit }
                    } else {
                        // Agent没有返回DingQue事件，自动为Agent选择定缺
                        // 选择手牌中最少的花色作为定缺
                        let state = &self.player_states[actor];
                        let man_count: u8 = (0..9).map(|i| state.tehai[i]).sum();
                        let pin_count: u8 = (9..18).map(|i| state.tehai[i]).sum();
                        let sou_count: u8 = (18..27).map(|i| state.tehai[i]).sum();
                        
                        let suit = if man_count <= pin_count && man_count <= sou_count {
                            crate::mjai::Suit::Man
                        } else if pin_count <= sou_count {
                            crate::mjai::Suit::Pin
                        } else {
                            crate::mjai::Suit::Sou
                        };
                        
                        Event::DingQue { actor: actor as u8, suit }
                    };
                    
                    // 更新玩家状态
                    self.player_states[actor]
                        .update(&ding_que_event)
                        .with_context(|| {
                            format!(
                                "failed to update player {} state with DingQue event",
                                actor
                            )
                        })?;
                    
                    // 标记该玩家已选择定缺
                    self.ding_que_selected[actor] = true;
                    
                    // 记录日志
                    self.add_log(EventExt::no_meta(ding_que_event));
                }
            }
            
            // 检查是否所有玩家都选择了定缺
            if self.ding_que_selected.iter().all(|&x| x) {
                // 所有玩家都选择了定缺，退出定缺选择阶段，开始第一轮摸牌
                self.ding_que_phase = false;
                
                let tile = self
                    .board
                    .yama
                    .pop()
                    .context("invalid yama: empty at init")?;
                self.tiles_left -= 1;
                
                // 基础规则：tiles_left 和 yama.len() 必须保持一致
                assert_eq!(
                    self.tiles_left as usize,
                    self.board.yama.len(),
                    "After initial tsumo, tiles_left ({}) and yama.len() ({}) are inconsistent. This indicates a fundamental bug in game state management.",
                    self.tiles_left,
                    self.board.yama.len()
                );
                
                let first_tsumo = Event::Tsumo {
                    actor: self.oya,
                    pai: tile,
                };
                self.broadcast(&first_tsumo);
                self.add_log_no_meta(first_tsumo);
                
                // 初始化tsumo_actor为oya（第一轮摸牌后，下一个摸牌的是oya）
                self.tsumo_actor = self.oya;
                
                // 第一轮摸牌后，摸牌的玩家应该可以打牌，所以can_act()应该返回true
                // 直接返回InGame，让poll()检查can_act()
                return Ok(Poll::InGame);
            }
            
            // 如果还在定缺选择阶段，返回 InGame 等待更多玩家选择
            return Ok(Poll::InGame);
        }

        // Validate reactions (only for players who haven't agari)
        for (actor, ev) in reactions.iter().enumerate() {
            if !self.players_agari[actor] {
                self.player_states[actor]
                    .validate_reaction(&ev.event)
                    .with_context(|| {
                        format!(
                            "invalid action: {ev:?}\nstate:\n{}",
                            self.player_states[actor].brief_info(),
                        )
                    })?;
            }
        }

        // 确保至少有一个玩家还没有和牌（基础规则：3人和牌时游戏结束）
        // 如果所有玩家都已和牌，说明游戏状态不一致，应该已经结束
        ensure!(
            self.agari_count < 4,
            "All players have agari (agari_count = {}), but game hasn't ended. This indicates a fundamental bug in game state management.",
            self.agari_count
        );

        // Multi-Ron Support (Yi Pao Duo Xiang)
        // Check if there are any Hora events first
        let specific_hora_events: Vec<_> = reactions
            .iter()
            .enumerate()
            .filter(|(actor, ev)| !self.players_agari[*actor] && matches!(ev.event, Event::Hora { .. }))
            .collect();

        if !specific_hora_events.is_empty() {
             let is_multi_ron = specific_hora_events.len() > 1;

             // Handle all Hora events
             for (actor, ev) in specific_hora_events {
                 if let Event::Hora { target, .. } = ev.event {
                     self.handle_hora(actor as u8, target, reactions, is_multi_ron)?;
                 }
             }

             // Check game end condition (3 players agari)
             if self.agari_count >= 3 {
                 return Ok(Poll::End);
             }

             // Continue game: find next actor
             // If multiple people ron, the next turn goes to the player right after the discarding player
             // (unless that player won, then skip them)
             // But wait, in standard rules, if Ron occurs, the next turn logic is tricky with Multi-Ron.
             // Usually, the player 'closest' to the discarder (in turn order) who declared Ron 'intercepts' the turn?
             // Actually in Bloody Battle, the game continues. The discarder loses the turn.
             // The next player is normally the one after the discarder.
             // The `handle_hora` logic skips the winner in turn rotation anyway.
             // So we just need to ensure `tsumo_actor` is set correctly.
             
             // Current logic in existing single-Hora handler:
             // let mut next_actor = (self.tsumo_actor + 1) % 4;
             // But valid `tsumo_actor` here might be stale if it was a Ron from `Dahai`.
             // In `Dahai` handler, `tsumo_actor` is updated to `next_actor`.
             // But wait, `step` is called AFTER reactions are collected.
             // If invalid `Dahai` happened previously, `tsumo_actor` is already pointing to next person?
             // No, `poll` calls `step`. `step` processes the best reaction.
             // If the best reaction is Hora, it overrides whatever `Dahai` might have implied about next turn.
             
             // Let's look at `Dahai` handler:
             // It updates `tsumo_actor` immediately: `self.tsumo_actor = next_actor;`
             // Then it returns `Ok(Poll::InGame)`.
             // THEN `poll` calls `step` again with reactions to that Dahai? 
             // Wait, `poll` loop:
             // 1. `step(&reactions)` -> returns event that JUST happened (e.g. Dahai).
             // 2. But `reactions` passed to `step` are the reactions TO the previous state?
             // No, standard MJAI loop:
             // Agent returns reaction to event.
             // Server processes reaction.
             
             // Let's re-read `poll` and `step` loop.
             // `poll` takes `reactions`.
             // `step` uses `reactions` to decide what happens next.
             // If `Dahai` just happened, `step` was called with `Dahai` event from active player.
             // It broadcasts `Dahai`, updates `tsumo_actor` to next.
             // Then returns `InGame`.
             // Agent gets `Dahai` event. Returns reactions (Ron/Pon/None).
             // `poll` is called again with these new reactions.
             // `step` is called with these reactions.
             // Un-agari players return reactions.
             // If Hora is present, we handle it.
             
             // So `tsumo_actor` currently points to the player AFTER the discarder (because Dahai handler updated it).
             // If Ron happens, does `tsumo_actor` change?
             // In Bloody Battle, after Ron(s), the next person to draw is the one after the discarder (who just played).
             // Which is exactly where `tsumo_actor` is currently pointing (updated by Dahai).
             // We just need to make sure that if THAT person won (Ron), they are skipped.
             
             // So we iterate to find valid next actor starting from current `tsumo_actor` (which is already discarder + 1)
             
             let mut next_actor = self.tsumo_actor;
             let mut attempts = 0;
             while self.players_agari[next_actor as usize] {
                 next_actor = (next_actor + 1) % 4;
                 attempts += 1;
                 if attempts >= 4 {
                     return Ok(Poll::End); 
                 }
             }
             self.tsumo_actor = next_actor;
             
             return Ok(Poll::InGame);
        }

        // Standard logic for non-Hora events (Pon, Daiminkan, etc.) - single winner priority
        let ev = reactions
            .iter()
            .enumerate()
            .filter(|(actor, _)| !self.players_agari[*actor]) // Skip players who have agari
            .map(|(_, ev)| ev)
            .min_by_key(|ev| match ev.event {
                Event::Hora { .. } => 0, // Should be caught above, but kept for safety
                Event::Daiminkan { .. } | Event::Pon { .. } => 1,
                Event::None => 3,
                _ => 2,
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No valid reaction found. All players have agari (agari_count = {}), but game hasn't ended. This indicates a fundamental bug in game state management.",
                    self.agari_count
                )
            })?;



        match ev.event {
            Event::None => {
                // 基础规则：tiles_left 和 yama.len() 必须保持一致
                // 如果两者不一致，说明游戏状态有严重错误，必须panic
                assert!(
                    self.tiles_left as usize == self.board.yama.len(),
                    "tiles_left ({}) and yama.len() ({}) are inconsistent. This indicates a fundamental bug in game state management.",
                    self.tiles_left,
                    self.board.yama.len()
                );
                
                // Check both tiles_left and yama to ensure consistency
                if self.tiles_left == 0 || self.board.yama.is_empty() {
                    self.exhaustive_ryukyoku();
                    return Ok(Poll::End);
                }

                // Skip players who have agari
                // 基础规则：最多3人和牌，所以最多循环3次就能找到未和牌玩家
                // 如果循环4次还没找到，说明所有玩家都已和牌，游戏状态不一致
                let mut attempts = 0;
                while self.players_agari[self.tsumo_actor as usize] {
                    self.tsumo_actor = (self.tsumo_actor + 1) % 4;
                    attempts += 1;
                    if attempts >= 4 {
                        // 所有玩家都已和牌，应该已经结束游戏（agari_count >= 3）
                        // 这是基础规则违反，必须panic
                        bail!(
                            "All players have agari (agari_count = {}), but game hasn't ended. This indicates a fundamental bug in game state management.",
                            self.agari_count
                        );
                    }
                }

                // Double-check before popping from yama
                // 基础规则：tiles_left 和 yama.len() 必须保持一致
                if self.board.yama.is_empty() {
                    // 如果 yama 为空，tiles_left 也应该是 0
                    assert_eq!(
                        self.tiles_left, 0,
                        "yama is empty but tiles_left ({}) is not 0. This indicates a fundamental bug in game state management.",
                        self.tiles_left
                    );
                    self.exhaustive_ryukyoku();
                    return Ok(Poll::End);
                }

                let tile = self.board.yama.pop().with_context(|| {
                    format!("tiles left > 0 ({}) but yama is empty", self.tiles_left)
                })?;
                self.tiles_left -= 1;
                
                // 基础规则：pop 后 tiles_left 和 yama.len() 必须保持一致
                assert_eq!(
                    self.tiles_left as usize,
                    self.board.yama.len(),
                    "After popping from yama, tiles_left ({}) and yama.len() ({}) are inconsistent. This indicates a fundamental bug in game state management.",
                    self.tiles_left,
                    self.board.yama.len()
                );
                let tsumo = Event::Tsumo {
                    actor: self.tsumo_actor,
                    pai: tile,
                };

                self.broadcast(&tsumo);
                self.add_log_no_meta(tsumo);
                
                // Reset Score Transfer state on Tsumo (new turn)
                self.last_kan_actor = None;
                self.last_kan_revenue = 0;
            }

            Event::Dahai { actor, pai: _pai, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                
                // 基础规则：如果 tiles_left == 0，应该触发流局
                // 但不能在这里立即触发，必须等待其他玩家对Dahai的反应（如Ron）
                // 真正的流局检查会在下一次step的Event::None中进行（如果没有人Ron）
                /*
                if self.tiles_left == 0 || self.board.yama.is_empty() {
                    self.exhaustive_ryukyoku();
                    return Ok(Poll::End);
                }
                */
                
                let mut next_actor = (actor + 1) % 4;
                // 基础规则：最多3人和牌，所以最多循环3次就能找到未和牌玩家
                // 如果循环4次还没找到，说明所有玩家都已和牌，游戏状态不一致
                let mut attempts = 0;
                while self.players_agari[next_actor as usize] {
                    next_actor = (next_actor + 1) % 4;
                    attempts += 1;
                    if attempts >= 4 {
                        // 所有玩家都已和牌，应该已经结束游戏（agari_count >= 3）
                        // 这是基础规则违反，必须panic
                        bail!(
                            "All players have agari (agari_count = {}), but game hasn't ended. This indicates a fundamental bug in game state management.",
                            self.agari_count
                        );
                    }
                }
                self.tsumo_actor = next_actor;


            }

            Event::Pon { .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                
                // Reset Score Transfer state on Pon (interruption)
                self.last_kan_actor = None;
                self.last_kan_revenue = 0;
            }

            Event::Ankan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.kans += 1;

                // Instant Payment (Xia Yu): All non-agari opponents pay 2000 points
                let mut payment_deltas = [0i32; 4];
                let mut total_revenue = 0;
                let payment_per_person = 2000;
                
                for i in 0..4 {
                    if i != actor as usize && !self.players_agari[i] {
                         payment_deltas[i] = -payment_per_person;
                         total_revenue += payment_per_person;
                    }
                }
                payment_deltas[actor as usize] = total_revenue;
                vec_add_assign(&mut self.kyoku_deltas, &payment_deltas);

                // Score Transfer tracking
                self.last_kan_actor = Some(actor);
                self.last_kan_revenue = total_revenue;

                self.gang_history.push(GangRecord {
                    actor,
                    deltas: payment_deltas,
                });
            }

            Event::Daiminkan { actor, target, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.kans += 1;
                
                // Instant Payment (Gua Feng - Ming Kan): Discarder pays 2000 points
                // Only if discarder has not agari (usually true, but check just in case?)
                // Actually, discarder just played, so they are active.
                let mut payment_deltas = [0i32; 4];
                let payment = 2000;
                
                payment_deltas[target as usize] = -payment;
                payment_deltas[actor as usize] = payment;
                
                vec_add_assign(&mut self.kyoku_deltas, &payment_deltas);

                // Score Transfer tracking
                self.last_kan_actor = Some(actor);
                self.last_kan_revenue = payment;

                self.gang_history.push(GangRecord {
                    actor,
                    deltas: payment_deltas,
                });
            }

            Event::Kakan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.kans += 1;
                
                // Instant Payment (Gua Feng - Wan Gang): All non-agari opponents pay 1000 points
                let mut payment_deltas = [0i32; 4];
                let mut total_revenue = 0;
                let payment_per_person = 1000;
                
                for i in 0..4 {
                    if i != actor as usize && !self.players_agari[i] {
                         payment_deltas[i] = -payment_per_person;
                         total_revenue += payment_per_person;
                    }
                }
                payment_deltas[actor as usize] = total_revenue;
                vec_add_assign(&mut self.kyoku_deltas, &payment_deltas);

                // Score Transfer tracking
                self.last_kan_actor = Some(actor);
                self.last_kan_revenue = total_revenue;

                self.gang_history.push(GangRecord {
                    actor,
                    deltas: payment_deltas,
                });
            }

            Event::Hora { .. } => {
                // This branch should ideally be unreachable now because Hora is handled at the top of step() for Multi-Ron.
                // However, we keep it as a fallback or for single Tsumo cases if they fall through (Tsumo is usually Event from actor, not reaction?)
                // Wait, Tsumo agari is an ACTION by the current player (self-reaction?), or a reaction to Tsumo event?
                // In MJAI, Tsumo agari is usually a 'Hora' event sent by the actor as a reaction to 'Tsumo' event.
                // So it WOULD appear in `reactions`.
                // And it would be caught by the `specific_hora_events` block above.
                
                // So this block might be redundant for normal flow, but let's keep it consistent just in case
                // `ev` was selected by min_by_key (e.g. if we modified the top block to only catch Ron).
                // Actually, the top block catches matches!(ev.event, Event::Hora). Tsumo IS Event::Hora.
                // So Tsumo agari is also handled there.
                
                // But wait, for Tsumo, there is only 1 actor (the current player). 
                // So `specific_hora_events` will have size 1.
                // Logic holds.
                
                // We can't easily remove this match arm without refactoring the whole match block, 
                // so we can just duplicate the logic or make it a no-op if we are sure it's handled.
                // But `ev` is already defined as a reference to one of the reactions.
                // If we handled it above, we returned `Poll::InGame` or `Poll::End`. 
                // So we won't reach here if `ev` was Hora.
                
                // Ah, `ev` variable definition uses `min_by_key` on `reactions`.
                // If we handled `specific_hora_events` and returned, we don't reach `let ev = ...`
                // EXCEPT if `specific_hora_events` was empty, in which case `ev` will NOT be Hora.
                // So this match arm is actually unreachable for Hora events if the top block works correctly.
                
                // We can verify this with a panic or log, or just leave it as dead code.
                // Let's panic to ensure our assumption is correct and we aren't missing edge cases.
                bail!("Unreachable Event::Hora in match block. Should have been handled by Multi-Ron logic.");
            }

            Event::Ryukyoku { .. } => {
                // 九種九牌 - 血战到底不支持中途流局
                bail!("Unexpected Event::Ryukyoku in Bloody Battle");
            }

            _ => {
                bail!("unexpected event: {:?}", ev.event);
            }
        };


        Ok(Poll::InGame)
    }

    pub fn encode_oracle_obs(&self, perspective: u8, version: u32) -> Array2<f32> {
        let shape = oracle_obs_shape(version);
        let mut arr = Simple2DArray::<27, f32>::new(shape.0);
        let mut idx = 0;

        self.player_states
            .iter()
            .cycle()
            .skip(perspective as usize + 1)
            .take(3)
            .for_each(|state| {
                state
                    .tehai()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .for_each(|(tile_id, &count)| {
                        arr.assign_rows(idx, tile_id, count as usize, 1.);
                    });
                idx += 4;

                idx += 3;

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
            });

        let mut encode_tile = |idx: usize, tile: Tile| -> usize {
            let tile_id = tile.as_usize();
            arr.assign(idx, tile_id, 1.);
            idx + 1
        };

        // Encode remaining tiles in yama
        for &tile in self.board.yama.iter().rev().take(self.tiles_left as usize) {
            idx = encode_tile(idx, tile);
        }
        // Skip remaining yama slots (no aka encoding, so only 1 dimension per tile)
        let max_yama_tiles = 69;
        idx += (max_yama_tiles - self.tiles_left as usize) * 1;

        idx += 4 * 1;

        idx += 5 * 1;

        idx += 5 * 1;

        assert_eq!(idx, shape.0);
        arr.build()
    }
}

#[rustfmt::skip]
const UNSHUFFLED: [Tile; 108] = [
    t!(1m), t!(1m), t!(1m), t!(1m),
    t!(2m), t!(2m), t!(2m), t!(2m),
    t!(3m), t!(3m), t!(3m), t!(3m),
    t!(4m), t!(4m), t!(4m), t!(4m),
    t!(5m), t!(5m), t!(5m), t!(5m),
    t!(6m), t!(6m), t!(6m), t!(6m),
    t!(7m), t!(7m), t!(7m), t!(7m),
    t!(8m), t!(8m), t!(8m), t!(8m),
    t!(9m), t!(9m), t!(9m), t!(9m),

    t!(1p), t!(1p), t!(1p), t!(1p),
    t!(2p), t!(2p), t!(2p), t!(2p),
    t!(3p), t!(3p), t!(3p), t!(3p),
    t!(4p), t!(4p), t!(4p), t!(4p),
    t!(5p), t!(5p), t!(5p), t!(5p),
    t!(6p), t!(6p), t!(6p), t!(6p),
    t!(7p), t!(7p), t!(7p), t!(7p),
    t!(8p), t!(8p), t!(8p), t!(8p),
    t!(9p), t!(9p), t!(9p), t!(9p),

    t!(1s), t!(1s), t!(1s), t!(1s),
    t!(2s), t!(2s), t!(2s), t!(2s),
    t!(3s), t!(3s), t!(3s), t!(3s),
    t!(4s), t!(4s), t!(4s), t!(4s),
    t!(5s), t!(5s), t!(5s), t!(5s),
    t!(6s), t!(6s), t!(6s), t!(6s),
    t!(7s), t!(7s), t!(7s), t!(7s),
    t!(8s), t!(8s), t!(8s), t!(8s),
    t!(9s), t!(9s), t!(9s), t!(9s),
];

#[cfg(test)]
mod test {
    use super::*;
    use crate::mjai::Suit;
    use crate::t;

    /// Test exhaustive_ryukyoku with 查花猪 (huazhu) logic
    /// 
    /// Test case 1: 1 player is huazhu (has ding_que tiles remaining)
    /// Expected: Huazhu player pays 16000, each non-huazhu player gets 16000/3 ≈ 5333
    #[test]
    fn test_exhaustive_ryukyoku_one_huazhu() {
        let mut board_state = create_test_board_state();
        
        // Setup: Player 0 is huazhu (has Man tiles, ding_que is Man)
        board_state.player_states[0].ding_que = Some(Suit::Man);
        // Add some Man tiles to player 0's hand to make them huazhu
        board_state.player_states[0].tehai[0] = 1; // 1m
        board_state.player_states[0].tehai[1] = 1; // 2m
        
        // Players 1, 2, 3 are not huazhu (no ding_que tiles)
        board_state.player_states[1].ding_que = Some(Suit::Pin);
        board_state.player_states[2].ding_que = Some(Suit::Sou);
        board_state.player_states[3].ding_que = Some(Suit::Man);
        
        // Set tiles_left to 0 to trigger exhaustive_ryukyoku
        board_state.tiles_left = 0;
        
        // Call exhaustive_ryukyoku
        board_state.exhaustive_ryukyoku();
        
        // Verify: Player 0 (huazhu) pays 16000 to EACH non-huazhu player
        // 3 non-huazhu players, so 16000 * 3 = 48000
        // Each of players 1, 2, 3 gets 16000
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], -48000, "Huazhu player should pay 48000");
        assert_eq!(deltas[1], 16000, "Non-huazhu player 1 should get 16000");
        assert_eq!(deltas[2], 16000, "Non-huazhu player 2 should get 16000");
        assert_eq!(deltas[3], 16000, "Non-huazhu player 3 should get 16000");
        
        // Verify total is 0 (conservation of points)
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Total deltas should be 0");
    }

    /// Test exhaustive_ryukyoku with 查大叫 (tenpai) logic
    /// 
    /// Test case 2: 1 player is tenpai (excluding huazhu)
    /// Expected: Tenpai player gets +3000, each non-tenpai player pays -1000
    #[test]
    fn test_exhaustive_ryukyoku_one_tenpai() {
        let mut board_state = create_test_board_state();
        
        // Setup: All players have completed ding_que (no huazhu)
        board_state.player_states[0].ding_que = Some(Suit::Man);
        board_state.player_states[1].ding_que = Some(Suit::Pin);
        board_state.player_states[2].ding_que = Some(Suit::Sou);
        board_state.player_states[3].ding_que = Some(Suit::Man);
        
        // Player 0 is tenpai (shanten = 0)
        board_state.player_states[0].shanten = 0;
        // Players 1, 2, 3 are not tenpai (shanten > 0)
        board_state.player_states[1].shanten = 1;
        board_state.player_states[2].shanten = 2;
        board_state.player_states[3].shanten = 1;
        
        // Set tiles_left to 0 to trigger exhaustive_ryukyoku
        board_state.tiles_left = 0;
        
        // Call exhaustive_ryukyoku
        board_state.exhaustive_ryukyoku();
        
        // Verify: Player 0 (tenpai) gets calculated points (PingHu = 1 Fan = 1000) from EACH non-tenpai.
        // 3 non-tenpai players -> 3000 total.
        // Players 1, 2, 3 (non-tenpai) each pay -1000
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], 3000, "Tenpai player should get +3000");
        assert_eq!(deltas[1], -1000, "Non-tenpai player 1 should pay -1000");
        assert_eq!(deltas[2], -1000, "Non-tenpai player 2 should pay -1000");
        assert_eq!(deltas[3], -1000, "Non-tenpai player 3 should pay -1000");
        
        // Verify total is 0
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Total deltas should be 0");
    }

    /// Test exhaustive_ryukyoku with both 查花猪 and 查大叫
    /// 
    /// Test case 3: 1 player is huazhu, 1 player is tenpai (excluding huazhu)
    /// Expected: 
    /// - Huazhu player pays 16000 (distributed to 3 non-huazhu players)
    /// - Tenpai player gets +3000, other non-huazhu non-tenpai players pay -1000
    #[test]
    fn test_exhaustive_ryukyoku_huazhu_and_tenpai() {
        let mut board_state = create_test_board_state();
        
        // Setup: Player 0 is huazhu (has Man tiles, ding_que is Man)
        board_state.player_states[0].ding_que = Some(Suit::Man);
        board_state.player_states[0].tehai[0] = 1; // 1m (huazhu)
        
        // Player 1 is tenpai (no huazhu)
        board_state.player_states[1].ding_que = Some(Suit::Pin);
        board_state.player_states[1].shanten = 0;
        
        // Players 2, 3 are not huazhu and not tenpai
        board_state.player_states[2].ding_que = Some(Suit::Sou);
        board_state.player_states[2].shanten = 1;
        board_state.player_states[3].ding_que = Some(Suit::Man);
        board_state.player_states[3].shanten = 1;
        
        // Set tiles_left to 0 to trigger exhaustive_ryukyoku
        board_state.tiles_left = 0;
        
        // Call exhaustive_ryukyoku
        board_state.exhaustive_ryukyoku();
        
        // Verify: 
        // Verify: 
        // - Player 0 (huazhu) pays 16000 * 3 = 48000
        // - Player 1 (tenpai, non-huazhu):
        //    - From Huazhu: +16000
        //    - From No-Ten (2, 3): +1000 * 2 = +2000  (Calculated points for PingHu)
        //    - Total: +18000
        // - Player 2 (non-huazhu, non-tenpai):
        //    - From Huazhu: +16000
        //    - To Tenpai: -1000
        //    - Total: +15000
        // - Player 3 (non-huazhu, non-tenpai):
        //    - From Huazhu: +16000
        //    - To Tenpai: -1000
        //    - Total: +15000
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], -48000, "Huazhu player should pay 48000");
        assert_eq!(deltas[1], 18000, "Tenpai non-huazhu player should get 16000+2000");
        assert_eq!(deltas[2], 15000, "Non-tenpai non-huazhu player should get 16000-1000");
        assert_eq!(deltas[3], 15000, "Non-tenpai non-huazhu player should get 16000-1000");
        
        // Verify total is 0
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Total deltas should be 0");
    }

    /// Test exhaustive_ryukyoku with 2 players tenpai
    /// 
    /// Test case 4: 2 players are tenpai (no huazhu)
    /// Expected: Each tenpai player gets +1500, each non-tenpai player pays -1500
    #[test]
    fn test_exhaustive_ryukyoku_two_tenpai() {
        let mut board_state = create_test_board_state();
        
        // Setup: All players have completed ding_que (no huazhu)
        board_state.player_states[0].ding_que = Some(Suit::Man);
        board_state.player_states[1].ding_que = Some(Suit::Pin);
        board_state.player_states[2].ding_que = Some(Suit::Sou);
        board_state.player_states[3].ding_que = Some(Suit::Man);
        
        // Players 0, 1 are tenpai
        board_state.player_states[0].shanten = 0;
        board_state.player_states[1].shanten = 0;
        // Players 2, 3 are not tenpai
        board_state.player_states[2].shanten = 1;
        board_state.player_states[3].shanten = 1;
        
        // Set tiles_left to 0 to trigger exhaustive_ryukyoku
        board_state.tiles_left = 0;
        
        // Call exhaustive_ryukyoku
        board_state.exhaustive_ryukyoku();
        
        // Verify: Players 0, 1 (tenpai) each get +2000 (1000 from player 2, 1000 from player 3)
        // Players 2, 3 (non-tenpai) each pay -2000 (1000 to player 0, 1000 to player 1)
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], 2000, "Tenpai player 0 should get +2000");
        assert_eq!(deltas[1], 2000, "Tenpai player 1 should get +2000");
        assert_eq!(deltas[2], -2000, "Non-tenpai player 2 should pay -2000");
        assert_eq!(deltas[3], -2000, "Non-tenpai player 3 should pay -2000");
        
        // Verify total is 0
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Total deltas should be 0");
    }

    /// Test exhaustive_ryukyoku with 3 players tenpai
    /// 
    /// Test case 5: 3 players are tenpai (no huazhu)
    /// Expected: Each tenpai player gets +1000, non-tenpai player pays -3000
    #[test]
    fn test_exhaustive_ryukyoku_three_tenpai() {
        let mut board_state = create_test_board_state();
        
        // Setup: All players have completed ding_que (no huazhu)
        board_state.player_states[0].ding_que = Some(Suit::Man);
        board_state.player_states[1].ding_que = Some(Suit::Pin);
        board_state.player_states[2].ding_que = Some(Suit::Sou);
        board_state.player_states[3].ding_que = Some(Suit::Man);
        
        // Players 0, 1, 2 are tenpai
        board_state.player_states[0].shanten = 0;
        board_state.player_states[1].shanten = 0;
        board_state.player_states[2].shanten = 0;
        // Player 3 is not tenpai
        board_state.player_states[3].shanten = 1;
        
        // Set tiles_left to 0 to trigger exhaustive_ryukyoku
        board_state.tiles_left = 0;
        
        // Call exhaustive_ryukyoku
        board_state.exhaustive_ryukyoku();
        
        // Verify: Players 0, 1, 2 (tenpai) each get +1000
        // Player 3 (non-tenpai) pays -3000
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], 1000, "Tenpai player 0 should get +1000");
        assert_eq!(deltas[1], 1000, "Tenpai player 1 should get +1000");
        assert_eq!(deltas[2], 1000, "Tenpai player 2 should get +1000");
        assert_eq!(deltas[3], -3000, "Non-tenpai player 3 should pay -3000");
        
        // Verify total is 0
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Total deltas should be 0");
    }

    /// Test exhaustive_ryukyoku with 2 huazhu players
    /// 
    /// Test case 6: 2 players are huazhu
    /// Expected: Each huazhu player pays 16000, each non-huazhu player gets 16000
    #[test]
    fn test_exhaustive_ryukyoku_two_huazhu() {
        let mut board_state = create_test_board_state();
        
        // Setup: Players 0, 1 are huazhu
        board_state.player_states[0].ding_que = Some(Suit::Man);
        board_state.player_states[0].tehai[0] = 1; // 1m (huazhu)
        board_state.player_states[1].ding_que = Some(Suit::Pin);
        board_state.player_states[1].tehai[9] = 1; // 1p (huazhu)
        
        // Players 2, 3 are not huazhu
        board_state.player_states[2].ding_que = Some(Suit::Sou);
        board_state.player_states[3].ding_que = Some(Suit::Man);
        
        // Set tiles_left to 0 to trigger exhaustive_ryukyoku
        board_state.tiles_left = 0;
        
        // Call exhaustive_ryukyoku
        board_state.exhaustive_ryukyoku();
        
        // Verify: Players 0, 1 (huazhu) each pay 16000 * 2 = 32000 (to 2, 3)
        // Players 2, 3 (non-huazhu) each get 16000 (from 0) + 16000 (from 1) = 32000
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], -32000, "Huazhu player 0 should pay 32000");
        assert_eq!(deltas[1], -32000, "Huazhu player 1 should pay 32000");
        assert_eq!(deltas[2], 32000, "Non-huazhu player 2 should get 32000");
        assert_eq!(deltas[3], 32000, "Non-huazhu player 3 should get 32000");
        
        // Verify total is 0
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Total deltas should be 0");
    }

    /// Helper function to create a test BoardState
    fn create_test_board_state() -> BoardState {
        let mut board = Board::default();
        board.kyoku = 0;
        board.scores = [25000; 4];
        
        // Create simple haipai (all players have same hand for simplicity)
        // Use a hand that doesn't have all three suits to avoid issues
        let test_hand = [
            t!(1m), t!(2m), t!(3m), t!(4m), t!(5m), t!(6m), t!(7m), t!(8m), t!(9m),
            t!(1p), t!(2p), t!(3p), t!(4p),
        ];
        for i in 0..4 {
            board.haipai[i] = test_hand;
        }
        
        board.yama = vec![];
        
        let mut board_state = board.into_state();
        board_state.kyoku_deltas = [0; 4];
        board_state
    }
}
