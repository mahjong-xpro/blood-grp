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

/// 血战到底：3人和牌时游戏结束，牌墙耗尽时结算（查花猪、查大叫）。
#[derive(Debug, Default)]
pub struct Board {
    /// Counts from 0 (for recording only, no game flow impact)
    pub kyoku: u8,
    /// [INITIAL_SCORE; 4] (see crate::consts)
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
    /// 和牌顺序，先和者在前。用于同分时排名。
    agari_order: Vec<u8>,
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

    /// 出牌（Dahai）计数，用于天胡/地胡精准判定。
    /// - 天胡: dahai_count == 0（庄家摸牌后直接自摸，无人出过牌）
    /// - 地胡: dahai_count == 1（庄家第一次打牌即被荣和）
    ///
    /// 不能只用 `tiles_left == 55 && kans == 0`，因为碰不消耗牌墙：
    /// 庄家出牌 → 碰链 → 庄家碰 → 庄家再出牌 → 荣和，tiles_left 仍为 55，
    /// 但此时并非庄家第一次出牌，不应触发地胡。
    #[derivative(Default(value = "0"))]
    dahai_count: u16,

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
    /// Whether this gang record is still valid (e.g. kakan robbed by chankan invalidates it).
    #[serde(default = "default_true")]
    pub valid: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Clone, Copy)]
pub enum Poll {
    InGame,
    End,
}

impl Board {
    pub fn init_from_seed(&mut self, game_seed: (u64, u64)) {
        let (nonce, key) = game_seed;
        // Include kyoku number in seed to ensure different shuffle each kyoku
        let kyoku_seed = Sha3_256::new()
            .chain_update(nonce.to_le_bytes())
            .chain_update(key.to_le_bytes())
            .chain_update(self.kyoku.to_le_bytes())  // Add kyoku number!
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
    pub fn end(&self) -> KyokuResult {
        KyokuResult {
            kyoku: self.board.kyoku,
            scores: self.board.scores,
            agari_order: self.agari_order.clone(),
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

        // 新流程：庄家先补一张到 14 张，再进入定缺阶段
        // StartKyoku(13x4) -> Tsumo(oya补牌) -> DingQue(四家提交) -> oya打牌 -> 正常轮转
        let tile = self
            .board
            .yama
            .pop()
            .context("invalid yama: empty at init")?;
        self.tiles_left -= 1;

        assert_eq!(
            self.tiles_left as usize,
            self.board.yama.len(),
            "After initial dealer tsumo, tiles_left ({}) and yama.len() ({}) are inconsistent.",
            self.tiles_left,
            self.board.yama.len()
        );

        let first_tsumo = Event::Tsumo {
            actor: self.oya,
            pai: tile,
        };
        self.broadcast(&first_tsumo);
        self.add_log_no_meta(first_tsumo);
        self.tsumo_actor = self.oya;

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
                s.ding_que.is_some() && !s.check_ding_que_complete()
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
            // 花猪罚分目标：所有非花猪玩家（包括已和牌者）
            // 四川麻将规则：花猪赔给每家非花猪，已和牌者也应收取花猪罚分
            let non_huazhu_targets: Vec<usize> = (0..4)
                .filter(|&i| !huazhu_actors.contains(&i))
                .collect();
            let target_count = non_huazhu_targets.len();

            if target_count > 0 {
                // 花猪罚分：花猪向每个非花猪支付极刑（封顶分）
                // 四川麻将规则：查花猪赔给非花猪每家满分（通常是极刑，这里定为16000）
                // Pay 16000 to EACH non-huazhu player (including agari players).
                let penalty_per_target = 16000;
                
                let mut huazhu_deltas = [0; 4];
                
                // Each Huazhu pays penalty_per_target * target_count
                for &huazhu in &huazhu_actors {
                    huazhu_deltas[huazhu] = -(penalty_per_target * target_count as i32);
                }
                
                // Each Non-Huazhu (including agari) receives penalty_per_target * huazhu_count
                for &target in &non_huazhu_targets {
                     huazhu_deltas[target] += penalty_per_target as i32 * huazhu_actors.len() as i32;
                }
                
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
                 // FIX: 查大叫计算理论最大手牌价值，不受牌可用性影响。
                 // state.waits() 会过滤 tiles_seen >= 4 的牌（实际游戏中不可能摸到），
                 // 但查大叫是理论罚款，应遍历所有结构上能和的牌。
                 // 例如：听 1m，但 4 张 1m 全部已见，state.waits()[0] = false，
                 // 导致 max_points = 0，该听牌玩家被低估或漏算。
                 let mut max_points = 0;
                 let tehai_sum: u8 = state.tehai.iter().sum();
                 let len_div3_after_draw = ((tehai_sum.saturating_add(1)) / 3) as u8;
                 for tid in 0..27 {
                     if state.tehai[tid] >= 4 {
                         continue; // 手牌已有 4 张，无法再加
                     }
                     let mut temp_tehai = state.tehai;
                     temp_tehai[tid] += 1;
                     // 结构上是否能和（不检查 tiles_seen）
                     if crate::algo::shanten::calc_all(&temp_tehai, len_div3_after_draw, state.ding_que) != -1 {
                         continue;
                     }
                         
                     let agari_calc = crate::algo::agari::AgariCalculator {
                         tehai: &temp_tehai,
                         pons: &state.pons,
                         minkans: &state.minkans,
                         ankans: &state.ankans,
                         winning_tile: tid as u8,
                         is_ron: true,
                         ding_que: state.ding_que,
                         is_after_kan: false,
                         is_kan_discard: false,
                         is_chankan: false,
                         exclude_gen_tile: None,
                         is_haidi: false,
                         is_tianhu: false,
                         is_dihu: false,
                     };
                         
                     if let Some(agari) = agari_calc.agari() {
                         let p = agari.point(false).ron;
                         if p > max_points {
                             max_points = p;
                         }
                     }
                 }
                 
                 // If max_points is 0 (can't agari even if tenpai? e.g. no yaku?), treat as No-Ten?
                 // We'll trust `max_points`.
                 // Even if max_points is 0 (No Yaku), we count them as Tenpai to avoid Tui Shui (Refund).
                 // They just won't receive Cha Da Jiao payment (or receive 0).
                 tenpai_details.push((i, max_points));
             }
        }

        // Who are No-Ten (Wei Ting / 被查大叫)?
        // Alive, non-Huazhu, and NOT in tenpai_details.
        let no_ten_actors: Vec<usize> = (0..4)
            .filter(|&i| {
                !self.players_agari[i]
                    && !huazhu_actors.contains(&i)
                    && !tenpai_details.iter().any(|(t, _)| *t == i)
            })
            .collect();
        
        if !tenpai_details.is_empty() {
            let mut chadajiao_deltas = [0; 4];
            
            // Calculate penalty for No-Ten players
            // Who are No-Ten?
            // Alive, Non-Huazhu, and NOT in tenpai_details
                
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
            .filter(|&i| {
                !self.players_agari[i]
                    && (huazhu_actors.contains(&i) || no_ten_actors.contains(&i))
            })
            .collect();
             
        // Iterate gang history (skip invalidated records, e.g. robbed kong)
        for record in &self.gang_history {
             if !record.valid {
                 continue;
             }
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
                 log::debug!("Tui Shui: Player {} refunds Gang (Revenue: {})", record.actor, record.deltas[record.actor as usize]);
             }
        }

        vec_add_assign(&mut self.kyoku_deltas, &final_deltas);
        let ryukyoku = Event::Ryukyoku {
            deltas: Some(final_deltas),
        };
        self.broadcast(&ryukyoku); // 同步各 PlayerState 的 scores（流局查花猪/查大叫/退税）
        self.add_log_no_meta(ryukyoku);
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
        let players_agari_before = self.players_agari;
        


        if !self.players_agari[single_actor as usize] {
            self.players_agari[single_actor as usize] = true;
            self.agari_count += 1;
            self.agari_order.push(single_actor);
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
        // 天胡：庄家摸牌后直接自摸（无人出过牌，dahai_count == 0）
        // 地胡：庄家第一次打牌即被荣和（仅庄家出了一张牌，dahai_count == 1）
        //
        // 使用 dahai_count 而非 tiles_left，因为碰不消耗牌墙：
        // 碰链可使 tiles_left 保持 55 但已经不是第一巡。
        let is_tianhu = self.dahai_count == 0 && !is_ron && single_actor == self.oya;
        let is_dihu = self.dahai_count == 1 && self.kans == 0 && is_ron && single_target == self.oya;

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
                         // FIX: 使用 gang_history 记录精确退款，而非硬编码 1000。
                         // 这与杠上炮多荣退款路径保持一致，避免金额不同步的风险。
                         if let Some(rec) = self
                             .gang_history
                             .iter_mut()
                             .rev()
                             .find(|r| r.valid && r.actor == kan_actor)
                         {
                             for i in 0..4 {
                                 deltas[i] -= rec.deltas[i];
                             }
                             rec.valid = false;
                         } else {
                             log::warn!(
                                 "Chankan refund: missing gang record for actor {}, revenue {}. Falling back to manual refund.",
                                 kan_actor,
                                 self.last_kan_revenue
                             );
                             // Fallback: 手动计算退款（兼容旧日志）
                             let refund_per_person = 1000;
                             let mut total_refund = 0;
                             for i in 0..4 {
                                 if i != single_target as usize && !players_agari_before[i] {
                                     deltas[i] += refund_per_person;
                                     total_refund += refund_per_person;
                                 }
                             }
                             deltas[single_target as usize] -= total_refund;
                         }
                         
                         // Clear revenue so subsequent winners (Multi-Ron) don't double refund
                         self.last_kan_revenue = 0;
                         // FIX: 清除 last_kan_actor，避免后续代码路径误判状态
                         self.last_kan_actor = None;

                         // 职责划分：PlayerState 级别的抢杠回退（minkans→pons、fuuro_overview、
                         // intermediate_kan、kans_on_board 等）统一由 update.rs::hora() 在
                         // broadcast(&hora) 时处理。board.rs 只负责棋盘级状态：
                         //   - gang_history 失效（上方已完成）
                         //   - last_kan_actor/last_kan_revenue 清除（上方已完成）
                         //   - kans 计数器递减（下方）
                         //   - tsumo_actor 推进（下方）
                         // 这避免了 board.rs 和 update.rs 对同一状态做重叠修改。

                         // FIX: 以下两行移入内层 guard，确保多荣抢杠时只执行一次。
                         // 之前放在外层 if is_chankan 中，导致 N 个荣和者各执行一次，
                         // kans 被多次递减，可能导致天胡/地胡误判。
                         self.tsumo_actor = (single_target + 1) % 4;
                         self.kans = self.kans.saturating_sub(1);
                     }
                 }
            }
            
            // Kong Revenue Handling (Hujiaozhuanyi / GangShangPao multi-ron rule)
            //
            // - Single Ron (杠上炮): the kan revenue is transferred to the winner.
            // - Multi Ron on the same discard (一炮多响杠上炮): the kan revenue is refunded to original payers
            //   (原路退回), and NOT transferred to any winner.
            //
            // Note: This is separate from chankan (抢杠) refund, which is handled above and also
            // clears last_kan_revenue / invalidates the gang record.
            if let Some(kan_actor) = self.last_kan_actor {
                if kan_actor == single_target && self.last_kan_revenue > 0 {
                    if _is_multi_ron {
                        // Refund exactly by reversing the last valid gang record for this actor.
                        if let Some(rec) = self
                            .gang_history
                            .iter_mut()
                            .rev()
                            .find(|r| r.valid && r.actor == kan_actor)
                        {
                            for i in 0..4 {
                                deltas[i] -= rec.deltas[i];
                            }
                            rec.valid = false;
                        } else {
                            log::warn!(
                                "Multi-ron gangpao refund: missing gang record for actor {}, revenue {}",
                                kan_actor,
                                self.last_kan_revenue
                            );
                        }
                        self.last_kan_revenue = 0;
                        self.last_kan_actor = None;
                    } else {
                        // Transfer: subtract from discarder, add to winner.
                        deltas[single_target as usize] -= self.last_kan_revenue;
                        deltas[single_actor as usize] += self.last_kan_revenue;
                        self.last_kan_revenue = 0;
                        self.last_kan_actor = None;

                        // FIX: 杠上炮（单家荣和）后必须失效 gang_history 记录，
                        // 否则退税（exhaustive_ryukyoku）会再次反转该笔交易，导致杠者重复扣款。
                        // 与多家荣和路径保持一致。
                        if let Some(rec) = self
                            .gang_history
                            .iter_mut()
                            .rev()
                            .find(|r| r.valid && r.actor == kan_actor)
                        {
                            rec.valid = false;
                        }
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
        self.broadcast(&hora); // 同步各 PlayerState 的 players_agari，否则 agent 拿到的 state 中「谁已和牌」会滞后
        self.add_log_no_meta(hora);
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

        // 处理定缺选择阶段（基础规则：血战到底必须在打牌前选择定缺）
        // 严格模式：必须由 AI 提交 DingQue；服务端不做任何自动定缺。
        if self.ding_que_phase {
            for actor in 0..4 {
                if self.ding_que_selected[actor] {
                    continue;
                }
                if let Event::DingQue { actor: ev_actor, suit } = reactions[actor].event {
                    ensure!(
                        ev_actor as usize == actor,
                        "ding_que actor mismatch: reaction actor={} but slot={}",
                        ev_actor,
                        actor
                    );
                    let ding_que_event = Event::DingQue { actor: ev_actor, suit };
                    self.broadcast(&ding_que_event);
                    self.add_log_no_meta(ding_que_event);
                    self.ding_que_selected[actor] = true;
                }
            }

            if self.ding_que_selected.iter().all(|&b| b) {
                self.ding_que_phase = false;
                // 强规则：定缺结束后必须轮到庄家打牌（庄家应为 14 张，3n+2）。
                // 如果庄家此时不能打牌，下一步会误走 Event::None -> 再摸一张，导致 15 张错误。
                let oya = self.oya as usize;
                let cans = self.player_states[oya].last_cans();
                assert!(
                    cans.can_discard && cans.target_actor == self.oya,
                    "DingQue finished but oya cannot discard (or wrong target_actor). \
                    This would cause an extra tsumo (15 tiles bug). oya={}, cans={:?}",
                    self.oya,
                    cans
                );
            }
            return Ok(Poll::InGame);
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
             for &(actor, ev) in &specific_hora_events {
                 if let Event::Hora { target, .. } = ev.event {
                     self.handle_hora(actor as u8, target, reactions, is_multi_ron)?;
                 }
             }

             // Check game end condition (3 players agari)
             if self.agari_count >= 3 {
                 return Ok(Poll::End);
             }

             // 血战到底规则：和牌后轮到和牌者的下家摸牌继续
            // - 荣和（Ron）：从和牌者的下家开始找下一位未和牌玩家
            // - 自摸（Tsumo）：tsumo_actor 就是自摸者本人，跳过后即为其下家，逻辑一致
            // - 多家荣和：取离放铳者最远的和牌者（按轮转顺序），从其下家开始

            // 获取放铳者（target），判断是荣和还是自摸
            let (target, is_ron) = match specific_hora_events[0].1.event {
                Event::Hora { actor, target, .. } => (target, actor != target),
                _ => unreachable!(),
            };

            let start_from = if is_ron {
                // 荣和：找到离放铳者最远的和牌者（按轮转顺序），从其下家开始
                let mut furthest_winner = target; // fallback
                let mut max_distance = 0u8;
                for &(actor_idx, _) in &specific_hora_events {
                    let actor = actor_idx as u8;
                    let distance = (actor + 4 - target) % 4;
                    if distance > max_distance {
                        max_distance = distance;
                        furthest_winner = actor;
                    }
                }
                (furthest_winner + 1) % 4
            } else {
                // 自摸：tsumo_actor 就是自摸者，已在 agari 列表中，会被跳过
                self.tsumo_actor
            };

            let mut next_actor = start_from;
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
                
                // Reset Score Transfer state only when this Tsumo is a normal turn draw,
                // not the rinshan draw after a kong. 杠上炮转雨 requires last_kan_actor/last_kan_revenue
                // to remain set until the discarder is ronned (or the next normal turn).
                let is_rinshan_draw = self.last_kan_actor == Some(self.tsumo_actor);
                if !is_rinshan_draw {
                    self.last_kan_actor = None;
                    self.last_kan_revenue = 0;
                }
            }

            Event::Dahai { actor, pai: _pai, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                self.dahai_count += 1;
                
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

            Event::Pon { actor, .. } => {
                // FIX: 碰牌后必须设置 tsumo_actor 为碰牌者，与大明杠/加杠/暗杠保持一致。
                // 虽然正常流程中碰牌者下一步一定打牌（Dahai 会重新设 tsumo_actor），
                // 但如果不设置，在碰与打牌之间 tsumo_actor 指向错误玩家，
                // 可能在未来新增代码路径中引发逻辑错误。
                self.tsumo_actor = actor;
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                
                // Reset Score Transfer state on Pon (interruption)
                self.last_kan_actor = None;
                self.last_kan_revenue = 0;
            }

            Event::Ankan { actor, .. } => {
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
                    valid: true,
                });
                
                // Construct modified event with deltas and broadcast
                let mut new_event = ev.event.clone();
                match &mut new_event {
                    Event::Ankan { deltas, .. } => *deltas = Some(payment_deltas),
                    _ => unreachable!(),
                }
                self.broadcast(&new_event);
                
                let mut new_ev_ext = ev.clone();
                new_ev_ext.event = new_event;
                self.add_log(new_ev_ext);
            }

            Event::Daiminkan { actor, target, .. } => {
                self.tsumo_actor = actor;
                self.kans += 1;
                
                // Instant Payment (Gua Feng - Ming Kan): Discarder pays 2000 points
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
                    valid: true,
                });
                
                // Construct modified event with deltas and broadcast
                let mut new_event = ev.event.clone();
                match &mut new_event {
                    Event::Daiminkan { deltas, .. } => *deltas = Some(payment_deltas),
                    _ => unreachable!(),
                }
                self.broadcast(&new_event);
                
                let mut new_ev_ext = ev.clone();
                new_ev_ext.event = new_event;
                self.add_log(new_ev_ext);
            }

            Event::Kakan { actor, .. } => {
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
                    valid: true,
                });
                
                // Construct modified event with deltas and broadcast
                let mut new_event = ev.event.clone();
                match &mut new_event {
                    Event::Kakan { deltas, .. } => *deltas = Some(payment_deltas),
                    _ => unreachable!(),
                }
                self.broadcast(&new_event);
                
                let mut new_ev_ext = ev.clone();
                new_ev_ext.event = new_event;
                self.add_log(new_ev_ext);
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
            
            Event::DingQue { .. } => {
                // 如果在非定缺阶段收到DingQue事件，理论上也是不合法的（应该在step开头部分处理）
                // 但为了避免崩溃（Crash instead of Deadlock），我们这里将其视为无操作
                // 这种情况可能发生在状态不同步时：can_ding_que=true 但 ding_que_phase=false
                // 我们记录警告但不panic
                log::warn!("Ignored unexpected DingQue event in step() main match block (not in ding_que_phase). Event: {:?}", ev.event);
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

                // 对手定缺花色 one-hot（3 通道：Man/Pin/Sou）。
                // 定缺已选择时写入对应通道；未选择时全零。
                // oracle 观察可直接看到对手手牌，但显式编码定缺省去模型推理开销，
                // 且与 obs_repr.rs 中自己 ding_que 的编码方式对齐。
                if let Some(suit) = state.ding_que {
                    arr.fill(idx + crate::ding_que::suit_id(suit), 1.);
                }
                idx += 3;

                // FIX: shanten 可能因定缺惩罚超过 6（最大可达 ~9），
                // 但 one-hot 编码只分配了 7 个通道（v2+）或 6 个（v1）。
                // 不裁剪会写入后续通道（shanten_rescale / waits），污染 oracle 观察。
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

                        // 定缺惩罚可使 raw_shanten > 6；需先 clamp 再归一化，
                        // 否则 v > 1.0 会打破特征归一化假设。
                        let v = raw_shanten.min(6) as f32 / 6.;
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

                // 定缺完成状态（1 通道）：对手是否已清除所有定缺花色牌。
                // 用于 value 预估：完成定缺的对手能和牌，对本家威胁更大。
                if state.check_ding_que_complete() {
                    arr.fill(idx, 1.);
                }
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
        // FIX: v1 使用 69 (继承自日麻 136 张)，v2+ 使用 56 (血战 108 张)。
        // 与 invisible.rs 保持一致，否则 v1 时 assert_eq!(idx, shape.0) 会 panic。
        let max_yama_tiles: usize = if version >= 2 { 56 } else { 69 };
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
        board.scores = [crate::consts::INITIAL_SCORE; 4];
        
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
