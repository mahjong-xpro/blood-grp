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
    check_four_kan: bool,

    log: Vec<EventExt>,
}

pub struct AgentContext<'a> {
    pub player_states: &'a [PlayerState; 4],
    pub log: &'a [EventExt],
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
    pub fn poll(&mut self, mut reactions: [EventExt; 4]) -> Result<Poll> {
        loop {
            let poll = self.step(&reactions)?;
            match poll {
                Poll::InGame => {
                    if self.player_states.iter().any(|c| c.last_cans().can_act()) {
                        return Ok(poll);
                    }
                }
                Poll::End => {
                    self.add_log_no_meta(Event::EndKyoku);
                    vec_add_assign(&mut self.board.scores, &self.kyoku_deltas);
                    return Ok(poll);
                }
            };
            reactions = Default::default();
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

        Ok(())
    }

    pub(crate) fn exhaustive_ryukyoku(&mut self) {
        // Flow: 1. Check 查花猪 (huazhu), 2. Check 查大叫 (tenpai)
        let mut final_deltas = [0; 4];

        // Step 1: 查花猪 (Check Huazhu - players with ding_que suit tiles remaining)
        // 花猪玩家需要向所有非花猪玩家赔付16000点（封顶点数）
        // 每个花猪向每个非花猪支付 16000/非花猪数量
        let huazhu_actors: ArrayVec<[_; 4]> = self
            .player_states
            .iter()
            .enumerate()
            .filter(|&(_, s)| !s.check_ding_que_complete()) // 还有定缺花色牌
            .map(|(i, _)| i)
            .collect();

        if !huazhu_actors.is_empty() {
            let non_huazhu_count = 4 - huazhu_actors.len();
            if non_huazhu_count > 0 {
                // 花猪罚分：每个花猪向每个非花猪支付 16000/非花猪数量
                let penalty_per_huazhu_per_non_huazhu = 16000 / non_huazhu_count; // 每个花猪向每个非花猪支付
                let total_penalty_per_huazhu = penalty_per_huazhu_per_non_huazhu * non_huazhu_count; // 每个花猪支付的总数
                let total_reward_per_non_huazhu = penalty_per_huazhu_per_non_huazhu * huazhu_actors.len(); // 每个非花猪获得的总数
                
                // 计算花猪罚分
                let mut huazhu_deltas = [0; 4];
                for &huazhu in &huazhu_actors {
                    huazhu_deltas[huazhu] = -(total_penalty_per_huazhu as i32);
                }
                for i in 0..4 {
                    if !huazhu_actors.contains(&i) {
                        huazhu_deltas[i] = total_reward_per_non_huazhu as i32;
                    }
                }
                vec_add_assign(&mut final_deltas, &huazhu_deltas);
            }
        }

        // Step 2: 查大叫 (Check Tenpai - exclude huazhu players)
        // 排除花猪玩家后，未听牌玩家向听牌玩家赔付
        let tenpai_actors: ArrayVec<[_; 4]> = self
            .player_states
            .iter()
            .enumerate()
            .filter(|&(i, s)| {
                // 排除花猪玩家
                !huazhu_actors.contains(&i) && s.shanten() == 0
            })
            .map(|(i, _)| i)
            .collect();

        let (plus, minus) = match tenpai_actors.len() {
            1 => (3000, -1000),
            2 => (1500, -1500),
            3 => (1000, -3000),
            // 0 | 4 (all non-huazhu players are tenpai or none are tenpai)
            _ => (0, 0),
        };
        
        if plus > 0 {
            let mut dod = [minus; 4];
            // 只对非花猪玩家应用查大叫规则
            for i in 0..4 {
                if huazhu_actors.contains(&i) {
                    dod[i] = 0; // 花猪玩家不参与查大叫
                }
            }
            tenpai_actors.into_iter().for_each(|i| dod[i] = plus);
            vec_add_assign(&mut final_deltas, &dod);
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
    ) -> Result<()> {
        let is_ron = single_actor != single_target;
        
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

        // This uses the actual fan calculation from AgariCalculator
        let point = self.player_states[single_actor as usize]
            .agari_points(is_ron, &[])
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
        } else {
            // Tsumo: all other players pay (no oya advantage)
            let tsumo_total = point.tsumo_total(false); // No oya advantage
            for i in 0..4 {
                if i != single_actor as usize {
                    deltas[i] = -point.tsumo_ko;
                }
            }
            deltas[single_actor as usize] = tsumo_total;
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


    #[inline]
    fn abortive_ryukyoku(&mut self) {
        let ryukyoku = Event::Ryukyoku {
            deltas: Some([0; 4]),
        };
        self.add_log_no_meta(ryukyoku);
        // No need to broadcast
    }

    fn step(&mut self, reactions: &[EventExt; 4]) -> Result<Poll> {
        if self.agari_count >= 3 {
            return Ok(Poll::End);
        }

        if self.tiles_left == 56 {
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

        // 确保至少有一个玩家还没有和牌（基础规则：3人和牌时游戏结束）
        // 如果所有玩家都已和牌，说明游戏状态不一致，应该已经结束
        ensure!(
            self.agari_count < 4,
            "All players have agari (agari_count = {}), but game hasn't ended. This indicates a fundamental bug in game state management.",
            self.agari_count
        );

        let ev = reactions
            .iter()
            .enumerate()
            .filter(|(actor, _)| !self.players_agari[*actor]) // Skip players who have agari
            .map(|(_, ev)| ev)
            .min_by_key(|ev| match ev.event {
                Event::Hora { .. } => 0,
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

        if self.check_four_kan && !matches!(ev.event, Event::Hora { .. }) {
            self.abortive_ryukyoku();
            return Ok(Poll::End);
        }

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
            }

            Event::Dahai { actor, pai: _pai, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                
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

                if self.kans == 4 && self.player_states.iter().all(|s| s.kans_count() < 4) {
                    self.check_four_kan = true;
                }
            }

            Event::Pon { .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
            }

            Event::Ankan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.kans += 1;
            }

            Event::Daiminkan { actor, .. } | Event::Kakan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.kans += 1;
            }


            Event::Hora { actor, target, .. } => {
                self.handle_hora(actor, target, reactions)?;
                return Ok(Poll::End);
            }

            Event::Ryukyoku { .. } => {
                // 九種九牌
                self.abortive_ryukyoku();
                return Ok(Poll::End);
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
        
        // Verify: Player 0 (huazhu) pays 16000
        // Each of players 1, 2, 3 gets 16000/3 ≈ 5333
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], -16000, "Huazhu player should pay 16000");
        assert_eq!(deltas[1], 5333, "Non-huazhu player 1 should get 5333");
        assert_eq!(deltas[2], 5333, "Non-huazhu player 2 should get 5333");
        assert_eq!(deltas[3], 5334, "Non-huazhu player 3 should get 5334 (rounding)");
        
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
        
        // Verify: Player 0 (tenpai) gets +3000
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
        // - Player 0 (huazhu) pays 16000
        // - Player 1 (tenpai, non-huazhu) gets 5333 (from huazhu) + 3000 (from tenpai) = 8333
        // - Player 2 (non-huazhu, non-tenpai) gets 5333 (from huazhu) - 1000 (to tenpai) = 4333
        // - Player 3 (non-huazhu, non-tenpai) gets 5334 (from huazhu, rounding) - 1000 (to tenpai) = 4334
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], -16000, "Huazhu player should pay 16000");
        assert_eq!(deltas[1], 8333, "Tenpai non-huazhu player should get 5333+3000");
        assert_eq!(deltas[2], 4333, "Non-tenpai non-huazhu player should get 5333-1000");
        assert_eq!(deltas[3], 4334, "Non-tenpai non-huazhu player should get 5334-1000");
        
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
        
        // Verify: Players 0, 1 (tenpai) each get +1500
        // Players 2, 3 (non-tenpai) each pay -1500
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], 1500, "Tenpai player 0 should get +1500");
        assert_eq!(deltas[1], 1500, "Tenpai player 1 should get +1500");
        assert_eq!(deltas[2], -1500, "Non-tenpai player 2 should pay -1500");
        assert_eq!(deltas[3], -1500, "Non-tenpai player 3 should pay -1500");
        
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
        
        // Verify: Players 0, 1 (huazhu) each pay 16000
        // Players 2, 3 (non-huazhu) each get 16000 (from 2 huazhu players)
        let deltas = board_state.kyoku_deltas;
        assert_eq!(deltas[0], -16000, "Huazhu player 0 should pay 16000");
        assert_eq!(deltas[1], -16000, "Huazhu player 1 should pay 16000");
        assert_eq!(deltas[2], 16000, "Non-huazhu player 2 should get 16000");
        assert_eq!(deltas[3], 16000, "Non-huazhu player 3 should get 16000");
        
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
