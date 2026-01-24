use super::result::KyokuResult;
use crate::array::Simple2DArray;
use crate::consts::oracle_obs_shape;
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::{matches_tu8, must_tile, t, tu8};
use std::convert::TryInto;
use std::{array, mem};

use anyhow::{Context, Result, bail};
use derivative::Derivative;
use ndarray::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use sha3::{Digest, Sha3_256};
use tinyvec::ArrayVec;

/// Bloody Battle Mahjong Board
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

    /// Bloody Battle Mahjong: track which players have agari
    #[derivative(Default(value = "[false; 4]"))]
    players_agari: [bool; 4],
    /// Bloody Battle Mahjong: count of players who have agari
    agari_count: u8,
    
    has_abortive_ryukyoku: bool,
    kyoku_deltas: [i32; 4],

    #[derivative(Default(value = "56"))]
    tiles_left: u8,
    tsumo_actor: u8,
    // Just a fancy bool
    deal_from_rinshan: Option<()>,
    kans: u8,
    check_four_kan: bool,
    paos: [Option<u8>; 4],

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
        // Bloody Battle Mahjong: no honba, use only kyoku for seed
        let kyoku_seed = Sha3_256::new()
            .chain_update(nonce.to_le_bytes())
            .chain_update(key.to_le_bytes())
            .chain_update([self.kyoku, 0]) // honba always 0 in Bloody Battle
            .finalize()
            .into();
        let mut rng = ChaCha12Rng::from_seed(kyoku_seed);
        let mut seq = UNSHUFFLED;
        seq.shuffle(&mut rng);

        // Deal 13 tiles to each of 4 players
        self.haipai = array::from_fn(|i| seq[i * 13..(i + 1) * 13].try_into().unwrap());
        let mut idx = 13 * 4;

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
                    // Bloody Battle: No renchan
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
            can_renchan: false, // Bloody Battle: no renchan
            has_hora: self.agari_count > 0,
            has_abortive_ryukyoku: self.has_abortive_ryukyoku,
            kyotaku_left: 0, // Bloody Battle: no kyotaku
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
        // Bloody Battle Mahjong: StartKyoku without bakaze, dora_marker, honba, kyotaku
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
        let first_tsumo = Event::Tsumo {
            actor: self.oya,
            pai: tile,
        };
        self.broadcast(&first_tsumo);
        self.add_log_no_meta(first_tsumo);

        Ok(())
    }

    fn exhaustive_ryukyoku(&mut self) {
        // Bloody Battle Mahjong: Exhaustive draw (流局)
        // No special scoring for exhaustive draw in Bloody Battle
        let deltas = [0; 4];

        // Bloody Battle: No nagashi mangan (流局满贯)
        // Just check tenpai for scoring
        let tenpai_actors: ArrayVec<[_; 4]> = self
            .player_states
            .iter()
            .enumerate()
            .filter(|&(_, s)| s.shanten() == 0)
            .map(|(i, _)| i)
            .collect();

        let (plus, minus) = match tenpai_actors.len() {
            1 => (3000, -1000),
            2 => (1500, -1500),
            3 => (1000, -3000),
            // 0 | 4
            _ => (0, 0),
        };
        let mut final_deltas = deltas;
        if plus > 0 {
            let mut dod = [minus; 4];
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

    // Bloody Battle Mahjong: No nagashi mangan, four wind, riichi, or dora
    // These functions are removed

    fn handle_hora(
        &mut self,
        single_actor: u8,
        single_target: u8,
        reactions: &[EventExt; 4],
    ) -> Result<()> {
        let is_ron = single_actor != single_target;
        
        // Bloody Battle Mahjong: Mark player as agari
        if !self.players_agari[single_actor as usize] {
            self.players_agari[single_actor as usize] = true;
            self.agari_count += 1;
            self.player_states[single_actor as usize].has_agari = true;
        }

        // TODO: Calculate points using Bloody Battle scoring system
        // For now, use placeholder calculation
        // This will be replaced when we rewrite the scoring system
        let points = reactions
            .iter()
            .map(|ev| match ev.event {
                Event::Hora { actor, .. } => {
                    // TODO: Replace with Bloody Battle agari_points calculation
                    // For now, return placeholder
                    Ok(Some(crate::algo::point::Point {
                        ron: 1000,
                        tsumo_ko: 500,
                        tsumo_oya: 500,
                    }))
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        // TODO: Implement Bloody Battle scoring system
        // For now, use placeholder calculation
        // Bloody Battle: 点数 = 1000 × 2^(番数-1), 5番封顶 = 16000点
        // Bloody Battle: No oya advantage in scoring
        
        let point = points[single_actor as usize].unwrap();
        let mut deltas = [0; 4];
        
        if is_ron {
            // Ron: target pays full amount
            deltas[single_target as usize] = -point.ron;
            deltas[single_actor as usize] = point.ron;
        } else {
            // Tsumo: all other players pay (no oya advantage)
            for i in 0..4 {
                if i != single_actor as usize {
                    deltas[i] = -point.tsumo_ko;
                }
            }
            deltas[single_actor as usize] = point.tsumo_ko * 3;
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

    // Bloody Battle: No pao (no jihai), this function is removed

    #[inline]
    fn abortive_ryukyoku(&mut self) {
        let ryukyoku = Event::Ryukyoku {
            deltas: Some([0; 4]),
        };
        self.add_log_no_meta(ryukyoku);
        self.has_abortive_ryukyoku = true;
        // No need to broadcast
    }

    fn step(&mut self, reactions: &[EventExt; 4]) -> Result<Poll> {
        // Bloody Battle Mahjong: Check if 3 players have agari
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
            .unwrap(); // Unwrap is safe because at least one player hasn't agari

        if self.check_four_kan && !matches!(ev.event, Event::Hora { .. }) {
            // 四槓散了 (still applies in Bloody Battle)
            self.abortive_ryukyoku();
            return Ok(Poll::End);
        }

        match ev.event {
            Event::None => {
                // Bloody Battle Mahjong: Check for exhaustive draw (流局)
                if self.tiles_left == 0 {
                    self.exhaustive_ryukyoku();
                    return Ok(Poll::End);
                }

                // Skip players who have agari
                while self.players_agari[self.tsumo_actor as usize] {
                    self.tsumo_actor = (self.tsumo_actor + 1) % 4;
                }

                let tile = if self.deal_from_rinshan.take().is_some() {
                    // Bloody Battle: kan draws from yama (no rinshan)
                    // This should not happen, but handle it gracefully
                    self.board.yama.pop().context("illegal kan: yama is empty")?
                } else {
                    self.board.yama.pop().with_context(|| {
                        format!("tiles left > 0 ({}) but yama is empty", self.tiles_left)
                    })?
                };
                self.tiles_left -= 1;
                let tsumo = Event::Tsumo {
                    actor: self.tsumo_actor,
                    pai: tile,
                };

                self.broadcast(&tsumo);
                self.add_log_no_meta(tsumo);
            }

            Event::Dahai { actor, pai, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                
                // Bloody Battle: Skip players who have agari when rotating
                let mut next_actor = (actor + 1) % 4;
                while self.players_agari[next_actor as usize] {
                    next_actor = (next_actor + 1) % 4;
                }
                self.tsumo_actor = next_actor;

                if self.kans == 4 && self.player_states.iter().all(|s| s.kans_count() < 4) {
                    // 四槓散了 (still applies in Bloody Battle)
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

                // Bloody Battle: kan draws from yama (no rinshan, no new dora)
                self.tsumo_actor = actor;
                self.deal_from_rinshan = Some(()); // Mark that next draw is after kan
                self.kans += 1;
            }

            Event::Daiminkan { actor, .. } | Event::Kakan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.deal_from_rinshan = Some(()); // Mark that next draw is after kan
                self.kans += 1;
            }

            // Event::Reach removed - Bloody Battle Mahjong does not have riichi

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

        // Bloody Battle: No pao (no jihai), removed update_paos call

        Ok(Poll::InGame)
    }

    pub fn encode_oracle_obs(&self, perspective: u8, version: u32) -> Array2<f32> {
        let shape = oracle_obs_shape(version);
        let mut arr = Simple2DArray::<34, f32>::new(shape.0);
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

                // Bloody Battle: No red 5s, skip akas_in_hand encoding
                idx += 3; // Keep same index offset for compatibility

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

                if state.at_furiten() {
                    arr.fill(idx, 1.);
                }
                idx += 1;
            });

        let mut encode_tile = |idx: usize, tile: Tile| {
            let tile_id = tile.deaka().as_usize();
            arr.assign(idx, tile_id, 1.);
            // Bloody Battle: No red 5s, skip is_aka check
        };

        self.board
            .yama
            .iter()
            .copied()
            .rev()
            .take(self.tiles_left as usize)
            .for_each(|tile| {
                encode_tile(idx, tile);
                idx += 2;
            });
        idx += (69 - self.tiles_left as usize) * 2;

        // Bloody Battle: No rinshan, skip encoding
        idx += 4 * 2;

        // Bloody Battle: No dora indicators, skip encoding
        idx += 5 * 2;

        // Bloody Battle: No ura indicators, skip encoding
        idx += 5 * 2;

        assert_eq!(idx, shape.0);
        arr.build()
    }
}

#[rustfmt::skip]
// Bloody Battle Mahjong: 108 tiles (3 suits × 9 numbers × 4 copies)
// No jihai (wind/dragon tiles), no red 5s
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
