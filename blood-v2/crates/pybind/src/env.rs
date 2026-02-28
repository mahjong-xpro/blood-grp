use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray1;

use engine::consts::*;
use engine::state::board::{BoardState, Phase};
use engine::state::action::Action;
use engine::state::event::Event;
use engine::tile::{Suit, Tile, is_terminal};
use engine::hand::{HandCounts, MeldType, suit_tile_count};
use engine::algo::shanten::{calc_shanten, waiting_tiles};
use engine::algo::agari::{calc_fan, calc_gen_count, WinContext};
use engine::obs::{encode_student_obs, encode_oracle_obs, encode_action_mask, OracleSpCache};

fn event_to_json(e: &Event) -> String {
    match e {
        Event::Deal { player, tiles } => {
            let arr = tiles.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
            format!(r#"{{"type":"deal","player":{},"tiles":[{}]}}"#, player, arr)
        }
        Event::DingQue { player, suit } => {
            let s = match suit { Suit::Man => "man", Suit::Pin => "pin", Suit::Sou => "sou" };
            format!(r#"{{"type":"ding_que","player":{},"suit":"{}"}}"#, player, s)
        }
        Event::Draw { player, tile } =>
            format!(r#"{{"type":"draw","player":{},"tile":{}}}"#, player, tile),
        Event::Discard { player, tile, is_tsumogiri } =>
            format!(r#"{{"type":"discard","player":{},"tile":{},"is_tsumogiri":{}}}"#, player, tile, is_tsumogiri),
        Event::Pon { player, from, tile } =>
            format!(r#"{{"type":"pon","player":{},"from":{},"tile":{}}}"#, player, from, tile),
        Event::MinKan { player, from, tile } =>
            format!(r#"{{"type":"min_kan","player":{},"from":{},"tile":{}}}"#, player, from, tile),
        Event::AnKan { player, tile } =>
            format!(r#"{{"type":"an_kan","player":{},"tile":{}}}"#, player, tile),
        Event::KaKan { player, tile, is_jishiyu } =>
            format!(r#"{{"type":"ka_kan","player":{},"tile":{},"is_jishiyu":{}}}"#, player, tile, is_jishiyu),
        Event::Tsumo { player, tile } =>
            format!(r#"{{"type":"tsumo","player":{},"tile":{}}}"#, player, tile),
        Event::Ron { player, from, tile } =>
            format!(r#"{{"type":"ron","player":{},"from":{},"tile":{}}}"#, player, from, tile),
        Event::KanPayment { payer, receiver, amount } =>
            format!(r#"{{"type":"kan_payment","payer":{},"receiver":{},"amount":{}}}"#, payer, receiver, amount),
        Event::GameEnd =>
            r#"{"type":"game_end"}"#.to_string(),
    }
}


use crate::opponent::OpponentPolicy;

#[pyclass]
pub struct RustMahjongEnv {
    state: BoardState,
    player_id: usize,
    prev_score: i32,
    opponent_policy: OpponentPolicy,
    seed: u64,
    initial_score: i32,
    oracle_sp_cache: OracleSpCache,
}

#[pymethods]
impl RustMahjongEnv {
    #[new]
    #[pyo3(signature = (seed=42, opponent_mode="rulebot", initial_score=100_000))]
    fn new(seed: u64, opponent_mode: &str, initial_score: i32) -> Self {
        let state = BoardState::with_initial_score(seed, initial_score);
        let policy = match opponent_mode {
            "random" => OpponentPolicy::Random(fastrand::Rng::with_seed(seed.wrapping_add(12345))),
            "external" => OpponentPolicy::External,
            _ => OpponentPolicy::RuleBot,
        };
        Self {
            player_id: 0,
            prev_score: state.players[0].score,
            state,
            opponent_policy: policy,
            seed,
            initial_score,
            oracle_sp_cache: OracleSpCache::default(),
        }
    }

    fn reset<'py>(&mut self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyDict>> {
        self.seed = seed;
        self.state = BoardState::with_initial_score(seed, self.initial_score);
        self.prev_score = self.state.players[self.player_id].score;
        self.oracle_sp_cache.clear();

        if let OpponentPolicy::Random(ref mut rng) = self.opponent_policy {
            *rng = fastrand::Rng::with_seed(seed.wrapping_add(12345));
        }

        self.advance_opponents();

        let obs = encode_student_obs(&self.state, self.player_id);
        let oracle_obs = encode_oracle_obs(&self.state, self.player_id, Some(&mut self.oracle_sp_cache));
        let mask = encode_action_mask(&self.state, self.player_id);
        let mask_f32: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let (shanten_labels, ow_labels) = self.compute_aux_labels();

        let dict = PyDict::new_bound(py);
        dict.set_item("obs", PyArray1::from_vec_bound(py, obs))?;
        dict.set_item("oracle_obs", PyArray1::from_vec_bound(py, oracle_obs))?;
        dict.set_item("action_mask", PyArray1::from_vec_bound(py, mask_f32))?;
        dict.set_item("shanten_labels", PyArray1::from_vec_bound(py, shanten_labels))?;
        dict.set_item("ow_labels", PyArray1::from_vec_bound(py, ow_labels))?;

        // Initial info fields for arena compatibility (no winners yet at reset)
        let info = PyDict::new_bound(py);
        info.set_item("winners", Vec::<usize>::new())?;
        let scores: Vec<i32> = (0..NUM_PLAYERS)
            .map(|i| self.state.players[i].score)
            .collect();
        info.set_item("scores", scores)?;
        dict.set_item("info", info)?;

        Ok(dict)
    }

    fn step<'py>(&mut self, py: Python<'py>, action_idx: usize) -> PyResult<PyObject> {
        let action = Action::from_index(action_idx)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("invalid action index in step: {}", action_idx)
            ))?;
        self.state.apply_action(self.player_id, action);

        self.advance_opponents();

        if self.state.phase == Phase::Scoring {
            self.state.finalize_scoring();
        }

        let reward = self.state.get_rewards(self.player_id, self.prev_score);
        self.prev_score = self.state.players[self.player_id].score;

        let terminated = self.state.is_done() || self.state.players[self.player_id].has_won;
        let truncated = false;

        let obs = encode_student_obs(&self.state, self.player_id);
        let oracle_obs = encode_oracle_obs(&self.state, self.player_id, Some(&mut self.oracle_sp_cache));
        let mask = encode_action_mask(&self.state, self.player_id);
        let mask_f32: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let (shanten_labels, ow_labels) = self.compute_aux_labels();

        let dict = PyDict::new_bound(py);
        dict.set_item("obs", PyArray1::from_vec_bound(py, obs))?;
        dict.set_item("oracle_obs", PyArray1::from_vec_bound(py, oracle_obs))?;
        dict.set_item("action_mask", PyArray1::from_vec_bound(py, mask_f32))?;
        dict.set_item("shanten_labels", PyArray1::from_vec_bound(py, shanten_labels))?;
        dict.set_item("ow_labels", PyArray1::from_vec_bound(py, ow_labels))?;

        let info = PyDict::new_bound(py);
        info.set_item("win_count", self.state.win_count)?;
        info.set_item("player_won", self.state.players[self.player_id].has_won)?;

        // Per-seat win status for arena evaluation with seat rotation
        let winners: Vec<usize> = (0..NUM_PLAYERS)
            .filter(|&i| self.state.players[i].has_won)
            .collect();
        info.set_item("winners", winners)?;

        // Per-seat final scores (always available, useful for arena)
        let scores: Vec<i32> = (0..NUM_PLAYERS)
            .map(|i| self.state.players[i].score)
            .collect();
        info.set_item("scores", scores)?;

        Ok((dict, reward, terminated, truncated, info).to_object(py))
    }

    fn get_oracle_obs<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let obs = encode_oracle_obs(&self.state, self.player_id, Some(&mut self.oracle_sp_cache));
        Ok(PyArray1::from_vec_bound(py, obs))
    }

    fn get_scores(&self) -> [i32; NUM_PLAYERS] {
        self.state.get_scores()
    }

    fn is_done(&self) -> bool {
        self.state.is_done()
    }

    // --- Low-level API for external opponent control ---

    fn get_phase(&self) -> &'static str {
        match self.state.phase {
            Phase::DingQue => "ding_que",
            Phase::SelfCheck => "self_check",
            Phase::KanSelect => "kan_select",
            Phase::Discard => "discard",
            Phase::Reaction => "reaction",
            Phase::Scoring => "scoring",
            Phase::Done => "done",
        }
    }

    fn get_current_player(&self) -> usize {
        self.state.current_player
    }

    fn get_ding_que_done(&self) -> [bool; NUM_PLAYERS] {
        self.state.ding_que_done()
    }

    fn get_reaction_pending(&self) -> [bool; NUM_PLAYERS] {
        self.state.reaction_pending
    }

    fn has_decision(&self, player_id: usize) -> bool {
        self.state.get_decision_request(player_id).is_some()
    }

    fn get_player_obs<'py>(&self, py: Python<'py>, player_id: usize) -> PyResult<Bound<'py, PyDict>> {
        let obs = encode_student_obs(&self.state, player_id);
        let mask = encode_action_mask(&self.state, player_id);
        let mask_f32: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let dict = PyDict::new_bound(py);
        dict.set_item("obs", PyArray1::from_vec_bound(py, obs))?;
        dict.set_item("action_mask", PyArray1::from_vec_bound(py, mask_f32))?;
        Ok(dict)
    }

    fn apply_ext_action(&mut self, player_id: usize, action_idx: usize) -> PyResult<()> {
        let action = Action::from_index(action_idx)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("invalid action index: {}", action_idx)
            ))?;
        self.state.apply_action(player_id, action);
        Ok(())
    }

    fn finalize_scoring(&mut self) {
        if self.state.phase != Phase::Done {
            self.state.phase = Phase::Scoring;
            self.state.finalize_scoring();
        }
    }

    fn get_reward_for(&self, player_id: usize) -> f32 {
        let current = self.state.players[player_id].score;
        (current - self.initial_score) as f32 / engine::consts::REWARD_NORM as f32
    }

    fn player_has_won(&self, player_id: usize) -> bool {
        self.state.players[player_id].has_won
    }

    fn get_win_count(&self) -> u8 {
        self.state.win_count
    }

    fn get_agent_shanten(&self) -> i32 {
        let p = &self.state.players[self.player_id];
        calc_shanten(&p.hand, p.melds.len()).into()
    }

    /// Returns [man_count, pin_count, sou_count] for the agent's hand.
    fn get_agent_suit_counts(&self) -> [u8; 3] {
        let p = &self.state.players[self.player_id];
        [
            suit_tile_count(&p.hand, Suit::Man),
            suit_tile_count(&p.hand, Suit::Pin),
            suit_tile_count(&p.hand, Suit::Sou),
        ]
    }

    /// Heuristic fan estimation for the agent's current hand.
    ///
    /// For tenpai hands (shanten=0), computes the best achievable fan across all
    /// waiting tiles using the full agari calculator. For non-tenpai hands,
    /// estimates fan potential from structural patterns (qingyise, menqing,
    /// duanyaojiu, gen count) without requiring a complete hand.
    ///
    /// Returns a float in [0, MAX_FAN] representing estimated fan count.
    fn get_agent_estimated_fan(&self) -> f32 {
        let p = &self.state.players[self.player_id];
        let shanten = calc_shanten(&p.hand, p.melds.len());

        if shanten == 0 {
            // Tenpai: compute exact best fan across all waiting tiles
            let waits = waiting_tiles(&p.hand, p.melds.len());
            let mut best_fan = 0u8;
            for &wt in &waits {
                let mut h = p.hand;
                h[wt as usize] += 1;
                let ctx = WinContext {
                    tehai: h,
                    melds: p.melds.clone(),
                    winning_tile: wt,
                    is_ron: false, // assume tsumo (higher fan) for optimistic estimate
                    ding_que: p.ding_que,
                    is_after_kan: false,
                    is_kan_discard: false,
                    is_chankan: false,
                    is_haidi: false,
                    is_tianhu: false,
                    is_dihu: false,
                    exclude_gen_tile: None,
                    fan_config: self.state.fan_config.clone(),
                };
                if let Some(result) = calc_fan(&ctx) {
                    best_fan = best_fan.max(result.fan);
                }
            }
            return best_fan as f32;
        }

        // Non-tenpai: heuristic estimation from hand structure
        estimate_fan_heuristic(&p.hand, &p.melds, p.ding_que)
    }

    /// Returns all recorded events as a JSONL string (one JSON object per line).
    /// Call after finalize_scoring() to get the complete game log.
    fn get_events_jsonl(&self) -> String {
        self.state.events.iter()
            .map(|e| event_to_json(e))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns the game header JSON string for the replay file.
    fn get_game_header_json(&self, names: Vec<String>) -> String {
        let scores_str = (0..NUM_PLAYERS)
            .map(|_| self.initial_score.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"type":"game_start","seed":{},"names":[{}],"initial_scores":[{}],"dealer":{}}}"#,
            self.seed,
            names.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(","),
            scores_str,
            self.state.dealer,
        )
    }

    /// Returns the final scores for all players.
    fn get_final_scores(&self) -> [i32; NUM_PLAYERS] {
        self.state.get_scores()
    }

    fn get_aux_labels<'py>(&self, py: Python<'py>, player_id: usize) -> PyResult<Bound<'py, PyDict>> {
        let _ = player_id; // always computed from self.player_id (seat 0)
        let (shanten_labels, ow_labels) = self.compute_aux_labels();
        let dict = PyDict::new_bound(py);
        dict.set_item("shanten_labels", PyArray1::from_vec_bound(py, shanten_labels))?;
        dict.set_item("ow_labels", PyArray1::from_vec_bound(py, ow_labels))?;
        Ok(dict)
    }
}

impl RustMahjongEnv {
    fn compute_aux_labels(&self) -> (Vec<f32>, Vec<f32>) {
        // shanten_labels: 3 opponents x 5 classes (0/1/2/3/4+ shanten), one-hot, shape [15]
        let mut shanten_labels = vec![0.0f32; 15];
        // ow_labels: 3 opponents x 27 tiles, shape [81]
        let mut ow = vec![0.0f32; 81];

        for opp_off in 1..NUM_PLAYERS {
            let opp_id = (self.player_id + opp_off) % NUM_PLAYERS;
            let p = &self.state.players[opp_id];

            let s = calc_shanten(&p.hand, p.melds.len());
            let sh = s.max(0).min(4) as usize;
            shanten_labels[(opp_off - 1) * 5 + sh] = 1.0;

            if s == 0 {
                let waits = waiting_tiles(&p.hand, p.melds.len());
                for &wt in &waits {
                    ow[(opp_off - 1) * NUM_TILE_TYPES + wt as usize] = 1.0;
                }
            }
        }

        (shanten_labels, ow)
    }

    fn advance_opponents(&mut self) {
        if matches!(self.opponent_policy, OpponentPolicy::External) {
            return;
        }
        let mut prev_turn = self.state.turn_count;
        let mut stall_count = 0u32;
        for _guard in 0..500 {
            if self.state.is_done() || self.state.phase == Phase::Scoring {
                break;
            }

            // Detect stuck state: if turn_count hasn't advanced in 8 consecutive
            // iterations, the game is deadlocked — force scoring to unblock.
            if self.state.turn_count == prev_turn {
                stall_count += 1;
                if stall_count >= 8 {
                    self.state.phase = Phase::Scoring;
                    break;
                }
            } else {
                prev_turn = self.state.turn_count;
                stall_count = 0;
            }

            match self.state.phase {
                Phase::DingQue => {
                    if self.state.players[self.player_id].ding_que.is_none() {
                        break;
                    }
                    for i in 0..NUM_PLAYERS {
                        if i == self.player_id || self.state.players[i].ding_que.is_some() { continue; }
                        let action = self.opponent_policy.choose_ding_que(&self.state, i);
                        self.state.apply_action(i, action);
                    }
                    if self.state.players[self.player_id].ding_que.is_none() {
                        break;
                    }
                }
                Phase::SelfCheck => {
                    if self.state.current_player == self.player_id {
                        if self.state.get_decision_request(self.player_id).is_some() {
                            break;
                        }
                        // No special action at SelfCheck → auto-advance to Discard
                        self.state.apply_action(self.player_id, Action::Pass);
                        continue;
                    }
                    let cp = self.state.current_player;
                    let action = self.opponent_policy.choose_action(&self.state, cp);
                    self.state.apply_action(cp, action);
                }
                Phase::Discard => {
                    if self.state.current_player == self.player_id {
                        break;
                    }
                    let cp = self.state.current_player;
                    let action = self.opponent_policy.choose_action(&self.state, cp);
                    self.state.apply_action(cp, action);
                }
                Phase::Reaction => {
                    if self.state.reaction_pending[self.player_id] {
                        break;
                    }
                    for i in 0..NUM_PLAYERS {
                        if i == self.player_id || !self.state.reaction_pending[i] { continue; }
                        let action = self.opponent_policy.choose_reaction(&self.state, i);
                        self.state.apply_action(i, action);
                    }
                    if self.state.reaction_pending[self.player_id] {
                        break;
                    }
                }
                Phase::KanSelect => {
                    if self.state.current_player == self.player_id {
                        break;
                    }
                    let cp = self.state.current_player;
                    let action = self.opponent_policy.choose_action(&self.state, cp);
                    self.state.apply_action(cp, action);
                }
                _ => break,
            }
        }
    }
}

/// Heuristic fan estimation for non-tenpai hands.
///
/// Checks structural patterns that contribute to fan in blood mahjong:
/// - pinghu (base): +1 (always)
/// - tsumo: +1 (optimistic assumption)
/// - menqing (no open melds): +1
/// - qingyise (single suit): +2
/// - duanyaojiu (all 2-8): +1
/// - gen (4 copies of a tile): +1 each
///
/// Returns estimated fan as f32, capped at MAX_FAN.
fn estimate_fan_heuristic(hand: &HandCounts, melds: &[MeldType], ding_que: Option<Suit>) -> f32 {
    let mut fan = 1.0f32; // pinghu base
    fan += 1.0; // optimistic tsumo assumption

    // Menqing: no open melds
    let is_menqing = melds.iter().all(|m| !m.is_open());
    if is_menqing {
        fan += 1.0;
    }

    // Qingyise: all tiles in hand + melds belong to a single suit
    // (excluding ding_que suit tiles which will be discarded)
    let mut suit_counts = [0u32; 3]; // man, pin, sou
    for t in 0..NUM_TILE_TYPES {
        if hand[t] > 0 {
            let suit_idx = t / TILES_PER_SUIT;
            // Skip tiles of ding_que suit (they'll be discarded)
            if let Some(dq) = ding_que {
                if suit_idx == dq as usize { continue; }
            }
            suit_counts[suit_idx] += hand[t] as u32;
        }
    }
    for m in melds {
        let suit_idx = m.tile() as usize / TILES_PER_SUIT;
        suit_counts[suit_idx] += 1;
    }
    let active_suits = suit_counts.iter().filter(|&&c| c > 0).count();
    if active_suits == 1 {
        fan += 2.0;
    }

    // Duanyaojiu: all tiles are rank 2-8 (no terminals)
    let mut all_inner = true;
    for t in 0..NUM_TILE_TYPES {
        if hand[t] > 0 {
            if let Some(dq) = ding_que {
                if t / TILES_PER_SUIT == dq as usize { continue; }
            }
            if is_terminal(t as Tile) {
                all_inner = false;
                break;
            }
        }
    }
    if all_inner {
        for m in melds {
            if is_terminal(m.tile()) {
                all_inner = false;
                break;
            }
        }
    }
    if all_inner && (suit_counts.iter().sum::<u32>() > 0) {
        fan += 1.0;
    }

    // Gen count: tiles appearing 4 times
    let gen = calc_gen_count(hand, melds, None);
    fan += gen as f32;

    fan.min(MAX_FAN as f32)
}
