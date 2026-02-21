use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray1;

use engine::consts::*;
use engine::state::board::{BoardState, Phase};
use engine::state::action::Action;
use engine::hand::calc_shanten;
use engine::algo::shanten::waiting_tiles;
use engine::obs::{encode_student_obs, encode_oracle_obs, encode_action_mask};


use crate::opponent::OpponentPolicy;

#[pyclass]
pub struct RustMahjongEnv {
    state: BoardState,
    player_id: usize,
    prev_score: i32,
    opponent_policy: OpponentPolicy,
    seed: u64,
}

#[pymethods]
impl RustMahjongEnv {
    #[new]
    #[pyo3(signature = (seed=42, opponent_mode="rulebot"))]
    fn new(seed: u64, opponent_mode: &str) -> Self {
        let state = BoardState::new(seed);
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
        }
    }

    fn reset<'py>(&mut self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyDict>> {
        self.seed = seed;
        self.state = BoardState::new(seed);
        self.prev_score = self.state.players[self.player_id].score;

        if let OpponentPolicy::Random(ref mut rng) = self.opponent_policy {
            *rng = fastrand::Rng::with_seed(seed.wrapping_add(12345));
        }

        self.advance_opponents();

        let obs = encode_student_obs(&self.state, self.player_id);
        let oracle_obs = encode_oracle_obs(&self.state, self.player_id);
        let mask = encode_action_mask(&self.state, self.player_id);
        let mask_f32: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let (dq_labels, ow_labels) = self.compute_aux_labels();

        let dict = PyDict::new_bound(py);
        dict.set_item("obs", PyArray1::from_vec_bound(py, obs))?;
        dict.set_item("oracle_obs", PyArray1::from_vec_bound(py, oracle_obs))?;
        dict.set_item("action_mask", PyArray1::from_vec_bound(py, mask_f32))?;
        dict.set_item("dq_labels", PyArray1::from_vec_bound(py, dq_labels))?;
        dict.set_item("ow_labels", PyArray1::from_vec_bound(py, ow_labels))?;
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
        let oracle_obs = encode_oracle_obs(&self.state, self.player_id);
        let mask = encode_action_mask(&self.state, self.player_id);
        let mask_f32: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let (dq_labels, ow_labels) = self.compute_aux_labels();

        let dict = PyDict::new_bound(py);
        dict.set_item("obs", PyArray1::from_vec_bound(py, obs))?;
        dict.set_item("oracle_obs", PyArray1::from_vec_bound(py, oracle_obs))?;
        dict.set_item("action_mask", PyArray1::from_vec_bound(py, mask_f32))?;
        dict.set_item("dq_labels", PyArray1::from_vec_bound(py, dq_labels))?;
        dict.set_item("ow_labels", PyArray1::from_vec_bound(py, ow_labels))?;

        let info = PyDict::new_bound(py);
        info.set_item("win_count", self.state.win_count)?;
        info.set_item("player_won", self.state.players[self.player_id].has_won)?;

        Ok((dict, reward, terminated, truncated, info).to_object(py))
    }

    fn get_oracle_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let obs = encode_oracle_obs(&self.state, self.player_id);
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
        self.state.ding_que_done
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
        if self.state.phase == Phase::Scoring {
            self.state.finalize_scoring();
        }
    }

    fn get_reward_for(&self, player_id: usize) -> f32 {
        let current = self.state.players[player_id].score;
        (current - engine::consts::INITIAL_SCORE) as f32 / 16000.0
    }

    fn player_has_won(&self, player_id: usize) -> bool {
        self.state.players[player_id].has_won
    }

    fn get_win_count(&self) -> u8 {
        self.state.win_count
    }
}

impl RustMahjongEnv {
    fn compute_aux_labels(&self) -> (Vec<f32>, Vec<f32>) {
        let mut dq = vec![3.0f32; 3]; // 3 = "unknown", used as ignore_index in CE
        let mut ow = vec![0.0f32; 81];

        for opp_off in 1..NUM_PLAYERS {
            let opp_id = (self.player_id + opp_off) % NUM_PLAYERS;
            let p = &self.state.players[opp_id];

            if let Some(suit) = p.ding_que {
                dq[opp_off - 1] = suit as u8 as f32; // 0=Man, 1=Pin, 2=Sou
            }

            let s = calc_shanten(&p.hand, p.melds.len());
            if s == 0 {
                let waits = waiting_tiles(&p.hand, p.melds.len());
                for &wt in &waits {
                    ow[(opp_off - 1) * NUM_TILE_TYPES + wt as usize] = 1.0;
                }
            }
        }

        (dq, ow)
    }

    fn advance_opponents(&mut self) {
        if matches!(self.opponent_policy, OpponentPolicy::External) {
            return;
        }
        for _guard in 0..500 {
            if self.state.is_done() || self.state.phase == Phase::Scoring {
                break;
            }

            match self.state.phase {
                Phase::DingQue => {
                    if !self.state.ding_que_done[self.player_id] {
                        break;
                    }
                    for i in 0..NUM_PLAYERS {
                        if i == self.player_id || self.state.ding_que_done[i] { continue; }
                        let action = self.opponent_policy.choose_ding_que(&self.state, i);
                        self.state.apply_action(i, action);
                    }
                    if !self.state.ding_que_done[self.player_id] {
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
