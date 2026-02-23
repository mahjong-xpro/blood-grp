use crate::consts::*;
use crate::tile::{Tile, Suit};
use crate::hand::{HandCounts, MeldType, remove_tile, add_tile, waiting_tiles, has_suit_tiles};
use crate::algo::shanten::{calc_shanten, clear_shanten_cache};
use crate::algo::agari::{WinContext, FanConfig, calc_fan};
use crate::algo::point::calc_score;
use super::candidate::Candidate;
use super::state::{SPInitState, SPState};

/// Single Player Table calculator.
///
/// Computes tenpai probability, win probability, and expected value
/// for each discard candidate over future turns.
pub struct SPCalculator {
    pub num_melds: usize,
    pub melds: Vec<MeldType>,
    pub ding_que: Option<Suit>,
    pub n_active_payers: u8,
    pub fan_config: FanConfig,
    pub is_at_rinshan: bool,
    pub max_turns: usize,
}

impl SPCalculator {
    pub fn new(num_melds: usize, ding_que: Option<Suit>) -> Self {
        Self {
            num_melds,
            melds: Vec::new(),
            ding_que,
            n_active_payers: 3,
            fan_config: FanConfig::default(),
            is_at_rinshan: false,
            max_turns: MAX_TURNS,
        }
    }

    /// Calculate SP table for a given initial state.
    /// Returns candidates sorted by total EV descending.
    pub fn calc(&self, init: &SPInitState) -> Vec<Candidate> {
        let state = SPState::from_init(init);
        // Clear memoization cache once per SP calculation pass.
        clear_shanten_cache();
        let shanten = calc_shanten(&state.tehai, self.num_melds);

        if shanten < 0 {
            return Vec::new(); // already complete
        }

        // 14-tile hand at shanten 0 needs to discard first (can't call
        // waiting_tiles on a 14-tile hand). Route through discard evaluation.
        let tile_count: u8 = state.tehai.iter().sum();
        let tenpai_tiles = ((4 - self.num_melds) * 3 + 1) as u8;
        if shanten == 0 && tile_count <= tenpai_tiles {
            return self.calc_tenpai(&state);
        }

        self.calc_discard_candidates(&state, shanten)
    }

    fn calc_tenpai(&self, state: &SPState) -> Vec<Candidate> {
        let waits = waiting_tiles(&state.tehai, self.num_melds);
        let mut total_remaining = 0u32;
        let mut total_ev = 0.0f32;

        for &wt in &waits {
            let avail = state.available_count(wt) as u32;
            total_remaining += avail;

            let mut h = state.tehai;
            add_tile(&mut h, wt);
            let score = self.get_win_score(&h, wt, false);
            total_ev += score as f32 * avail as f32;
        }

        if total_remaining == 0 {
            return Vec::new();
        }

        let turns_left = state.remaining.min(self.max_turns as u8);
        let n_left = state.remaining as f32;

        let avg_score = total_ev / total_remaining.max(1) as f32;
        let mut cand = Candidate::new(0, 0);
        let mut cum_not_win = 1.0f32;
        for t in 0..turns_left as usize {
            let draw_prob = (total_remaining as f32 / (n_left - t as f32).max(1.0)).min(1.0);
            let p_win_this_turn = cum_not_win * draw_prob;
            cum_not_win *= 1.0 - draw_prob;
            cand.tenpai_probs[t] = 1.0;
            cand.win_probs[t] = p_win_this_turn;
            cand.exp_values[t] = avg_score * p_win_this_turn;
        }

        vec![cand]
    }

    fn calc_discard_candidates(&self, state: &SPState, shanten: i8) -> Vec<Candidate> {
        let mut candidates = Vec::new();

        let must_discard_dq = self.ding_que
            .map_or(false, |dq| has_suit_tiles(&state.tehai, dq));

        for t in 0..NUM_TILE_TYPES as u8 {
            if state.tehai[t as usize] == 0 { continue; }

            if let Some(dq) = self.ding_que {
                let is_dq_tile = Suit::from_tile(t) == dq;
                if must_discard_dq && !is_dq_tile { continue; }
                if !must_discard_dq && is_dq_tile { continue; }
            }

            let mut h = state.tehai;
            remove_tile(&mut h, t);
            let new_shanten = calc_shanten(&h, self.num_melds);
            let diff = new_shanten - shanten;

            if diff > 0 { continue; }

            let mut cand = Candidate::new(t, diff);
            let turns = state.remaining.min(self.max_turns as u8) as usize;
            let n_left = state.remaining as f32;

            if new_shanten == 0 {
                self.fill_tenpai_candidate(&mut cand, &h, state, turns, n_left);
            } else if new_shanten == 1 {
                // 1-ply lookahead: iishanten → tenpai transition
                self.fill_lookahead_candidate(&mut cand, &h, state, new_shanten, turns, n_left);
            } else {
                // shanten >= 2: fast estimate based on effective tile count
                self.fill_deep_estimate(&mut cand, &h, state, new_shanten, turns, n_left);
            }

            candidates.push(cand);
        }

        candidates.sort_by(|a, b| b.total_ev().partial_cmp(&a.total_ev()).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Fill candidate with precise tenpai-level win prob / EV
    fn fill_tenpai_candidate(
        &self, cand: &mut Candidate, hand: &HandCounts,
        state: &SPState, turns: usize, n_left: f32,
    ) {
        let waits = waiting_tiles(hand, self.num_melds);
        let mut total_outs = 0u32;
        let mut weighted_score = 0.0f32;

        for &wt in &waits {
            let avail = state.available_count(wt) as u32;
            total_outs += avail;
            let mut h = *hand;
            add_tile(&mut h, wt);
            let score = self.get_win_score(&h, wt, false);
            weighted_score += score as f32 * avail as f32;
        }

        if total_outs == 0 { return; }
        let avg_score = weighted_score / total_outs as f32;

        let mut cum_not_win = 1.0f32;
        for turn in 0..turns {
            let draw_prob = (total_outs as f32 / (n_left - turn as f32).max(1.0)).min(1.0);
            let p_win_this_turn = cum_not_win * draw_prob;
            cum_not_win *= 1.0 - draw_prob;
            cand.tenpai_probs[turn] = 1.0;
            cand.win_probs[turn] = p_win_this_turn;
            cand.exp_values[turn] = avg_score * p_win_this_turn;
        }
    }

    /// Fast estimate for shanten >= 2: uses effective tile count without deep expansion
    fn fill_deep_estimate(
        &self, cand: &mut Candidate, hand: &HandCounts,
        state: &SPState, current_shanten: i8, turns: usize, n_left: f32,
    ) {
        let mut total_eff = 0u32;
        for eff_t in 0..NUM_TILE_TYPES as u8 {
            let avail = state.available_count(eff_t);
            if avail == 0 || hand[eff_t as usize] >= 4 { continue; }
            let mut hh = *hand;
            add_tile(&mut hh, eff_t);
            if calc_shanten(&hh, self.num_melds) < current_shanten {
                total_eff += avail as u32;
            }
        }
        if total_eff == 0 { return; }

        let steps_to_tenpai = current_shanten as f32;
        let mut cum_not_eff = 1.0f32;
        for turn in 0..turns {
            let p = (total_eff as f32 / (n_left - turn as f32).max(1.0)).min(1.0);
            cum_not_eff *= 1.0 - p;
            let tenpai_prob = (1.0 - cum_not_eff) / steps_to_tenpai;
            cand.tenpai_probs[turn] = tenpai_prob.min(1.0);
            let win_given_tenpai = (total_eff as f32 * 0.3 / (n_left - turn as f32).max(1.0)).min(0.5);
            cand.win_probs[turn] = tenpai_prob * win_given_tenpai;
            let base_score = if self.num_melds == 0 { 2000.0 } else { 1000.0 };
            cand.exp_values[turn] = cand.win_probs[turn] * base_score;
        }
    }

    /// 1-ply lookahead for iishanten: estimate tenpai→win path using sampled eff tiles
    fn fill_lookahead_candidate(
        &self, cand: &mut Candidate, hand: &HandCounts,
        state: &SPState, current_shanten: i8, turns: usize, n_left: f32,
    ) {
        let mut total_eff = 0u32;
        let mut sample_outs = 0u32;
        let mut sample_score = 0i64;
        let mut n_sampled = 0u32;

        /// Set to 0 to skip waiting_tiles calls in the lookahead path.
        /// waiting_tiles calls calc_shanten 27 times per sample; with 14 candidates
        /// and 8 samples each that is ~3000 calc_shanten calls per observation,
        /// causing multi-second hangs during rollout collection.
        /// The fast fill_deep_estimate path is used instead when n_sampled == 0.
        const MAX_SAMPLES: u32 = 0;
        for eff_t in 0..NUM_TILE_TYPES as u8 {
            let avail = state.available_count(eff_t);
            if avail == 0 || hand[eff_t as usize] >= 4 { continue; }

            let mut hh = *hand;
            add_tile(&mut hh, eff_t);
            let s = calc_shanten(&hh, self.num_melds);
            if s >= current_shanten { continue; }

            total_eff += avail as u32;

            if n_sampled < MAX_SAMPLES {
                // Lightweight: count available tiles for waiting types without calling waiting_tiles
                let waits = waiting_tiles(&hh, self.num_melds);
                let outs: u32 = waits.iter().map(|&wt| {
                    if wt == eff_t {
                        state.available_count(wt).saturating_sub(1) as u32
                    } else {
                        state.available_count(wt) as u32
                    }
                }).sum();
                sample_outs += outs;
                let base_score = if self.num_melds == 0 { 2000 } else { 1000 };
                sample_score += base_score * outs as i64;
                n_sampled += 1;
            }
        }

        if total_eff == 0 { return; }

        let avg_outs = if n_sampled > 0 { sample_outs as f32 / n_sampled as f32 } else { 4.0 };
        let avg_score = if n_sampled > 0 && sample_outs > 0 {
            sample_score as f32 / sample_outs as f32
        } else {
            2000.0
        };

        let mut cum_not_eff = 1.0f32;
        for turn in 0..turns {
            let p = (total_eff as f32 / (n_left - turn as f32).max(1.0)).min(1.0);
            cum_not_eff *= 1.0 - p;
            cand.tenpai_probs[turn] = 1.0 - cum_not_eff;

            let remaining = turns.saturating_sub(turn + 1);
            let p_win = if remaining > 0 && avg_outs > 0.0 {
                let p_per = (avg_outs / (n_left - (turn + 1) as f32).max(1.0)).min(1.0);
                1.0 - (1.0 - p_per).powi(remaining as i32)
            } else { 0.0 };
            cand.win_probs[turn] = cand.tenpai_probs[turn] * p_win;
            cand.exp_values[turn] = cand.win_probs[turn] * avg_score;
        }
    }

    fn get_win_score(&self, hand: &HandCounts, winning_tile: Tile, is_ron: bool) -> i32 {
        let ctx = WinContext {
            tehai: *hand,
            melds: self.melds.clone(),
            winning_tile,
            is_ron,
            ding_que: self.ding_que,
            is_after_kan: self.is_at_rinshan,
            is_kan_discard: false,
            is_chankan: false,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            exclude_gen_tile: None,
            fan_config: self.fan_config,
        };

        match calc_fan(&ctx) {
            Some(result) => {
                let base = calc_score(result.fan);
                if is_ron {
                    base
                } else {
                    base * self.n_active_payers as i32
                }
            }
            None => 0,
        }
    }
}
