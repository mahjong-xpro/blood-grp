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

    /// 三层精度的打牌候选评估：
    /// - 听牌(shanten=0)：精确计算待牌、番型得分
    /// - 一向听(shanten=1)：采样前瞻，精确计算部分进牌的待牌和得分
    /// - 二向听以上(shanten≥2)：基于有效牌数的快速估计
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
                // 听牌：精确计算所有待牌的枚数和番型得分
                self.fill_tenpai_candidate(&mut cand, &h, state, turns, n_left);
            } else if new_shanten == 1 {
                // 一向听：采样前瞻，对最多5个有效进牌精确计算待牌和得分
                self.fill_lookahead_candidate(&mut cand, &h, state, new_shanten, turns, n_left);
            } else {
                // 二向听以上：基于有效牌数的快速估计（不展开下一层）
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

    /// 一向听的1-ply前瞻：通过采样有效进牌来估计听牌→和牌路径。
    ///
    /// 性能说明：
    /// - waiting_tiles() 每次调用 calc_shanten() 27次（遍历所有牌种）
    /// - calc_shanten() 有线程局部 FxHashMap 缓存（SHANTEN_CACHE），
    ///   同一次 SP 计算内的重复手牌状态会命中缓存
    /// - 采样上限 MAX_SAMPLES=5，最多 5×27=135 次 calc_shanten 调用，
    ///   其中大量会命中缓存，实际开销远低于理论值
    /// - 按可用枚数降序排列有效牌，优先采样最常见的进牌以提高估计覆盖率
    fn fill_lookahead_candidate(
        &self, cand: &mut Candidate, hand: &HandCounts,
        state: &SPState, current_shanten: i8, turns: usize, n_left: f32,
    ) {
        // 第一遍：收集所有有效进牌（能降低向听数的牌）及其可用枚数
        let mut eff_tiles: Vec<(u8, u8)> = Vec::new(); // (tile_id, available_count)
        for eff_t in 0..NUM_TILE_TYPES as u8 {
            let avail = state.available_count(eff_t);
            if avail == 0 || hand[eff_t as usize] >= 4 { continue; }

            let mut hh = *hand;
            add_tile(&mut hh, eff_t);
            let s = calc_shanten(&hh, self.num_melds);
            if s >= current_shanten { continue; }

            eff_tiles.push((eff_t, avail));
        }

        if eff_tiles.is_empty() { return; }

        // 总有效进牌数（用于听牌概率计算）
        let total_eff: u32 = eff_tiles.iter().map(|&(_, a)| a as u32).sum();

        // 按可用枚数降序排列，优先采样最多的进牌
        // 这样能用最少的采样覆盖最大的概率质量
        eff_tiles.sort_by(|a, b| b.1.cmp(&a.1));

        // 采样上限：5个有效牌 × 27次calc_shanten/个 = 最多135次调用
        // 实际因缓存命中，开销约为50-80次未缓存调用
        const MAX_SAMPLES: usize = 5;

        let mut sample_outs_total = 0u32;    // 采样牌的总待牌枚数
        let mut sample_weighted_score = 0f64; // 按枚数加权的得分总和
        let mut n_sampled = 0usize;

        // 第二遍：对前 MAX_SAMPLES 个有效牌调用 waiting_tiles 精确计算
        for &(eff_t, _avail) in eff_tiles.iter().take(MAX_SAMPLES) {
            let mut hh = *hand;
            add_tile(&mut hh, eff_t);

            // waiting_tiles 内部调用 calc_shanten 27次，但大部分会命中缓存
            let waits = waiting_tiles(&hh, self.num_melds);

            let mut this_outs = 0u32;
            let mut this_weighted_score = 0f64;

            for &wt in &waits {
                // 进牌 eff_t 已被消耗一枚，需要从可用数中扣除
                let wt_avail = if wt == eff_t {
                    state.available_count(wt).saturating_sub(1) as u32
                } else {
                    state.available_count(wt) as u32
                };

                if wt_avail == 0 { continue; }

                this_outs += wt_avail;

                // 用实际番型计算得分：hh 已是 14 张（13+eff_t），直接作为和牌手牌
                // 不再 add_tile(wt)，否则变成 15 张导致 calc_fan 返回 None (Issue #R7-C1)
                let score = self.get_win_score(&hh, wt, false);
                this_weighted_score += score as f64 * wt_avail as f64;
            }

            sample_outs_total += this_outs;
            sample_weighted_score += this_weighted_score;
            n_sampled += 1;
        }

        // 计算平均待牌数和平均得分
        // 如果采样了有效牌但所有待牌都不可用（极端情况），使用启发式估计
        let avg_outs = if n_sampled > 0 && sample_outs_total > 0 {
            sample_outs_total as f32 / n_sampled as f32
        } else {
            // 启发式回退：基于有效牌数量估计
            // 典型一向听手牌的待牌数约为有效牌数的40-60%
            (total_eff as f32 * 0.5).max(2.0)
        };

        let avg_score = if sample_outs_total > 0 {
            // 按枚数加权的平均得分（精确值）
            (sample_weighted_score / sample_outs_total as f64) as f32
        } else {
            // 启发式回退：门清手牌基础分较高
            if self.num_melds == 0 { 2000.0 } else { 1000.0 }
        };

        // 逐巡计算听牌概率、和牌概率、期望值
        let mut cum_not_eff = 1.0f32;
        for turn in 0..turns {
            let p = (total_eff as f32 / (n_left - turn as f32).max(1.0)).min(1.0);
            cum_not_eff *= 1.0 - p;
            cand.tenpai_probs[turn] = 1.0 - cum_not_eff;

            // 听牌后剩余巡数内的和牌概率
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
