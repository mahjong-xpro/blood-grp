use crate::consts::*;
use crate::tile::*;
use crate::hand::*;
use crate::algo::shanten::{calc_shanten, waiting_tiles};
use crate::algo::sp::{SPCalculator, SPInitState};
use crate::algo::agari::calc_gen_count;
use crate::state::board::{BoardState, Phase};

const OBS_SIZE: usize = NUM_STUDENT_CHANNELS * NUM_TILE_TYPES;

/// Encode full student observation: 473 channels x 27 tiles
/// 注意：通道数变更（464→470→473）需要重新训练模型。
pub fn encode_student_obs(board: &BoardState, player_id: usize) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_SIZE];
    let mut ch = 0usize;

    let p = &board.players[player_id];

    // Helper to write a value at (channel, tile)
    macro_rules! w {
        ($channel:expr, $tile:expr, $val:expr) => {
            obs[$channel * NUM_TILE_TYPES + $tile] = $val;
        };
    }

    // Fill entire channel with a scalar value
    macro_rules! fill_ch {
        ($channel:expr, $val:expr) => {
            for t in 0..NUM_TILE_TYPES {
                obs[$channel * NUM_TILE_TYPES + t] = $val;
            }
        };
    }

    // === Section 1: HAND (5 ch) ===
    for t in 0..NUM_TILE_TYPES {
        for k in 0..4usize {
            if p.hand[t] as usize > k {
                w!(ch + k, t, 1.0);
            }
        }
    }
    ch += 4;

    // Last drawn tile
    if let Some(tile) = p.last_drawn_tile {
        w!(ch, tile as usize, 1.0);
    }
    ch += 1;

    // === Section 2: GAME CONTEXT (13 ch) ===
    let total_score = (board.initial_score * NUM_PLAYERS as i32) as f32;
    for pi in 0..NUM_PLAYERS {
        let idx = (player_id + pi) % NUM_PLAYERS;
        fill_ch!(ch + pi, board.players[idx].score as f32 / total_score);
    }
    ch += 4;

    // Ranks (one-hot)
    let mut scores: Vec<(usize, i32)> = (0..NUM_PLAYERS).map(|i| {
        let idx = (player_id + i) % NUM_PLAYERS;
        (i, board.players[idx].score)
    }).collect();
    scores.sort_by(|a, b| b.1.cmp(&a.1));
    for (rank, &(rel_idx, _)) in scores.iter().enumerate() {
        fill_ch!(ch + rel_idx, (rank as f32) / 3.0);
    }
    ch += 4;

    // Is dealer
    let is_dealer = ((board.dealer + NUM_PLAYERS - player_id) % NUM_PLAYERS == 0) as u8 as f32;
    fill_ch!(ch, is_dealer);
    ch += 1;

    // Turn progress (1 ch)
    fill_ch!(ch, (board.turn_count as f32 / MAX_TURNS as f32).min(1.0));
    ch += 1;

    // Score gap to leader: how far behind 1st place (0 if agent is leader)
    let max_score = board.players.iter().map(|p| p.score).max().unwrap_or(0);
    fill_ch!(ch, (max_score - board.players[player_id].score).max(0) as f32 / total_score);
    ch += 1;

    // Score gap to last: how far ahead of last place (0 if agent is last)
    let min_score = board.players.iter().map(|p| p.score).min().unwrap_or(0);
    fill_ch!(ch, (board.players[player_id].score - min_score).max(0) as f32 / total_score);
    ch += 1;

    // Relative score vs mean (signed, normalized)
    let mean_score = board.players.iter().map(|p| p.score).sum::<i32>() as f32 / NUM_PLAYERS as f32;
    fill_ch!(ch, (board.players[player_id].score as f32 - mean_score) / total_score);
    ch += 1;

    // === Section 3: DING QUE (17 ch) ===
    if let Some(suit) = p.ding_que {
        // DingQue已完成：标记选择的花色
        for t in suit.start()..suit.end() {
            w!(ch + suit as usize, t, 1.0);
        }
    } else if board.phase == Phase::DingQue {
        // DingQue阶段：提供花色统计信息，帮助模型做出明智的选择
        // 通道0: Man花色的牌数量（归一化到 [0,1]）
        // 通道1: Pin花色的牌数量（归一化到 [0,1]）
        // 通道2: Sou花色的牌数量（归一化到 [0,1]）
        for suit in Suit::all() {
            let count = (suit.start()..suit.end())
                .filter(|&t| p.hand[t] > 0)
                .map(|t| p.hand[t] as u32)
                .sum::<u32>();
            fill_ch!(ch + suit as usize, count as f32 / 13.0);
        }
    }
    ch += 3;

    // Ding que completed
    fill_ch!(ch, p.ding_que_completed() as u8 as f32);
    ch += 1;

    // Ding que remaining
    fill_ch!(ch, p.ding_que_remaining() as f32 / 13.0);
    ch += 1;

    // Opponent ding que (3 x 3)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        if let Some(suit) = board.players[opp_id].ding_que {
            for t in suit.start()..suit.end() {
                w!(ch + suit as usize, t, 1.0);
            }
        }
        ch += 3;
    }

    // Opponent agari status (3 ch)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        fill_ch!(ch, board.players[opp_id].has_won as u8 as f32);
        ch += 1;
    }

    // === Section 4: GAME STATE (5 ch) ===
    // Max initial wall = TOTAL_TILES - HAND_SIZE*NUM_PLAYERS - 1(dealer extra) = 108 - 52 - 1 = 55
    fill_ch!(ch, board.wall_remaining() as f32 / 55.0);
    ch += 1;

    // Compute shanten and waits once; reuse below
    let shanten = calc_shanten(&p.hand, p.melds.len());
    let waits: Vec<Tile> = if shanten == 0 {
        waiting_tiles(&p.hand, p.melds.len())
    } else {
        Vec::new()
    };

    // Tenpai width: number of distinct wait tiles (normalized).
    // Blood mahjong has no furiten rule; more waits = better tenpai quality.
    if shanten == 0 {
        fill_ch!(ch, waits.len() as f32 / NUM_TILE_TYPES as f32);
    }
    ch += 1;

    // Furiten passed ron fan (过手加番): non-zero = passed on ron, can win if fan increases
    // This encodes the temporary furiten state more precisely than a binary flag
    fill_ch!(ch, p.furiten_passed_ron_fan.unwrap_or(0) as f32 / MAX_FAN as f32);
    ch += 1;

    fill_ch!(ch, p.is_rinshan as u8 as f32);
    ch += 1;

    // Total kans on board
    let total_kans: usize = board.players.iter().map(|p| {
        p.melds.iter().filter(|m| matches!(m, MeldType::MinKan(_) | MeldType::AnKan(_) | MeldType::KaKan(_))).count()
    }).sum();
    fill_ch!(ch, total_kans as f32 / 16.0);
    ch += 1;

    // === Section 5: SELF KAWA (58 ch) ===
    // Window = MAX_TURNS (28): covers worst-case where 2 players win early,
    // leaving 2 players to share the remaining wall (~27 discards each).
    let self_discards = &p.discards;
    let self_tsumogiri = &p.tsumogiri;
    let n = self_discards.len();
    let start_idx = n.saturating_sub(MAX_TURNS);
    for (pos, idx) in (start_idx..n).enumerate() {
        let tile = self_discards[idx];
        w!(ch + pos * 2, tile as usize, 1.0);
        if idx < self_tsumogiri.len() && self_tsumogiri[idx] {
            fill_ch!(ch + pos * 2 + 1, 1.0);
        }
    }
    ch += MAX_TURNS * 2; // 28 * 2 = 56

    // Exponential decay overview (clamped to [0, 1])
    for (idx, &tile) in self_discards.iter().enumerate() {
        let decay = 0.9f32.powi((n - 1 - idx) as i32);
        let prev = obs[ch * NUM_TILE_TYPES + tile as usize];
        w!(ch, tile as usize, (prev + decay).min(1.0));
    }
    ch += 1;
    for (idx, (&_tile, &is_tg)) in self_discards.iter().zip(self_tsumogiri.iter()).enumerate() {
        if is_tg {
            let decay = 0.9f32.powi((n - 1 - idx) as i32);
            let tile = self_discards[idx];
            let prev = obs[ch * NUM_TILE_TYPES + tile as usize];
            w!(ch, tile as usize, (prev + decay).min(1.0));
        }
    }
    ch += 1;

    // === Section 6: OPPONENT KAWA (174 ch) ===
    // Window = MAX_TURNS (28): same reasoning as Section 5.
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let opp_n = opp.discards.len();
        let opp_start = opp_n.saturating_sub(MAX_TURNS);
        for (pos, idx) in (opp_start..opp_n).enumerate() {
            let tile = opp.discards[idx];
            w!(ch + pos * 2, tile as usize, 1.0);
            if idx < opp.tsumogiri.len() && opp.tsumogiri[idx] {
                fill_ch!(ch + pos * 2 + 1, 1.0);
            }
        }
        ch += MAX_TURNS * 2; // 56

        // Exponential decay for this opponent (clamped to [0, 1])
        for (idx, &tile) in opp.discards.iter().enumerate() {
            let decay = 0.9f32.powi((opp_n - 1 - idx) as i32);
            let prev = obs[ch * NUM_TILE_TYPES + tile as usize];
            w!(ch, tile as usize, (prev + decay).min(1.0));
        }
        ch += 1;

        // Tsumogiri decay for this opponent (mirrors self kawa structure)
        for (idx, (&tile, &is_tg)) in opp.discards.iter().zip(opp.tsumogiri.iter()).enumerate() {
            if is_tg {
                let decay = 0.9f32.powi((opp_n - 1 - idx) as i32);
                let prev = obs[ch * NUM_TILE_TYPES + tile as usize];
                w!(ch, tile as usize, (prev + decay).min(1.0));
            }
        }
        ch += 1;
    }

    // === Section 7: VISIBLE TILES (48 ch) ===
    // Kawa overview per opponent (3 x 4 one-hot); self kawa fully covered by Section 5
    for pi in 1..NUM_PLAYERS {
        let idx = (player_id + pi) % NUM_PLAYERS;
        let mut counts = [0u8; NUM_TILE_TYPES];
        for &tile in &board.players[idx].discards {
            counts[tile as usize] += 1;
        }
        for t in 0..NUM_TILE_TYPES {
            for k in 0..4usize {
                if counts[t] as usize > k {
                    w!(ch + k, t, 1.0);
                }
            }
        }
        ch += 4;
    }

    // Fuuro overview (4 players x 2ch x 4 melds = 32 ch)
    for pi in 0..NUM_PLAYERS {
        let idx = (player_id + pi) % NUM_PLAYERS;
        for (mi, meld) in board.players[idx].melds.iter().enumerate() {
            let t = meld.tile() as usize;
            w!(ch + mi * 2, t, 1.0);
            let meld_type_val = match meld {
                MeldType::Pon(_) => 0.25,
                MeldType::MinKan(_) => 0.5,
                MeldType::AnKan(_) => 0.75,
                MeldType::KaKan(_) => 1.0,
            };
            fill_ch!(ch + mi * 2 + 1, meld_type_val);
        }
        ch += 8; // 4 melds * 2 ch
    }

    // AnKan overview (1 ch, all players combined — kans are rare, 4ch was wasteful)
    for pi in 0..NUM_PLAYERS {
        let idx = (player_id + pi) % NUM_PLAYERS;
        for meld in &board.players[idx].melds {
            if let MeldType::AnKan(t) = meld {
                w!(ch, *t as usize, 1.0);
            }
        }
    }
    ch += 1;

    // Dahai count (total discards made; 0 = first discard = tianhu/dihu window)
    fill_ch!(ch, (board.dahai_count as f32 / (MAX_TURNS * NUM_PLAYERS) as f32).min(1.0));
    ch += 1;

    // Current player relative position (0 = self, 1-3 = others in turn order)
    let rel_current = (board.current_player + NUM_PLAYERS - player_id) % NUM_PLAYERS;
    fill_ch!(ch, rel_current as f32 / (NUM_PLAYERS - 1) as f32);
    ch += 1;

    // Game phase scalar (DingQue=0 .. Done=6, normalized)
    let phase_val = match board.phase {
        Phase::DingQue  => 0u8,
        Phase::SelfCheck => 1,
        Phase::KanSelect => 2,
        Phase::Discard   => 3,
        Phase::Reaction  => 4,
        Phase::Scoring   => 5,
        Phase::Done      => 6,
    };
    fill_ch!(ch, phase_val as f32 / 6.0);
    ch += 1;

    // === Section 8: DEFENSE (9 ch) ===
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let total_discards = opp.discards.len().max(1) as f32;
        // Single-pass suit count (avoids 3× O(n) filter per opponent)
        let mut suit_counts = [0u32; NUM_SUITS];
        for &t in &opp.discards {
            suit_counts[t as usize / TILES_PER_SUIT] += 1;
        }
        for suit in Suit::all() {
            fill_ch!(ch, suit_counts[suit as usize] as f32 / total_discards);
            ch += 1;
        }
    }

    // === Section 9: DERIVED FEATURES (11 ch) ===
    // Wall remaining per tile
    for t in 0..NUM_TILE_TYPES {
        let remaining = (COPIES_PER_TILE as u8).saturating_sub(p.tiles_seen[t]);
        w!(ch, t, remaining as f32 / COPIES_PER_TILE as f32);
    }
    ch += 1;

    fill_ch!(ch, p.is_menzen() as u8 as f32);
    ch += 1;

    fill_ch!(ch, p.melds.len() as f32 / MAX_MELDS as f32);
    ch += 1;

    // Win count: how many players have already won (endgame pressure signal)
    fill_ch!(ch, board.win_count as f32 / (NUM_PLAYERS - 1) as f32);
    ch += 1;

    // Opponent fuuro counts
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        fill_ch!(ch, board.players[opp_id].melds.len() as f32 / MAX_MELDS as f32);
        ch += 1;
    }

    // Opponent terminal discard ratio (3 ch): fraction of terminals in each opponent's discards;
    // high ratio → opponent likely building tanyao (all-simples) hand
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let total = opp.discards.len().max(1) as f32;
        let terminal_count = opp.discards.iter()
            .filter(|&&t| { let r = Suit::rank(t); r == 1 || r == 9 })
            .count();
        fill_ch!(ch, terminal_count as f32 / total);
        ch += 1;
    }

    // Self discard count (normalized): explicit scalar complementing Section 5's positional window
    fill_ch!(ch, p.discards.len() as f32 / (MAX_TURNS as f32));
    ch += 1;

    // === Section 10: HAND ANALYSIS (6 ch) ===
    // Reuse cached waits
    if shanten == 0 {
        for &wt in &waits {
            w!(ch, wt as usize, 1.0);
        }
    }
    ch += 1;

    // Shanten one-hot (0-4)
    if shanten >= 0 {
        let sh = (shanten as usize).min(4);
        fill_ch!(ch + sh, 1.0);
    }
    ch += 5;

    // At kan select — now redundant with phase scalar; replaced by opponent rinshan below
    // (removed 1 ch, reallocated)

    // === Section 11: ACTION CONTEXT (12 ch) ===
    if let Some((_, tile)) = board.last_discard {
        if board.phase == Phase::Reaction {
            w!(ch, tile as usize, 1.0);
        }
    }
    ch += 1;

    // Discard candidates
    if board.phase == Phase::Discard && board.current_player == player_id {
        let candidates = p.discard_candidates();
        for &t in &candidates {
            w!(ch, t as usize, 1.0);
        }
    }
    ch += 1;

    // Shanten-based discard classification (2 ch):
    //   ch+0: tiles that improve shanten (progress discards) or reach tenpai
    //   ch+1: tiles that reach tenpai after discard
    // Only populated during Discard phase; zeros otherwise.
    if board.phase == Phase::Discard && board.current_player == player_id {
        for t in 0..NUM_TILE_TYPES as u8 {
            if p.hand[t as usize] == 0 { continue; }
            let mut h = p.hand;
            h[t as usize] -= 1;
            let ns = calc_shanten(&h, p.melds.len());
            if ns < shanten  { w!(ch,     t as usize, 1.0); }
            if ns == 0       { w!(ch + 1, t as usize, 1.0); }
        }
    }
    ch += 2;

    // Opponent rinshan state (2 ch): any-opponent-rinshan + rinshan count
    let opp_rinshan_count = (1..NUM_PLAYERS)
        .filter(|&off| board.players[(player_id + off) % NUM_PLAYERS].is_rinshan)
        .count();
    fill_ch!(ch, (opp_rinshan_count > 0) as u8 as f32);
    ch += 1;
    fill_ch!(ch, opp_rinshan_count as f32 / (NUM_PLAYERS - 1) as f32);
    ch += 1;

    // Can pon/kan/agari
    if let Some(ac) = board.get_decision_request(player_id) {
        if ac.can_pon { fill_ch!(ch, 1.0); }
        ch += 1;
        if ac.can_kan { fill_ch!(ch, 1.0); }
        ch += 1;
        // AnKan candidates
        for &t in &p.can_ankan_tiles() {
            w!(ch, t as usize, 1.0);
        }
        ch += 1;
        // KaKan candidates
        for &t in &p.can_kakan_tiles() {
            w!(ch, t as usize, 1.0);
        }
        ch += 1;
        if ac.can_agari { fill_ch!(ch, 1.0); }
        ch += 1;
    } else {
        ch += 5;
    }

    // Current ron fan
    if board.phase == Phase::Reaction {
        if let Some((_, tile)) = board.last_discard {
            let ctx = board.make_win_context_for_obs(player_id, tile);
            if let Some(result) = crate::algo::agari::calc_fan(&ctx) {
                fill_ch!(ch, result.fan as f32 / MAX_FAN as f32);
            }
        }
    }
    ch += 1;

    // === Section 12: SP TABLE (99 ch) ===
    {
        let sp_init = SPInitState {
            tehai: p.hand,
            tiles_seen: p.tiles_seen,
            tiles_left: board.wall_remaining() as u8,
            num_melds: p.melds.len(),
            ding_que: p.ding_que,
        };
        let mut sp_calc = SPCalculator::new(p.melds.len(), p.ding_que);
        sp_calc.melds = p.melds.clone();
        sp_calc.fan_config = board.fan_config;
        sp_calc.is_at_rinshan = p.is_rinshan;
        sp_calc.n_active_payers = board.active_player_count().saturating_sub(1) as u8;

        let candidates = sp_calc.calc(&sp_init);

        // ch+0: Max EV per discard tile (per-tile, normalized by REWARD_NORM=32000)
        // ch+1: Max win prob per discard tile (per-tile)
        // ch+2: Shanten-maintaining/improving discard indicator (per-tile)
        // ch+3: Best discard marker (per-tile)
        for c in &candidates {
            let ti = c.tile as usize;
            w!(ch, ti, c.total_ev() / 32000.0);
            w!(ch + 1, ti, c.total_win_prob().min(1.0));
            if c.shanten_diff <= 0 {
                w!(ch + 2, ti, 1.0);
            }
        }
        if let Some(best) = candidates.first() {
            w!(ch + 3, best.tile as usize, 1.0);
        }
        ch += 4;

        // ch+4..ch+31: Best candidate tenpai probs per turn (28 ch, fill_ch!)
        // ch+32..ch+59: Best candidate win probs per turn (28 ch, fill_ch!)
        // ch+60..ch+87: Best candidate EV per turn (28 ch, fill_ch!, normalized)
        if let Some(best) = candidates.first() {
            for t in 0..MAX_TURNS {
                fill_ch!(ch + t, best.tenpai_probs[t]);
                fill_ch!(ch + MAX_TURNS + t, best.win_probs[t]);
                fill_ch!(ch + 2 * MAX_TURNS + t, best.exp_values[t] / 32000.0);
            }
        }
        ch += 3 * MAX_TURNS; // 84

        // ch+88..ch+98: Summary features (11 ch)
        if let Some(best) = candidates.first() {
            fill_ch!(ch, best.total_ev() / 32000.0);
            ch += 1;
            fill_ch!(ch, best.total_win_prob().min(1.0));
            ch += 1;
        } else {
            ch += 2;
        }
        fill_ch!(ch, shanten.max(0) as f32 / 4.0);
        ch += 1;
        fill_ch!(ch, candidates.len() as f32 / NUM_TILE_TYPES as f32);
        ch += 1;
        // EV spread: best - worst (how decisive the discard choice is)
        let best_ev = candidates.first().map_or(0.0, |c| c.total_ev());
        let worst_ev = candidates.last().map_or(0.0, |c| c.total_ev());
        fill_ch!(ch, (best_ev - worst_ev).max(0.0) / 32000.0);
        ch += 1;
        // Win prob spread: best - worst
        let best_wp = candidates.first().map_or(0.0, |c| c.total_win_prob());
        let worst_wp = candidates.last().map_or(0.0, |c| c.total_win_prob());
        fill_ch!(ch, (best_wp - worst_wp).max(0.0).min(1.0));
        ch += 1;
        // Shanten-improving candidate count
        let improve_count = candidates.iter().filter(|c| c.shanten_diff < 0).count();
        fill_ch!(ch, improve_count as f32 / NUM_TILE_TYPES as f32);
        ch += 1;
        // Second-best candidate EV (for relative comparison)
        if let Some(second) = candidates.get(1) {
            fill_ch!(ch, second.total_ev() / 32000.0);
        }
        ch += 1;
        // Best candidate peak win prob (max over turns, vs cumulative)
        if let Some(best) = candidates.first() {
            fill_ch!(ch, best.max_win_prob().min(1.0));
        }
        ch += 1;
        // Gen count (四归一): tiles appearing 4 times across hand + melds
        let gen = calc_gen_count(&p.hand, &p.melds, None);
        fill_ch!(ch, gen as f32 / 4.0);
        ch += 1;
        // Last discard was from a kan draw (杠上炮 context)
        fill_ch!(ch, board.last_discard_is_kan as u8 as f32);
        ch += 1;
    }

    // === Section 13: FAN CONFIG (7 ch) ===
    let fc = &board.fan_config;
    fill_ch!(ch, fc.menqing as u8 as f32); ch += 1;
    fill_ch!(ch, fc.duanyaojiu as u8 as f32); ch += 1;
    fill_ch!(ch, fc.daiyaojiu as u8 as f32); ch += 1;
    fill_ch!(ch, fc.yitiaolong as u8 as f32); ch += 1;
    fill_ch!(ch, fc.jiaxinwu as u8 as f32); ch += 1;
    fill_ch!(ch, fc.haidi as u8 as f32); ch += 1;
    fill_ch!(ch, fc.tianhu_dihu as u8 as f32); ch += 1;

    // === Section 14: 对手手牌信息 (6 ch) ===
    // 注意：此 Section 为新增通道（464→470），需要重新训练模型。
    //
    // 通道 ch+0..ch+2: 3个对手的手牌数量（归一化，hand_count / 13.0）
    // 通道 ch+3..ch+5: 3个对手最近一次副露的来源玩家相对位置（归一化，source / 3.0）
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        // 对手手牌数量：归一化到 [0, 1]，除以 13（初始手牌数）
        fill_ch!(ch, opp.hand_count() as f32 / 13.0);
        ch += 1;
    }
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        // 副露来源编码：取最近一次副露的来源玩家相对位置
        // 相对位置：1=下家, 2=对家, 3=上家（相对于观测玩家 player_id）
        // 归一化为 source / 3.0；无副露则为 0
        let meld_source_val = opp.meld_from.iter().rev()
            .find_map(|&from| from)
            .map(|abs_from| {
                // Fix R12-M1: use (rel + 1) / 4.0 so rel=0 (from observer) encodes as 0.25,
                // distinguishable from no-meld (0.0). Old encoding: rel=0 → 0.0 = no-meld.
                let rel = (abs_from + NUM_PLAYERS - player_id) % NUM_PLAYERS;
                (rel as f32 + 1.0) / 4.0
            })
            .unwrap_or(0.0);
        fill_ch!(ch, meld_source_val);
        ch += 1;
    }

    // === Section 15: GENBUTSU / 現物 (3 ch) ===
    // 每个对手一个通道：该对手弃过的牌标记为 1.0（100% 安全牌）。
    // 血战无振听规则，对手弃过的牌永远不会被该对手荣和。
    // 显式编码避免网络从 174ch 牌河中自行学习此规则。
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        for &tile in &board.players[opp_id].discards {
            w!(ch, tile as usize, 1.0);
        }
        ch += 1;
    }

    assert_eq!(ch, NUM_STUDENT_CHANNELS, "used {} channels, expected {}", ch, NUM_STUDENT_CHANNELS);

    obs
}
