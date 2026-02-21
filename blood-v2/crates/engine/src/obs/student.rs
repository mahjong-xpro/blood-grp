use crate::consts::*;
use crate::tile::*;
use crate::hand::*;
use crate::algo::shanten::{calc_shanten, waiting_tiles};
use crate::algo::sp::{SPCalculator, SPInitState};
use crate::state::board::{BoardState, Phase};

const OBS_SIZE: usize = NUM_STUDENT_CHANNELS * NUM_TILE_TYPES;

/// Encode full student observation: ~438 channels x 27 tiles
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

    // === Section 2: GAME CONTEXT (14 ch) ===
    let total_score = (INITIAL_SCORE * NUM_PLAYERS as i32) as f32;
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

    // Active player count
    fill_ch!(ch, board.active_player_count() as f32 / NUM_PLAYERS as f32);
    ch += 1;

    // Padding for 14 ch total
    ch += 4; // remaining context channels (placeholder)

    // === Section 3: DING QUE (17 ch) ===
    if let Some(suit) = p.ding_que {
        for t in suit.start()..suit.end() {
            w!(ch + suit as usize, t, 1.0);
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

    // Forbidden tiles (furiten map)
    if shanten == 0 {
        if p.temporary_furiten || p.is_permanent_furiten(&waits) {
            for &wt in &waits {
                w!(ch, wt as usize, 1.0);
            }
        }
    }
    ch += 1;

    fill_ch!(ch, p.temporary_furiten as u8 as f32);
    ch += 1;

    fill_ch!(ch, p.is_rinshan as u8 as f32);
    ch += 1;

    // Total kans on board
    let total_kans: usize = board.players.iter().map(|p| {
        p.melds.iter().filter(|m| matches!(m, MeldType::MinKan(_) | MeldType::AnKan(_) | MeldType::KaKan(_))).count()
    }).sum();
    fill_ch!(ch, total_kans as f32 / 16.0);
    ch += 1;

    // === Section 5: SELF KAWA (38 ch) ===
    let self_discards = &p.discards;
    let self_tsumogiri = &p.tsumogiri;
    let n = self_discards.len();
    let start_idx = n.saturating_sub(18);
    for (pos, idx) in (start_idx..n).enumerate() {
        let tile = self_discards[idx];
        w!(ch + pos * 2, tile as usize, 1.0);
        if idx < self_tsumogiri.len() && self_tsumogiri[idx] {
            fill_ch!(ch + pos * 2 + 1, 1.0);
        }
    }
    ch += 36; // 18 * 2

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

    // === Section 6: OPPONENT KAWA (111 ch) ===
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let opp_n = opp.discards.len();
        let opp_start = opp_n.saturating_sub(18);
        for (pos, idx) in (opp_start..opp_n).enumerate() {
            let tile = opp.discards[idx];
            w!(ch + pos * 2, tile as usize, 1.0);
            if idx < opp.tsumogiri.len() && opp.tsumogiri[idx] {
                fill_ch!(ch + pos * 2 + 1, 1.0);
            }
        }
        ch += 36;

        // Exponential decay for this opponent (clamped to [0, 1])
        for (idx, &tile) in opp.discards.iter().enumerate() {
            let decay = 0.9f32.powi((opp_n - 1 - idx) as i32);
            let prev = obs[ch * NUM_TILE_TYPES + tile as usize];
            w!(ch, tile as usize, (prev + decay).min(1.0));
        }
        ch += 1;
    }

    // === Section 7: VISIBLE TILES (53 ch) ===
    // Kawa overview per player (4 x 4 one-hot)
    for pi in 0..NUM_PLAYERS {
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

    // AnKan overview (4 x 1)
    for pi in 0..NUM_PLAYERS {
        let idx = (player_id + pi) % NUM_PLAYERS;
        for meld in &board.players[idx].melds {
            if let MeldType::AnKan(t) = meld {
                w!(ch, *t as usize, 1.0);
            }
        }
        ch += 1;
    }

    // Tiles seen ratio
    for t in 0..NUM_TILE_TYPES {
        w!(ch, t, p.tiles_seen[t] as f32 / COPIES_PER_TILE as f32);
    }
    ch += 1;

    // === Section 8: DEFENSE (9 ch) ===
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let total_discards = opp.discards.len().max(1) as f32;
        for suit in Suit::all() {
            let suit_count = opp.discards.iter().filter(|&&t| Suit::from_tile(t) == suit).count();
            fill_ch!(ch, suit_count as f32 / total_discards);
            ch += 1;
        }
    }

    // === Section 9: DERIVED FEATURES (8 ch) ===
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

    fill_ch!(ch, (board.turn_count as f32 / MAX_TURNS as f32).min(1.0));
    ch += 1;

    // Acceptance count at tenpai (reuse cached waits)
    if shanten == 0 && !waits.is_empty() {
        let acceptance: u32 = waits.iter().map(|&wt| {
            (COPIES_PER_TILE as u8).saturating_sub(p.tiles_seen[wt as usize]) as u32
        }).sum();
        fill_ch!(ch, acceptance as f32 / 20.0);
    }
    ch += 1;

    // Opponent fuuro counts
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        fill_ch!(ch, board.players[opp_id].melds.len() as f32 / MAX_MELDS as f32);
        ch += 1;
    }

    // === Section 10: HAND ANALYSIS (7 ch) ===
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

    // At kan select
    if board.phase == Phase::KanSelect && board.current_player == player_id {
        fill_ch!(ch, 1.0);
    }
    ch += 1;

    // === Section 11: ACTION CONTEXT (11 ch) ===
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

    // keep_shanten / improve / tenpai candidates (simplified)
    ch += 3;

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

    // === Section 12: SP TABLE (100 ch) ===
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
        sp_calc.fan_config = board.fan_config.clone();
        sp_calc.is_at_rinshan = p.is_rinshan;
        sp_calc.n_active_payers = board.active_player_count().saturating_sub(1) as u8;

        let candidates = sp_calc.calc(&sp_init);

        // ch+0: Max EV per discard tile (per-tile, normalized by 16000)
        // ch+1: Max win prob per discard tile (per-tile)
        // ch+2: Shanten-maintaining/improving discard indicator (per-tile)
        // ch+3: Best discard marker (per-tile)
        for c in &candidates {
            let ti = c.tile as usize;
            w!(ch, ti, c.total_ev() / 48000.0);
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
                fill_ch!(ch + 2 * MAX_TURNS + t, best.exp_values[t] / 48000.0);
            }
        }
        ch += 3 * MAX_TURNS; // 84

        // ch+88..ch+99: Summary features (12 ch)
        if let Some(best) = candidates.first() {
            fill_ch!(ch, best.total_ev() / 48000.0);
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
        ch += 7; // reserved
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

    assert_eq!(ch, NUM_STUDENT_CHANNELS, "used {} channels, expected {}", ch, NUM_STUDENT_CHANNELS);

    obs
}
