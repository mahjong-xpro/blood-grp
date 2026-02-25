use rustc_hash::FxHashMap;

use crate::consts::*;
use crate::algo::shanten::{calc_shanten, waiting_tiles};
use crate::algo::agari::calc_fan;
use crate::algo::sp::{SPCalculator, SPInitState};
use crate::state::board::BoardState;
use super::student::encode_student_obs;

/// Cache key for opponent SP results.
/// Keyed on hand state + meld count + tiles_left.
/// Intentionally omits `tiles_seen` — it changes on every discard but has
/// minimal impact on SP results (±1 tile visibility). `tiles_left` captures
/// the most impactful change. This is a deliberate trade-off: slightly stale
/// SP results in exchange for ~50% reduction in SP computation.
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct OppSpCacheKey {
    pub tehai: [u8; NUM_TILE_TYPES],
    pub num_melds: usize,
    pub tiles_left: usize,
}

/// Cached SP result for one opponent.
#[derive(Clone)]
pub struct OppSpCacheEntry {
    pub best_ev: f32,
    pub best_win_prob: f32,
    pub best_peak_win_prob: f32,
}

/// Type alias for the oracle SP cache.
/// Keyed on `(opponent_id, OppSpCacheKey)`.
pub type OracleSpCache = FxHashMap<(usize, OppSpCacheKey), OppSpCacheEntry>;

/// 编码 Oracle 观测：学生观测 + 额外的完美信息通道
///
/// Oracle 额外通道布局（52 通道）：
///   - 对手真实手牌（3×4 = 12 ch）
///   - 对手 SP Table 摘要（3×3 = 9 ch）：最佳弃牌 EV / 听牌概率 / 胜率
///   - 对手真实向听数（3×5 = 15 ch）
///   - 对手听牌（3×1 = 3 ch）
///   - 牌山剩余牌（4 ch）
///   - 对手危险度评分（3 ch）
///   - 对手最佳番数（3 ch）
///   - 对手最后摸牌（3 ch）
pub fn encode_oracle_obs(
    board: &BoardState,
    player_id: usize,
    mut sp_cache: Option<&mut OracleSpCache>,
) -> Vec<f32> {
    let mut obs = encode_student_obs(board, player_id);
    obs.resize(NUM_ORACLE_CHANNELS * NUM_TILE_TYPES, 0.0);

    let base = NUM_STUDENT_CHANNELS;
    let mut ch = base;

    macro_rules! w {
        ($channel:expr, $tile:expr, $val:expr) => {
            obs[$channel * NUM_TILE_TYPES + $tile] = $val;
        };
    }
    macro_rules! fill_ch {
        ($channel:expr, $val:expr) => {
            for t in 0..NUM_TILE_TYPES {
                obs[$channel * NUM_TILE_TYPES + t] = $val;
            }
        };
    }

    // ── 对手真实手牌（3 × 4 one-hot = 12 ch）──
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        for t in 0..NUM_TILE_TYPES {
            for k in 0..4usize {
                if opp.hand[t] as usize > k {
                    w!(ch + k, t, 1.0);
                }
            }
        }
        ch += 4;
    }

    // ── 对手 SP Table 摘要（3 × 3 = 9 ch）──
    // 替换原冗余的"对手定缺"通道（已在学生观测 Section 3 中编码）。
    // 每个对手 3 通道：最佳弃牌 EV / 最佳弃牌听牌概率 / 最佳弃牌胜率
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];

        let cache_key = OppSpCacheKey {
            tehai: opp.hand,
            num_melds: opp.melds.len(),
            tiles_left: board.wall_remaining(),
        };

        // Check cache first
        let cached = sp_cache.as_ref().and_then(|c| c.get(&(opp_id, cache_key.clone())).cloned());

        if let Some(entry) = cached {
            fill_ch!(ch, entry.best_ev);
            fill_ch!(ch + 1, entry.best_win_prob);
            fill_ch!(ch + 2, entry.best_peak_win_prob);
        } else {
            // 为对手构建 SP 计算上下文
            let sp_init = SPInitState {
                tehai: opp.hand,
                tiles_seen: opp.tiles_seen,
                tiles_left: board.wall_remaining() as u8,
                num_melds: opp.melds.len(),
                ding_que: opp.ding_que,
            };
            let mut sp_calc = SPCalculator::new(opp.melds.len(), opp.ding_que);
            sp_calc.melds = opp.melds.clone();
            sp_calc.fan_config = board.fan_config;
            sp_calc.is_at_rinshan = opp.is_rinshan;
            sp_calc.n_active_payers = board.active_player_count().saturating_sub(1) as u8;

            let candidates = sp_calc.calc(&sp_init);

            let (best_ev, best_win_prob, best_peak_win_prob) = if let Some(best) = candidates.first() {
                (
                    best.total_ev() / 32000.0,
                    best.total_win_prob().min(1.0),
                    best.max_win_prob().min(1.0),
                )
            } else {
                (0.0, 0.0, 0.0)
            };

            fill_ch!(ch, best_ev);
            fill_ch!(ch + 1, best_win_prob);
            fill_ch!(ch + 2, best_peak_win_prob);

            // Store in cache
            if let Some(ref mut cache) = sp_cache {
                cache.insert((opp_id, cache_key), OppSpCacheEntry {
                    best_ev,
                    best_win_prob,
                    best_peak_win_prob,
                });
            }
        }
        ch += 3;
    }

    // ── 预计算对手向听数（复用于向听 one-hot 和听牌通道）──
    let mut opp_shantens = [0i8; NUM_PLAYERS - 1];
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        opp_shantens[opp_off - 1] = calc_shanten(&opp.hand, opp.melds.len());
    }

    // ── 对手真实向听数（3 × 5 = 15 ch）──
    for opp_off in 1..NUM_PLAYERS {
        let sh = opp_shantens[opp_off - 1].max(0).min(4) as usize;
        fill_ch!(ch + sh, 1.0);
        ch += 5;
    }

    // ── 对手听牌（3 × 1 = 3 ch）──
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        if opp_shantens[opp_off - 1] == 0 {
            let waits = waiting_tiles(&opp.hand, opp.melds.len());
            for &wt in &waits {
                w!(ch, wt as usize, 1.0);
            }
        }
        ch += 1;
    }

    // ── 牌山剩余牌计数（4 ch）──
    let mut wall_counts = [0u8; NUM_TILE_TYPES];
    for i in board.wall_idx..board.wall_back_idx {
        wall_counts[board.wall[i] as usize] += 1;
    }
    for t in 0..NUM_TILE_TYPES {
        for k in 0..4usize {
            if wall_counts[t] as usize > k {
                w!(ch + k, t, 1.0);
            }
        }
    }
    ch += 4;

    // ── 对手危险度评分（3 ch）──
    // 替换原冗余的"对手定缺完成"通道（可从学生观测推断）。
    // 综合考虑：向听数、副露数、定缺完成度、打牌回合数
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let sh = opp_shantens[opp_off - 1];

        // 基础危险度：向听数越低越危险（0=听牌→1.0, 1→0.7, 2→0.4, 3+→0.2）
        let shanten_danger = match sh {
            i if i <= 0 => 1.0f32,
            1 => 0.7,
            2 => 0.4,
            _ => 0.2,
        };

        // 副露加成：副露越多手牌越少，进攻意图越明显
        let meld_bonus = opp.melds.len() as f32 * 0.1;

        // 定缺完成加成：已完成定缺说明手牌更纯，更接近和牌
        let dq_bonus = if opp.ding_que_completed() { 0.15 } else { 0.0 };

        // 打牌回合加成：打出的牌越多，手牌越精炼
        let discard_progress = (opp.discards.len() as f32 / MAX_TURNS as f32).min(1.0) * 0.1;

        let danger = (shanten_danger + meld_bonus + dq_bonus + discard_progress).min(1.0);
        fill_ch!(ch, danger);
        ch += 1;
    }

    // ── 对手最佳番数估计（3 ch）：听牌时归一化番数，否则为 0 ──
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        if opp_shantens[opp_off - 1] == 0 {
            let waits = waiting_tiles(&board.players[opp_id].hand, board.players[opp_id].melds.len());
            let best_fan = waits.iter().filter_map(|&wt| {
                let ctx = board.make_win_context_for_obs(opp_id, wt);
                calc_fan(&ctx).map(|r| r.fan)
            }).max().unwrap_or(0);
            fill_ch!(ch, best_fan as f32 / MAX_FAN as f32);
        }
        ch += 1;
    }

    // ── 对手最后摸牌（3 ch）：对学生隐藏，Oracle 可见 ──
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        if let Some(tile) = board.players[opp_id].last_drawn_tile {
            w!(ch, tile as usize, 1.0);
        }
        ch += 1;
    }

    assert_eq!(ch, NUM_ORACLE_CHANNELS, "oracle channel count mismatch: got {}, expected {}", ch, NUM_ORACLE_CHANNELS);

    obs
}
