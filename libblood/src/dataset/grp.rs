use crate::algo::shanten;
use crate::mjai::{Event, Suit};
use crate::rankings::Rankings;
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use std::array;

use anyhow::{Context, Result};
use pyo3::prelude::*;
use serde_json as json;

/// Replaces original Grp struct
#[pyclass]
#[derive(Clone, Default, Debug)]
pub struct GameScore {
    /// Scores at the START of each kyoku.
    pub scores_history: Vec<[i32; 4]>,
    pub final_scores: [i32; 4],
    pub rank_by_player: [u8; 4],
    
    /// Ding Que selection quality for each player per kyoku.
    /// Value: +1.0 = best choice, 0.0 = middle, -1.0 = worst choice
    /// If no DingQue event found for a kyoku, value is 0.0
    pub ding_que_quality: Vec<[f32; 4]>,

    /// Heuristic best suit index per kyoku per player for Ding Que CE auxiliary.
    /// 0 = Man, 1 = Pin, 2 = Sou. Only valid for kyokus where DingQue occurred.
    pub ding_que_best_suit: Vec<[u8; 4]>,
    
    /// Agari (win) count per player per kyoku (for action rewards)
    pub agari_count: Vec<[u8; 4]>,
    
    /// Houjuu (deal-in) count per player per kyoku (for action rewards)
    pub houjuu_count: Vec<[u8; 4]>,
}

/// Count how many complete 顺子 (sequences) can be formed in a suit's 9 tile counts.
/// Greedy: repeatedly extract one 顺子 (three consecutive indices with count >= 1).
pub(crate) fn count_sequences_in_suit(counts: &[u8; 9]) -> u8 {
    let mut c = *counts;
    let mut num: u8 = 0;
    loop {
        let mut found = false;
        for i in 0..7 {
            if c[i] >= 1 && c[i + 1] >= 1 && c[i + 2] >= 1 {
                c[i] -= 1;
                c[i + 1] -= 1;
                c[i + 2] -= 1;
                num += 1;
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
    }
    num
}

/// Count how many tile kinds improve the hand when added (reduce shanten).
/// Used as a bonus: more improvement kinds = better shape after removing the suit.
///
/// `voided_suit` — 被定缺的花色范围 `(start, end)`，该花色牌全部为零，
/// 添加一张不可能降低向听（无法与其他花色组搭），跳过以节省 ~33% 向听计算。
fn count_improvement_kinds(
    tehai_without: &[u8; 27],
    remaining_count: u8,
    shanten: i8,
    voided_suit: (usize, usize),
) -> u8 {
    // FIX: 使用与 base shanten 相同的 len_div3（= remaining_count / 3）。
    let new_len_div3 = remaining_count / 3;
    let mut count: u8 = 0;
    for tid in 0..27 {
        // 跳过被定缺的花色：该花色牌全部为零，添加孤立一张不可能降低向听。
        if tid >= voided_suit.0 && tid < voided_suit.1 {
            continue;
        }
        if tehai_without[tid] >= 4 {
            continue; // cannot add one more of this kind
        }
        let mut t = *tehai_without;
        t[tid] += 1;
        if shanten::calc_all(&t, new_len_div3, None) < shanten {
            count += 1;
        }
    }
    count
}

/// Calculate the cost of choosing a suit as Ding Que.
/// Lower cost = better choice.
///
/// Factors:
/// 1. Shanten after removing the suit (base)
/// 2. Triplet penalty (刻子: each triplet removed is a significant loss)
/// 3. Sequence penalty (顺子: each complete 顺子 removed is a loss)
/// 4. Pair penalty (对子: each pair removed is a smaller loss; 刻子不重复计对子)
/// 5. 搭子 penalty (partial sequence: 两张相邻/间隔的牌，只差一张成顺)
/// 6. ToiToi potential bonus (remaining hand has sufficient triplet potential)
/// 7. Improvement-kinds bonus (进张种类数: more tile kinds that reduce shanten = better shape)
/// 8. Quantity penalty / bonus (数量惩罚: 张数越多越不想定缺)
pub(crate) fn calc_ding_que_cost(tehai: &[u8; 27], suit: Suit) -> f32 {
    let (start, end) = crate::ding_que::suit_range(suit);
    let suit_range = start..end;

    // Count tiles to be removed, 刻子 (triplets), and 对子 (pairs, count==2 only; 刻子 already counted)
    let mut removed_count: u8 = 0;
    let mut removed_triplets: u8 = 0;
    let mut removed_pairs: u8 = 0;
    let mut suit_counts = [0u8; 9];
    for (j, i) in suit_range.clone().enumerate() {
        let count = tehai[i];
        removed_count += count;
        suit_counts[j] = count;
        if count >= 3 {
            removed_triplets += 1;
        } else if count == 2 {
            removed_pairs += 1;
        }
    }

    // If no tiles in this suit, it's the perfect choice
    if removed_count == 0 {
        return -10.0; // Very low cost (bonus)
    }

    // 顺子 (sequence) count in this suit: how many complete 123/234/... we're removing
    let removed_sequences = count_sequences_in_suit(&suit_counts);

    // 搭子 (partial sequence) count: 相邻(gap=1) 和 间隔(gap=2) 的配对
    // 只在 count >= 1 的位置计数，避免与刻子/顺子过度重叠
    let mut taatsu_count: u8 = 0;
    for i in 0..8 {
        if suit_counts[i] >= 1 && suit_counts[i + 1] >= 1 {
            taatsu_count += 1; // 相邻搭子 (e.g., 12, 23, ...)
        }
    }
    for i in 0..7 {
        if suit_counts[i] >= 1 && suit_counts[i + 2] >= 1 {
            taatsu_count += 1; // 间隔搭子 (e.g., 13, 24, ...)
        }
    }
    // 减去已经成顺的搭子（每个顺子贡献 3 个搭子：2 个相邻 + 1 个间隔，避免重复计算）
    // 例如 123 → 相邻(12,23) + 间隔(13) = 3 个搭子
    taatsu_count = taatsu_count.saturating_sub(removed_sequences * 3);

    // Create tehai without the suit
    let mut tehai_without = *tehai;
    for i in suit_range {
        tehai_without[i] = 0;
    }

    // Calculate new len_div3 (number of complete groups possible)
    let remaining_count: u8 = tehai_without.iter().sum();
    let new_len_div3 = remaining_count / 3;

    // Calculate shanten after removal (pass None for ding_que since we're evaluating the choice itself)
    let shanten = shanten::calc_all(&tehai_without, new_len_div3, None);

    // 刻子 penalty: each 刻子 removed is a significant loss (we lose one complete group)
    let triplet_penalty = removed_triplets as f32 * 0.8;

    // 顺子 penalty: each complete 顺子 removed is also a loss (we lose one complete group)
    let sequence_penalty = removed_sequences as f32 * 0.7;

    // 对子 penalty: each 对子 (pair, count==2) removed is a smaller loss (将牌/进刻潜力); 刻子不重复计
    let pair_penalty = removed_pairs as f32 * 0.35;

    // 搭子 penalty: partial sequences have half-group potential (one draw from completion)
    let taatsu_penalty = taatsu_count as f32 * 0.2;

    // ToiToi potential: count actual triplets and pairs in remaining hand
    // FIX: 之前 pair_triplet_count >= 4 时给 bonus，但 4 个对子 + 0 个刻子
    // 没有碰碰胡潜力。现在要求至少 2 个刻子才给 bonus。
    let mut remaining_triplets: u8 = 0;
    let mut remaining_pairs_or_more: u8 = 0;
    for &count in tehai_without.iter() {
        if count >= 3 {
            remaining_triplets += 1;
            remaining_pairs_or_more += 1;
        } else if count == 2 {
            remaining_pairs_or_more += 1;
        }
    }
    let toitoi_bonus = if remaining_triplets >= 3
        || (remaining_triplets >= 2 && remaining_pairs_or_more >= 5)
    {
        0.5
    } else {
        0.0
    };

    // Quantity Penalty / Bonus
    // - Count >= 7: Hard Ban (+20.0). Almost impossible to clear quickly.
    // - Count == 6: Heavy Penalty (+5.0). 留 6 张极为困难但不是绝对不可能。
    // - Count == 5: Moderate Penalty (+0.8).
    // - Count == 4: Slight Penalty (+0.2).
    // - Count == 1: Slight Bonus (-0.3). 只需打一张就完成定缺，极高效。
    let count_penalty = match removed_count {
        0 => unreachable!(), // handled above
        1 => -0.3,
        2 | 3 => 0.0,
        4 => 0.2,
        5 => 0.8,
        6 => 5.0,
        _ => 20.0, // 7+
    };

    // Improvement-kinds bonus (进张种类): more tile kinds that reduce shanten = better hand shape
    let improvement_kinds = count_improvement_kinds(
        &tehai_without, remaining_count, shanten, (start, end),
    );
    let improvement_bonus = (improvement_kinds as f32 * 0.12).min(2.0); // cap so one factor doesn't dominate

    // Final cost: structure in the removed suit increases cost; remaining hand quality decreases cost
    shanten as f32
        + triplet_penalty
        + sequence_penalty
        + pair_penalty
        + taatsu_penalty
        - toitoi_bonus
        - improvement_bonus
        + count_penalty
}

/// Compute the costs for all 3 suits, returning `[cost_Man, cost_Pin, cost_Sou]`.
/// Shared by `best_ding_que_suit_index` and `evaluate_ding_que_quality` to avoid
/// redundant `calc_ding_que_cost` calls.
fn compute_ding_que_costs(tehai: &[u8; 27]) -> [f32; 3] {
    [
        calc_ding_que_cost(tehai, Suit::Man),
        calc_ding_que_cost(tehai, Suit::Pin),
        calc_ding_que_cost(tehai, Suit::Sou),
    ]
}

/// Returns the heuristic best suit index for Ding Que: 0 = Man, 1 = Pin, 2 = Sou.
/// On tie, returns the first with minimum cost.
fn best_ding_que_suit_index_from_costs(costs: &[f32; 3]) -> u8 {
    let (best_idx, _) = costs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    best_idx as u8
}

/// Evaluate the quality of a Ding Que selection.
/// Returns a continuous value in [-1.0, +1.0]:
///   +1.0 = best possible choice (lowest cost)
///   -1.0 = worst possible choice (highest cost)
///    0.0 = all choices equivalent
///
/// 改为连续值而非离散 {-1, 0, +1}，避免丢失梯度信息。
/// 例如 costs = [3.0, 3.01, 15.0] 时选 3.01 应得 ≈+0.998 而非 0.0。
fn evaluate_ding_que_quality_from_costs(costs: &[f32; 3], chosen_suit: Suit) -> f32 {
    let chosen_idx = crate::ding_que::suit_id(chosen_suit);
    let chosen_cost = costs[chosen_idx];

    let min_cost = costs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_cost = costs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_cost - min_cost;

    if range < 0.001 {
        0.0 // All choices equivalent
    } else {
        // Linear map: best → +1.0, worst → -1.0
        1.0 - 2.0 * (chosen_cost - min_cost) / range
    }
}

/// Convert Tile array to u8 count array
fn tiles_to_tehai(tiles: &[Tile; 13]) -> [u8; 27] {
    let mut tehai = [0u8; 27];
    for &tile in tiles {
        let idx = tile.as_usize();
        if idx < 27 {
            tehai[idx] += 1;
        }
    }
    tehai
}

#[pymethods]
impl GameScore {
    #[staticmethod]
    fn load_log(raw_log: &str) -> Result<Self> {
        let events = raw_log
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .context("failed to parse log")?;
        Self::load_events(&events)
    }

    /// Returns list of score arrays (one per kyoku)
    pub fn take_scores_history(&mut self) -> Vec<[i32; 4]> {
        std::mem::take(&mut self.scores_history)
    }

    pub const fn take_final_scores(&self) -> [i32; 4] {
        self.final_scores
    }
    
    pub const fn take_rank_by_player(&self) -> [u8; 4] {
        self.rank_by_player
    }
    
    /// Returns list of Ding Que quality scores per kyoku per player
    /// Value: +1.0 = best choice, 0.0 = middle, -1.0 = worst choice
    pub fn take_ding_que_quality(&mut self) -> Vec<[f32; 4]> {
        std::mem::take(&mut self.ding_que_quality)
    }

    /// Returns heuristic best suit index per kyoku per player: 0=Man, 1=Pin, 2=Sou.
    /// Used for Ding Que CE auxiliary loss.
    pub fn take_ding_que_best_suit(&mut self) -> Vec<[u8; 4]> {
        std::mem::take(&mut self.ding_que_best_suit)
    }
    
    /// Returns agari (win) count per player per kyoku for action rewards
    pub fn take_agari_count(&mut self) -> Vec<[u8; 4]> {
        std::mem::take(&mut self.agari_count)
    }
    
    /// Returns houjuu (deal-in) count per player per kyoku for action rewards
    pub fn take_houjuu_count(&mut self) -> Vec<[u8; 4]> {
        std::mem::take(&mut self.houjuu_count)
    }
}

impl GameScore {
    pub fn load_events(events: &[Event]) -> Result<Self> {
        let mut scores_history = vec![];
        let mut rank_by_player_opt = None;
        let mut final_deltas = [0; 4];
        let mut final_scores = [0; 4];
        
        // For Ding Que quality and best-suit tracking
        let mut ding_que_quality: Vec<[f32; 4]> = vec![];
        let mut ding_que_best_suit: Vec<[u8; 4]> = vec![];
        // Track the effective hand at DingQue time (庄家在定缺前会多一张补牌)
        let mut current_tehais: Option<[[u8; 27]; 4]> = None;
        let mut current_kyoku_quality = [0.0f32; 4];
        let mut current_kyoku_best_suit = [0u8; 4]; // 0=Man, 1=Pin, 2=Sou
        
        // For agari (win) and houjuu (deal-in) tracking
        let mut agari_count: Vec<[u8; 4]> = vec![];
        let mut houjuu_count: Vec<[u8; 4]> = vec![];
        let mut current_kyoku_agari = [0u8; 4];
        let mut current_kyoku_houjuu = [0u8; 4];

        // Forward pass to collect scores and Ding Que quality
        for ev in events.iter() {
            match ev {
                Event::StartKyoku { scores, tehais, .. } => {
                    // Save previous kyoku's quality, best suit, and action counts (if any)
                    if current_tehais.is_some() {
                        ding_que_quality.push(current_kyoku_quality);
                        ding_que_best_suit.push(current_kyoku_best_suit);
                        agari_count.push(current_kyoku_agari);
                        houjuu_count.push(current_kyoku_houjuu);
                    }

                    // Reset for new kyoku
                    scores_history.push(*scores);
                    current_tehais = Some(array::from_fn(|i| tiles_to_tehai(&tehais[i])));
                    current_kyoku_quality = [0.0f32; 4]; // Default neutral
                    current_kyoku_best_suit = [0; 4];
                    current_kyoku_agari = [0u8; 4];
                    current_kyoku_houjuu = [0u8; 4];
                }
                Event::Tsumo { actor, pai } => {
                    if let Some(ref mut tehais) = current_tehais {
                        let player_idx = *actor as usize;
                        if player_idx < 4 {
                            let tid = pai.as_usize();
                            if tid < 27 {
                                tehais[player_idx][tid] += 1;
                            }
                        }
                    }
                }
                Event::DingQue { actor, suit } => {
                    if let Some(ref tehais) = current_tehais {
                        let player_idx = *actor as usize;
                        if player_idx < 4 {
                            // 计算一次 costs，供 quality 和 best_suit 复用
                            let costs = compute_ding_que_costs(&tehais[player_idx]);
                            current_kyoku_quality[player_idx] =
                                evaluate_ding_que_quality_from_costs(&costs, *suit);
                            current_kyoku_best_suit[player_idx] =
                                best_ding_que_suit_index_from_costs(&costs);
                        }
                    }
                }
                Event::Hora { actor, target, .. } => {
                    // Track agari (win) for actor
                    let winner_idx = *actor as usize;
                    if winner_idx < 4 {
                        current_kyoku_agari[winner_idx] = current_kyoku_agari[winner_idx].saturating_add(1);
                    }
                    // Track houjuu (deal-in) for target if it's a Ron (target != actor)
                    let target_idx = *target as usize;
                    if target_idx < 4 && target_idx != winner_idx {
                        current_kyoku_houjuu[target_idx] = current_kyoku_houjuu[target_idx].saturating_add(1);
                    }
                }
                _ => {}
            }
        }
        
        // Save the last kyoku's quality, best suit, and action counts
        if current_tehais.is_some() {
            ding_que_quality.push(current_kyoku_quality);
            ding_que_best_suit.push(current_kyoku_best_suit);
            agari_count.push(current_kyoku_agari);
            houjuu_count.push(current_kyoku_houjuu);
        }

        // Reverse pass to find final scores (existing logic)
        // FIX: 同时收集和牌顺序，用于同分时按先和牌者排名靠前
        let mut agari_order_rev: Vec<u8> = Vec::new();
        for ev in events.iter().rev() {
            match *ev {
                // Bloody Battle scoring events can happen outside Hora/Ryukyoku.
                // In particular, kan events carry "instant payment" deltas, which must be
                // included to reconstruct the true final scores (and thus correct rewards).
                Event::Hora { actor, deltas, .. } => {
                    if rank_by_player_opt.is_none() {
                        let ds = deltas.context(
                            "invalid log: field `deltas` is required for scoring events",
                        )?;
                        vec_add_assign(&mut final_deltas, &ds);
                        // 反向遍历，所以后和的先被收集
                        if !agari_order_rev.contains(&actor) {
                            agari_order_rev.push(actor);
                        }
                    }
                }
                Event::Ryukyoku { deltas, .. }
                | Event::Daiminkan { deltas, .. }
                | Event::Kakan { deltas, .. }
                | Event::Ankan { deltas, .. } => {
                    if rank_by_player_opt.is_none() {
                        let ds = deltas.context(
                            "invalid log: field `deltas` is required for scoring events",
                        )?;
                        vec_add_assign(&mut final_deltas, &ds);
                    }
                }
                Event::StartKyoku { scores, .. } => {
                    if rank_by_player_opt.is_none() {
                        final_scores = scores;
                        vec_add_assign(&mut final_scores, &final_deltas);

                        // 反转得到正确的和牌顺序（先和牌者在前）
                        agari_order_rev.reverse();
                        let agari_order = if agari_order_rev.is_empty() {
                            None
                        } else {
                            Some(agari_order_rev.as_slice())
                        };
                        let rk = Rankings::new_with_agari_order(final_scores, agari_order);

                        // assume the sum of scores to be TOTAL_SCORE (zero-sum: 4×INITIAL_SCORE)
                        let sum: i32 = final_scores.iter().sum();
                        if sum != crate::consts::TOTAL_SCORE {
                            let top = rk.player_by_rank[0] as usize;
                            final_scores[top] += crate::consts::TOTAL_SCORE - sum;
                        }

                        rank_by_player_opt = Some(rk.rank_by_player);
                    }
                }
                _ => (),
            }
        }

        let rank_by_player =
            rank_by_player_opt.context("invalid log: no scoring event found after a StartKyoku")?;

        Ok(Self {
            scores_history,
            final_scores,
            rank_by_player,
            ding_que_quality,
            ding_que_best_suit,
            agari_count,
            houjuu_count,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts::INITIAL_SCORE;
    use crate::mjai::Suit;
    use crate::t;

    #[test]
    fn final_scores_include_kan_deltas() {
        let init: [i32; 4] = [INITIAL_SCORE; 4];
        let events = vec![
            Event::StartGame {
                names: [
                    "A".to_string(),
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                ],
                seed: Some((1, 2)),
            },
            Event::StartKyoku {
                kyoku: 1,
                oya: 0,
                scores: init,
                tehais: [[t!(1m); 13]; 4],
            },
            // Ankan: instant payment (+6000 to actor, -2000 each from others)
            Event::Ankan {
                actor: 0,
                consumed: [t!(9m), t!(9m), t!(9m), t!(9m)],
                deltas: Some([6000, -2000, -2000, -2000]),
            },
            // End by ryukyoku with 0 deltas (keeps only kan deltas)
            Event::Ryukyoku {
                deltas: Some([0, 0, 0, 0]),
            },
            Event::EndKyoku,
            Event::EndGame,
        ];

        let gs = GameScore::load_events(&events).unwrap();
        assert_eq!(gs.scores_history, vec![init]);
        assert_eq!(
            gs.final_scores,
            [
                INITIAL_SCORE + 6000,
                INITIAL_SCORE - 2000,
                INITIAL_SCORE - 2000,
                INITIAL_SCORE - 2000,
            ]
        );
        // rank_by_player is 0 for top, 3 for last
        assert_eq!(gs.rank_by_player[0], 0);
    }

    #[test]
    fn ding_que_count_sequences_in_suit() {
        // No sequence: 1m 5m 9m
        assert_eq!(count_sequences_in_suit(&[1, 0, 0, 0, 1, 0, 0, 0, 1]), 0);
        // One sequence: 123
        assert_eq!(count_sequences_in_suit(&[1, 1, 1, 0, 0, 0, 0, 0, 0]), 1);
        // Two sequences: 123 + 234
        assert_eq!(count_sequences_in_suit(&[1, 1, 1, 1, 1, 1, 0, 0, 0]), 2);
        // 111234 -> one 顺子 (123), one 2 and 3 left but no 4 for 234
        assert_eq!(count_sequences_in_suit(&[3, 1, 1, 1, 0, 0, 0, 0, 0]), 1);
    }

    #[test]
    fn ding_que_cost_penalizes_triplet_and_sequence() {
        // Man: 111m (3 张, 刻子), Pin: 123p+77p (5 张, 顺子+对子)
        // Sou: 135s+88s (5 张, 散牌+对子). 总: 3+5+5 = 13
        let mut tehai = [0u8; 27];
        tehai[0] = 3;                                // 111m (3 张, 刻子)
        tehai[9] = 1; tehai[10] = 1; tehai[11] = 1; // 123p (3 张, 顺子)
        tehai[18] = 1; tehai[20] = 1; tehai[22] = 1; // 135s (3 张, 散牌)
        tehai[15] = 2;                               // 77p  (2 张)
        tehai[25] = 2;                               // 88s  (2 张)
        // 总: 3 + 3 + 3 + 2 + 2 = 13

        let cost_man = calc_ding_que_cost(&tehai, Suit::Man);
        let cost_pin = calc_ding_que_cost(&tehai, Suit::Pin);
        let cost_sou = calc_ding_que_cost(&tehai, Suit::Sou);

        // 定缺 Man (3 张): 失去 1 个刻子 → 结构损失大
        // 定缺 Pin (3+2=5 张): 失去 1 个顺子 + 1 个对子 → 牌多 + 结构
        // 定缺 Sou (3+2=5 张): 失去散牌 + 1 个对子 → 牌多但结构损失小
        //
        // 刻子 vs 顺子（同为 3 张时）：刻子罚分 0.8 > 顺子 0.7
        // 但 Pin/Sou 各有 5 张 (含 pair)，数量惩罚使它们 cost 更高。
        // 因此: cost_pin > cost_man（Pin 5 张 > Man 3 张），
        //       cost_sou > cost_man（同理），
        //       cost_pin > cost_sou（顺子+对子 > 散牌+对子）。
        assert!(cost_man < cost_pin,
            "定缺 Man(3张) cost({cost_man}) 应 < Pin(5张) cost({cost_pin})");
        assert!(cost_man < cost_sou,
            "定缺 Man(3张) cost({cost_man}) 应 < Sou(5张) cost({cost_sou})");
        // 顺子有结构 → 定缺代价更高
        assert!(cost_pin > cost_sou,
            "定缺 Pin(顺子+对子) cost({cost_pin}) 应 > Sou(散牌+对子) cost({cost_sou})");
    }

    #[test]
    fn ding_que_cost_prefers_suit_with_no_groups() {
        // Hand: 1m 5m 9m (Man, no 刻子/顺子), 123p (Pin, one 顺子), 111s (Sou, one 刻子)
        let mut tehai = [0u8; 27];
        tehai[0] = 1;
        tehai[4] = 1;
        tehai[8] = 1; // Man: 散牌
        tehai[9] = 1;
        tehai[10] = 1;
        tehai[11] = 1; // Pin: 顺子
        tehai[18] = 3; // Sou: 刻子
        let cost_man = calc_ding_que_cost(&tehai, Suit::Man);
        let cost_pin = calc_ding_que_cost(&tehai, Suit::Pin);
        let cost_sou = calc_ding_que_cost(&tehai, Suit::Sou);
        // 定缺 Man (only 散牌) should have lowest cost
        assert!(cost_man < cost_pin, "定缺无成组门应 cost 最低");
        assert!(cost_man < cost_sou);
    }
}
