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

/// Count how many tile kinds (0..27) improve the hand when added by one (reduce shanten).
/// Used as a bonus: more improvement kinds = better shape after removing the suit.
fn count_improvement_kinds(tehai_without: &[u8; 27], remaining_count: u8, shanten: i8) -> u8 {
    // FIX: 使用与 base shanten 相同的 len_div3（= remaining_count / 3）。
    // 之前用 (remaining_count + 1) / 3，当 remaining_count % 3 == 2 时目标组数多 1，
    // 导致加牌后的向听被高估，进而系统性低估受入种类。
    let new_len_div3 = remaining_count / 3;
    let mut count: u8 = 0;
    for tid in 0..27 {
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
/// 5. ToiToi potential bonus (if remaining hand has 4+ pairs/triplets)
/// 6. Improvement-kinds bonus (进张种类数: more tile kinds that reduce shanten = better shape)
pub(crate) fn calc_ding_que_cost(tehai: &[u8; 27], suit: Suit) -> f32 {
    let suit_range = match suit {
        Suit::Man => 0..9,
        Suit::Pin => 9..18,
        Suit::Sou => 18..27,
    };

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

    // ToiToi potential: count pairs and triplets in remaining hand
    let mut pair_triplet_count: u8 = 0;
    for &count in tehai_without.iter() {
        if count >= 2 {
            pair_triplet_count += 1;
        }
    }
    let toitoi_bonus = if pair_triplet_count >= 4 { 0.5 } else { 0.0 };

    // Quantity Penalty (Golden Balance Plan + User Request)
    // - Count >= 6: Soft Ban (+20.0). It's suicidal to void >=6 tiles.
    // - Count == 5: Moderate Penalty (+0.6). Worse than pair penalty(0.35), close to sequence(0.7).
    // - Count == 4: Slight Penalty (+0.15). Slight bias against voiding 4 tiles if 3-tile option exists.
    let count_penalty = if removed_count >= 6 {
        20.0
    } else if removed_count == 5 {
        0.8
    } else if removed_count == 4 {
        0.2
    } else {
        0.0
    };

    // Improvement-kinds bonus (进张种类): more tile kinds that reduce shanten = better hand shape
    let improvement_kinds = count_improvement_kinds(&tehai_without, remaining_count, shanten);
    let improvement_bonus = (improvement_kinds as f32 * 0.12).min(2.0); // cap so one factor doesn't dominate

    // Final cost: 刻子/顺子/对子 in the removed suit all increase cost (worse to 定缺 a suit that has useful structure)
    shanten as f32 + triplet_penalty + sequence_penalty + pair_penalty - toitoi_bonus - improvement_bonus + count_penalty
}

/// Returns the heuristic best suit index for Ding Que: 0 = Man, 1 = Pin, 2 = Sou.
/// On tie, returns the first with minimum cost.
fn best_ding_que_suit_index(tehai: &[u8; 27]) -> u8 {
    let suits = [Suit::Man, Suit::Pin, Suit::Sou];
    let costs: Vec<f32> = suits.iter().map(|&s| calc_ding_que_cost(tehai, s)).collect();
    let (best_idx, _) = costs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    best_idx as u8
}

/// Evaluate the quality of a Ding Que selection.
/// Returns: +1.0 (best), 0.0 (middle), -1.0 (worst)
fn evaluate_ding_que_quality(tehai: &[u8; 27], chosen_suit: Suit) -> f32 {
    let suits = [Suit::Man, Suit::Pin, Suit::Sou];
    let costs: Vec<f32> = suits.iter().map(|&s| calc_ding_que_cost(tehai, s)).collect();

    let chosen_idx = match chosen_suit {
        Suit::Man => 0,
        Suit::Pin => 1,
        Suit::Sou => 2,
    };

    let chosen_cost = costs[chosen_idx];
    let min_cost = costs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_cost = costs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Handle edge case: all costs are equal
    if (max_cost - min_cost).abs() < 0.001 {
        return 0.0; // No difference, neutral
    }

    // Check if chosen is best, worst, or middle
    if (chosen_cost - min_cost).abs() < 0.001 {
        1.0 // Best choice
    } else if (chosen_cost - max_cost).abs() < 0.001 {
        -1.0 // Worst choice
    } else {
        0.0 // Middle choice
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
                            current_kyoku_quality[player_idx] =
                                evaluate_ding_que_quality(&tehais[player_idx], *suit);
                            current_kyoku_best_suit[player_idx] =
                                best_ding_que_suit_index(&tehais[player_idx]);
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
        // Hand: 111m 123p 1s 2s 3s 4s 5s (Man has 刻子, Pin has 顺子, Sou has 顺子)
        let mut tehai = [0u8; 27];
        tehai[0] = 3; // 111m
        tehai[9] = 1;
        tehai[10] = 1;
        tehai[11] = 1; // 123p
        tehai[18] = 1;
        tehai[19] = 1;
        tehai[20] = 1;
        tehai[21] = 1;
        tehai[22] = 1; // 12345s
        let cost_man = calc_ding_que_cost(&tehai, Suit::Man);
        let cost_pin = calc_ding_que_cost(&tehai, Suit::Pin);
        let cost_sou = calc_ding_que_cost(&tehai, Suit::Sou);
        // 定缺 Man: lose one 刻子 -> high cost
        // 定缺 Pin: lose one 顺子 -> medium cost
        // 定缺 Sou: lose one 顺子 (123) and some loose tiles -> cost depends on shanten
        assert!(cost_man > cost_pin, "定缺刻子门应比定缺顺子门 cost 更高");
        assert!(cost_man > cost_sou);
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
