use crate::algo::shanten;
use crate::mjai::{Event, Suit};
use crate::rankings::Rankings;
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
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
}

/// Calculate the cost of choosing a suit as Ding Que.
/// Lower cost = better choice.
/// 
/// Factors:
/// 1. Shanten after removing the suit (base)
/// 2. Triplet penalty (if removing triplets)
/// 3. ToiToi potential bonus (if remaining hand has 4+ pairs/triplets)
fn calc_ding_que_cost(tehai: &[u8; 27], suit: Suit) -> f32 {
    let suit_range = match suit {
        Suit::Man => 0..9,
        Suit::Pin => 9..18,
        Suit::Sou => 18..27,
    };
    
    // Count tiles to be removed
    let mut removed_count: u8 = 0;
    let mut removed_triplets: u8 = 0;
    for i in suit_range.clone() {
        let count = tehai[i];
        removed_count += count;
        if count >= 3 {
            removed_triplets += 1;
        }
    }
    
    // If no tiles in this suit, it's the perfect choice
    if removed_count == 0 {
        return -10.0; // Very low cost (bonus)
    }
    
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
    
    // Triplet penalty: each triplet removed is a significant loss
    let triplet_penalty = removed_triplets as f32 * 0.8;
    
    // ToiToi potential: count pairs and triplets in remaining hand
    let mut pair_triplet_count: u8 = 0;
    for &count in tehai_without.iter() {
        if count >= 2 {
            pair_triplet_count += 1;
        }
    }
    let toitoi_bonus = if pair_triplet_count >= 4 { 0.5 } else { 0.0 };
    
    // Final cost
    shanten as f32 + triplet_penalty - toitoi_bonus
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
}

impl GameScore {
    pub fn load_events(events: &[Event]) -> Result<Self> {
        let mut scores_history = vec![];
        let mut rank_by_player_opt = None;
        let mut final_deltas = [0; 4];
        let mut final_scores = [0; 4];
        
        // For Ding Que quality tracking
        let mut ding_que_quality: Vec<[f32; 4]> = vec![];
        let mut current_tehais: Option<[[Tile; 13]; 4]> = None;
        let mut current_kyoku_quality = [0.0f32; 4];

        // Forward pass to collect scores and Ding Que quality
        for ev in events.iter() {
            match ev {
                Event::StartKyoku { scores, tehais, .. } => {
                    // Save previous kyoku's quality (if any)
                    if current_tehais.is_some() {
                        ding_que_quality.push(current_kyoku_quality);
                    }
                    
                    // Reset for new kyoku
                    scores_history.push(*scores);
                    current_tehais = Some(*tehais);
                    current_kyoku_quality = [0.0f32; 4]; // Default neutral
                }
                Event::DingQue { actor, suit } => {
                    if let Some(ref tehais) = current_tehais {
                        let player_idx = *actor as usize;
                        if player_idx < 4 {
                            let tehai = tiles_to_tehai(&tehais[player_idx]);
                            current_kyoku_quality[player_idx] = evaluate_ding_que_quality(&tehai, *suit);
                        }
                    }
                }
                _ => {}
            }
        }
        
        // Save the last kyoku's quality
        if current_tehais.is_some() {
            ding_que_quality.push(current_kyoku_quality);
        }

        // Reverse pass to find final scores (existing logic)
        for ev in events.iter().rev() {
            match *ev {
                // Bloody Battle scoring events can happen outside Hora/Ryukyoku.
                // In particular, kan events carry "instant payment" deltas, which must be
                // included to reconstruct the true final scores (and thus correct rewards).
                Event::Hora { deltas, .. }
                | Event::Ryukyoku { deltas, .. }
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

                        let rk = Rankings::new(final_scores);

                        // assume the sum of scores to be 100k
                        let sum: i32 = final_scores.iter().sum();
                        if sum < 100_000 {
                            final_scores[rk.player_by_rank[0] as usize] += 100_000 - sum;
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
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::t;

    #[test]
    fn final_scores_include_kan_deltas() {
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
                scores: [25000, 25000, 25000, 25000],
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
        assert_eq!(gs.scores_history, vec![[25000, 25000, 25000, 25000]]);
        assert_eq!(gs.final_scores, [31000, 23000, 23000, 23000]);
        // rank_by_player is 0 for top, 3 for last
        assert_eq!(gs.rank_by_player[0], 0);
    }
}
