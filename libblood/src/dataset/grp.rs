use crate::mjai::Event;
use crate::rankings::Rankings;
use crate::vec_ops::vec_add_assign;
use anyhow::{Context, Result};
use pyo3::prelude::*;
use serde_json as json;
// use crate::consts::GRP_SIZE; // Removed

// Replaces original Grp struct
#[pyclass]
#[derive(Clone, Default, Debug)]
pub struct GameScore {
    // Scores at the START of each kyoku.
    // Normalized to 10000 = 1.0 in Python? No, keep as raw i32 here.
    pub scores_history: Vec<[i32; 4]>, 
    pub final_scores: [i32; 4],
    pub rank_by_player: [u8; 4],
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
}

impl GameScore {
    pub fn load_events(events: &[Event]) -> Result<Self> {
        let mut scores_history = vec![];
        let mut rank_by_player_opt = None;
        let mut final_deltas = [0; 4];
        let mut final_scores = [0; 4];

        // Forward pass to collect scores at StartKyoku
        for ev in events.iter() {
            if let Event::StartKyoku { scores, .. } = ev {
                scores_history.push(*scores);
            }
        }

        // Reverse pass (or simplified logic) to finding final scores
        // The original logic used reverse traversal to properly handle AL (All Last) deltas
        // Let's stick to the proven logic for calculating final scores to be safe.
        
        for ev in events.iter().rev() {
            match *ev {
                Event::Hora { deltas, .. } | Event::Ryukyoku { deltas, .. } => {
                    if rank_by_player_opt.is_none() {
                        let ds = deltas.context(
                            "invalid log: field `deltas` is required for Hora and Ryukyoku of AL",
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
            rank_by_player_opt.context("invalid log: no Hora or Ryukyoku after a StartKyoku")?;

        Ok(Self {
            scores_history,
            final_scores,
            rank_by_player,
        })
    }
}
