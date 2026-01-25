use crate::consts::GRP_SIZE;
use crate::mjai::{Event, Suit};
use crate::rankings::Rankings;
use crate::vec_ops::vec_add_assign;
use std::fs::File;
use std::io;
use std::mem;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use rayon::prelude::*;
use serde_json as json;
use tinyvec::array_vec;

#[pyclass]
#[derive(Clone, Default)]
pub struct Grp {
    // Bloody Battle: [kyoku, [score[i] / 10000], [agari[i]], [ding_que[i]]] where i is player_id
    // agari[i] = 1.0 if player i has agari, 0.0 otherwise
    // ding_que[i] = 0.0 for Man, 0.5 for Pin, 1.0 for Sou (normalized)
    // No grand_kyoku, honba, kyotaku
    pub feature: Array2<f64>,
    pub rank_by_player: [u8; 4],
    pub final_scores: [i32; 4],
}

#[pymethods]
impl Grp {
    #[staticmethod]
    fn load_log(raw_log: &str) -> Result<Self> {
        let events = raw_log
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .context("failed to parse log")?;
        Self::load_events(&events)
    }

    #[staticmethod]
    #[pyo3(name = "load_gz_log_files")]
    fn load_gz_log_files_py(gzip_filenames: Vec<PyBackedStr>) -> Result<Vec<Self>> {
        Self::load_gz_log_files(gzip_filenames)
    }

    /// Returns List[List[np.ndarray]]
    pub fn take_feature<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, mem::take(&mut self.feature))
    }
    pub const fn take_rank_by_player(&self) -> [u8; 4] {
        self.rank_by_player
    }
    pub const fn take_final_scores(&self) -> [i32; 4] {
        self.final_scores
    }
}

impl Grp {
    #[inline]
    pub fn len(&self) -> usize {
        self.feature.len_of(Axis(0))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn load_gz_log_files<V, S>(gzip_filenames: V) -> Result<Vec<Self>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        gzip_filenames
            .into_par_iter()
            .map(|f| {
                let filename = f.as_ref();
                let inner = || {
                    let file = File::open(filename)?;
                    let gz = GzDecoder::new(file);
                    let raw = io::read_to_string(gz)?;
                    Self::load_log(&raw)
                };
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect()
    }

    pub fn load_events(events: &[Event]) -> Result<Self> {
        let mut game_info = vec![];
        let mut rank_by_player_opt = None;
        let mut final_deltas = [0; 4];
        let mut final_scores = [0; 4];
        
        // Track which players have agari and their ding_que at the START of each StartKyoku
        // In Bloody Battle, agari state persists across kyokus (once a player agari, they stay agari)
        // Ding_que is set at the start of each kyoku and persists for that kyoku
        // We need to build this by forward traversal first
        let mut players_agari_at_kyoku: Vec<[bool; 4]> = vec![];
        let mut players_ding_que_at_kyoku: Vec<[Option<Suit>; 4]> = vec![];
        let mut current_players_agari = [false; 4];
        let mut current_players_ding_que = [None; 4];
        
        // First pass: forward traversal to track agari and ding_que state at the START of each StartKyoku
        for ev in events.iter() {
            match ev {
                Event::StartKyoku { .. } => {
                    // Record the agari and ding_que state at the START of this kyoku (before any actions in this kyoku)
                    // This is the state BEFORE this StartKyoku event
                    players_agari_at_kyoku.push(current_players_agari);
                    players_ding_que_at_kyoku.push(current_players_ding_que);
                    // Reset ding_que for new kyoku (each kyoku has its own ding_que)
                    current_players_ding_que = [None; 4];
                }
                Event::Hora { actor, .. } => {
                    // Mark this player as having agari (this persists for future kyokus)
                    current_players_agari[*actor as usize] = true;
                }
                Event::DingQue { actor, suit } => {
                    // Record ding_que for this player (persists for this kyoku)
                    current_players_ding_que[*actor as usize] = Some(*suit);
                }
                _ => (),
            }
        }
        
        // Reverse the tracking for reverse traversal (since we traverse events in reverse)
        players_agari_at_kyoku.reverse();
        players_ding_que_at_kyoku.reverse();
        let mut agari_idx = 0;

        // Second pass: reverse traversal to extract features
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
                // Event::ReachAccepted removed - Bloody Battle Mahjong does not have riichi (立直)
                Event::StartKyoku {
                    kyoku,
                    scores,
                    ..
                } => {
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

                    // Bloody Battle Mahjong: GRP feature is [kyoku, [score[i] / 10000], [agari[i]], [ding_que[i]]]
                    // agari[i] = 1.0 if player i has agari, 0.0 otherwise
                    // ding_que[i] = 0.0 for Man, 0.5 for Pin, 1.0 for Sou (normalized), or 0.0 if not set
                    // No grand_kyoku, honba, or kyotaku
                    let mut kyoku_info = array_vec!([_; GRP_SIZE]);
                    kyoku_info.push(kyoku as f64);
                    kyoku_info.extend(scores.iter().map(|&score| score as f64 / 10000.));
                    
                    // Add agari information
                    let players_agari = if agari_idx < players_agari_at_kyoku.len() {
                        players_agari_at_kyoku[agari_idx]
                    } else {
                        // Fallback: if we don't have tracking data, use empty (shouldn't happen)
                        [false; 4]
                    };
                    kyoku_info.extend(players_agari.iter().map(|&agari| if agari { 1.0 } else { 0.0 }));
                    
                    // Add ding_que information
                    let players_ding_que = if agari_idx < players_ding_que_at_kyoku.len() {
                        players_ding_que_at_kyoku[agari_idx]
                    } else {
                        // Fallback: if we don't have tracking data, use None (shouldn't happen)
                        [None; 4]
                    };
                    kyoku_info.extend(players_ding_que.iter().map(|&ding_que| {
                        match ding_que {
                            Some(Suit::Man) => 0.0,
                            Some(Suit::Pin) => 0.5,
                            Some(Suit::Sou) => 1.0,
                            None => 0.0, // Default to 0.0 if not set (shouldn't happen in normal games)
                        }
                    }));
                    agari_idx += 1;
                    
                    assert_eq!(kyoku_info.len(), GRP_SIZE);

                    game_info.insert(0, kyoku_info);
                }
                _ => (),
            }
        }

        let rank_by_player =
            rank_by_player_opt.context("invalid log: no Hora or Ryukyoku after a StartKyoku")?;
        let shape = (game_info.len(), GRP_SIZE);
        let feature =
            Array::from_iter(game_info.into_iter().flatten()).into_shape_with_order(shape)?;

        Ok(Self {
            feature,
            rank_by_player,
            final_scores,
        })
    }
}
