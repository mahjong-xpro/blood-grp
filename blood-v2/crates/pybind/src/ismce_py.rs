use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

use engine::consts::*;
use engine::tile::Suit;
use engine::algo::ismce::{self, IsmceConfig, PlayerInfo};

/// Evaluate candidate discards using ISMCE.
///
/// Args:
///     hand: [u8; 27] tile counts
///     melds_count: number of melds
///     ding_que: 0=Man, 1=Pin, 2=Sou, -1=none
///     tiles_seen: [u8; 27] tiles visible to this player
///     candidates: list of tile indices to evaluate
///     wall_remaining: tiles left in wall
///     num_worlds: number of sampled worlds (default 64)
///     rollout_depth: draw depth per world (default 4)
///
/// Returns: list of (tile, win_rate, tenpai_rate, improvement)
#[pyfunction]
#[pyo3(signature = (hand, melds_count, ding_que, tiles_seen, candidates, wall_remaining, num_worlds=64, rollout_depth=4, base_seed=0))]
pub fn ismce_evaluate(
    _py: Python<'_>,
    hand: PyReadonlyArray1<u8>,
    melds_count: usize,
    ding_que: i8,
    tiles_seen: PyReadonlyArray1<u8>,
    candidates: Vec<u8>,
    wall_remaining: usize,
    num_worlds: usize,
    rollout_depth: usize,
    base_seed: u64,
) -> PyResult<Vec<(u8, f64, f64, f64)>> {
    let hand_slice = hand.as_slice()?;
    let seen_slice = tiles_seen.as_slice()?;

    if hand_slice.len() < NUM_TILE_TYPES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("hand must have at least {} elements, got {}", NUM_TILE_TYPES, hand_slice.len())
        ));
    }
    if seen_slice.len() < NUM_TILE_TYPES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("tiles_seen must have at least {} elements, got {}", NUM_TILE_TYPES, seen_slice.len())
        ));
    }

    let mut h = [0u8; NUM_TILE_TYPES];
    h.copy_from_slice(&hand_slice[..NUM_TILE_TYPES]);
    let mut s = [0u8; NUM_TILE_TYPES];
    s.copy_from_slice(&seen_slice[..NUM_TILE_TYPES]);

    let dq = match ding_que {
        0 => Some(Suit::Man),
        1 => Some(Suit::Pin),
        2 => Some(Suit::Sou),
        _ => None,
    };

    let info = PlayerInfo {
        hand: h,
        melds_count,
        ding_que: dq,
        tiles_seen: s,
        wall_remaining,
    };

    let config = IsmceConfig {
        num_worlds,
        rollout_depth,
        base_seed,
    };

    let results = ismce::evaluate_discards(&info, &candidates, &config);

    Ok(results
        .iter()
        .map(|r| (r.tile, r.win_rate, r.tenpai_rate, r.avg_shanten_improvement))
        .collect())
}

/// Compute danger scores for all 27 tile types.
///
/// Returns: [f32; 27] danger score per tile type.
#[pyfunction]
pub fn ismce_danger<'py>(
    py: Python<'py>,
    tiles_seen: PyReadonlyArray1<u8>,
    opp1_discards: Vec<u8>,
    opp2_discards: Vec<u8>,
    opp3_discards: Vec<u8>,
    wall_remaining: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let seen_slice = tiles_seen.as_slice()?;
    let mut s = [0u8; NUM_TILE_TYPES];
    s.copy_from_slice(&seen_slice[..NUM_TILE_TYPES]);

    let opp_discards = [opp1_discards, opp2_discards, opp3_discards];
    let danger = ismce::danger_scores(&s, &opp_discards, wall_remaining);

    Ok(PyArray1::from_vec_bound(py, danger.to_vec()))
}
