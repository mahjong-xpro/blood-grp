use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

use engine::consts::*;
use engine::tile::Suit;
use engine::algo::ismce::{self, IsmceConfig, PlayerInfo, OpponentInfo, OpponentDangerInfo};

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
/// Returns: list of (tile, win_rate, tenpai_rate, improvement, expected_score, tenpai_value, danger_cost)
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
) -> PyResult<Vec<(u8, f64, f64, f64, f64, f64, f64)>> {
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

    // Validate candidate tile indices are in range [0, 27)
    for &c in &candidates {
        if (c as usize) >= NUM_TILE_TYPES {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("candidate tile {} out of range [0, {})", c, NUM_TILE_TYPES)
            ));
        }
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
        .map(|r| (r.tile, r.win_rate, r.tenpai_rate, r.avg_shanten_improvement,
                   r.expected_score, r.tenpai_value, r.danger_cost))
        .collect())
}

/// Full ISMCE evaluation with constrained sampling and defense-aware rollouts.
///
/// This is the recommended entry point that combines:
/// 1. Constrained world sampling (respects opponent ding-que)
/// 2. Enhanced danger score computation
/// 3. Defense-aware rollout (uses danger as tiebreaker among equal-shanten discards)
///
/// Args:
///     hand: [u8; 27] tile counts
///     melds_count: number of melds
///     ding_que: 0=Man, 1=Pin, 2=Sou, -1=none
///     tiles_seen: [u8; 27] tiles visible to this player
///     candidates: list of tile indices to evaluate
///     wall_remaining: tiles left in wall
///     opponent_ding_que: list of 3 ding-que values (-1=none, 0=Man, 1=Pin, 2=Sou)
///     opponent_meld_counts: list of 3 meld counts
///     opponent_discard_counts: list of 3 discard counts
///     opponent_discards: list of 3 lists of tile indices (recent discards per opponent)
///     num_worlds: number of sampled worlds (default 64)
///     rollout_depth: draw depth per world (default 4)
///     base_seed: RNG seed (default 0)
///
/// Returns: list of (tile, win_rate, tenpai_rate, improvement)
#[pyfunction]
#[pyo3(signature = (hand, melds_count, ding_que, tiles_seen, candidates, wall_remaining,
                    opponent_ding_que, opponent_meld_counts, opponent_discard_counts,
                    opponent_discards, num_worlds=64, rollout_depth=4, base_seed=0))]
pub fn ismce_evaluate_full(
    _py: Python<'_>,
    hand: PyReadonlyArray1<u8>,
    melds_count: usize,
    ding_que: i8,
    tiles_seen: PyReadonlyArray1<u8>,
    candidates: Vec<u8>,
    wall_remaining: usize,
    opponent_ding_que: Vec<i8>,
    opponent_meld_counts: Vec<usize>,
    opponent_discard_counts: Vec<usize>,
    opponent_discards: Vec<Vec<u8>>,
    num_worlds: usize,
    rollout_depth: usize,
    base_seed: u64,
) -> PyResult<Vec<(u8, f64, f64, f64, f64, f64, f64)>> {
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

    // Validate candidate tile indices are in range [0, 27)
    for &c in &candidates {
        if (c as usize) >= NUM_TILE_TYPES {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("candidate tile {} out of range [0, {})", c, NUM_TILE_TYPES)
            ));
        }
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

    // Build OpponentInfo array for constrained sampling
    let mut opponents: Vec<OpponentInfo> = Vec::new();
    for i in 0..3.min(opponent_ding_que.len()) {
        let opp_dq = match opponent_ding_que.get(i).copied().unwrap_or(-1) {
            0 => Some(Suit::Man),
            1 => Some(Suit::Pin),
            2 => Some(Suit::Sou),
            _ => None,
        };
        let discards_vec: Vec<u8> = opponent_discards.get(i).cloned().unwrap_or_default();
        let mc = opponent_meld_counts.get(i).copied().unwrap_or(0);
        // Estimate hand count: pon uses 2 tiles from hand (reveal 2 + 1 from opponent),
        // kan uses 3 tiles from hand. Without meld type info, use 2 per meld as
        // conservative estimate (pon is more common than kan in Sichuan mahjong).
        let hand_count = 13u8.saturating_sub((2 * mc) as u8);
        opponents.push(OpponentInfo {
            ding_que: opp_dq,
            discards: discards_vec,
            melds_count: mc,
            hand_count,
        });
    }

    // Build OpponentDangerInfo array for danger scoring
    let mut danger_infos: [OpponentDangerInfo; 3] = [
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: 0 },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: 0 },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: 0 },
    ];
    for i in 0..3 {
        danger_infos[i].ding_que = match opponent_ding_que.get(i).copied().unwrap_or(-1) {
            0 => Some(Suit::Man),
            1 => Some(Suit::Pin),
            2 => Some(Suit::Sou),
            _ => None,
        };
        danger_infos[i].melds_count = opponent_meld_counts.get(i).copied().unwrap_or(0);
        danger_infos[i].discard_count = opponent_discard_counts.get(i).copied().unwrap_or(0);
    }

    // Build opponent_discards array for danger scoring
    let opp_disc: [Vec<u8>; 3] = [
        opponent_discards.get(0).cloned().unwrap_or_default(),
        opponent_discards.get(1).cloned().unwrap_or_default(),
        opponent_discards.get(2).cloned().unwrap_or_default(),
    ];

    let results = ismce::evaluate_discards_full(
        &info, &candidates, &opponents, &opp_disc, &danger_infos, &config,
    );

    Ok(results
        .iter()
        .map(|r| (r.tile, r.win_rate, r.tenpai_rate, r.avg_shanten_improvement,
                   r.expected_score, r.tenpai_value, r.danger_cost))
        .collect())
}

/// Full ISMCE evaluation with informed sampling from opponent hand predictions.
///
/// Same as ismce_evaluate_full but additionally accepts opponent hand probability
/// predictions (from OpponentHandPredictor) to improve world sampling quality.
///
/// Args:
///     opponent_hand_probs: list of 3 lists of 27 floats — P(tile in opponent hand)
///     (all other args same as ismce_evaluate_full)
///
/// Returns: list of (tile, win_rate, tenpai_rate, improvement, expected_score, tenpai_value, danger_cost)
#[pyfunction]
#[pyo3(signature = (hand, melds_count, ding_que, tiles_seen, candidates, wall_remaining,
                    opponent_ding_que, opponent_meld_counts, opponent_discard_counts,
                    opponent_discards, opponent_hand_probs,
                    num_worlds=64, rollout_depth=4, base_seed=0))]
pub fn ismce_evaluate_informed(
    _py: Python<'_>,
    hand: PyReadonlyArray1<u8>,
    melds_count: usize,
    ding_que: i8,
    tiles_seen: PyReadonlyArray1<u8>,
    candidates: Vec<u8>,
    wall_remaining: usize,
    opponent_ding_que: Vec<i8>,
    opponent_meld_counts: Vec<usize>,
    opponent_discard_counts: Vec<usize>,
    opponent_discards: Vec<Vec<u8>>,
    opponent_hand_probs: Vec<Vec<f32>>,
    num_worlds: usize,
    rollout_depth: usize,
    base_seed: u64,
) -> PyResult<Vec<(u8, f64, f64, f64, f64, f64, f64)>> {
    let hand_slice = hand.as_slice()?;
    let seen_slice = tiles_seen.as_slice()?;

    if hand_slice.len() < NUM_TILE_TYPES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("hand must have at least {} elements", NUM_TILE_TYPES)
        ));
    }
    if seen_slice.len() < NUM_TILE_TYPES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("tiles_seen must have at least {} elements", NUM_TILE_TYPES)
        ));
    }

    // Validate candidate tile indices are in range [0, 27)
    for &c in &candidates {
        if (c as usize) >= NUM_TILE_TYPES {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("candidate tile {} out of range [0, {})", c, NUM_TILE_TYPES)
            ));
        }
    }

    let mut h = [0u8; NUM_TILE_TYPES];
    h.copy_from_slice(&hand_slice[..NUM_TILE_TYPES]);
    let mut s = [0u8; NUM_TILE_TYPES];
    s.copy_from_slice(&seen_slice[..NUM_TILE_TYPES]);

    let dq = match ding_que {
        0 => Some(Suit::Man), 1 => Some(Suit::Pin), 2 => Some(Suit::Sou), _ => None,
    };

    let info = PlayerInfo { hand: h, melds_count, ding_que: dq, tiles_seen: s, wall_remaining };
    let config = IsmceConfig { num_worlds, rollout_depth, base_seed };

    // Build OpponentInfo
    let mut opponents: Vec<OpponentInfo> = Vec::new();
    for i in 0..3.min(opponent_ding_que.len()) {
        let opp_dq = match opponent_ding_que.get(i).copied().unwrap_or(-1) {
            0 => Some(Suit::Man), 1 => Some(Suit::Pin), 2 => Some(Suit::Sou), _ => None,
        };
        let discards_vec = opponent_discards.get(i).cloned().unwrap_or_default();
        let mc = opponent_meld_counts.get(i).copied().unwrap_or(0);
        let hand_count = 13u8.saturating_sub((2 * mc) as u8);
        opponents.push(OpponentInfo { ding_que: opp_dq, discards: discards_vec, melds_count: mc, hand_count });
    }

    // Build OpponentDangerInfo
    let mut danger_infos: [OpponentDangerInfo; 3] = [
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: 0 },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: 0 },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: 0 },
    ];
    for i in 0..3 {
        danger_infos[i].ding_que = match opponent_ding_que.get(i).copied().unwrap_or(-1) {
            0 => Some(Suit::Man), 1 => Some(Suit::Pin), 2 => Some(Suit::Sou), _ => None,
        };
        danger_infos[i].melds_count = opponent_meld_counts.get(i).copied().unwrap_or(0);
        danger_infos[i].discard_count = opponent_discard_counts.get(i).copied().unwrap_or(0);
    }

    let opp_disc: [Vec<u8>; 3] = [
        opponent_discards.get(0).cloned().unwrap_or_default(),
        opponent_discards.get(1).cloned().unwrap_or_default(),
        opponent_discards.get(2).cloned().unwrap_or_default(),
    ];

    // Convert opponent hand probs from Vec<Vec<f32>> to [[f32; 27]; 3]
    let mut probs = [[0.0f32; NUM_TILE_TYPES]; 3];
    for i in 0..3.min(opponent_hand_probs.len()) {
        for j in 0..NUM_TILE_TYPES.min(opponent_hand_probs[i].len()) {
            probs[i][j] = opponent_hand_probs[i][j];
        }
    }

    let results = ismce::evaluate_discards_informed(
        &info, &candidates, &opponents, &opp_disc, &danger_infos, &config, &probs,
    );

    Ok(results
        .iter()
        .map(|r| (r.tile, r.win_rate, r.tenpai_rate, r.avg_shanten_improvement,
                   r.expected_score, r.tenpai_value, r.danger_cost))
        .collect())
}

/// Compute danger scores for all 27 tile types (enhanced version).
///
/// Uses the enhanced danger scoring that considers opponent ding-que,
/// meld counts, and discard patterns.
///
/// Args:
///     tiles_seen: [u8; 27] tiles visible
///     opp1_discards, opp2_discards, opp3_discards: discard lists per opponent
///     wall_remaining: tiles left in wall
///     opp_ding_que: list of 3 ding-que values (-1=none, 0=Man, 1=Pin, 2=Sou)
///     opp_meld_counts: list of 3 meld counts
///     opp_discard_counts: list of 3 discard counts
///
/// Returns: [f32; 27] danger score per tile type.
#[pyfunction]
#[pyo3(signature = (tiles_seen, opp1_discards, opp2_discards, opp3_discards, wall_remaining,
                    opp_ding_que=None, opp_meld_counts=None, opp_discard_counts=None))]
pub fn ismce_danger<'py>(
    py: Python<'py>,
    tiles_seen: PyReadonlyArray1<u8>,
    opp1_discards: Vec<u8>,
    opp2_discards: Vec<u8>,
    opp3_discards: Vec<u8>,
    wall_remaining: usize,
    opp_ding_que: Option<Vec<i8>>,
    opp_meld_counts: Option<Vec<usize>>,
    opp_discard_counts: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let seen_slice = tiles_seen.as_slice()?;
    if seen_slice.len() < NUM_TILE_TYPES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("tiles_seen must have at least {} elements, got {}", NUM_TILE_TYPES, seen_slice.len())
        ));
    }
    let mut s = [0u8; NUM_TILE_TYPES];
    s.copy_from_slice(&seen_slice[..NUM_TILE_TYPES]);

    let opp_discards = [opp1_discards, opp2_discards, opp3_discards];

    // Build OpponentDangerInfo from optional parameters
    let mut danger_infos: [OpponentDangerInfo; 3] = [
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: opp_discards[0].len() },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: opp_discards[1].len() },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: opp_discards[2].len() },
    ];

    if let Some(ref dqs) = opp_ding_que {
        for i in 0..3.min(dqs.len()) {
            danger_infos[i].ding_que = match dqs[i] {
                0 => Some(Suit::Man),
                1 => Some(Suit::Pin),
                2 => Some(Suit::Sou),
                _ => None,
            };
        }
    }
    if let Some(ref mcs) = opp_meld_counts {
        for i in 0..3.min(mcs.len()) {
            danger_infos[i].melds_count = mcs[i];
        }
    }
    if let Some(ref dcs) = opp_discard_counts {
        for i in 0..3.min(dcs.len()) {
            danger_infos[i].discard_count = dcs[i];
        }
    }

    let danger = ismce::danger_scores_enhanced(&s, &opp_discards, wall_remaining, &danger_infos);

    Ok(PyArray1::from_vec_bound(py, danger.to_vec()))
}
