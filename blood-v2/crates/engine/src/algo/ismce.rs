//! Information Set Monte Carlo Evaluation (ISMCE).
//!
//! Samples consistent opponent hands from the information set, then
//! evaluates each candidate discard by scoring how often it leads to
//! a tenpai / win state within `depth` turns.
//!
//! Used at inference time to refine the policy network's action selection.

use crate::consts::*;
use crate::tile::{Tile, Suit};
use crate::hand::*;
use crate::algo::shanten::calc_shanten;

/// Result for a single candidate discard.
#[derive(Debug, Clone)]
pub struct IsmceScore {
    pub tile: Tile,
    pub win_rate: f64,
    pub tenpai_rate: f64,
    pub avg_shanten_improvement: f64,
}

/// Configuration for ISMCE evaluation.
pub struct IsmceConfig {
    pub num_worlds: usize,
    pub rollout_depth: usize,
    pub base_seed: u64,
}

impl Default for IsmceConfig {
    fn default() -> Self {
        Self {
            num_worlds: 64,
            rollout_depth: 4,
            base_seed: 0,
        }
    }
}

/// Information known to the evaluating player.
pub struct PlayerInfo {
    pub hand: HandCounts,
    pub melds_count: usize,
    pub ding_que: Option<Suit>,
    pub tiles_seen: [u8; NUM_TILE_TYPES],
    pub wall_remaining: usize,
}

/// Sample a consistent opponent hand configuration from the information set.
///
/// Distributes unseen tiles randomly among opponents and the remaining wall.
fn sample_world(info: &PlayerInfo, rng_seed: u64) -> Vec<Tile> {
    let mut rng = fastrand::Rng::with_seed(rng_seed);

    let mut pool: Vec<Tile> = Vec::new();
    for t in 0..NUM_TILE_TYPES {
        let seen = info.tiles_seen[t];
        let total = COPIES_PER_TILE as u8;
        let unseen = total.saturating_sub(seen);
        for _ in 0..unseen {
            pool.push(t as Tile);
        }
    }

    // Fisher-Yates shuffle
    let n = pool.len();
    for i in (1..n).rev() {
        let j = rng.usize(..=i);
        pool.swap(i, j);
    }

    pool
}

/// Evaluate all candidate discards using ISMCE.
///
/// For each discard candidate, we sample `num_worlds` random consistent
/// worlds and simulate `rollout_depth` draws to estimate win probability
/// and tenpai rate.
pub fn evaluate_discards(
    info: &PlayerInfo,
    candidates: &[Tile],
    config: &IsmceConfig,
) -> Vec<IsmceScore> {
    let base_shanten = calc_shanten(&info.hand, info.melds_count);

    candidates
        .iter()
        .map(|&discard| {
            let mut hand_after = info.hand;
            remove_tile(&mut hand_after, discard);

            let shanten_after = calc_shanten(&hand_after, info.melds_count);
            let improvement = (base_shanten as f64 - shanten_after as f64)
                .max(-5.0)
                .min(5.0);

            let mut wins = 0u64;
            let mut tenpais = 0u64;

            for world_idx in 0..config.num_worlds {
                let seed = config.base_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((discard as u64) * 100_000 + world_idx as u64);
                let wall = sample_world(info, seed);

                let (won, is_tenpai) =
                    simulate_draws(&hand_after, info.melds_count, info.ding_que, &wall, config.rollout_depth);

                if won {
                    wins += 1;
                }
                if is_tenpai {
                    tenpais += 1;
                }
            }

            let n = config.num_worlds as f64;
            IsmceScore {
                tile: discard,
                win_rate: wins as f64 / n,
                tenpai_rate: tenpais as f64 / n,
                avg_shanten_improvement: improvement,
            }
        })
        .collect()
}

/// Simulate random draws from the wall and check for tenpai/agari.
fn simulate_draws(
    hand: &HandCounts,
    melds: usize,
    ding_que: Option<Suit>,
    wall: &[Tile],
    depth: usize,
) -> (bool, bool) {
    let mut h = *hand;
    let max_draws = depth.min(wall.len());

    for i in 0..max_draws {
        let drawn = wall[i];
        add_tile(&mut h, drawn);

        if is_complete(&h, melds) {
            let complete_ok = match ding_que {
                Some(s) => !has_suit_tiles(&h, s),
                None => true,
            };
            if complete_ok {
                return (true, true);
            }
        }

        // Discard: must discard ding-que suit tiles first, then greedy shanten
        let mut best_discard = drawn;
        let mut best_s = calc_shanten(&h, melds);

        // Phase 1: if any ding-que tiles remain, must discard one of those
        let mut forced_dq = false;
        if let Some(suit) = ding_que {
            let start = suit.start();
            let end = suit.end();
            for t in start..end {
                if h[t] == 0 { continue; }
                forced_dq = true;
                let mut hh = h;
                remove_tile(&mut hh, t as Tile);
                let s = calc_shanten(&hh, melds);
                if s < best_s || (s == best_s && Suit::from_tile(best_discard) != suit) {
                    best_s = s;
                    best_discard = t as Tile;
                }
            }
        }

        // Phase 2: if no ding-que tiles remain, pick best among all tiles
        if !forced_dq {
            for t in 0..NUM_TILE_TYPES {
                if h[t] == 0 { continue; }
                let mut hh = h;
                remove_tile(&mut hh, t as Tile);
                let s = calc_shanten(&hh, melds);
                if s < best_s {
                    best_s = s;
                    best_discard = t as Tile;
                }
            }
        }

        remove_tile(&mut h, best_discard);
    }

    let final_s = calc_shanten(&h, melds);
    (false, final_s == 0)
}

/// Compute danger scores for each tile based on ISMCE opponent modeling.
///
/// For each tile that the agent might discard, estimates how likely
/// each opponent is to win on that tile.
pub fn danger_scores(
    tiles_seen: &[u8; NUM_TILE_TYPES],
    opponent_discards: &[Vec<Tile>; 3],
    wall_remaining: usize,
) -> [f32; NUM_TILE_TYPES] {
    let mut danger = [0.0f32; NUM_TILE_TYPES];

    // Heuristic: tiles that no opponent has discarded and are scarce
    // in the visible pool are more dangerous (opponents may be waiting)
    for t in 0..NUM_TILE_TYPES {
        let total_copies = COPIES_PER_TILE as u8;
        let seen = tiles_seen[t];
        let unseen = total_copies.saturating_sub(seen) as f32;

        let discarded_by_any = opponent_discards
            .iter()
            .any(|ds| ds.contains(&(t as Tile)));

        if discarded_by_any {
            // Opponents discarded this → likely safe
            danger[t] = 0.1 * unseen / total_copies as f32;
        } else {
            // No one discarded → potentially dangerous
            danger[t] = 0.3 + 0.2 * unseen / total_copies as f32;
        }

        // Late-game tiles are more dangerous
        if wall_remaining < 20 {
            danger[t] *= 1.5;
        }
    }

    danger
}
