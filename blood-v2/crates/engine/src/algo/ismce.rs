//! Information Set Monte Carlo Evaluation (ISMCE).
//!
//! 从信息集中采样一致的对手手牌，然后评估每个候选弃牌在 `depth` 回合内
//! 达到听牌/和牌状态的概率。
//!
//! 推理时用于优化策略网络的动作选择。

use crate::consts::*;
use crate::tile::{Tile, Suit};
use crate::hand::*;
use crate::algo::shanten::calc_shanten;
use crate::algo::point::calc_score;

/// Rollout 结果（替代原来的 (bool, bool) 元组）
#[derive(Debug, Clone, Copy)]
struct RolloutResult {
    won: bool,
    tenpai: bool,
    score: f64,       // 和牌时的期望得分
    tenpai_waits: usize, // 听牌时的待牌数（0 = 未听牌或已和牌）
}

/// 单个候选弃牌的评估结果
#[derive(Debug, Clone)]
pub struct IsmceScore {
    pub tile: Tile,
    pub win_rate: f64,
    pub expected_score: f64,          // 番数加权期望得分
    pub tenpai_rate: f64,
    pub tenpai_value: f64,            // 听牌质量 (待数×剩余轮归一化)
    pub danger_cost: f64,             // 防守代价
    pub avg_shanten_improvement: f64,
    // Legacy fields for backward compatibility
    pub win_rate_raw: f64,
}

/// ISMCE 评估配置
pub struct IsmceConfig {
    pub num_worlds: usize,
    pub rollout_depth: usize,
    pub base_seed: u64,
}

impl Default for IsmceConfig {
    fn default() -> Self {
        Self {
            num_worlds: 64,
            rollout_depth: 8,       // v2: 4 → 8 for deeper lookahead
            base_seed: 0,
        }
    }
}

/// 评估玩家已知的信息
pub struct PlayerInfo {
    pub hand: HandCounts,
    pub melds_count: usize,
    pub ding_que: Option<Suit>,
    pub tiles_seen: [u8; NUM_TILE_TYPES],
    pub wall_remaining: usize,
}

/// 对手信息，用于约束采样
pub struct OpponentInfo {
    /// 对手的定缺花色（如果已知）
    pub ding_que: Option<Suit>,
    /// 对手已打出的牌
    pub discards: Vec<Tile>,
    /// 对手的副露数量
    pub melds_count: usize,
    /// 对手需要的手牌数量
    pub hand_count: u8,
}

/// 约束采样：分配未见牌给对手时考虑定缺花色约束
///
/// 改进点：
/// - 对手已定缺的花色不分配给该对手（硬约束）
/// - 对手已打出的牌不分配给该对手（隐含在 tiles_seen 中）
/// - 剩余牌按约束分配给各对手和牌山
fn sample_world_constrained(
    info: &PlayerInfo,
    opponents: &[OpponentInfo],
    rng_seed: u64,
) -> (Vec<Vec<Tile>>, Vec<Tile>) {
    let mut rng = fastrand::Rng::with_seed(rng_seed);

    // 收集所有未见牌
    let mut pool: Vec<Tile> = Vec::new();
    for t in 0..NUM_TILE_TYPES {
        let seen = info.tiles_seen[t];
        let total = COPIES_PER_TILE as u8;
        let unseen = total.saturating_sub(seen);
        for _ in 0..unseen {
            pool.push(t as Tile);
        }
    }

    // Fisher-Yates 洗牌
    let n = pool.len();
    for i in (1..n).rev() {
        let j = rng.usize(..=i);
        pool.swap(i, j);
    }

    // 为每个对手分配手牌（考虑定缺约束）
    let mut opp_hands: Vec<Vec<Tile>> = opponents.iter().map(|_| Vec::new()).collect();
    let mut remaining_pool: Vec<Tile> = Vec::new();

    // 第一轮：为每个对手从池中挑选合法牌
    let mut used = vec![false; pool.len()];

    // Fix R10-M1: randomize opponent processing order to avoid systematic bias
    // (opponent 2 previously always got leftover tiles after 0 and 1's ding-que filters)
    let num_opp = opponents.len();
    let mut opp_order: Vec<usize> = (0..num_opp).collect();
    for i in (1..opp_order.len()).rev() {
        let j = rng.usize(..=i);
        opp_order.swap(i, j);
    }

    for &opp_idx in &opp_order {
        let opp = &opponents[opp_idx];
        let needed = opp.hand_count as usize;
        let mut assigned = 0usize;

        for (pool_idx, &tile) in pool.iter().enumerate() {
            if assigned >= needed { break; }
            if used[pool_idx] { continue; }

            // 定缺约束：不分配定缺花色的牌给该对手
            if let Some(dq_suit) = opp.ding_que {
                if Suit::from_tile(tile) == dq_suit {
                    continue;
                }
            }

            used[pool_idx] = true;
            opp_hands[opp_idx].push(tile);
            assigned += 1;
        }

        // 如果约束过强导致无法分配足够的牌，放宽约束从剩余牌中补充
        if assigned < needed {
            for (pool_idx, &tile) in pool.iter().enumerate() {
                if assigned >= needed { break; }
                if used[pool_idx] { continue; }
                used[pool_idx] = true;
                opp_hands[opp_idx].push(tile);
                assigned += 1;
            }
        }
    }

    // 剩余未分配的牌作为牌山
    for (pool_idx, &tile) in pool.iter().enumerate() {
        if !used[pool_idx] {
            remaining_pool.push(tile);
        }
    }

    (opp_hands, remaining_pool)
}

/// 信息引导的世界采样：使用对手手牌概率分布加权分配
///
/// 当 OpponentHandPredictor 提供了每个对手持有每种牌的概率时，
/// 优先将高概率的牌分配给对应对手。
/// 对手处理顺序随机化以避免系统性偏向 opponent 0。
pub fn sample_world_informed(
    info: &PlayerInfo,
    opponents: &[OpponentInfo],
    opponent_hand_probs: &[[f32; NUM_TILE_TYPES]; 3],
    rng_seed: u64,
) -> (Vec<Vec<Tile>>, Vec<Tile>) {
    let mut rng = fastrand::Rng::with_seed(rng_seed);

    // Collect unseen tiles
    let mut pool: Vec<Tile> = Vec::new();
    for t in 0..NUM_TILE_TYPES {
        let unseen = (COPIES_PER_TILE as u8).saturating_sub(info.tiles_seen[t]);
        for _ in 0..unseen {
            pool.push(t as Tile);
        }
    }

    // Shuffle pool
    let n = pool.len();
    for i in (1..n).rev() {
        let j = rng.usize(..=i);
        pool.swap(i, j);
    }

    let num_opp = opponents.len().min(3);
    let mut opp_hands: Vec<Vec<Tile>> = opponents.iter().map(|_| Vec::new()).collect();
    let mut used = vec![false; pool.len()];

    // Randomize opponent processing order to avoid systematic bias
    let mut opp_order: Vec<usize> = (0..num_opp).collect();
    for i in (1..opp_order.len()).rev() {
        let j = rng.usize(..=i);
        opp_order.swap(i, j);
    }

    // For each opponent (in random order), assign tiles weighted by predicted probability
    // Fix R10-H1: use Gumbel-max trick for weighted sampling instead of greedy top-k.
    // Greedy assignment collapses the predicted distribution to a single mode,
    // defeating the purpose of Monte Carlo sampling diversity.
    for &opp_idx in &opp_order {
        let opp = &opponents[opp_idx];
        let needed = opp.hand_count as usize;
        let probs = &opponent_hand_probs[opp_idx];

        // Build candidates with Gumbel-perturbed log-probabilities for weighted sampling
        let mut candidates: Vec<(usize, f32)> = pool.iter().enumerate()
            .filter(|(idx, _)| !used[*idx])
            .map(|(idx, &tile)| {
                let p = probs[tile as usize];
                // Apply ding-que constraint: zero probability for ding-que suit
                let p = if let Some(dq) = opp.ding_que {
                    if Suit::from_tile(tile) == dq { 0.0 } else { p }
                } else {
                    p
                };
                // Gumbel-max trick: add Gumbel noise to log(p) for weighted sampling
                // without replacement. For p=0, use -inf so these are never selected.
                let gumbel_score = if p > 1e-8 {
                    let u = (rng.u32(1..u32::MAX) as f32) / (u32::MAX as f32);
                    p.ln() - (-u.ln()).ln()
                } else {
                    f32::NEG_INFINITY
                };
                (idx, gumbel_score)
            })
            .collect();

        // Sort by Gumbel-perturbed score descending (top-k = weighted sample)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut assigned = 0usize;
        for (pool_idx, _prob) in &candidates {
            if assigned >= needed { break; }
            used[*pool_idx] = true;
            opp_hands[opp_idx].push(pool[*pool_idx]);
            assigned += 1;
        }

        // Fallback: if ding-que constraint left insufficient tiles, relax and fill from remaining
        if assigned < needed {
            for (pool_idx, &tile) in pool.iter().enumerate() {
                if assigned >= needed { break; }
                if used[pool_idx] { continue; }
                used[pool_idx] = true;
                opp_hands[opp_idx].push(tile);
                assigned += 1;
            }
        }
    }

    // Remaining tiles form the wall
    let remaining_pool: Vec<Tile> = pool.iter().enumerate()
        .filter(|(idx, _)| !used[*idx])
        .map(|(_, &tile)| tile)
        .collect();

    (opp_hands, remaining_pool)
}

/// 从信息集中采样一致的对手手牌配置（无约束版本，向后兼容）
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

    // Fisher-Yates 洗牌
    let n = pool.len();
    for i in (1..n).rev() {
        let j = rng.usize(..=i);
        pool.swap(i, j);
    }

    pool
}

/// 轻量番数估算：仅检查高频番種，避免完整 calc_fan 的开销。
///
/// 检查: 自摸(+1), 清一色(+2), 对对和(+1), 门清(+1), 七对子(+2), 根(+1 each)
/// 返回估计番数 (1-6)
fn estimate_fan_quick(hand: &HandCounts, melds_count: usize, is_tsumo: bool) -> u8 {
    let mut fan: u8 = 1; // base: 平胡

    // 自摸 +1 (per rules.md and agari.rs: tsumo is a separate yaku)
    if is_tsumo {
        fan += 1;
    }

    // 门清: no melds
    if melds_count == 0 {
        fan += 1;
    }

    // 清一色: all tiles in one suit
    let mut suits_present = [false; 3];
    for t in 0..NUM_TILE_TYPES {
        if hand[t] > 0 {
            suits_present[t / TILES_PER_SUIT] = true;
        }
    }
    let suit_count = suits_present.iter().filter(|&&s| s).count();
    if suit_count == 1 {
        fan += 2; // 清一色 = 2 fan (per rules.md and agari.rs)
    }

    // 七对子 (qidui): 7 pairs, no melds — +2 fan
    // Fix R10-H2: missing qidui caused 2-4x score underestimate for chitoi hands
    if melds_count == 0 {
        let mut is_chitoi = true;
        let mut pair_count = 0u8;
        for t in 0..NUM_TILE_TYPES {
            match hand[t] {
                0 => {},
                2 => pair_count += 1,
                4 => pair_count += 2,  // 4-of-a-kind counts as 2 pairs (龙七对)
                _ => { is_chitoi = false; break; }
            }
        }
        if is_chitoi && pair_count == 7 {
            fan += 2;
        }
    }

    // 根 (gen): 4-of-a-kind in hand — +1 fan each
    // Fix R10-M5: missing gen check
    for t in 0..NUM_TILE_TYPES {
        if hand[t] == 4 {
            fan += 1;
        }
    }

    // 对对和: all groups are triplets (no sequences), exactly one pair
    // Quick check: every tile count in hand must be 0, 2, or 3.
    // count=1 implies a sequence component → not toitoi.
    // count=4 is ambiguous (triplet+single or pair+pair) → skip toitoi.
    {
        let mut pairs = 0u8;
        let mut triplets = 0u8;
        let mut is_toitoi = true;
        for t in 0..NUM_TILE_TYPES {
            match hand[t] {
                0 => {},
                2 => pairs += 1,
                3 => triplets += 1,
                _ => { is_toitoi = false; break; }  // count 1 or 4 → not pure toitoi
            }
        }
        // Hand portion: triplets + 1 pair; total with melds: triplets + melds >= 4
        if is_toitoi && pairs == 1 && triplets + melds_count as u8 >= 4 {
            fan += 1;
        }
    }

    fan.min(MAX_FAN)
}

/// 使用 ISMCE 评估所有候选弃牌（基础版本，无约束采样，无防守）
///
/// 对每个弃牌候选，采样 `num_worlds` 个随机一致世界，
/// 模拟 `rollout_depth` 次摸牌来估计胜率和听牌率。
///
/// 推荐使用 [`evaluate_discards_full`] 获得完整的约束采样+防守评估。
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
            let mut total_score = 0.0f64;

            for world_idx in 0..config.num_worlds {
                // Seed mixing: wrapping_add(1) ensures non-zero input to the multiplier,
                // preventing base_seed=0 from producing deterministic (zero) seeds.
                let seed = config.base_seed.wrapping_add(1)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((discard as u64) * 100_000 + world_idx as u64);
                let wall = sample_world(info, seed);

                let result = simulate_draws_with_defense(
                    &hand_after, info.melds_count, info.ding_que, &wall, config.rollout_depth, None,
                );

                if result.won {
                    wins += 1;
                    total_score += result.score;
                }
                if result.tenpai {
                    tenpais += 1;
                }
            }

            let n = config.num_worlds as f64;
            IsmceScore {
                tile: discard,
                win_rate: wins as f64 / n,
                expected_score: total_score / n,
                tenpai_rate: tenpais as f64 / n,
                tenpai_value: 0.0,  // basic version doesn't compute tenpai quality
                danger_cost: 0.0,   // basic version doesn't compute danger
                avg_shanten_improvement: improvement,
                win_rate_raw: wins as f64 / n,
            }
        })
        .collect()
}

/// 使用约束采样的 ISMCE 评估（考虑对手定缺信息，无防守）
///
/// 与 evaluate_discards 相同，但采样时考虑对手的定缺花色约束，
/// 生成更真实的对手手牌分布。
///
/// 推荐使用 [`evaluate_discards_full`] 获得完整的约束采样+防守评估。
pub fn evaluate_discards_constrained(
    info: &PlayerInfo,
    candidates: &[Tile],
    opponents: &[OpponentInfo],
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
            let mut total_score = 0.0f64;

            for world_idx in 0..config.num_worlds {
                let seed = config.base_seed.wrapping_add(1)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((discard as u64) * 100_000 + world_idx as u64);

                let (_opp_hands, wall) = sample_world_constrained(info, opponents, seed);

                let result = simulate_draws_with_defense(
                    &hand_after, info.melds_count, info.ding_que, &wall, config.rollout_depth, None,
                );

                if result.won {
                    wins += 1;
                    total_score += result.score;
                }
                if result.tenpai {
                    tenpais += 1;
                }
            }

            let n = config.num_worlds as f64;
            IsmceScore {
                tile: discard,
                win_rate: wins as f64 / n,
                expected_score: total_score / n,
                tenpai_rate: tenpais as f64 / n,
                tenpai_value: 0.0,
                danger_cost: 0.0,
                avg_shanten_improvement: improvement,
                win_rate_raw: wins as f64 / n,
            }
        })
        .collect()
}

/// Full ISMCE evaluation with constrained sampling and defense-aware rollouts.
///
/// This is the recommended entry point that combines all three capabilities:
/// 1. Constrained world sampling (respects opponent ding-que)
///    — or informed sampling when `opponent_hand_probs` is provided
/// 2. Enhanced danger score computation
/// 3. Defense-aware rollout (uses danger as tiebreaker among equal-shanten discards)
///
/// When `opponents` is empty, falls back to unconstrained sampling.
/// When `opponent_danger_info` is empty, rollouts run without defense awareness.
/// When `opponent_hand_probs` is Some, uses probability-weighted sampling.
pub fn evaluate_discards_full(
    info: &PlayerInfo,
    candidates: &[Tile],
    opponents: &[OpponentInfo],
    opponent_discards: &[Vec<Tile>; 3],
    opponent_danger_info: &[OpponentDangerInfo; 3],
    config: &IsmceConfig,
) -> Vec<IsmceScore> {
    evaluate_discards_full_inner(info, candidates, opponents, opponent_discards,
                                 opponent_danger_info, config, None)
}

/// Full ISMCE evaluation with informed sampling from opponent hand predictions.
pub fn evaluate_discards_informed(
    info: &PlayerInfo,
    candidates: &[Tile],
    opponents: &[OpponentInfo],
    opponent_discards: &[Vec<Tile>; 3],
    opponent_danger_info: &[OpponentDangerInfo; 3],
    config: &IsmceConfig,
    opponent_hand_probs: &[[f32; NUM_TILE_TYPES]; 3],
) -> Vec<IsmceScore> {
    evaluate_discards_full_inner(info, candidates, opponents, opponent_discards,
                                 opponent_danger_info, config, Some(opponent_hand_probs))
}

fn evaluate_discards_full_inner(
    info: &PlayerInfo,
    candidates: &[Tile],
    opponents: &[OpponentInfo],
    opponent_discards: &[Vec<Tile>; 3],
    opponent_danger_info: &[OpponentDangerInfo; 3],
    config: &IsmceConfig,
    opponent_hand_probs: Option<&[[f32; NUM_TILE_TYPES]; 3]>,
) -> Vec<IsmceScore> {
    let base_shanten = calc_shanten(&info.hand, info.melds_count);

    // 1. Compute danger scores once using enhanced method
    let has_danger_info = opponent_danger_info.iter().any(|o| o.ding_que.is_some() || o.melds_count > 0 || o.discard_count > 0);
    let danger = if has_danger_info {
        Some(danger_scores_enhanced(
            &info.tiles_seen,
            opponent_discards,
            info.wall_remaining,
            opponent_danger_info,
        ))
    } else {
        None
    };

    // Estimate remaining turns for tenpai value calculation
    let remaining_turns = info.wall_remaining.min(MAX_TURNS) as f64;

    // 2. For each candidate discard, evaluate across sampled worlds
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
            let mut total_score = 0.0f64;
            let mut total_tenpai_waits = 0.0f64;

            for world_idx in 0..config.num_worlds {
                let seed = config.base_seed.wrapping_add(1)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((discard as u64) * 100_000 + world_idx as u64);

                // Use informed/constrained sampling if opponent info available
                let wall = if let Some(probs) = opponent_hand_probs {
                    let (_opp_hands, wall) = sample_world_informed(info, opponents, probs, seed);
                    wall
                } else if !opponents.is_empty() {
                    let (_opp_hands, wall) = sample_world_constrained(info, opponents, seed);
                    wall
                } else {
                    sample_world(info, seed)
                };

                // Use defense-aware rollout with opponent behavior model
                let rollout_seed = seed.wrapping_add(world_idx as u64 * 7919);
                let opp_info_ref = if has_danger_info { Some(opponent_danger_info) } else { None };
                let result = simulate_draws_with_opponents(
                    &hand_after,
                    info.melds_count,
                    info.ding_que,
                    &wall,
                    config.rollout_depth,
                    danger.as_ref(),
                    opp_info_ref,
                    rollout_seed,
                );

                if result.won {
                    wins += 1;
                    total_score += result.score;
                }
                if result.tenpai {
                    tenpais += 1;
                    // Use wait count from rollout terminal hand state (not hand_after)
                    total_tenpai_waits += result.tenpai_waits as f64;
                }
            }

            let n = config.num_worlds as f64;
            let win_rate = wins as f64 / n;
            let tenpai_rate = tenpais as f64 / n;

            // Tenpai value: average wait count × remaining turns (normalized)
            let avg_waits = if tenpais > 0 { total_tenpai_waits / tenpais as f64 } else { 0.0 };
            let tenpai_value = (avg_waits / NUM_TILE_TYPES as f64) * (remaining_turns / MAX_TURNS as f64);

            // Danger cost: danger score of the discard tile itself
            let danger_cost = danger.as_ref().map_or(0.0, |d| d[discard as usize] as f64);

            IsmceScore {
                tile: discard,
                win_rate,
                expected_score: total_score / n,
                tenpai_rate,
                tenpai_value,
                danger_cost,
                avg_shanten_improvement: improvement,
                win_rate_raw: win_rate,
            }
        })
        .collect()
}

/// 估算对手荣和弃牌的概率
///
/// 基于对手的危险度信息和弃牌的危险度评分，估算对手荣和的概率。
/// 对每个对手独立估算听牌概率，取最大值。
/// 使用非线性缩放：danger² 使高危险牌的放铳概率显著高于中等危险牌。
fn estimate_ron_probability(
    discard_tile: Tile,
    danger_tiles: Option<&[f32; NUM_TILE_TYPES]>,
    opponent_danger_info: Option<&[OpponentDangerInfo; 3]>,
) -> f32 {
    let danger = match danger_tiles {
        Some(d) => d[discard_tile as usize],
        None => return 0.0,
    };

    // Only consider ron if danger is non-trivial
    if danger < 0.2 {
        return 0.0;
    }

    // Per-opponent tenpai probability estimation
    let max_tenpai_prob = match opponent_danger_info {
        Some(infos) => {
            let mut max_p = 0.0f32;
            for o in infos.iter() {
                let est_shanten = (4.0 - o.melds_count as f32 - o.discard_count as f32 / 5.0).max(0.0);
                // Tenpai probability: continuous mapping from estimated shanten.
                // Use float comparison to avoid integer truncation bias
                // (e.g., est_shanten=0.9 should NOT map to tenpai_p=0.8).
                let tenpai_p = if est_shanten < 0.5 {
                    0.8     // very likely tenpai
                } else if est_shanten < 1.5 {
                    0.3     // possibly tenpai
                } else if est_shanten < 2.5 {
                    0.05    // unlikely
                } else {
                    0.0
                };
                max_p = max_p.max(tenpai_p);
            }
            max_p
        },
        None => if danger > 0.5 { 0.3 } else { 0.0 },
    };

    if max_tenpai_prob < 0.01 {
        return 0.0;
    }

    // Non-linear scaling: danger² × tenpai_prob × ceiling
    // danger=0.2 → 0.04, danger=0.5 → 0.25, danger=0.8 → 0.64, danger=1.0 → 1.0
    // Ceiling 0.45: max ron prob = 0.45 × 0.8 (tenpai) × 1.0 (danger²) = 0.36
    let ron_ceiling = 0.45;
    (danger * danger * max_tenpai_prob * ron_ceiling).min(0.5)
}

/// 带防守意识的摸牌模拟
///
/// `danger_tiles`: 可选的每张牌危险度评分 [0, 1]。
/// 在向听数相同的候选弃牌中，选择危险度最低的牌。
/// 当 danger_tiles 为 None 时退化为纯贪心向听最小化（向后兼容）。
///
/// 返回 RolloutResult，和牌时包含基于 estimate_fan_quick 的期望得分。
fn simulate_draws_with_defense(
    hand: &HandCounts,
    melds: usize,
    ding_que: Option<Suit>,
    wall: &[Tile],
    depth: usize,
    danger_tiles: Option<&[f32; NUM_TILE_TYPES]>,
) -> RolloutResult {
    simulate_draws_with_opponents(hand, melds, ding_que, wall, depth, danger_tiles, None, 0)
}

/// 带对手行为模型的摸牌模拟
///
/// 在弃牌后检查对手是否可能荣和，模拟放铳风险。
/// `rng_seed` 用于对手荣和的随机判定。
fn simulate_draws_with_opponents(
    hand: &HandCounts,
    melds: usize,
    ding_que: Option<Suit>,
    wall: &[Tile],
    depth: usize,
    danger_tiles: Option<&[f32; NUM_TILE_TYPES]>,
    opponent_danger_info: Option<&[OpponentDangerInfo; 3]>,
    rng_seed: u64,
) -> RolloutResult {
    let mut h = *hand;
    let max_draws = depth.min(wall.len());
    let mut rng = fastrand::Rng::with_seed(rng_seed);

    for i in 0..max_draws {
        let drawn = wall[i];
        add_tile(&mut h, drawn);

        if is_complete(&h, melds) {
            let complete_ok = match ding_que {
                Some(s) => !has_suit_tiles(&h, s),
                None => true,
            };
            if complete_ok {
                let fan = estimate_fan_quick(&h, melds, true);
                let score = calc_score(fan) as f64;
                return RolloutResult { won: true, tenpai: true, score, tenpai_waits: 0 };
            }
        }

        // 弃牌：优先弃定缺花色的牌，然后综合向听数+危险度选择最优弃牌
        // 初始化：默认弃摸到的牌，计算弃掉摸牌后的向听数作为基准
        let mut best_discard = drawn;
        let mut init_h = h;
        remove_tile(&mut init_h, drawn);
        let mut best_s = calc_shanten(&init_h, melds);
        let mut best_danger = danger_tiles.map_or(0.0f32, |d| d[drawn as usize]);

        // 阶段1：如果还有定缺花色的牌，必须弃其中一张
        let mut forced_dq = false;
        if let Some(suit) = ding_que {
            let start = suit.start();
            let end = suit.end();
            // First check if any ding-que tiles exist in hand
            let has_dq_tiles = (start..end).any(|t| h[t] > 0);
            if has_dq_tiles {
                forced_dq = true;
                // Reset baseline: must pick from ding-que tiles only.
                // Initialize to worst possible so any ding-que tile wins.
                best_s = i8::MAX;
                best_danger = f32::MAX;
                best_discard = start as Tile; // fallback to first ding-que position
                for t in start..end {
                    if h[t] == 0 { continue; }
                    let mut hh = h;
                    remove_tile(&mut hh, t as Tile);
                    let s = calc_shanten(&hh, melds);
                    let d = danger_tiles.map_or(0.0f32, |dt| dt[t]);
                    // 优先选择向听数更低的；向听数相同时选危险度更低的
                    if s < best_s || (s == best_s && d < best_danger) {
                        best_s = s;
                        best_discard = t as Tile;
                        best_danger = d;
                    }
                }
            }
        }

        // 阶段2：如果没有定缺花色的牌，从所有牌中选最优
        // 向听数优先，同向听数时选危险度最低的牌
        if !forced_dq {
            for t in 0..NUM_TILE_TYPES {
                if h[t] == 0 { continue; }
                let mut hh = h;
                remove_tile(&mut hh, t as Tile);
                let s = calc_shanten(&hh, melds);
                let d = danger_tiles.map_or(0.0f32, |dt| dt[t]);
                if s < best_s || (s == best_s && d < best_danger) {
                    best_s = s;
                    best_discard = t as Tile;
                    best_danger = d;
                }
            }
        }

        remove_tile(&mut h, best_discard);

        // A4: 简化对手行为模型 — 弃牌后检查对手是否可能荣和
        let ron_prob = estimate_ron_probability(best_discard, danger_tiles, opponent_danger_info);
        if ron_prob > 0.0 && rng.f32() < ron_prob {
            // 对手荣和：放铳惩罚。番数按危险度缩放：
            // 低危险(0.2) → 1番(1000), 中危险(0.5) → 2番(2000), 高危险(0.8+) → 3-4番(4000-8000)
            let danger_val = danger_tiles.map_or(0.5f32, |d| d[best_discard as usize]);
            let est_fan = if danger_val >= 0.8 { 4u8 }
                          else if danger_val >= 0.6 { 3 }
                          else if danger_val >= 0.4 { 2 }
                          else { 1 };
            let penalty_score = -(calc_score(est_fan) as f64);
            return RolloutResult { won: false, tenpai: false, score: penalty_score, tenpai_waits: 0 };
        }
    }

    let final_s = calc_shanten(&h, melds);
    let is_tenpai = final_s == 0;
    let waits = if is_tenpai {
        crate::algo::shanten::waiting_tiles(&h, melds).len()
    } else {
        0
    };
    RolloutResult { won: false, tenpai: is_tenpai, score: 0.0, tenpai_waits: waits }
}

/// 增强版危险度评分：综合考虑对手行为模型
///
/// 改进点：
/// - 考虑对手的副露数量（副露越多越危险）
/// - 考虑对手的定缺花色（定缺花色的牌对该对手无危险）
/// - 考虑对手的打牌模式（最近几巡打安全牌 → 可能听牌防守）
/// - 加入对手向听数的粗略估计（基于副露数和打牌回合数）
pub fn danger_scores(
    tiles_seen: &[u8; NUM_TILE_TYPES],
    opponent_discards: &[Vec<Tile>; 3],
    wall_remaining: usize,
) -> [f32; NUM_TILE_TYPES] {
    // 向后兼容：无额外信息时使用默认值
    let default_info = [
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: opponent_discards[0].len() },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: opponent_discards[1].len() },
        OpponentDangerInfo { ding_que: None, melds_count: 0, discard_count: opponent_discards[2].len() },
    ];
    danger_scores_enhanced(tiles_seen, opponent_discards, wall_remaining, &default_info)
}

/// 对手危险度计算所需的额外信息
pub struct OpponentDangerInfo {
    /// 对手的定缺花色
    pub ding_que: Option<Suit>,
    /// 对手的副露数量
    pub melds_count: usize,
    /// 对手已打出的牌数
    pub discard_count: usize,
}

/// 增强版危险度评分
///
/// 对每张牌计算综合危险度，考虑所有对手的状态。
/// 返回值范围 [0, 1]，越高越危险。
pub fn danger_scores_enhanced(
    tiles_seen: &[u8; NUM_TILE_TYPES],
    opponent_discards: &[Vec<Tile>; 3],
    wall_remaining: usize,
    opponent_info: &[OpponentDangerInfo; 3],
) -> [f32; NUM_TILE_TYPES] {
    let mut danger = [0.0f32; NUM_TILE_TYPES];

    for t in 0..NUM_TILE_TYPES {
        let total_copies = COPIES_PER_TILE as u8;
        let seen = tiles_seen[t];
        let unseen = total_copies.saturating_sub(seen) as f32;
        let tile_suit = Suit::from_tile(t as Tile);

        let mut max_opp_danger = 0.0f32;

        for (opp_idx, opp) in opponent_info.iter().enumerate() {
            let opp_discards = &opponent_discards[opp_idx];

            // ── 定缺花色检查：该花色对此对手无危险 ──
            if let Some(dq) = opp.ding_que {
                if tile_suit == dq {
                    // 对手定缺此花色，这张牌对该对手完全安全
                    continue;
                }
            }

            // ── 基础危险度：对手是否打过此牌 ──
            let discarded_by_opp = opp_discards.contains(&(t as Tile));
            let base_danger = if discarded_by_opp {
                // 对手打过此牌 → 大概率安全（现物）
                0.05 * unseen / total_copies as f32
            } else {
                // 对手未打过 → 有潜在危险
                0.25 + 0.15 * unseen / total_copies as f32
            };

            // ── 副露加成：副露越多，手牌越少，越可能听牌 ──
            // 0副露→0, 1副露→0.1, 2副露→0.2, 3副露→0.3, 4副露→0.35
            let meld_bonus = (opp.melds_count as f32 * 0.1).min(0.35);

            // ── 向听数粗略估计：基于副露数和打牌回合数 ──
            // 副露多+打牌多 → 向听数低 → 更危险
            // 粗略公式：estimated_shanten ≈ max(0, 4 - melds - discards/5)
            let est_shanten = (4.0 - opp.melds_count as f32
                - opp.discard_count as f32 / 5.0).max(0.0);
            // Fix R10-M3: use float thresholds instead of integer truncation.
            // `as u8` truncated 0.9 to 0 (tenpai), overestimating danger.
            // Consistent with estimate_ron_probability which uses 0.5/1.5/2.5.
            let shanten_danger = if est_shanten < 0.5 {
                0.3     // 估计听牌
            } else if est_shanten < 1.5 {
                0.15    // 估计一向听
            } else if est_shanten < 2.5 {
                0.05    // 估计二向听
            } else {
                0.0     // 三向听以上不加成
            };

            // ── 打牌模式分析：最近几巡是否在打安全牌 ──
            // 如果对手最近 3 巡都在打已见牌（现物），可能在防守/听牌
            let recent_safe_ratio = if opp_discards.len() >= 3 {
                let recent = &opp_discards[opp_discards.len().saturating_sub(3)..];
                let safe_count = recent.iter().filter(|&&tile| {
                    // "安全牌"：已被其他人打过的牌（可见牌数 >= 2）
                    tiles_seen[tile as usize] >= 2
                }).count();
                safe_count as f32 / recent.len() as f32
            } else {
                0.0
            };
            // 对手打安全牌比例高 → 可能在听牌防守 → 更危险
            let pattern_bonus = recent_safe_ratio * 0.1;

            let opp_danger = (base_danger + meld_bonus + shanten_danger + pattern_bonus).min(1.0);
            max_opp_danger = max_opp_danger.max(opp_danger);
        }

        danger[t] = max_opp_danger;

        // ── 终盘加成：牌山剩余少时整体更危险 ──
        if wall_remaining < 20 {
            let late_game_mult = 1.0 + (20.0 - wall_remaining as f32) / 20.0 * 0.5;
            danger[t] = (danger[t] * late_game_mult).min(1.0);
        }
    }

    danger
}
