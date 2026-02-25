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

/// 单个候选弃牌的评估结果
#[derive(Debug, Clone)]
pub struct IsmceScore {
    pub tile: Tile,
    pub win_rate: f64,
    pub tenpai_rate: f64,
    pub avg_shanten_improvement: f64,
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
            rollout_depth: 4,
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

    for (opp_idx, opp) in opponents.iter().enumerate() {
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

            for world_idx in 0..config.num_worlds {
                let seed = config.base_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((discard as u64) * 100_000 + world_idx as u64);

                // 使用约束采样，牌山部分用于模拟摸牌
                let (_opp_hands, wall) = sample_world_constrained(info, opponents, seed);

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

/// Full ISMCE evaluation with constrained sampling and defense-aware rollouts.
///
/// This is the recommended entry point that combines all three capabilities:
/// 1. Constrained world sampling (respects opponent ding-que)
/// 2. Enhanced danger score computation
/// 3. Defense-aware rollout (uses danger as tiebreaker among equal-shanten discards)
///
/// When `opponents` is empty, falls back to unconstrained sampling.
/// When `opponent_danger_info` is empty, rollouts run without defense awareness.
pub fn evaluate_discards_full(
    info: &PlayerInfo,
    candidates: &[Tile],
    opponents: &[OpponentInfo],
    opponent_discards: &[Vec<Tile>; 3],
    opponent_danger_info: &[OpponentDangerInfo; 3],
    config: &IsmceConfig,
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

            for world_idx in 0..config.num_worlds {
                let seed = config.base_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((discard as u64) * 100_000 + world_idx as u64);

                // Use constrained sampling if opponent info available
                let wall = if !opponents.is_empty() {
                    let (_opp_hands, wall) = sample_world_constrained(info, opponents, seed);
                    wall
                } else {
                    sample_world(info, seed)
                };

                // Use defense-aware rollout
                let (won, is_tenpai) = simulate_draws_with_defense(
                    &hand_after,
                    info.melds_count,
                    info.ding_que,
                    &wall,
                    config.rollout_depth,
                    danger.as_ref(),
                );

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

/// 模拟从牌山摸牌并检查听牌/和牌
///
/// 改进：加入基础防守意识。当 `danger_tiles` 非空时，在向听数相同的候选弃牌中
/// 优先选择低危险度的牌，避免在对手可能听牌时打出高危险牌。
/// 这比纯贪心向听最小化更接近真实玩家的行为模型。
fn simulate_draws(
    hand: &HandCounts,
    melds: usize,
    ding_que: Option<Suit>,
    wall: &[Tile],
    depth: usize,
) -> (bool, bool) {
    simulate_draws_with_defense(hand, melds, ding_que, wall, depth, None)
}

/// 带防守意识的摸牌模拟
///
/// `danger_tiles`: 可选的每张牌危险度评分 [0, 1]。
/// 在向听数相同的候选弃牌中，选择危险度最低的牌。
/// 当 danger_tiles 为 None 时退化为纯贪心向听最小化（向后兼容）。
fn simulate_draws_with_defense(
    hand: &HandCounts,
    melds: usize,
    ding_que: Option<Suit>,
    wall: &[Tile],
    depth: usize,
    danger_tiles: Option<&[f32; NUM_TILE_TYPES]>,
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

        // 弃牌：优先弃定缺花色的牌，然后综合向听数+危险度选择最优弃牌
        let mut best_discard = drawn;
        let mut best_s = calc_shanten(&h, melds);
        let mut best_danger = danger_tiles.map_or(0.0f32, |d| d[drawn as usize]);

        // 阶段1：如果还有定缺花色的牌，必须弃其中一张
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
                let d = danger_tiles.map_or(0.0f32, |dt| dt[t]);
                // 优先选择向听数更低的；向听数相同时选危险度更低的
                if s < best_s
                    || (s == best_s && d < best_danger)
                    || (s == best_s && Suit::from_tile(best_discard) != suit)
                {
                    best_s = s;
                    best_discard = t as Tile;
                    best_danger = d;
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
    }

    let final_s = calc_shanten(&h, melds);
    (false, final_s == 0)
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
            let shanten_danger = match est_shanten as u8 {
                0 => 0.3,     // 估计听牌
                1 => 0.15,    // 估计一向听
                2 => 0.05,    // 估计二向听
                _ => 0.0,     // 三向听以上不加成
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
