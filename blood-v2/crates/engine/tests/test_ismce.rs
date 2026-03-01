//! Integration tests for ISMCE (Information Set Monte Carlo Evaluation).

use engine::consts::*;
use engine::tile::Suit;
use engine::hand::*;
use engine::algo::ismce::{evaluate_discards, IsmceConfig, PlayerInfo};

fn build_hand(tiles: &[(u8, u8)]) -> HandCounts {
    let mut hand = [0u8; NUM_TILE_TYPES];
    for &(t, c) in tiles {
        hand[t as usize] = c;
    }
    hand
}

/// Build a tiles_seen array from the hand (player sees their own tiles).
fn tiles_seen_from_hand(hand: &HandCounts) -> [u8; NUM_TILE_TYPES] {
    let mut seen = [0u8; NUM_TILE_TYPES];
    for t in 0..NUM_TILE_TYPES {
        seen[t] = hand[t];
    }
    seen
}

#[test]
fn test_ismce_basic_evaluate() {
    // Tenpai hand: 123m 456m 789m 12p + 44p (waiting for 3p)
    // 13 tiles in hand, need to evaluate which discard is best
    // Actually for evaluate_discards we need 14 tiles (just drew one).
    // Hand: 123m 456m 789m 12p 44p + drew 8p (tile 16)
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),   // 123m
        (3, 1), (4, 1), (5, 1),   // 456m
        (6, 1), (7, 1), (8, 1),   // 789m
        (9, 1), (10, 1),          // 12p
        (12, 2),                   // 44p
        (16, 1),                   // 8p (drew)
    ]);

    let tiles_seen = tiles_seen_from_hand(&hand);

    let info = PlayerInfo {
        hand,
        melds_count: 0,
        ding_que: Some(Suit::Sou),
        tiles_seen,
        wall_remaining: 40,
    };

    // Candidates: all tiles in hand that we could discard
    let candidates: Vec<u8> = (0..NUM_TILE_TYPES as u8)
        .filter(|&t| hand[t as usize] > 0)
        .collect();

    let config = IsmceConfig {
        num_worlds: 32,
        rollout_depth: 6,
        base_seed: 42,
    };

    let results = evaluate_discards(&info, &candidates, &config);

    // Results should be non-empty (one per candidate)
    assert!(!results.is_empty(), "evaluate_discards should return results");
    assert_eq!(results.len(), candidates.len());

    // At least one discard should have win_rate > 0 (hand is near tenpai)
    let max_win_rate = results.iter().map(|r| r.win_rate).fold(0.0f64, f64::max);
    assert!(
        max_win_rate > 0.0,
        "at least one discard should have positive win_rate, got max={}",
        max_win_rate
    );
}

#[test]
fn test_ismce_better_discard_ranked_higher() {
    // Hand: 123m 456m 789m 12p 44p + 9s (tile 26, ding_que suit)
    // Discarding 9s (useless ding_que tile) should rank higher than
    // discarding 1m (breaks a complete mentsu).
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),   // 123m
        (3, 1), (4, 1), (5, 1),   // 456m
        (6, 1), (7, 1), (8, 1),   // 789m
        (9, 1), (10, 1),          // 12p
        (12, 2),                   // 44p
        (26, 1),                   // 9s (ding_que suit tile)
    ]);

    let tiles_seen = tiles_seen_from_hand(&hand);

    let info = PlayerInfo {
        hand,
        melds_count: 0,
        ding_que: Some(Suit::Sou),
        tiles_seen,
        wall_remaining: 40,
    };

    // Compare discarding 9s (tile 26) vs discarding 1m (tile 0)
    let candidates = vec![0u8, 26u8]; // 1m vs 9s

    let config = IsmceConfig {
        num_worlds: 64,
        rollout_depth: 8,
        base_seed: 123,
    };

    let results = evaluate_discards(&info, &candidates, &config);
    assert_eq!(results.len(), 2);

    let score_discard_1m = &results[0]; // discarding 1m (tile 0)
    let score_discard_9s = &results[1]; // discarding 9s (tile 26)

    // Discarding 9s should be better: higher win_rate or better shanten improvement.
    // 9s is a ding_que tile that must be discarded anyway, and discarding it
    // preserves the near-tenpai structure. Discarding 1m breaks 123m.
    let value_1m = score_discard_1m.win_rate + score_discard_1m.tenpai_rate * 0.5
        + score_discard_1m.avg_shanten_improvement * 0.1;
    let value_9s = score_discard_9s.win_rate + score_discard_9s.tenpai_rate * 0.5
        + score_discard_9s.avg_shanten_improvement * 0.1;

    assert!(
        value_9s > value_1m,
        "discarding 9s (ding_que tile) should rank higher than discarding 1m (breaks mentsu): \
         9s value={:.4} (wr={:.4}, tr={:.4}, si={:.4}) vs \
         1m value={:.4} (wr={:.4}, tr={:.4}, si={:.4})",
        value_9s, score_discard_9s.win_rate, score_discard_9s.tenpai_rate,
        score_discard_9s.avg_shanten_improvement,
        value_1m, score_discard_1m.win_rate, score_discard_1m.tenpai_rate,
        score_discard_1m.avg_shanten_improvement,
    );
}
