use engine::consts::*;
use engine::tile::Suit;
use engine::hand::*;
use engine::algo::sp::{SPCalculator, SPInitState};

fn build_hand(tiles: &[(u8, u8)]) -> HandCounts {
    let mut hand = [0u8; 27];
    for &(t, c) in tiles {
        hand[t as usize] = c;
    }
    hand
}

#[test]
fn test_sp_tenpai() {
    // Tenpai hand: 123m 456m 789m 12p 44p - waiting for 3p
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (9, 1), (10, 1),
        (12, 2),
    ]);
    let tiles_seen = [0u8; NUM_TILE_TYPES]; // simplified: no tiles seen
    let init = SPInitState {
        tehai: hand,
        tiles_seen,
        tiles_left: 40,
        num_melds: 0,
        ding_que: Some(Suit::Sou),
    };

    let calc = SPCalculator::new(0, Some(Suit::Sou));
    let candidates = calc.calc(&init);

    // Should produce at least one candidate (the tenpai itself)
    assert!(!candidates.is_empty(), "tenpai hand should produce SP candidates");

    // Win probs should be > 0 for first turns
    let c = &candidates[0];
    assert!(c.win_probs[0] > 0.0, "should have nonzero win prob at turn 0");
}

#[test]
fn test_sp_complete() {
    // Complete hand: should return empty (already won)
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (9, 1), (10, 1), (11, 1),
        (12, 2),
    ]);
    let init = SPInitState {
        tehai: hand,
        tiles_seen: [0u8; NUM_TILE_TYPES],
        tiles_left: 40,
        num_melds: 0,
        ding_que: Some(Suit::Sou),
    };

    let calc = SPCalculator::new(0, Some(Suit::Sou));
    let candidates = calc.calc(&init);
    assert!(candidates.is_empty(), "complete hand should have no discard candidates");
}

#[test]
fn test_sp_discard_candidates() {
    // Iishanten hand: needs improvement
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1),
        (9, 1), (10, 1),
        (12, 2), (13, 1),
    ]);
    let init = SPInitState {
        tehai: hand,
        tiles_seen: [0u8; NUM_TILE_TYPES],
        tiles_left: 40,
        num_melds: 0,
        ding_que: Some(Suit::Sou),
    };

    let calc = SPCalculator::new(0, Some(Suit::Sou));
    let candidates = calc.calc(&init);

    // Should have some discard candidates
    assert!(!candidates.is_empty());

    // Candidates should be sorted by total EV (descending) — matches calc.rs sort order
    for i in 1..candidates.len() {
        assert!(
            candidates[i - 1].total_ev() >= candidates[i].total_ev(),
            "candidates should be sorted by total EV descending"
        );
    }
}

#[test]
fn test_sp_respects_ding_que() {
    // Hand with sou tiles, ding_que is sou
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (9, 1), (10, 1),
        (12, 2),
    ]);
    let init = SPInitState {
        tehai: hand,
        tiles_seen: [0u8; NUM_TILE_TYPES],
        tiles_left: 40,
        num_melds: 0,
        ding_que: Some(Suit::Sou),
    };

    let calc = SPCalculator::new(0, Some(Suit::Sou));
    let candidates = calc.calc(&init);

    // No candidate should be a sou tile
    for c in &candidates {
        let suit = Suit::from_tile(c.tile);
        assert_ne!(suit, Suit::Sou, "should not suggest discarding ding que suit");
    }
}
