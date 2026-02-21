use engine::hand::*;

fn build_hand(tiles: &[(u8, u8)]) -> HandCounts {
    let mut hand = [0u8; 27];
    for &(t, c) in tiles {
        hand[t as usize] = c;
    }
    hand
}

#[test]
fn test_complete_hand() {
    // 123m 456m 789m 123p 44p - complete
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (9, 1), (10, 1), (11, 1),
        (12, 2),
    ]);
    assert_eq!(calc_shanten(&hand, 0), -1);
    assert!(is_complete(&hand, 0));
}

#[test]
fn test_tenpai() {
    // 123m 456m 789m 12p 44p - tenpai waiting for 3p
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (9, 1), (10, 1),
        (12, 2),
    ]);
    assert_eq!(calc_shanten(&hand, 0), 0);
}

#[test]
fn test_iishanten() {
    // 12m 45m 78m 12p 44p (9 tiles, needs melds or more)
    let hand = build_hand(&[
        (0, 1), (1, 1),
        (3, 1), (4, 1),
        (6, 1), (7, 1),
        (9, 1), (10, 1),
        (12, 2),
    ]);
    // 10 tiles = 3 mentsu + 1 pair needed; but only has partial sequences
    let shanten = calc_shanten(&hand, 0);
    assert!(shanten >= 1);
}

#[test]
fn test_seven_pairs_complete() {
    let hand = build_hand(&[
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
    ]);
    assert_eq!(calc_shanten(&hand, 0), -1);
    assert!(is_complete(&hand, 0));
}

#[test]
fn test_seven_pairs_tenpai() {
    // 6 pairs + 1 single tile
    let hand = build_hand(&[
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 1), (7, 1),
    ]);
    let shanten = calc_shanten(&hand, 0);
    assert_eq!(shanten, 0); // one tile away from 7 pairs
}

#[test]
fn test_with_melds() {
    // With 1 meld (pon), hand has 10 tiles: 3 mentsu + pair needed from 10 tiles
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (12, 2),
    ]);
    assert_eq!(calc_shanten(&hand, 1), -1);
}

#[test]
fn test_waiting_tiles() {
    // 123m 456m 789m 12p 44p - waiting for 3p
    let hand = build_hand(&[
        (0, 1), (1, 1), (2, 1),
        (3, 1), (4, 1), (5, 1),
        (6, 1), (7, 1), (8, 1),
        (9, 1), (10, 1),
        (12, 2),
    ]);
    let waits = waiting_tiles(&hand, 0);
    assert!(waits.contains(&11)); // 3p
}

#[test]
fn test_empty_hand_with_four_melds() {
    // 4 melds + pair only
    let hand = build_hand(&[
        (0, 2), // pair only
    ]);
    assert_eq!(calc_shanten(&hand, 4), -1);
}

#[test]
fn test_seven_pairs_with_four_of_a_kind() {
    // 龙七对: 5 types at 2 + 1 type at 4 = 7 pairs, 14 tiles
    let hand = build_hand(&[
        (0, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
    ]);
    assert_eq!(calc_shanten(&hand, 0), -1);
    assert!(is_complete(&hand, 0));
}

#[test]
fn test_seven_pairs_with_four_of_a_kind_tenpai() {
    // 龙七对 tenpai: 4 types at 2 + 1 type at 4 + 1 single = 13 tiles
    let hand = build_hand(&[
        (0, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 1),
    ]);
    assert_eq!(calc_shanten(&hand, 0), 0);
    let waits = waiting_tiles(&hand, 0);
    assert!(waits.contains(&5));
}
