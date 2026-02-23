use crate::consts::*;
use crate::tile::{Tile, Suit};

pub type HandCounts = [u8; NUM_TILE_TYPES];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeldType {
    Pon(Tile),
    MinKan(Tile),
    AnKan(Tile),
    KaKan(Tile),
}

impl MeldType {
    pub fn tile(&self) -> Tile {
        match self {
            MeldType::Pon(t) | MeldType::MinKan(t) | MeldType::AnKan(t) | MeldType::KaKan(t) => *t,
        }
    }

    pub fn is_open(&self) -> bool {
        matches!(self, MeldType::Pon(_) | MeldType::MinKan(_) | MeldType::KaKan(_))
    }

    pub fn tile_count(&self) -> u8 {
        match self {
            MeldType::Pon(_) => 3,
            _ => 4,
        }
    }
}

pub fn add_tile(hand: &mut HandCounts, t: Tile) {
    debug_assert!(
        hand[t as usize] < COPIES_PER_TILE as u8,
        "add_tile: tile {} already at max count {}",
        t, hand[t as usize]
    );
    hand[t as usize] += 1;
}

pub fn remove_tile(hand: &mut HandCounts, t: Tile) {
    debug_assert!(hand[t as usize] > 0, "tile {} not in hand", t);
    hand[t as usize] -= 1;
}

pub fn total_tiles(hand: &HandCounts) -> u8 {
    hand.iter().sum()
}

pub fn has_suit_tiles(hand: &HandCounts, suit: Suit) -> bool {
    for i in suit.start()..suit.end() {
        if hand[i] > 0 {
            return true;
        }
    }
    false
}

pub fn suit_tile_count(hand: &HandCounts, suit: Suit) -> u8 {
    let mut count = 0u8;
    for i in suit.start()..suit.end() {
        count += hand[i];
    }
    count
}

/// Shanten calculation (recursive).
/// Returns -1 when complete, 0 for tenpai, 1 for iishanten, etc.
pub fn calc_shanten(hand: &HandCounts, num_melds: usize) -> i8 {
    let target = 4 - num_melds;
    let mut best = (target as i8) * 2; // worst case

    // Seven pairs (only if no melds); 4-of-a-kind counts as 2 pairs (龙七对)
    if num_melds == 0 {
        let total: u8 = hand.iter().sum();
        if total >= 13 {
            let pairs: i8 = hand.iter().map(|&c| (c / 2) as i8).sum();
            best = best.min(6 - pairs);
        }
    }

    let mut h = *hand;

    // Standard form with jantai (pair head)
    for head in 0..NUM_TILE_TYPES {
        if h[head] < 2 { continue; }
        h[head] -= 2;
        let mut local_best = best;
        scan_groups(&mut h, 0, target, 0, 0, &mut local_best, true);
        best = best.min(local_best);
        h[head] += 2;
    }

    // Standard form without jantai
    scan_groups(&mut h, 0, target, 0, 0, &mut best, false);

    best
}

fn scan_groups(
    hand: &mut HandCounts,
    start: usize,
    target: usize,
    mentsu: usize,
    tatsu: usize,
    best: &mut i8,
    has_jantai: bool,
) {
    if *best == -1 { return; }

    // Prune: more partial groups than we can use — cap tatsu and evaluate now.
    // eff_tatsu = tatsu.min(target - mentsu), so extra tatsu beyond (target - mentsu) are wasted.
    let remaining = target.saturating_sub(mentsu);
    if tatsu > remaining {
        let eff_tatsu = remaining;
        let s = (remaining as i8) * 2 - (eff_tatsu as i8) - if has_jantai { 1 } else { 0 };
        *best = (*best).min(s);
        return;
    }

    // Skip to next non-zero tile
    let mut pos = start;
    while pos < NUM_TILE_TYPES && hand[pos] == 0 {
        pos += 1;
    }

    // All tiles consumed → compute shanten
    if pos >= NUM_TILE_TYPES {
        let remaining = target.saturating_sub(mentsu);
        let eff_tatsu = tatsu.min(remaining);
        let s = (remaining as i8) * 2 - (eff_tatsu as i8) - if has_jantai { 1 } else { 0 };
        *best = (*best).min(s);
        return;
    }

    let same_suit = |a: usize, b: usize| a / TILES_PER_SUIT == b / TILES_PER_SUIT;

    // Kotsu (triplet)
    if hand[pos] >= 3 {
        hand[pos] -= 3;
        scan_groups(hand, pos, target, mentsu + 1, tatsu, best, has_jantai);
        hand[pos] += 3;
    }

    // Shuntsu (sequence)
    if pos + 2 < NUM_TILE_TYPES && same_suit(pos, pos + 2) && hand[pos + 1] > 0 && hand[pos + 2] > 0 {
        hand[pos] -= 1; hand[pos + 1] -= 1; hand[pos + 2] -= 1;
        scan_groups(hand, pos, target, mentsu + 1, tatsu, best, has_jantai);
        hand[pos] += 1; hand[pos + 1] += 1; hand[pos + 2] += 1;
    }

    // Toitsu (pair as tatsu)
    if hand[pos] >= 2 {
        hand[pos] -= 2;
        scan_groups(hand, pos, target, mentsu, tatsu + 1, best, has_jantai);
        hand[pos] += 2;
    }

    // Ryanmen/Penchan (consecutive pair)
    if pos + 1 < NUM_TILE_TYPES && same_suit(pos, pos + 1) && hand[pos + 1] > 0 {
        hand[pos] -= 1; hand[pos + 1] -= 1;
        scan_groups(hand, pos, target, mentsu, tatsu + 1, best, has_jantai);
        hand[pos] += 1; hand[pos + 1] += 1;
    }

    // Kanchan (skip one)
    if pos + 2 < NUM_TILE_TYPES && same_suit(pos, pos + 2) && hand[pos + 2] > 0 {
        hand[pos] -= 1; hand[pos + 2] -= 1;
        scan_groups(hand, pos, target, mentsu, tatsu + 1, best, has_jantai);
        hand[pos] += 1; hand[pos + 2] += 1;
    }

    // Isolate: leave this tile unused, move on
    hand[pos] -= 1;
    scan_groups(hand, pos, target, mentsu, tatsu, best, has_jantai);
    hand[pos] += 1;
}

pub fn is_complete(hand: &HandCounts, num_melds: usize) -> bool {
    calc_shanten(hand, num_melds) == -1
}

/// Get the tiles we are waiting for (only valid at tenpai i.e. shanten==0).
/// Uses the memoized shanten cache for performance.
pub fn waiting_tiles(hand: &HandCounts, num_melds: usize) -> Vec<Tile> {
    let mut waits = Vec::new();
    let mut h = *hand;
    for t in 0..NUM_TILE_TYPES as u8 {
        if h[t as usize] < 4 {
            h[t as usize] += 1;
            // Use memoized calc_shanten to avoid redundant recursive calls
            if crate::algo::shanten::calc_shanten(&h, num_melds) == -1 {
                waits.push(t);
            }
            h[t as usize] -= 1;
        }
    }
    waits
}
