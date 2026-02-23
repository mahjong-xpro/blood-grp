use crate::consts::*;
use crate::tile::{Tile, Suit};

pub type HandCounts = [u8; NUM_TILE_TYPES];

// ── Lookup-table shanten (tomohxx algorithm) ─────────────────────────────────
//
// Ported from V1 libblood/src/algo/shanten.rs.
// The table encodes per-suit shanten contributions for all possible tile
// distributions within a 9-tile suit (base-5 index, max count 4 per tile).
// Three suits are combined with add_suhai() to get the final shanten.
//
// Reference: https://github.com/tomohxx/shanten-number-calculator/

use std::sync::LazyLock;
use std::io::Read;

const SUHAI_TABLE_SIZE: usize = 1_940_777;

static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    let gzipped = include_bytes!("algo/data/shanten_suhai.bin.gz");
    let mut gz = flate2::read::GzDecoder::new(gzipped.as_ref());
    let mut raw = Vec::new();
    gz.read_to_end(&mut raw).expect("shanten table decompress failed");

    let mut ret = Vec::with_capacity(SUHAI_TABLE_SIZE);
    let mut entry = [0u8; 10];
    for (i, b) in raw.into_iter().enumerate() {
        entry[i * 2 % 10] = b & 0b1111;
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), SUHAI_TABLE_SIZE, "shanten table size mismatch");
    ret
});

#[inline]
fn suit_index(tiles: &[u8]) -> usize {
    tiles.iter().fold(0usize, |acc, &x| acc * 5 + x as usize)
}

fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let m = m.min(4);
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();
    for j in (5..=(5 + m)).rev() {
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            let jk = j - k;
            sht = sht.min(lhs[k] + tab[jk]).min(lhs[jk] + tab[k]);
        }
        lhs[j] = sht;
    }
    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            sht = sht.min(lhs[k] + tab[j - k]);
        }
        lhs[j] = sht;
    }
}

fn calc_normal_table(hand: &HandCounts, len_div3: usize) -> i8 {
    let mut ret = SUHAI_TABLE
        .get(suit_index(&hand[..9]))
        .copied()
        .unwrap_or_default();
    add_suhai(&mut ret, suit_index(&hand[9..18]), len_div3);
    add_suhai(&mut ret, suit_index(&hand[18..27]), len_div3);
    (ret[5 + len_div3] as i8) - 1
}

fn calc_chitoi_table(hand: &HandCounts) -> i8 {
    let mut pairs: u8 = 0;
    let mut kinds: u8 = 0;
    let mut quads: u8 = 0;
    for &c in hand.iter().filter(|&&c| c > 0) {
        kinds += 1;
        if c >= 4 { pairs += 2; quads += 1; }
        else if c >= 2 { pairs += 1; }
    }
    let needed_kinds = 7u8.saturating_sub(quads);
    let redunct = needed_kinds.saturating_sub(kinds) as i8;
    7 - (pairs as i8) + redunct - 1
}

// ─────────────────────────────────────────────────────────────────────────────

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

/// Shanten calculation via lookup table (tomohxx algorithm).
/// Returns -1 when complete, 0 for tenpai, 1 for iishanten, etc.
pub fn calc_shanten(hand: &HandCounts, num_melds: usize) -> i8 {
    // len_div3 = number of complete groups possible from hand tiles alone.
    // hand tiles = total - melds*3; len_div3 = hand_tiles / 3.
    let hand_tiles: usize = hand.iter().map(|&c| c as usize).sum();
    let len_div3 = hand_tiles / 3;

    let normal = calc_normal_table(hand, len_div3);
    let chitoi = if num_melds == 0 && hand_tiles >= 13 {
        calc_chitoi_table(hand)
    } else {
        i8::MAX
    };
    normal.min(chitoi)
}

pub fn is_complete(hand: &HandCounts, num_melds: usize) -> bool {
    calc_shanten(hand, num_melds) == -1
}

/// Get the tiles we are waiting for (only valid at tenpai i.e. shanten==0).
pub fn waiting_tiles(hand: &HandCounts, num_melds: usize) -> Vec<Tile> {
    let mut waits = Vec::new();
    let mut h = *hand;
    for t in 0..NUM_TILE_TYPES as u8 {
        if h[t as usize] < 4 {
            h[t as usize] += 1;
            if calc_shanten(&h, num_melds) == -1 {
                waits.push(t);
            }
            h[t as usize] -= 1;
        }
    }
    waits
}
