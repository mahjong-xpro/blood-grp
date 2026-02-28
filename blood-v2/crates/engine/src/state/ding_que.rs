//! Centralized ding_que (定缺) helpers.
//!
//! All ding_que-related logic (tile filtering, validation, conversion)
//! is collected here to eliminate duplication across player, board,
//! agari, sp, and ismce modules.

use crate::tile::{Tile, Suit};
use crate::hand::{HandCounts, MeldType, has_suit_tiles, suit_tile_count};

/// Whether `t` belongs to the ding_que suit.
#[inline]
pub fn is_ding_que_tile(ding_que: Option<Suit>, t: Tile) -> bool {
    match ding_que {
        Some(suit) => Suit::from_tile(t) == suit,
        None => false,
    }
}

/// Whether the player must discard a ding_que-suit tile
/// (i.e. ding_que is set and hand still contains that suit).
#[inline]
pub fn must_discard_ding_que(hand: &HandCounts, ding_que: Option<Suit>) -> bool {
    ding_que.is_some_and(|s| has_suit_tiles(hand, s))
}

/// Whether ding_que is completed (hand contains no tiles of the ding_que suit).
/// Returns `false` if ding_que has not been chosen yet.
#[inline]
pub fn is_completed(hand: &HandCounts, ding_que: Option<Suit>) -> bool {
    match ding_que {
        Some(suit) => !has_suit_tiles(hand, suit),
        None => false,
    }
}

/// Number of ding_que-suit tiles remaining in hand.
#[inline]
pub fn remaining_count(hand: &HandCounts, ding_que: Option<Suit>) -> u8 {
    match ding_que {
        Some(suit) => suit_tile_count(hand, suit),
        None => 0,
    }
}

/// Validate that a winning hand (tehai + melds) contains no ding_que-suit tiles.
pub fn validate_win(tehai: &HandCounts, melds: &[MeldType], ding_que: Option<Suit>) -> bool {
    let Some(suit) = ding_que else { return true };
    if has_suit_tiles(tehai, suit) {
        return false;
    }
    for m in melds {
        if Suit::from_tile(m.tile()) == suit {
            return false;
        }
    }
    true
}

/// Convert an i8 value (from Python) to `Option<Suit>`.
/// 0 → Man, 1 → Pin, 2 → Sou, anything else → None.
#[inline]
pub fn suit_from_i8(v: i8) -> Option<Suit> {
    match v {
        0 => Some(Suit::Man),
        1 => Some(Suit::Pin),
        2 => Some(Suit::Sou),
        _ => None,
    }
}

/// Convert `Option<Suit>` to i8 for Python interop.
/// Man → 0, Pin → 1, Sou → 2, None → -1.
#[inline]
pub fn suit_to_i8(s: Option<Suit>) -> i8 {
    match s {
        Some(Suit::Man) => 0,
        Some(Suit::Pin) => 1,
        Some(Suit::Sou) => 2,
        None => -1,
    }
}
