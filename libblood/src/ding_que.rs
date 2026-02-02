use crate::mjai::Suit;

/// Tile kind count for Bloody Battle (no honors): 3 suits × 9 numbers.
pub const TILE_KIND_COUNT: usize = 27;

#[inline]
#[must_use]
pub const fn suit_id(suit: Suit) -> usize {
    match suit {
        Suit::Man => 0,
        Suit::Pin => 1,
        Suit::Sou => 2,
    }
}

/// Returns the inclusive start and exclusive end of the tile-id range for the suit.
#[inline]
#[must_use]
pub const fn suit_range(suit: Suit) -> (usize, usize) {
    match suit {
        Suit::Man => (0, 9),
        Suit::Pin => (9, 18),
        Suit::Sou => (18, 27),
    }
}

#[inline]
#[must_use]
pub const fn tile_suit_id(tile_id: usize) -> usize {
    tile_id / 9
}

#[inline]
#[must_use]
pub fn has_suit_tiles(tehai: &[u8; TILE_KIND_COUNT], suit: Suit) -> bool {
    let (start, end) = suit_range(suit);
    (start..end).any(|i| tehai[i] > 0)
}

#[inline]
#[must_use]
pub fn has_ding_que_tiles(tehai: &[u8; TILE_KIND_COUNT], ding_que: Option<Suit>) -> bool {
    ding_que.is_some_and(|s| has_suit_tiles(tehai, s))
}

#[inline]
#[must_use]
pub fn is_ding_que_tile(tile_id: usize, ding_que: Option<Suit>) -> bool {
    ding_que.is_some_and(|s| tile_suit_id(tile_id) == suit_id(s))
}

/// DingQue discard rule (四川血战到底):
/// - If the hand currently contains any tiles of the chosen DingQue suit, the player MUST discard a tile of that suit.
/// - Otherwise (no DingQue-suit tiles in hand), there is no additional discard restriction from DingQue.
#[inline]
#[must_use]
pub fn discard_allowed(tile_id: usize, tehai: &[u8; TILE_KIND_COUNT], ding_que: Option<Suit>) -> bool {
    let Some(suit) = ding_que else {
        // If DingQue hasn't been chosen, the game flow should not allow discarding at all.
        // Keep this permissive here; caller should gate with can_discard.
        return true;
    };

    if has_suit_tiles(tehai, suit) {
        tile_suit_id(tile_id) == suit_id(suit)
    } else {
        true
    }
}

/// DingQue win rule (花猪): cannot agari if any DingQue-suit tile remains in hand.
#[inline]
#[must_use]
pub fn can_agari(tehai: &[u8; TILE_KIND_COUNT], ding_que: Option<Suit>) -> bool {
    !has_ding_que_tiles(tehai, ding_que)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn discard_rule_only_when_have() {
        let mut tehai = [0u8; TILE_KIND_COUNT];
        // Have 1m (ding_que Man) and 1p.
        tehai[0] = 1;
        tehai[9] = 1;
        assert!(discard_allowed(0, &tehai, Some(Suit::Man)));
        assert!(!discard_allowed(9, &tehai, Some(Suit::Man)));

        // After clearing Man tiles, anything is allowed.
        tehai[0] = 0;
        assert!(discard_allowed(9, &tehai, Some(Suit::Man)));
    }
}
