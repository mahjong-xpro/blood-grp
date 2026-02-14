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

/// 整手（手牌 + 副露）是否仍含定缺花色。用于和牌/花猪判定。
#[inline]
#[must_use]
pub fn has_ding_que_tiles_in_hand(
    tehai: &[u8; TILE_KIND_COUNT],
    pons: &[u8],
    minkans: &[u8],
    ankans: &[u8],
    ding_que: Option<Suit>,
) -> bool {
    if let Some(suit) = ding_que {
        if has_suit_tiles(tehai, suit) {
            return true;
        }
        for &t in pons.iter().chain(minkans.iter()).chain(ankans.iter()) {
            if (t as usize) < TILE_KIND_COUNT && is_ding_que_tile(t as usize, ding_que) {
                return true;
            }
        }
        false
    } else {
        false
    }
}

/// 整手（手牌 + 副露）无定缺花色时可和牌；否则为花猪不可和。
#[inline]
#[must_use]
pub fn can_agari_with_fuuro(
    tehai: &[u8; TILE_KIND_COUNT],
    pons: &[u8],
    minkans: &[u8],
    ankans: &[u8],
    ding_que: Option<Suit>,
) -> bool {
    !has_ding_que_tiles_in_hand(tehai, pons, minkans, ankans, ding_que)
}

#[inline]
#[must_use]
pub fn is_ding_que_tile(tile_id: usize, ding_que: Option<Suit>) -> bool {
    ding_que.is_some_and(|s| tile_suit_id(tile_id) == suit_id(s))
}

/// When the ding_que discard constraint is active (hand still holds ding_que suit tiles),
/// returns `Some((start, end))` — the inclusive-exclusive tile-id range of the ding_que suit
/// that the player is forced to discard from.
/// Returns `None` when unconstrained (no ding_que chosen, or all ding_que tiles already cleared).
#[inline]
#[must_use]
pub fn ding_que_forced_range(tehai: &[u8; TILE_KIND_COUNT], ding_que: Option<Suit>) -> Option<(usize, usize)> {
    ding_que.and_then(|s| has_suit_tiles(tehai, s).then(|| suit_range(s)))
}

/// DingQue discard rule (四川血战到底):
/// - If the hand currently contains any tiles of the chosen DingQue suit, the player MUST discard a tile of that suit.
/// - Otherwise (no DingQue-suit tiles in hand), there is no additional discard restriction from DingQue.
#[inline]
#[must_use]
pub fn discard_allowed(tile_id: usize, tehai: &[u8; TILE_KIND_COUNT], ding_que: Option<Suit>) -> bool {
    ding_que_forced_range(tehai, ding_que)
        .map_or(true, |(start, end)| tile_id >= start && tile_id < end)
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

    #[test]
    fn fuuro_ding_que_hand() {
        let mut tehai = [0u8; TILE_KIND_COUNT];
        tehai[9] = 2;
        tehai[10] = 2;
        tehai[11] = 2;
        let pons: &[u8] = &[0]; // 1m (index 0), 定缺 Man 时副露有万
        let minkans: &[u8] = &[];
        let ankans: &[u8] = &[];
        assert!(has_ding_que_tiles_in_hand(&tehai, pons, minkans, ankans, Some(Suit::Man)));
        assert!(!can_agari_with_fuuro(&tehai, pons, minkans, ankans, Some(Suit::Man)));
        let pons_no_man: &[u8] = &[9];
        assert!(!has_ding_que_tiles_in_hand(&tehai, pons_no_man, minkans, ankans, Some(Suit::Man)));
        assert!(can_agari_with_fuuro(&tehai, pons_no_man, minkans, ankans, Some(Suit::Man)));
    }
}
