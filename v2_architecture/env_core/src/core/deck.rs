use rand::seq::SliceRandom;
use rand::thread_rng;

/// Generates a full deck for Bloody Battle Mahjong.
/// 108 tiles total: 3 suits (Man 0..9, Pin 9..18, Sou 18..27), ranks 1-9, 4 copies each.
pub fn generate_deck() -> Vec<u8> {
    let mut deck = Vec::with_capacity(108);
    for suit_offset in [0, 9, 18] {
        for rank in 0..9 {
            let tile = suit_offset + rank;
            for _ in 0..4 {
                deck.push(tile);
            }
        }
    }
    deck
}

/// Shuffles the deck in place.
pub fn shuffle_deck(deck: &mut Vec<u8>) {
    let mut rng = thread_rng();
    deck.shuffle(&mut rng);
}
