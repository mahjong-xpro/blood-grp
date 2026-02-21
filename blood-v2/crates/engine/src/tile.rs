use crate::consts::*;

pub type Tile = u8; // 0..26

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Man = 0, // 万
    Pin = 1, // 筒
    Sou = 2, // 条
}

impl Suit {
    pub fn from_tile(t: Tile) -> Suit {
        debug_assert!((t as usize) < NUM_TILE_TYPES, "tile {} out of range", t);
        match t as usize / TILES_PER_SUIT {
            0 => Suit::Man,
            1 => Suit::Pin,
            2 => Suit::Sou,
            _ => unreachable!("invalid tile {}", t),
        }
    }

    pub fn from_index(i: usize) -> Suit {
        match i {
            0 => Suit::Man,
            1 => Suit::Pin,
            2 => Suit::Sou,
            _ => unreachable!("invalid suit index {}", i),
        }
    }

    pub fn start(self) -> usize {
        self as usize * TILES_PER_SUIT
    }

    pub fn end(self) -> usize {
        self.start() + TILES_PER_SUIT
    }

    pub fn rank(t: Tile) -> u8 {
        (t as usize % TILES_PER_SUIT) as u8 + 1 // 1-9
    }

    pub fn all() -> [Suit; 3] {
        [Suit::Man, Suit::Pin, Suit::Sou]
    }
}

pub fn make_tile(suit: Suit, rank: u8) -> Tile {
    debug_assert!((1..=9).contains(&rank));
    (suit as u8) * TILES_PER_SUIT as u8 + rank - 1
}

pub fn is_terminal(t: Tile) -> bool {
    let r = Suit::rank(t);
    r == 1 || r == 9
}

pub fn generate_deck(rng: &mut fastrand::Rng) -> Vec<Tile> {
    let mut deck: Vec<Tile> = Vec::with_capacity(TOTAL_TILES);
    for t in 0..NUM_TILE_TYPES as u8 {
        for _ in 0..COPIES_PER_TILE {
            deck.push(t);
        }
    }
    rng.shuffle(&mut deck);
    deck
}
