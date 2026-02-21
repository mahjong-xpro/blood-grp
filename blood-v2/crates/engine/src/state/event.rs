use crate::tile::{Tile, Suit};

/// Game events that drive state transitions
#[derive(Debug, Clone)]
pub enum Event {
    DingQue { player: usize, suit: Suit },
    Draw { player: usize, tile: Tile },
    Discard { player: usize, tile: Tile, is_tsumogiri: bool },
    Pon { player: usize, from: usize, tile: Tile },
    MinKan { player: usize, from: usize, tile: Tile },
    AnKan { player: usize, tile: Tile },
    KaKan { player: usize, tile: Tile, is_jishiyu: bool },
    Tsumo { player: usize, tile: Tile },
    Ron { player: usize, from: usize, tile: Tile },
    KanPayment { payer: usize, receiver: usize, amount: i32 },
    GameEnd,
}
