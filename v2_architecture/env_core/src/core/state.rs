//! Core data structures for Bloody Battle Mahjong.

pub const NUM_PLAYERS: usize = 4;
pub const NUM_TILES: usize = 108; // 3 suits (man, pin, sou) x 9 ranks x 4 copies

/// Tile representations (0-26):
/// 0-8:   1-9 Man (万)
/// 9-17:  1-9 Pin (筒)
/// 18-26: 1-9 Sou (条)
pub type TileIdx = u8;

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub hand: [u8; 27], // Counts of each tile in hand
    pub melds: Vec<Meld>, // Pons, Kans
    pub score: i32,
    pub missing_suit: Option<u8>, // 0: man, 1: pin, 2: sou
    pub has_won: bool,
}

#[derive(Clone, Debug)]
pub enum Meld {
    Pon(TileIdx),
    MinKan(TileIdx), // Open Kan
    AnKan(TileIdx),  // Closed Kan
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            hand: [0; 27],
            melds: Vec::new(),
            score: 0,
            missing_suit: None,
            has_won: false,
        }
    }
}

pub enum Action {
    Discard(TileIdx),
    Pon,
    Kan, // Either open or closed depending on context
    Ron,
    Tsumo,
    Pass,
    DingQue(u8), // Select missing suit (0, 1, or 2)
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub players: [PlayerState; NUM_PLAYERS],
    pub wall: Vec<TileIdx>, // Remaining tiles
    pub current_player: usize,
    pub turn_count: usize,
    pub is_done: bool,
}

impl GameState {
    pub fn new() -> Self {
        // Initialize an empty game state
        Self {
            players: [
                PlayerState::default(),
                PlayerState::default(),
                PlayerState::default(),
                PlayerState::default(),
            ],
            wall: Vec::new(),
            current_player: 0,
            turn_count: 0,
            is_done: false,
        }
    }

    pub fn reset(&mut self) {
        for p in self.players.iter_mut() {
            p.hand = [0; 27];
            p.melds.clear();
            p.missing_suit = None;
            p.has_won = false;
        }
        self.is_done = false;
        self.turn_count = 0;
        
        // Generate and shuffle deck
        let mut deck = crate::core::deck::generate_deck();
        crate::core::deck::shuffle_deck(&mut deck);
        
        // Deal 13 tiles to each player
        for player_idx in 0..NUM_PLAYERS {
            for _ in 0..13 {
                if let Some(tile) = deck.pop() {
                    self.players[player_idx].hand[tile as usize] += 1;
                }
            }
        }
        
        // Dealer (current_player) gets the 14th tile
        self.current_player = 0; // In a real game, this rotates
        if let Some(tile) = deck.pop() {
            self.players[self.current_player].hand[tile as usize] += 1;
        }
        
        self.wall = deck;
    }
}
