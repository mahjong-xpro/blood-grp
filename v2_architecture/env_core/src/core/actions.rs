use crate::core::state::{GameState, Action, TileIdx, NUM_PLAYERS};

/// Returns all legal actions for the current player in the given GameState.
pub fn get_legal_actions(state: &GameState) -> Vec<Action> {
    let mut actions = Vec::new();

    // If game is done, no actions
    if state.is_done {
        return actions;
    }

    let current_p = &state.players[state.current_player];

    // Phase 1: DingQue (Missing Suit Selection)
    // If any player hasn't selected a missing suit, they MUST select it.
    // In Blood Mahjong, this happens at the start of the game simultaneously.
    if current_p.missing_suit.is_none() {
        actions.push(Action::DingQue(0)); // Man
        actions.push(Action::DingQue(1)); // Pin
        actions.push(Action::DingQue(2)); // Sou
        return actions;
    }

    // Phase 2: Normal Turn (Discard / Tsumo / Kan)
    // For now, we only implement Discarding. A player can discard any tile in their hand.
    // NOTE: In Bloody Battle Mahjong, you MUST discard your chosen missing suit first if you have any.
    let missing_suit = current_p.missing_suit.unwrap();
    let suit_start = (missing_suit * 9) as usize;
    let suit_end = suit_start + 9;

    let mut has_missing_suit = false;
    for i in suit_start..suit_end {
        if current_p.hand[i] > 0 {
            has_missing_suit = true;
            actions.push(Action::Discard(i as u8));
        }
    }

    // If they don't have the missing suit, they can discard anything they hold.
    if !has_missing_suit {
        for i in 0..27 {
            if current_p.hand[i] > 0 {
                actions.push(Action::Discard(i as u8));
            }
        }
    }

    actions
}
