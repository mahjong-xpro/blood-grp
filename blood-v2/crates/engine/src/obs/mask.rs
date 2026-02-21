use crate::consts::*;
use crate::state::board::BoardState;

/// Encode action mask as a 34-dim bool array
pub fn encode_action_mask(board: &BoardState, player_id: usize) -> [bool; ACTION_SPACE] {
    match board.get_decision_request(player_id) {
        Some(candidate) => candidate.to_mask(),
        None => [false; ACTION_SPACE],
    }
}
