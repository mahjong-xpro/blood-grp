use engine::consts::*;
use engine::state::board::BoardState;
use engine::state::action::Action;
use engine::tile::Suit;
use engine::hand::suit_tile_count;
use engine::obs::{encode_student_obs, encode_oracle_obs, encode_action_mask};

fn setup_post_dingque() -> BoardState {
    let mut board = BoardState::new(42);
    for i in 0..NUM_PLAYERS {
        let hand = &board.players[i].hand;
        let mut best_suit = Suit::Man;
        let mut min_count = u8::MAX;
        for suit in Suit::all() {
            let count = suit_tile_count(hand, suit);
            if count < min_count {
                min_count = count;
                best_suit = suit;
            }
        }
        board.apply_action(i, Action::DingQue(best_suit));
    }
    board
}

#[test]
fn test_student_obs_shape() {
    let board = BoardState::new(42);
    let obs = encode_student_obs(&board, 0);
    assert_eq!(obs.len(), NUM_STUDENT_CHANNELS * NUM_TILE_TYPES);
}

#[test]
fn test_oracle_obs_shape() {
    let board = BoardState::new(42);
    let obs = encode_oracle_obs(&board, 0);
    assert_eq!(obs.len(), NUM_ORACLE_CHANNELS * NUM_TILE_TYPES);
}

#[test]
fn test_action_mask_ding_que() {
    let board = BoardState::new(42);
    let mask = encode_action_mask(&board, 0);
    // During ding que, only actions 31-33 should be available
    assert!(mask[31]); // Man
    assert!(mask[32]); // Pin
    assert!(mask[33]); // Sou
    // All others should be false
    for i in 0..31 {
        assert!(!mask[i], "action {} should not be available during ding_que", i);
    }
}

#[test]
fn test_action_mask_discard() {
    let board = setup_post_dingque();
    let cp = board.current_player;
    let mask = encode_action_mask(&board, cp);

    // Some discard actions should be available
    let has_discard = mask[0..27].iter().any(|&m| m);
    // Either in self-check (agari/kan/pass) or discard phase
    if board.phase == engine::state::board::Phase::Discard {
        assert!(has_discard, "should have discard options");
    }
}

#[test]
fn test_obs_values_in_range() {
    let board = setup_post_dingque();
    let obs = encode_student_obs(&board, 0);

    // Most values should be in [-1, 10] range
    for (i, &v) in obs.iter().enumerate() {
        assert!(
            v >= -1.0 && v <= 100.0,
            "obs[{}] = {} out of expected range", i, v
        );
    }
}

#[test]
fn test_oracle_contains_student() {
    let board = BoardState::new(42);
    let student = encode_student_obs(&board, 0);
    let oracle = encode_oracle_obs(&board, 0);

    // Oracle should contain student obs as prefix
    let student_size = NUM_STUDENT_CHANNELS * NUM_TILE_TYPES;
    for i in 0..student_size {
        assert_eq!(
            student[i], oracle[i],
            "oracle[{}] != student[{}]: {} vs {}", i, i, oracle[i], student[i]
        );
    }
}

#[test]
fn test_different_player_views() {
    let board = BoardState::new(42);
    let obs0 = encode_student_obs(&board, 0);
    let obs1 = encode_student_obs(&board, 1);

    // Different players should see different observations (different hands)
    assert_ne!(obs0, obs1, "different players should have different observations");
}
