use engine::consts::*;
use engine::state::board::{BoardState, Phase};
use engine::state::action::Action;
use engine::tile::Suit;
use engine::hand::*;

#[test]
fn test_new_game_state() {
    let board = BoardState::new(42);
    assert_eq!(board.phase, Phase::DingQue);
    for i in 0..NUM_PLAYERS {
        let hand_count = total_tiles(&board.players[i].hand);
        if i == board.dealer {
            assert_eq!(hand_count, 14, "dealer should have 14 tiles");
        } else {
            assert_eq!(hand_count, 13, "non-dealer should have 13 tiles");
        }
        assert_eq!(board.players[i].score, INITIAL_SCORE);
    }
}

#[test]
fn test_ding_que_phase() {
    let mut board = BoardState::new(42);
    assert_eq!(board.phase, Phase::DingQue);

    for i in 0..NUM_PLAYERS {
        assert!(board.players[i].ding_que.is_none());
        // Choose suit with fewest tiles
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

    assert!(board.all_ding_que_done());
    // After all ding que, should move to self check (dealer has 14 tiles)
    assert!(board.phase == Phase::SelfCheck || board.phase == Phase::Discard,
        "after ding_que, phase should be SelfCheck or Discard, got {:?}", board.phase);
}

#[test]
fn test_initial_scores() {
    let board = BoardState::new(123);
    for i in 0..NUM_PLAYERS {
        assert_eq!(board.players[i].score, 100000);
    }
}

#[test]
fn test_wall_count() {
    let board = BoardState::new(42);
    // 108 total - 13*4 - 1 (dealer extra) = 108 - 53 = 55
    assert_eq!(board.wall_remaining(), 55);
}

#[test]
fn test_full_game_simulation() {
    // Play a full game with simple bot logic
    let mut board = BoardState::new(99);

    // Ding que
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

    // Play until done (max 500 actions to prevent infinite loop)
    let mut steps = 0;
    while !board.is_done() && board.phase != Phase::Scoring && steps < 500 {
        steps += 1;
        let cp = board.current_player;

        match board.phase {
            Phase::SelfCheck => {
                if let Some(ac) = board.get_decision_request(cp) {
                    if ac.can_agari {
                        board.apply_action(cp, Action::Agari);
                        continue;
                    }
                }
                board.apply_action(cp, Action::Pass);
            }
            Phase::Discard => {
                let candidates = board.players[cp].discard_candidates();
                if !candidates.is_empty() {
                    board.apply_action(cp, Action::Discard(candidates[0]));
                }
            }
            Phase::Reaction => {
                for i in 0..NUM_PLAYERS {
                    if board.reaction_pending[i] {
                        board.apply_action(i, Action::Pass);
                    }
                }
            }
            _ => break,
        }
    }

    if board.phase == Phase::Scoring {
        board.finalize_scoring();
    }

    // Verify game ended properly
    let total_score: i32 = board.get_scores().iter().sum();
    // Total score should be preserved (zero-sum with initial 100000 each = 400000)
    assert_eq!(total_score, 400000, "total score should be preserved, got {}", total_score);
}

#[test]
fn test_decision_request() {
    let mut board = BoardState::new(42);
    // During ding que, each player should get a request
    for i in 0..NUM_PLAYERS {
        let req = board.get_decision_request(i);
        assert!(req.is_some(), "player {} should have ding_que request", i);
        let ac = req.unwrap();
        assert!(ac.can_ding_que);
    }
}

#[test]
fn test_multiple_games_deterministic() {
    let board1 = BoardState::new(42);
    let board2 = BoardState::new(42);
    assert_eq!(board1.wall, board2.wall);
    assert_eq!(board1.dealer, board2.dealer);
    for i in 0..NUM_PLAYERS {
        assert_eq!(board1.players[i].hand, board2.players[i].hand);
    }
}
