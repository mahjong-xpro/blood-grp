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
    let obs = encode_oracle_obs(&board, 0, None);
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
    let oracle = encode_oracle_obs(&board, 0, None);

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

// --- Test: Hand channel values (H9) ---

#[test]
fn test_hand_channel_values() {
    // Create a board and verify specific hand channel values.
    // Hand channels are at CH_HAND_BASE (0..4): one-hot encoding of tile counts.
    // If player has 3 copies of tile X, channels 0,1,2 at position X should be 1.0,
    // and channel 3 at position X should be 0.0.
    let board = BoardState::new(42);
    let obs = encode_student_obs(&board, 0);

    let hand = &board.players[0].hand;

    for t in 0..NUM_TILE_TYPES {
        let count = hand[t] as usize;
        for k in 0..4usize {
            let ch = CH_HAND_BASE + k;
            let val = obs[ch * NUM_TILE_TYPES + t];
            if k < count {
                assert_eq!(
                    val, 1.0,
                    "tile {} count={}, channel {} should be 1.0 but got {}",
                    t, count, k, val
                );
            } else {
                assert_eq!(
                    val, 0.0,
                    "tile {} count={}, channel {} should be 0.0 but got {}",
                    t, count, k, val
                );
            }
        }
    }
}

// --- Test: Midgame obs has non-zero kawa and visible tile channels (H10) ---

fn advance_game_past_dingque(board: &mut BoardState, num_discards: usize) {
    // Complete ding_que
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

    // Play some turns
    let mut discards_done = 0;
    let mut steps = 0;
    while discards_done < num_discards && !board.is_done()
        && board.phase != engine::state::board::Phase::Scoring
        && steps < 500
    {
        steps += 1;
        let cp = board.current_player;
        match board.phase {
            engine::state::board::Phase::SelfCheck => {
                board.apply_action(cp, Action::Pass);
            }
            engine::state::board::Phase::Discard => {
                let candidates = board.players[cp].discard_candidates();
                if !candidates.is_empty() {
                    board.apply_action(cp, Action::Discard(candidates[0]));
                    discards_done += 1;
                }
            }
            engine::state::board::Phase::Reaction => {
                for i in 0..NUM_PLAYERS {
                    if board.reaction_pending[i] {
                        board.apply_action(i, Action::Pass);
                    }
                }
            }
            _ => break,
        }
    }
}

#[test]
fn test_midgame_obs_nonzero() {
    let mut board = BoardState::new(42);
    advance_game_past_dingque(&mut board, 8); // play 8 discards

    let obs = encode_student_obs(&board, 0);

    // Self kawa channels (Section 5) should have non-zero values
    let self_kawa_start = CH_SELF_KAWA_BASE * NUM_TILE_TYPES;
    let self_kawa_end = (CH_SELF_KAWA_BASE + MAX_TURNS * 2 + 2) * NUM_TILE_TYPES;
    let self_kawa_sum: f32 = obs[self_kawa_start..self_kawa_end].iter().sum();
    assert!(
        self_kawa_sum > 0.0,
        "self kawa channels should be non-zero after discards, sum={}",
        self_kawa_sum
    );

    // Opponent kawa channels (Section 6) should have non-zero values
    let opp_kawa_start = CH_OPP_KAWA_BASE * NUM_TILE_TYPES;
    let opp_kawa_end = (CH_OPP_KAWA_BASE + 3 * CH_OPP_KAWA_STRIDE) * NUM_TILE_TYPES;
    let opp_kawa_sum: f32 = obs[opp_kawa_start..opp_kawa_end].iter().sum();
    assert!(
        opp_kawa_sum > 0.0,
        "opponent kawa channels should be non-zero after discards, sum={}",
        opp_kawa_sum
    );

    // Visible tiles (Section 7) should have non-zero values
    let vis_start = CH_VISIBLE_TILES_BASE * NUM_TILE_TYPES;
    // 3 opponents x 4 channels = 12 channels for kawa overview
    let vis_end = (CH_VISIBLE_TILES_BASE + 12) * NUM_TILE_TYPES;
    let vis_sum: f32 = obs[vis_start..vis_end].iter().sum();
    assert!(
        vis_sum > 0.0,
        "visible tile channels should be non-zero after discards, sum={}",
        vis_sum
    );
}

// --- Test: Genbutsu encoding (H9) ---

#[test]
fn test_genbutsu_encoding() {
    // Genbutsu (現物) channels mark opponent discards as 1.0.
    // Section 15 at CH_GENBUTSU_BASE: 3 channels, one per opponent.
    let mut board = BoardState::new(42);
    advance_game_past_dingque(&mut board, 12); // play enough turns for discards

    let player_id = 0;
    let obs = encode_student_obs(&board, player_id);

    // For each opponent, verify their discarded tiles are marked as 1.0
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let ch = CH_GENBUTSU_BASE + (opp_off - 1);

        for &tile in &board.players[opp_id].discards {
            let val = obs[ch * NUM_TILE_TYPES + tile as usize];
            assert_eq!(
                val, 1.0,
                "genbutsu channel for opp {} should mark tile {} as 1.0, got {}",
                opp_id, tile, val
            );
        }
    }
}

// --- Test: Reaction phase action mask (H10) ---

#[test]
fn test_reaction_phase_mask() {
    // Verify that during reaction phase, the action mask has pon/agari/pass bits
    // set correctly for a player who can react.
    let mut found_reaction = false;
    for seed in 0..500 {
        let mut board = BoardState::new(seed);
        // Do ding_que
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

        let mut steps = 0;
        while !board.is_done() && board.phase != engine::state::board::Phase::Scoring && steps < 200 {
            steps += 1;
            let cp = board.current_player;

            match board.phase {
                engine::state::board::Phase::SelfCheck => {
                    board.apply_action(cp, Action::Pass);
                }
                engine::state::board::Phase::Discard => {
                    let candidates = board.players[cp].discard_candidates();
                    if !candidates.is_empty() {
                        board.apply_action(cp, Action::Discard(candidates[0]));
                    }
                }
                engine::state::board::Phase::Reaction => {
                    for i in 0..NUM_PLAYERS {
                        if board.reaction_pending[i] {
                            let mask = encode_action_mask(&board, i);

                            // Pass should always be available during reaction
                            assert!(mask[30], "pass (action 30) should be available in reaction phase");

                            // At least one of pon/kan/agari should be available
                            // (otherwise the player wouldn't be in reaction_pending)
                            let has_special = mask[27] || mask[28] || mask[29];
                            assert!(
                                has_special,
                                "seed {}: reaction player {} should have pon/kan/agari available",
                                seed, i
                            );

                            // Discard tiles should NOT be available during reaction
                            let has_discard = mask[0..27].iter().any(|&m| m);
                            assert!(
                                !has_discard,
                                "seed {}: discard actions should not be available during reaction",
                                seed
                            );

                            found_reaction = true;
                            board.apply_action(i, Action::Pass);
                        }
                    }
                    if found_reaction { break; }
                }
                _ => break,
            }
            if found_reaction { break; }
        }
        if found_reaction { break; }
    }
    assert!(found_reaction, "should find a reaction phase in 500 seeds");
}
