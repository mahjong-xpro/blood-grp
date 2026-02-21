use crate::consts::*;
use crate::algo::shanten::{calc_shanten, waiting_tiles};
use crate::state::board::BoardState;
use super::student::encode_student_obs;

/// Encode oracle observation: student obs + extra perfect-info channels
pub fn encode_oracle_obs(board: &BoardState, player_id: usize) -> Vec<f32> {
    let mut obs = encode_student_obs(board, player_id);
    obs.resize(NUM_ORACLE_CHANNELS * NUM_TILE_TYPES, 0.0);

    let base = NUM_STUDENT_CHANNELS;
    let mut ch = base;

    macro_rules! w {
        ($channel:expr, $tile:expr, $val:expr) => {
            obs[$channel * NUM_TILE_TYPES + $tile] = $val;
        };
    }
    macro_rules! fill_ch {
        ($channel:expr, $val:expr) => {
            for t in 0..NUM_TILE_TYPES {
                obs[$channel * NUM_TILE_TYPES + t] = $val;
            }
        };
    }

    // Opponent true hands (3 x 4 one-hot = 12 ch)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        for t in 0..NUM_TILE_TYPES {
            for k in 0..4usize {
                if opp.hand[t] as usize > k {
                    w!(ch + k, t, 1.0);
                }
            }
        }
        ch += 4;
    }

    // Opponent true ding que (3 x 3 = 9 ch)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        if let Some(suit) = board.players[opp_id].ding_que {
            for t in suit.start()..suit.end() {
                w!(ch + suit as usize, t, 1.0);
            }
        }
        ch += 3;
    }

    // Opponent true shanten (3 x 5 = 15 ch)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let s = calc_shanten(&opp.hand, opp.melds.len());
        let sh = s.max(0).min(4) as usize;
        fill_ch!(ch + sh, 1.0);
        ch += 5;
    }

    // Opponent wait tiles (3 x 1 = 3 ch)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        let opp = &board.players[opp_id];
        let s = calc_shanten(&opp.hand, opp.melds.len());
        if s == 0 {
            let waits = waiting_tiles(&opp.hand, opp.melds.len());
            for &wt in &waits {
                w!(ch, wt as usize, 1.0);
            }
        }
        ch += 1;
    }

    // Wall tile counts (4 ch)
    let mut wall_counts = [0u8; NUM_TILE_TYPES];
    for i in board.wall_idx..board.wall_back_idx {
        wall_counts[board.wall[i] as usize] += 1;
    }
    for t in 0..NUM_TILE_TYPES {
        for k in 0..4usize {
            if wall_counts[t] as usize > k {
                w!(ch + k, t, 1.0);
            }
        }
    }
    ch += 4;

    // Opponent ding que completion (3 ch, one per opponent)
    for opp_off in 1..NUM_PLAYERS {
        let opp_id = (player_id + opp_off) % NUM_PLAYERS;
        fill_ch!(ch, board.players[opp_id].ding_que_completed() as u8 as f32);
        ch += 1;
    }

    debug_assert!(ch == NUM_ORACLE_CHANNELS, "oracle used {} channels, expected {}", ch, NUM_ORACLE_CHANNELS);

    obs
}
