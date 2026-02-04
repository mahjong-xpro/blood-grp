#[derive(Debug, Clone, Copy)]
pub struct Rankings {
    pub player_by_rank: [u8; 4],
    pub rank_by_player: [u8; 4],
}

impl Rankings {
    pub fn new(scores: [i32; 4]) -> Self {
        let mut player_by_rank = [0, 1, 2, 3];
        player_by_rank.sort_by_key(|&i| -scores[i as usize]);

        let mut rank_by_player = [0; 4];
        for (rank, id) in player_by_rank.into_iter().enumerate() {
            rank_by_player[id as usize] = rank as u8;
        }

        Self {
            player_by_rank,
            rank_by_player,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts::{INITIAL_SCORE, TOTAL_SCORE};

    #[test]
    fn rankings() {
        let mut scores = [INITIAL_SCORE, INITIAL_SCORE, INITIAL_SCORE + 5000, INITIAL_SCORE - 5000];

        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [2, 0, 1, 3]);
        assert_eq!(rk.rank_by_player, [1, 2, 0, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [INITIAL_SCORE, INITIAL_SCORE, 0, INITIAL_SCORE - 5000]);

        scores = [INITIAL_SCORE; 4];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 1, 2, 3]);
        assert_eq!(rk.rank_by_player, [0, 1, 2, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [0, INITIAL_SCORE, INITIAL_SCORE, INITIAL_SCORE]);

        scores = [INITIAL_SCORE - 7000, INITIAL_SCORE + 7000, INITIAL_SCORE + 7000, INITIAL_SCORE - 7000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 2, 0, 3]);
        assert_eq!(rk.rank_by_player, [2, 0, 1, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [INITIAL_SCORE - 7000, 0, INITIAL_SCORE + 7000, INITIAL_SCORE - 7000]);

        scores = [INITIAL_SCORE + 7000, INITIAL_SCORE - 7000, INITIAL_SCORE - 7000, INITIAL_SCORE + 7000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 3, 1, 2]);
        assert_eq!(rk.rank_by_player, [0, 2, 3, 1]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [0, INITIAL_SCORE - 7000, INITIAL_SCORE - 7000, INITIAL_SCORE + 7000]);

        scores = [0, TOTAL_SCORE, 0, 0];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 0, 2, 3]);
        assert_eq!(rk.rank_by_player, [1, 0, 2, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [0; 4]);
    }
}
