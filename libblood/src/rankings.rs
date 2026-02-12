#[derive(Debug, Clone, Copy)]
pub struct Rankings {
    pub player_by_rank: [u8; 4],
    pub rank_by_player: [u8; 4],
}

impl Rankings {
    /// 创建排名。同分按和牌顺序排名（先和者靠前）。
    ///
    /// - `agari_order`: 和牌顺序（可选）。例如 `&[2, 0, 3]` 表示 P2 先和，P0 次之，P3 最后。
    ///   不在列表中的玩家排在所有已和牌玩家之后（未和牌 / 流局时按座位号兜底）。
    ///   若为 `None`，同分时按座位号排序（向后兼容）。
    pub fn new_with_agari_order(scores: [i32; 4], agari_order: Option<&[u8]>) -> Self {
        // 为每个玩家构建辅助排序键：(分数取反, 和牌顺序, 座位号)
        // 分数高者优先；同分时先和牌者优先；都未和牌同分则座位号小者优先。
        let mut player_by_rank: [u8; 4] = [0, 1, 2, 3];

        player_by_rank.sort_by(|&a, &b| {
            let score_cmp = scores[b as usize].cmp(&scores[a as usize]); // 分数降序
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
            // 同分：按和牌顺序
            let order_a = agari_order
                .and_then(|order| order.iter().position(|&p| p == a))
                .unwrap_or(usize::MAX); // 未和牌者排在最后
            let order_b = agari_order
                .and_then(|order| order.iter().position(|&p| p == b))
                .unwrap_or(usize::MAX);
            let order_cmp = order_a.cmp(&order_b);
            if order_cmp != std::cmp::Ordering::Equal {
                return order_cmp;
            }
            a.cmp(&b) // 兜底：座位号
        });

        let mut rank_by_player = [0; 4];
        for (rank, id) in player_by_rank.iter().enumerate() {
            rank_by_player[*id as usize] = rank as u8;
        }

        Self {
            player_by_rank,
            rank_by_player,
        }
    }

    /// 向后兼容的简易构造器（同分按座位号排序）。
    pub fn new(scores: [i32; 4]) -> Self {
        Self::new_with_agari_order(scores, None)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts::{INITIAL_SCORE, TOTAL_SCORE};

    #[test]
    fn rankings_no_agari_order() {
        // 向后兼容：无和牌顺序时同分按座位号
        let scores = [INITIAL_SCORE, INITIAL_SCORE, INITIAL_SCORE + 5000, INITIAL_SCORE - 5000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [2, 0, 1, 3]);
        assert_eq!(rk.rank_by_player, [1, 2, 0, 3]);

        let scores = [INITIAL_SCORE; 4];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 1, 2, 3]);

        let scores = [INITIAL_SCORE - 7000, INITIAL_SCORE + 7000, INITIAL_SCORE + 7000, INITIAL_SCORE - 7000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 2, 0, 3]);

        let scores = [INITIAL_SCORE + 7000, INITIAL_SCORE - 7000, INITIAL_SCORE - 7000, INITIAL_SCORE + 7000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 3, 1, 2]);

        let scores = [0, TOTAL_SCORE, 0, 0];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 0, 2, 3]);
    }

    #[test]
    fn rankings_with_agari_order() {
        // 同分时先和牌者排名靠前
        let scores = [INITIAL_SCORE + 7000, INITIAL_SCORE + 7000, INITIAL_SCORE - 7000, INITIAL_SCORE - 7000];

        // P1 先和，P0 后和 → 同分时 P1 排在 P0 前面
        let rk = Rankings::new_with_agari_order(scores, Some(&[1, 0]));
        assert_eq!(rk.player_by_rank, [1, 0, 2, 3]);
        assert_eq!(rk.rank_by_player, [1, 0, 2, 3]);

        // P0 先和，P1 后和 → 同分时 P0 排在 P1 前面
        let rk = Rankings::new_with_agari_order(scores, Some(&[0, 1]));
        assert_eq!(rk.player_by_rank, [0, 1, 2, 3]);
        assert_eq!(rk.rank_by_player, [0, 1, 2, 3]);

        // 三人同分：P2 先和, P0 次之, P3 最后; P1 未和
        let scores = [INITIAL_SCORE; 4];
        let rk = Rankings::new_with_agari_order(scores, Some(&[2, 0, 3]));
        assert_eq!(rk.player_by_rank, [2, 0, 3, 1]);
        assert_eq!(rk.rank_by_player, [1, 3, 0, 2]);
    }
}
