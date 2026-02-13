/// 
/// Formula: 点数 = 1000 × 2^(番数-1)
/// Cap: 5番封顶 = 16000点
/// No oya advantage in scoring
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Point {
    pub ron: i32,
    pub tsumo_ko: i32,
}

impl Point {
    /// 
    /// Formula: 点数 = 1000 × 2^(番数-1)
    /// Cap: 5番封顶 = 16000点
    /// 
    /// # Arguments
    /// * `fan` - 番数 (1-5, capped at 5)
    /// 
    /// # Returns
    /// Point with ron and tsumo values (no oya distinction)
    #[must_use]
    pub fn calc_from_fan(fan: u8) -> Self {
        let fan = fan.min(5); // 5番封顶
        let base_points = if fan == 0 {
            0
        } else {
            1000 * 2_i32.pow((fan - 1) as u32)
        };
        
        Self {
            ron: base_points,
            tsumo_ko: base_points,
        }
    }


    #[must_use]
    pub const fn yakuman(_is_oya: bool, _count: i32) -> Self {
        // Return max points (5番封顶)
        Self {
            ron: 16000,
            tsumo_ko: 16000,
        }
    }

    /// Calculate total points for tsumo (with specified number of payers).
    ///
    /// 血战到底中自摸时，只有尚未和牌的对手支付。
    /// `n_payers`: 实际支付人数（1-3）。游戏开始时为 3，
    /// 每有一个玩家和牌就减 1。
    #[inline]
    #[must_use]
    pub const fn tsumo_total(self, n_payers: u32) -> i32 {
        self.tsumo_ko * n_payers as i32
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bloody_battle_scoring() {
        assert_eq!(Point::calc_from_fan(1).ron, 1000);  // 1番 = 1000点
        assert_eq!(Point::calc_from_fan(2).ron, 2000);  // 2番 = 2000点
        assert_eq!(Point::calc_from_fan(3).ron, 4000);  // 3番 = 4000点
        assert_eq!(Point::calc_from_fan(4).ron, 8000);  // 4番 = 8000点
        assert_eq!(Point::calc_from_fan(5).ron, 16000); // 5番 = 16000点（封顶）
        assert_eq!(Point::calc_from_fan(6).ron, 16000); // 6番 = 16000点（封顶）
        assert_eq!(Point::calc_from_fan(10).ron, 16000); // 10番 = 16000点（封顶）

        // Test no oya advantage
        let point = Point::calc_from_fan(3);
        assert_eq!(point.ron, point.tsumo_ko);
        
        // Test tsumo_total with different payer counts
        assert_eq!(point.tsumo_total(3), 4000 * 3); // 3 players pay (game start)
        assert_eq!(point.tsumo_total(2), 4000 * 2); // 2 players pay (1 player won)
        assert_eq!(point.tsumo_total(1), 4000 * 1); // 1 player pays (2 players won)
    }
}
