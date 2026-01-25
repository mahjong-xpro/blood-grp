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

    /// Legacy method for compatibility - calculates from fu and han (Japanese Mahjong/日本麻将)
    #[must_use]
    pub fn calc(_is_oya: bool, _fu: u8, han: u8) -> Self {
        Self::calc_from_fan(han)
    }

    #[must_use]
    pub const fn yakuman(_is_oya: bool, _count: i32) -> Self {
        // Return max points (5番封顶)
        Self {
            ron: 16000,
            tsumo_ko: 16000,
        }
    }

    /// Calculate total points for tsumo
    /// 
    #[inline]
    #[must_use]
    pub const fn tsumo_total(self, _is_oya: bool) -> i32 {
        self.tsumo_ko * 3
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
        
        // Test tsumo_total
        assert_eq!(point.tsumo_total(false), 4000 * 3); // 3 players pay
        assert_eq!(point.tsumo_total(true), 4000 * 3);  // Same for oya
    }
}
