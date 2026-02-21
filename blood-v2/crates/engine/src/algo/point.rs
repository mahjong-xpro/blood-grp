use crate::consts::MAX_FAN;

/// Calculate score from fan count: 1000 * 2^(fan-1), capped at 5 fan = 16000
pub fn calc_score(fan: u8) -> i32 {
    if fan == 0 {
        return 0;
    }
    let capped = fan.min(MAX_FAN);
    1000 * (1i32 << (capped - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_table() {
        assert_eq!(calc_score(0), 0);
        assert_eq!(calc_score(1), 1000);
        assert_eq!(calc_score(2), 2000);
        assert_eq!(calc_score(3), 4000);
        assert_eq!(calc_score(4), 8000);
        assert_eq!(calc_score(5), 16000);
        assert_eq!(calc_score(6), 16000); // capped
        assert_eq!(calc_score(10), 16000);
        assert_eq!(calc_score(u8::MAX), 16000); // extreme cap
    }
}
