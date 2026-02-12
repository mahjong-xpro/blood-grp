//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//!
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>

use std::io::prelude::*;
use std::sync::LazyLock;

use flate2::read::GzDecoder;

const SUHAI_TABLE_SIZE: usize = 1_940_777;

static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_suhai.bin.gz"),
        SUHAI_TABLE_SIZE,
    )
});

fn read_table(gzipped: &[u8], length: usize) -> Vec<[u8; 10]> {
    let mut gz = GzDecoder::new(gzipped);
    let mut raw = vec![];
    gz.read_to_end(&mut raw).unwrap();

    let mut ret = Vec::with_capacity(length);
    let mut entry = [0; 10];
    for (i, b) in raw.into_iter().enumerate() {
        entry[i * 2 % 10] = b & 0b1111;
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), length);

    ret
}

pub fn ensure_init() {
    assert_eq!(SUHAI_TABLE.len(), SUHAI_TABLE_SIZE);
}

fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    // len_div3 should be 0-4, so m should be 0-4
    // If m is out of range, clamp it to prevent index out of bounds
    // Note: m can be 255 (u8::MAX) if tehai_len_div3 underflowed, so we need to handle it
    let m = if m > 4 { 
        // Log warning and clamp to 4 (safest value)
        eprintln!("WARNING: len_div3 (m) should be 0-4, but got {}. This may indicate tehai_len_div3 underflow. Clamping to 4.", m);
        4 
    } else { 
        m 
    };
    
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();
    
    // j should be in range [5, 9] when m is 0-4
    for j in (5..=(5 + m)).rev() {
        // Ensure j is within tab bounds [0, 9]
        if j >= 10 {
            // This should never happen if m <= 4, but add safety check
            continue;
        }
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            // Ensure indices are within bounds
            let jk = j - k;
            if jk < 10 && k < 10 {
                sht = sht.min(lhs[k] + tab[jk]).min(lhs[jk] + tab[k]);
            }
        }
        lhs[j] = sht;
    }

    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            // Ensure j - k is within tab bounds [0, 9]
            let jk = j - k;
            if jk < 10 {
                sht = sht.min(lhs[k] + tab[jk]);
            }
        }
        lhs[j] = sht;
    }
}


fn sum_tiles(tiles: &[u8]) -> usize {
    tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)
}

/// `len_div3` must be within [0, 4].
#[must_use]
pub fn calc_normal(tiles: &[u8; 27], len_div3: u8) -> i8 {
    let len_div3 = len_div3 as usize;

    let mut ret = SUHAI_TABLE
        .get(sum_tiles(&tiles[..9]))
        .copied()
        .unwrap_or_default();
    add_suhai(&mut ret, sum_tiles(&tiles[9..2 * 9]), len_div3);
    add_suhai(&mut ret, sum_tiles(&tiles[2 * 9..3 * 9]), len_div3);

    (ret[5 + len_div3] as i8) - 1
}

#[must_use]
pub fn calc_chitoi(tiles: &[u8; 27]) -> i8 {
    let mut pairs: u8 = 0;
    let mut kinds: u8 = 0;
    let mut quads: u8 = 0;
    tiles.iter().filter(|&&c| c > 0).for_each(|&c| {
        kinds += 1;
        // 龙七对（Long QiDui）：4张相同牌算2个对子
        if c >= 4 {
            pairs += 2;
            quads += 1;
        } else if c >= 2 {
            pairs += 1;
        }
    });

    // 龙七对允许 4 张相同牌算 2 个对子，因此最少只需 (7 - quads) 种牌，
    // 而非标准七对的 7 种。例如：
    //   1111m 2233p 445566s → 6 种, quads=1, needed=6, redunct=0 → 完成(-1)
    //   1111m 2222p 334455s → 5 种, quads=2, needed=5, redunct=0 → 完成(-1)
    let needed_kinds = 7_u8.saturating_sub(quads);
    let redunct = needed_kinds.saturating_sub(kinds) as i8;
    7 - (pairs as i8) + redunct - 1
}

// 血战到底无国士无双（无字牌），calc_kokushi 已移除

#[must_use]
pub fn calc_all(tiles: &[u8; 27], len_div3: u8, ding_que: Option<crate::mjai::Suit>) -> i8 {
    let (clean_tiles, void_count) = if let Some(suit) = ding_que {
        let mut t = *tiles;
        let range = match suit {
            crate::mjai::Suit::Man => 0..9,
            crate::mjai::Suit::Pin => 9..18,
            crate::mjai::Suit::Sou => 18..27,
        };
        let count = range.map(|i| {
            let c = t[i];
            t[i] = 0;
            c
        }).sum();
        (t, count)
    } else {
        (*tiles, 0)
    };

    let clean_len_div3 = if void_count > 0 {
        clean_tiles.iter().sum::<u8>() / 3
    } else {
        len_div3
    };

    let mut shanten = calc_normal(&clean_tiles, clean_len_div3);
    
    // Add penalty for ding_que tiles still in hand
    shanten = shanten.saturating_add(void_count as i8);

    // Chitoi check (add penalty too)
    let chitoi = calc_chitoi(&clean_tiles).saturating_add(void_count as i8);
    shanten = shanten.min(chitoi);
    
    shanten
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;

    #[test]
    fn calc_3n_plus_1() {
        let tehai = hand("1111m 333p 222s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 1);
        let tehai = hand("147m 258p 369s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 6);
        let tehai = hand("468m 33346p 7s").unwrap();
        assert_eq!(calc_all(&tehai, 3, None), 2);
        let tehai = hand("147m 258p 3s").unwrap();
        assert_eq!(calc_all(&tehai, 2, None), 4);
        let tehai = hand("4455s").unwrap();
        assert_eq!(calc_all(&tehai, 1, None), 0);
        let tehai = hand("15559m 19p 19s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 3);
        let tehai = hand("9999m 6677p 88s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 2);
        let tehai = hand("19m 19p 159s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 1);
    }

    #[test]
    fn calc_3n_plus_2() {
        let tehai = hand("2344456m 14p 127s 7p").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 3);
        let tehai = hand("2344456m 14p 127s 5p").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 2);
        let tehai = hand("344455667p 1139s 9m").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 2);
        let tehai = hand("344455667p 1139s 9p").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 1);
        let tehai = hand("122334m 678p 37s 5s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 0);
        let tehai = hand("122334m 678p 12s 4s").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), 0);
        let tehai = hand("12223456m 78889p 2m").unwrap();
        assert_eq!(calc_all(&tehai, 4, None), -1);
        let tehai = hand("34778p").unwrap();
        assert_eq!(calc_all(&tehai, 1, None), 0);
        let tehai = hand("34s").unwrap();
        assert_eq!(calc_all(&tehai, 0, None), 0);
        let tehai = hand("55m").unwrap();
        assert_eq!(calc_all(&tehai, 0, None), -1);
    }
}
