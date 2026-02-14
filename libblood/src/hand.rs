//! Hand format conversions, usually only useful for testing and debugging.
//!
//! 血战到底麻将专用：仅支持万(m)、筒(p)、条(s) 三种花色，数字 1-9。
//! 无字牌(z)，无赤宝牌(0)。

use crate::tile::Tile;
use crate::must_tile;

use anyhow::{Result, bail, ensure};

/// Parse hand string (e.g. "123m 456p 789s") into tile count array.
/// Spaces are allowed. Only digits 1-9 with suit suffixes m/p/s are valid.
pub fn hand(s: &str) -> Result<[u8; 27]> {
    ensure!(s.is_ascii(), "hand {s} contains non-ascii content");

    let mut ret = [0; 27];
    let mut stack = vec![];

    for b in s.as_bytes() {
        match b {
            b'1'..=b'9' => stack.push((b - b'0') as usize),
            b'0' => bail!("血战到底无赤宝牌(aka-dora)，不支持 '0'，请使用 '5'"),
            b'm' | b'p' | b's' => {
                for t in stack.drain(..) {
                    let kind = match b {
                        b'm' => 0,
                        b'p' => 1,
                        b's' => 2,
                        _ => unreachable!(),
                    };
                    let idx = kind * 9 + t - 1;
                    if idx < 27 {
                        ret[idx] += 1;
                    } else {
                        bail!("tile index out of range: {idx}");
                    }
                }
            }
            b'z' => bail!("血战到底无字牌(honor tiles)，不支持 'z' 后缀"),
            b' ' | b'\t' | b'\n' => (),
            _ => bail!("unexpected byte {b}"),
        };
    }

    if !stack.is_empty() {
        bail!("trailing digits without suit suffix: {:?}", stack);
    }

    ensure!(
        ret.iter().all(|&c| c <= 4),
        "tile count exceeds 4: {}",
        tiles_to_string(&ret),
    );

    Ok(ret)
}

// Removed tile37_to_vec - Bloody Battle Mahjong only uses 27 tile types

#[must_use]
pub fn tile27_to_vec(tiles: &[u8; 27]) -> Vec<Tile> {
    let mut ret = vec![];
    tiles
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .for_each(|(tid, &count)| {
            ret.resize(ret.len() + count as usize, must_tile!(tid));
        });
    ret
}

/// Convert tehai count array to mjai-format string list (e.g. ["1m","2m","3m",...]).
#[must_use]
pub fn tehai_to_strings(tehai: &[u8; 27]) -> Vec<String> {
    tile27_to_vec(tehai)
        .iter()
        .map(|t| t.to_string())
        .collect()
}

#[must_use]
pub fn tiles_to_string(tiles: &[u8; 27]) -> String {
    let suhai = tiles[..3 * 9]
        .chunks_exact(9)
        .enumerate()
        .map(|(kind, chunk)| {
            let mut partial = String::new();
            let mut not_empty = false;
            chunk
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .for_each(|(num, &count)| {
                    let literal_num = num + 1;
                    partial += &literal_num.to_string().repeat(count as usize);
                    not_empty = true;
                });

            if not_empty {
                let c = match kind {
                    0 => 'm',
                    1 => 'p',
                    2 => 's',
                    _ => unreachable!(),
                };
                partial.push(c);
            }
            partial
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    suhai
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse() {
        assert_eq!(
            hand("1111m 333p 222s").unwrap(),
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, // m
                0, 0, 3, 0, 0, 0, 0, 0, 0, // p
                0, 3, 0, 0, 0, 0, 0, 0, 0, // s
            ]
        );

        assert_eq!(
            hand("456m 6p 7899p 987s 9p").unwrap(),
            [
                0, 0, 0, 1, 1, 1, 0, 0, 0, // m
                0, 0, 0, 0, 0, 1, 1, 1, 3, // p
                0, 0, 0, 0, 0, 0, 1, 1, 1, // s
            ]
        );
    }

    #[test]
    fn rejects_tile_count_exceeds_4() {
        // 1m 出现 5 次 → 物理上不可能
        assert!(hand("111444777999m 11m").is_err());
        // 9p 出现 5 次
        assert!(hand("9999p 9p").is_err());
        // 恰好 4 张应通过
        assert!(hand("1111m").is_ok());
    }

    #[test]
    fn string() {
        assert_eq!(
            tiles_to_string(&[
                0, 0, 2, 0, 1, 1, 1, 0, 0, // m
                0, 0, 1, 1, 1, 1, 1, 1, 0, // p
                0, 0, 0, 0, 0, 1, 1, 1, 0, // s
            ]),
            "33567m 345678p 678s"
        );
    }
}
