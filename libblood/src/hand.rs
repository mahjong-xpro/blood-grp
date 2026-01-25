//! Hand format conversions, usually only useful for testing and debugging.
//!
//! Note that all functions in this mod that take or produce strings are dealing
//! with tenhou.net/2 format tile description (like 0m 123z) instead of mjai (like
//! 5mr ESW).

use crate::tile::Tile;
use crate::must_tile;

use anyhow::{Result, bail, ensure};

/// Spaces are allowed.
pub fn hand_with_aka(s: &str) -> Result<[u8; 37]> {
    // Bloody Battle: This function is kept for compatibility but red 5s are not supported
    // We will be using bytes instead of chars afterwards.
    ensure!(s.is_ascii(), "hand {s} contains non-ascii content");

    let mut ret = [0; 37];
    let mut stack = vec![];

    for b in s.as_bytes() {
        match b {
            b'0'..=b'9' => stack.push((b - b'0') as usize),
            b'm' | b'p' | b's' | b'z' => {
                for t in stack.drain(..) {
                    // Bloody Battle: Treat 0 as regular 5 (no red 5s)
                    let num = if t == 0 { 5 } else { t };
                    let kind = match b {
                        b'm' => 0,
                        b'p' => 1,
                        b's' => 2,
                        b'z' => 3, // Bloody Battle: jihai not used but kept for compatibility
                        _ => unreachable!(),
                    };
                    let idx = if kind < 3 {
                        kind * 9 + num - 1
                    } else {
                        // jihai: 27 + (num - 1) for E/S/W/N/P/F/C
                        if num >= 1 && num <= 7 {
                            27 + num - 1
                        } else {
                            bail!("invalid jihai number: {num}");
                        }
                    };
                    if idx < 37 {
                        ret[idx] += 1;
                    } else {
                        bail!("tile index out of range: {idx}");
                    }
                }
            }
            b' ' | b'\t' | b'\n' => (),
            _ => bail!("unexpected byte {b}"),
        };
    }

    Ok(ret)
}

/// Spaces are allowed.
/// Bloody Battle: Returns [u8; 27] (no jihai, no red 5s)
pub fn hand(s: &str) -> Result<[u8; 27]> {
    // Bloody Battle: No jihai, no red 5s, only suhai (m, p, s)
    ensure!(s.is_ascii(), "hand {s} contains non-ascii content");

    let mut ret = [0; 27];
    let mut stack = vec![];

    for b in s.as_bytes() {
        match b {
            b'0'..=b'9' => stack.push((b - b'0') as usize),
            b'm' | b'p' | b's' => {
                for t in stack.drain(..) {
                    // Bloody Battle: Treat 0 as regular 5 (no red 5s)
                    let num = if t == 0 { 5 } else { t };
                    let kind = match b {
                        b'm' => 0,
                        b'p' => 1,
                        b's' => 2,
                        _ => unreachable!(),
                    };
                    let idx = kind * 9 + num - 1;
                    if idx < 27 {
                        ret[idx] += 1;
                    } else {
                        bail!("tile index out of range: {idx}");
                    }
                }
            }
            b'z' => {
                // Bloody Battle: No jihai, skip z tiles
                stack.clear();
            }
            b' ' | b'\t' | b'\n' => (),
            _ => bail!("unexpected byte {b}"),
        };
    }

    Ok(ret)
}

#[must_use]
pub fn tile37_to_vec(tiles: &[u8; 37]) -> Vec<Tile> {
    let mut ret = vec![];
    tiles
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .for_each(|(tid, &count)| {
            if tid < 34 {
                ret.resize(ret.len() + count as usize, must_tile!(tid));
            } else {
                ret.push(must_tile!(tid));
            }
        });
    ret
}

#[must_use]
/// Bloody Battle: Converts [u8; 27] tile array to Vec<Tile>
/// (27 tile kinds: 3 suits × 9 numbers, no jihai)
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

#[must_use]
// Bloody Battle: 27 tile kinds (no jihai, no red 5s)
pub fn tiles_to_string(tiles: &[u8; 27], _aka: [bool; 3]) -> String {
    // Bloody Battle: No jihai, only suhai (3 * 9 = 27)
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
                    // Bloody Battle: No red 5s (aka)
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

    // Bloody Battle: No jihai
    suhai
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse() {
        // Bloody Battle: No jihai, updated test cases
        assert_eq!(
            hand("1111m 333p 222s").unwrap(),
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, // m
                0, 0, 3, 0, 0, 0, 0, 0, 0, // p
                0, 3, 0, 0, 0, 0, 0, 0, 0, // s
            ]
        );

        assert_eq!(
            hand_with_aka("22334450m234p2s3s4s").unwrap(),
            [
                0, 2, 2, 2, 1, 0, 0, 0, 0, // m
                0, 1, 1, 1, 0, 0, 0, 0, 0, // p
                0, 1, 1, 1, 0, 0, 0, 0, 0, // s
                0, 0, 0, 0, 0, 0, 0, // z (kept for compatibility but not used)
                1, 0, 0, // a (kept for compatibility but not used)
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
    fn string() {
        // Bloody Battle: No jihai, updated test case
        assert_eq!(
            tiles_to_string(
                &[
                    0, 0, 2, 0, 1, 1, 1, 0, 0, // m
                    0, 0, 1, 1, 1, 1, 1, 1, 0, // p
                    0, 0, 0, 0, 0, 1, 1, 1, 0, // s
                ],
                [false, false, false] // Bloody Battle: No red 5s
            ),
            "33067m 345678p 678s"
        );
    }
}
