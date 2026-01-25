//! Hand format conversions, usually only useful for testing and debugging.
//!
//! Note that all functions in this mod that take or produce strings are dealing
//! with tenhou.net/2 format tile description (like 0m 123z) instead of mjai (like
//! 5mr ESW).

use crate::tile::Tile;
use crate::must_tile;

use anyhow::{Result, bail, ensure};

/// Spaces are allowed.
pub fn hand(s: &str) -> Result<[u8; 27]> {
    ensure!(s.is_ascii(), "hand {s} contains non-ascii content");

    let mut ret = [0; 27];
    let mut stack = vec![];

    for b in s.as_bytes() {
        match b {
            b'0'..=b'9' => stack.push((b - b'0') as usize),
            b'm' | b'p' | b's' => {
                for t in stack.drain(..) {
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
pub fn tiles_to_string(tiles: &[u8; 27], _aka: [bool; 3]) -> String {
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
    fn string() {
        assert_eq!(
            tiles_to_string(
                &[
                    0, 0, 2, 0, 1, 1, 1, 0, 0, // m
                    0, 0, 1, 1, 1, 1, 1, 1, 0, // p
                    0, 0, 0, 0, 0, 1, 1, 1, 0, // s
                ],
                [false, false, false]
            ),
            "33067m 345678p 678s"
        );
    }
}
