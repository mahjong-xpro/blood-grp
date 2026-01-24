use crate::{matches_tu8, t, tu8};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::sync::LazyLock;

use ahash::AHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MJAI_PAI_STRINGS_LEN: usize = 3 * 9 + 1; // 27 tiles + 1 unknown
const MJAI_PAI_STRINGS: [&str; MJAI_PAI_STRINGS_LEN] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", // m
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", // p
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", // s
    "?",   // unknown
];
const DISCARD_PRIORITIES: [u8; 28] = [
    6, 5, 4, 3, 2, 3, 4, 5, 6, // m
    6, 5, 4, 3, 2, 3, 4, 5, 6, // p
    6, 5, 4, 3, 2, 3, 4, 5, 6, // s
    0, // unknown
];

static MJAI_PAI_STRINGS_MAP: LazyLock<AHashMap<&'static str, Tile>> = LazyLock::new(|| {
    MJAI_PAI_STRINGS
        .iter()
        .enumerate()
        .map(|(id, &s)| (s, Tile::try_from(id).unwrap()))
        .collect()
});

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile(u8);

#[derive(Debug)]
pub enum InvalidTile {
    Number(usize),
    String(String),
}

impl Tile {
    /// # Safety
    /// Calling this method with an out-of-bounds tile ID is undefined behavior.
    #[inline]
    #[must_use]
    pub const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    pub const fn deaka(self) -> Self {
        // Bloody Battle Mahjong has no red 5s, so deaka() just returns self
        self
    }

    #[inline]
    #[must_use]
    pub const fn akaize(self) -> Self {
        // Bloody Battle Mahjong has no red 5s, so akaize() just returns self
        self
    }

    #[inline]
    #[must_use]
    pub const fn is_aka(self) -> bool {
        // Bloody Battle Mahjong has no red 5s
        false
    }

    #[inline]
    #[must_use]
    pub const fn is_jihai(self) -> bool {
        // Bloody Battle Mahjong has no jihai (wind/dragon tiles)
        false
    }

    #[inline]
    #[must_use]
    pub const fn is_yaokyuu(self) -> bool {
        // Bloody Battle Mahjong: only terminal tiles (1 and 9)
        matches_tu8!(self.0, 1m | 9m | 1p | 9p | 1s | 9s)
    }

    #[inline]
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.0 >= tu8!(?)
    }

    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let kind = tile.0 / 9;
        let num = tile.0 % 9;

        // Bloody Battle Mahjong: only number tiles (0-2 for m, p, s)
        if kind < 3 {
            Self(kind * 9 + (num + 1) % 9)
        } else {
            // Unknown tile, return as is
            self
        }
    }

    #[inline]
    #[must_use]
    pub const fn prev(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let kind = tile.0 / 9;
        let num = tile.0 % 9;
        // Bloody Battle Mahjong: only number tiles (0-2 for m, p, s)
        if kind < 3 {
            Self(kind * 9 + (num + 9 - 1) % 9)
        } else {
            // Unknown tile, return as is
            self
        }
    }

    #[inline]
    #[must_use]
    pub const fn augment(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let tid = tile.0;
        let kind = tid / 9;
        match kind {
            0 => Self(tid + 9), // m -> p
            1 => Self(tid - 9), // p -> m
            _ => tile,          // s or unknown
        }
    }

    /// `Ordering::Equal` iff `self == other`
    #[inline]
    #[must_use]
    pub fn cmp_discard_priority(self, other: Self) -> Ordering {
        let l = self.0 as usize;
        let r = other.0 as usize;
        match DISCARD_PRIORITIES[l].cmp(&DISCARD_PRIORITIES[r]) {
            Ordering::Equal => r.cmp(&l),
            o => o,
        }
    }
}

impl Default for Tile {
    fn default() -> Self {
        t!(?)
    }
}

impl TryFrom<u8> for Tile {
    type Error = InvalidTile;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        Self::try_from(v as usize)
    }
}

impl TryFrom<usize> for Tile {
    type Error = InvalidTile;

    fn try_from(v: usize) -> Result<Self, Self::Error> {
        if v >= MJAI_PAI_STRINGS_LEN {
            Err(InvalidTile::Number(v))
        } else {
            Ok(Self(v as u8))
        }
    }
}

impl FromStr for Tile {
    type Err = InvalidTile;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MJAI_PAI_STRINGS_MAP
            .get(s)
            .copied()
            .ok_or_else(|| InvalidTile::String(s.to_owned()))
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(MJAI_PAI_STRINGS[self.0 as usize])
    }
}

impl<'de> Deserialize<'de> for Tile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tile = String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)?;
        Ok(tile)
    }
}

impl Serialize for Tile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl fmt::Display for InvalidTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("not a valid tile: ")?;
        match self {
            Self::Number(n) => fmt::Display::fmt(n, f),
            Self::String(s) => write!(f, "not a valid tile: \"{s}\""),
        }
    }
}

impl Error for InvalidTile {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn convert() {
        "1m".parse::<Tile>().unwrap();
        "9s".parse::<Tile>().unwrap();
        "?".parse::<Tile>().unwrap();
        Tile::try_from(0_u8).unwrap();
        Tile::try_from(26_u8).unwrap();
        Tile::try_from(27_u8).unwrap();

        "".parse::<Tile>().unwrap_err();
        "0s".parse::<Tile>().unwrap_err();
        "!".parse::<Tile>().unwrap_err();
        Tile::try_from(28_u8).unwrap_err();
        Tile::try_from(u8::MAX).unwrap_err();
    }

    #[test]
    fn next_prev() {
        MJAI_PAI_STRINGS.iter().take(27).for_each(|&s| {
            let tile: Tile = s.parse().unwrap();
            assert_eq!(tile.prev().next(), tile.deaka());
            assert_eq!(tile.next().prev(), tile.deaka());
        });
    }
}
