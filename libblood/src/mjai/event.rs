use crate::tile::Tile;
use std::error::Error;
use std::fmt;

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use serde_with::{TryFromInto, serde_as, skip_serializing_none};

/// Suit for Ding Que (定缺) in Bloody Battle Mahjong
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Suit {
    Man,  // 万子
    Pin,  // 筒子
    Sou,  // 条子
}

/// Describes an event in mjai format.
///
/// Mjai protocol was originally defined in
/// <https://gimite.net/pukiwiki/index.php?Mjai%20%E9%BA%BB%E9%9B%80AI%E5%AF%BE%E6%88%A6%E3%82%B5%E3%83%BC%E3%83%90>.
/// This implementation does not contain the full specs defined in the original
/// one, and it has some extensions added.
#[serde_as]
#[skip_serializing_none]
#[derive(Debug, Default, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Event {
    #[default]
    None,

    StartGame {
        #[serde(default)]
        names: [String; 4],

        /// Consists of (nonce, key).
        seed: Option<(u64, u64)>,
    },
    StartKyoku {
        /// Counts from 1 (for recording only, no game flow impact)
        #[serde_as(deserialize_as = "TryFromInto<BoundedU8<1, 4>>")]
        kyoku: u8,
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        oya: u8,
        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
    },

    Tsumo {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        pai: Tile,
    },
    Dahai {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        pai: Tile,
        tsumogiri: bool,
    },

    // Chi event removed - Bloody Battle Mahjong does not have chi
    Pon {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        target: u8,
        pai: Tile,
        consumed: [Tile; 2],
    },
    Daiminkan {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        target: u8,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Kakan {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        pai: Tile,
        consumed: [Tile; 3],
    },
    Ankan {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        consumed: [Tile; 4],
    },
    // Dora event removed - Bloody Battle Mahjong does not have dora
    // Reach and ReachAccepted events removed - Bloody Battle Mahjong does not have riichi

    DingQue {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        suit: Suit,
    },

    Hora {
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        actor: u8,
        #[serde_as(deserialize_as = "TryFromInto<Actor>")]
        target: u8,

        deltas: Option<[i32; 4]>,
        // ura_markers removed - Bloody Battle Mahjong does not have ura dora
    },
    Ryukyoku {
        deltas: Option<[i32; 4]>,
    },

    EndKyoku,
    EndGame,
}

#[derive(Deserialize)]
struct BoundedU8<const MIN: u8, const MAX: u8>(u8);

type Actor = BoundedU8<0, 3>;

#[derive(Debug)]
pub struct OutOfBoundError(pub u8);

/// An extended version of `Event` which allows metadata recording.
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventExt {
    #[serde(flatten)]
    pub event: Event,
    pub meta: Option<Metadata>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub q_values: Option<Vec<f32>>,
    pub mask_bits: Option<u64>,
    pub is_greedy: Option<bool>,
    pub batch_size: Option<usize>,
    pub eval_time_ns: Option<u64>,
    pub shanten: Option<i8>,
    pub at_furiten: Option<bool>,
    pub kan_select: Option<Box<Metadata>>,
}

#[derive(Serialize, Deserialize)]
pub struct EventWithCanAct {
    #[serde(flatten)]
    pub event: Event,
    pub can_act: Option<bool>,
}

impl Event {
    #[inline]
    #[must_use]
    pub const fn actor(&self) -> Option<u8> {
        match *self {
            Self::Tsumo { actor, .. }
            | Self::Dahai { actor, .. }
            | Self::Pon { actor, .. }
            | Self::Daiminkan { actor, .. }
            | Self::Kakan { actor, .. }
            | Self::Ankan { actor, .. }
            | Self::DingQue { actor, .. }
            | Self::Hora { actor, .. } => Some(actor),
            _ => None,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_in_game_announce(&self) -> bool {
        matches!(self, Self::Hora { .. })
    }

    pub fn augment(&mut self) {
        const fn swap_tile(t: &mut Tile) {
            *t = t.augment();
        }

        match self {
            Self::StartKyoku { tehais, .. } => {
                tehais.iter_mut().flatten().for_each(swap_tile);
            }
            Self::Tsumo { pai, .. } | Self::Dahai { pai, .. } => swap_tile(pai),
            Self::Pon { pai, consumed, .. } => {
                swap_tile(pai);
                consumed.iter_mut().for_each(swap_tile);
            }
            Self::Daiminkan { pai, consumed, .. } | Self::Kakan { pai, consumed, .. } => {
                swap_tile(pai);
                consumed.iter_mut().for_each(swap_tile);
            }
            Self::Ankan { consumed, .. } => consumed.iter_mut().for_each(swap_tile),
            Self::Hora { .. } => {
                // No ura_markers in Bloody Battle Mahjong
            }
            _ => (),
        }
    }
}

impl<const MIN: u8, const MAX: u8> TryFrom<BoundedU8<MIN, MAX>> for u8 {
    type Error = OutOfBoundError;

    fn try_from(value: BoundedU8<MIN, MAX>) -> Result<Self, Self::Error> {
        if (MIN..=MAX).contains(&value.0) {
            Ok(value.0)
        } else {
            Err(OutOfBoundError(value.0))
        }
    }
}

impl fmt::Display for OutOfBoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "out-of-range number {}", self.0)
    }
}

impl Error for OutOfBoundError {}

impl EventExt {
    #[inline]
    #[must_use]
    pub const fn no_meta(event: Event) -> Self {
        Self { event, meta: None }
    }
}

impl From<Event> for EventExt {
    fn from(ev: Event) -> Self {
        Self::no_meta(ev)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use serde_json::{self as json, Map, Number, Value, json};

    #[test]
    fn json_consistency() {
        let lines = r#"
            {"type":"none"}
            {"type":"start_game","names":["Equim","Mortal","akochan","NoName"],"seed":[123,456]}
            {"type":"start_kyoku","kyoku":1,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","1m","2m","3m","4m"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]]}
            {"type":"ding_que","actor":0,"suit":"man"}
            {"type":"ding_que","actor":1,"suit":"pin"}
            {"type":"ding_que","actor":2,"suit":"sou"}
            {"type":"ding_que","actor":3,"suit":"man"}
            {"type":"tsumo","actor":0,"pai":"1m"}
            {"type":"dahai","actor":0,"pai":"2m","tsumogiri":true}
            {"type":"pon","actor":1,"target":0,"pai":"5p","consumed":["5p","5p"]}
            {"type":"daiminkan","actor":2,"target":0,"pai":"5s","consumed":["5s","5s","5s"]}
            {"type":"kakan","actor":3,"pai":"9m","consumed":["9m","9m","9m"]}
            {"type":"ankan","actor":0,"consumed":["9m","9m","9m","9m"]}
            {"type":"hora","actor":3,"target":1,"deltas":[0,-8000,0,8000]}
            {"type":"hora","actor":3,"target":1}
            {"type":"ryukyoku","deltas":[0,1500,0,-1500]}
            {"type":"ryukyoku"}
            {"type":"end_kyoku"}
            {"type":"end_game"}
        "#.trim();

        let expected: Vec<Value> = lines.lines().map(|l| json::from_str(l).unwrap()).collect();
        let actual: Vec<Value> = lines
            .lines()
            .map(|l| {
                let event: Event = json::from_str(l).unwrap();
                json::to_value(event).unwrap()
            })
            .collect();

        assert_eq!(expected, actual);
    }

    #[test]
    fn bound_check() {
        let value = json! ({
            "type": "ding_que",
            "actor": 4,
            "suit": "man",
        });
        json::from_value::<Event>(value).unwrap_err();

        let value = json! ({
            "type": "hora",
            "actor": 0,
            "target": 5,
        });
        json::from_value::<Event>(value).unwrap_err();

        let value = json!({
            "type": "start_kyoku",
            "kyoku": 1,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
                ["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],
                ["1s","2s","3s","4s","5s","6s","7s","8s","9s","1m","2m","3m","4m"],
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
            ],
        });
        let obj: Map<String, Value> = json::from_value(value).unwrap();
        json::from_value::<Event>(Value::Object(obj.clone())).unwrap();

        let mut test_obj = obj.clone();
        test_obj["kyoku"] = Value::Number(Number::from(0));
        json::from_value::<Event>(Value::Object(test_obj)).unwrap_err();

        let mut test_obj = obj;
        test_obj["kyoku"] = Value::Number(Number::from(5));
        json::from_value::<Event>(Value::Object(test_obj)).unwrap_err();
    }
}
