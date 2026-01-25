use super::action::ActionCandidate;
use super::item::KawaItem;
use crate::algo::sp::Candidate;
use crate::hand::tiles_to_string;
use crate::must_tile;
use crate::tile::Tile;
use std::iter;

use anyhow::Result;
use derivative::Derivative;
use pyo3::prelude::*;
use serde_json as json;
use tinyvec::{ArrayVec, TinyVec};

/// `PlayerState` is the core of the lib, which holds all the observable game
/// state information from a specific seat's perspective with the ability to
/// identify the legal actions the specified player can make upon an incoming
/// mjai event, along with some helper functions to build an actual agent.
/// Notably, `PlayerState` encodes observation features into numpy arrays which
/// serve as inputs for deep learning model.
#[pyclass]
#[derive(Clone, Derivative)]
#[derivative(Default)]
pub struct PlayerState {
    pub(super) player_id: u8,

    #[derivative(Default(value = "[0; 27]"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) tehai: [u8; 27],

    /// Tiles that can be used to win (waiting tiles).
    /// Does not consider yakunashi, but does consider other kinds of
    /// forbidden wins (similar to furiten in Japanese Mahjong).
    #[derivative(Default(value = "[false; 27]"))]
    pub(super) waits: [bool; 27],

    /// For calculating `waits`, also for SPCalculator.
    #[derivative(Default(value = "[0; 27]"))]
    pub(super) tiles_seen: [u8; 27],

    #[derivative(Default(value = "[false; 27]"))]
    pub(super) keep_shanten_discards: [bool; 27],

    #[derivative(Default(value = "[false; 27]"))]
    pub(super) next_shanten_discards: [bool; 27],

    #[derivative(Default(value = "[false; 27]"))]
    pub(super) forbidden_tiles: [bool; 27],

    /// Used for checking forbidden wins (similar to furiten in Japanese Mahjong).
    #[derivative(Default(value = "[false; 27]"))]
    pub(super) discarded_tiles: [bool; 27],

    /// Counts from 0 unlike mjai.
    pub(super) kyoku: u8,
    /// Rotated to be relative, so `scores[0]` is the score of the player.
    pub(super) scores: [i32; 4],
    pub(super) rank: u8,
    /// Relative to `player_id`.
    pub(super) oya: u8,
    /// 24 is the theoretical max size of kawa, however, since None is included
    /// in the kawa, in some very rare cases (about one in a million hanchans),
    /// the size can exceed 24.
    pub(super) kawa: [TinyVec<[Option<KawaItem>; 24]>; 4],

    pub(super) kawa_overview: [ArrayVec<[Tile; 24]>; 4],
    pub(super) fuuro_overview: [ArrayVec<[ArrayVec<[Tile; 4]>; 4]>; 4],
    /// In this field all `Tile` are normalized (no aka dora distinction in Bloody Battle Mahjong)
    pub(super) ankan_overview: [ArrayVec<[Tile; 4]>; 4],

    pub(super) at_turn: u8,
    pub(super) tiles_left: u8,
    pub(super) intermediate_kan: ArrayVec<[Tile; 4]>,

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) shanten: i8,

    pub(super) last_self_tsumo: Option<Tile>,
    pub(super) last_kawa_tile: Option<Tile>,
    pub(super) last_cans: ActionCandidate,

    pub(super) ankan_candidates: ArrayVec<[Tile; 3]>,
    pub(super) kakan_candidates: ArrayVec<[Tile; 3]>,
    pub(super) chankan_chance: Option<()>,
    /// Track which player performed kakan when chankan occurs (for excluding gen)
    /// This is set when chankan_chance is Some(())
    pub chankan_kakan_actor: Option<u8>,
    /// The tile that was kakan'd (for chankan gen exclusion)
    pub chankan_kakan_tile: Option<u8>,
    /// Track if the last discarded tile was after a kan (for 杠上炮)
    /// This is set in dahai() when intermediate_kan is not empty
    pub(super) last_discard_was_after_kan: bool,

    pub(super) at_rinshan: bool,

    /// Used for 4-kan check.
    pub(super) kans_on_board: u8,

    pub(super) is_menzen: bool,
    pub(super) pons: ArrayVec<[u8; 4]>,
    pub(super) minkans: ArrayVec<[u8; 4]>,
    pub(super) ankans: ArrayVec<[u8; 4]>,

    pub has_agari: bool,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) ding_que: Option<crate::mjai::Suit>,
    pub(super) other_ding_que: [Option<crate::mjai::Suit>; 3],

    /// For shanten calc.
    pub(super) tehai_len_div3: u8,

    /// Used in single-player features to get the shanten for 3n+2.
    pub(super) has_next_shanten_discard: bool,
}

#[pymethods]
impl PlayerState {
    /// Panics if `player_id` is outside of range [0, 3].
    #[new]
    #[must_use]
    pub fn new(player_id: u8) -> Self {
        assert!(player_id < 4, "{player_id} is not in range [0, 3]");
        Self {
            player_id,
            ..Default::default()
        }
    }

    /// Returns an `ActionCandidate`.
    #[pyo3(name = "update")]
    pub(super) fn update_json(&mut self, mjai_json: &str) -> Result<ActionCandidate> {
        let event = json::from_str(mjai_json)?;
        self.update(&event)
    }

    /// Raises an exception if the action is not valid.
    #[pyo3(name = "validate_reaction")]
    pub(super) fn validate_reaction_json(&self, mjai_json: &str) -> Result<()> {
        let action = json::from_str(mjai_json)?;
        self.validate_reaction(&action)
    }

    /// For debug only.
    ///
    /// Return a human readable description of the current state.
    #[must_use]
    pub fn brief_info(&self) -> String {
        let waits = self
            .waits
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .map(|(i, _)| must_tile!(i))
            .collect::<Vec<_>>();

        let zipped_kawa = self.kawa[0]
            .iter()
            .chain(iter::repeat(&None))
            .zip(self.kawa[1].iter().chain(iter::repeat(&None)))
            .zip(self.kawa[2].iter().chain(iter::repeat(&None)))
            .zip(self.kawa[3].iter().chain(iter::repeat(&None)))
            .take_while(|row| !matches!(row, &(((None, None), None), None)))
            .enumerate()
            .map(|(i, (((a, b), c), d))| {
                format!(
                    "{i:2}. {}\t{}\t{}\t{}",
                    a.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                    b.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                    c.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                    d.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let can_discard = self.last_cans.can_discard;
        let mut sp_tables = Candidate::csv_header(can_discard).join("\t");
        if let Ok(tables) = self.single_player_tables(None) {
            for candidate in tables.max_ev_table {
                sp_tables.push('\n');
                sp_tables.push_str(&candidate.csv_row(can_discard).join("\t"));
            }
        }

        format!(
            r#"player (abs): {}
oya (rel): {}
kyoku: {}-{}
turn: {}
score (rel): {:?}
tehai: {}
fuuro: {:?}
ankan: {:?}
tehai len: {}
shanten: {} (actual: {})
waits: {waits:?}
action candidates: {:#?}
last self tsumo: {:?}
last kawa tile: {:?}
tiles left: {}
kawa:
{zipped_kawa}
single player table (max EV):
{sp_tables}"#,
            self.player_id,
            self.oya,
            self.kyoku + 1,
            self.at_turn,
            format!("{:?}", self.scores),
            tiles_to_string(&self.tehai),
            format!("{:?}", self.fuuro_overview[0]),
            format!("{:?}", self.ankan_overview[0]),
            self.tehai_len_div3,
            self.shanten,
            self.real_time_shanten(),
            format!("{:#?}", self.last_cans),
            self.last_cans,
            self.last_self_tsumo,
            self.last_kawa_tile,
            self.tiles_left,
        )
    }
}

impl PlayerState {
    /// Check if ding que (定缺) is complete (no ding_que suit tiles in hand)
    #[must_use]
    pub fn check_ding_que_complete(&self) -> bool {
        if let Some(suit) = self.ding_que {
            let (start, end) = match suit {
                crate::mjai::Suit::Man => (0, 9),
                crate::mjai::Suit::Pin => (9, 18),
                crate::mjai::Suit::Sou => (18, 27),
            };
            (start..end).all(|i| self.tehai[i] == 0)
        } else {
            false // No ding_que selected
        }
    }

    /// Count remaining ding_que suit tiles in hand
    #[must_use]
    pub fn count_ding_que_tiles(&self) -> u8 {
        if let Some(suit) = self.ding_que {
            let (start, end) = match suit {
                crate::mjai::Suit::Man => (0, 9),
                crate::mjai::Suit::Pin => (9, 18),
                crate::mjai::Suit::Sou => (18, 27),
            };
            (start..end).map(|i| self.tehai[i]).sum()
        } else {
            0 // No ding_que selected
        }
    }
}
