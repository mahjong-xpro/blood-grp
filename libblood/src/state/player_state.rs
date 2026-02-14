use super::action::ActionCandidate;
use super::item::KawaItem;
use crate::algo::agari::FanConfig;
use crate::algo::sp::Candidate;
use crate::hand::tiles_to_string;
use crate::must_tile;
use crate::tile::Tile;
use std::iter;

use anyhow::Result;
use derivative::Derivative;
use pyo3::prelude::*;
use serde_json as json;
use tinyvec::ArrayVec;
use std::sync::Mutex;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// PERF-01: Mutex wrapper that implements Clone by cloning the inner value.
/// Needed because PlayerState derives Clone but std::sync::Mutex does not.
pub(super) struct ClonableMutex<T: Clone>(Mutex<T>);

impl<T: Clone> ClonableMutex<T> {
    pub fn new(val: T) -> Self { Self(Mutex::new(val)) }
    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, T>> { self.0.lock() }
}

impl<T: Clone> Clone for ClonableMutex<T> {
    fn clone(&self) -> Self {
        let inner = self.0.lock().map(|g| g.clone()).unwrap_or_else(|e| e.into_inner().clone());
        Self(Mutex::new(inner))
    }
}

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
    #[pyo3(get)]
    pub forbidden_tiles: [bool; 27],

    // discarded_tiles 已移除：血战到底无永久振听规则，
    // 临时过手状态由 furiten_passed_ron_fan 追踪。

    /// Counts from 0 unlike mjai.
    pub(super) kyoku: u8,
    /// Rotated to be relative, so `scores[0]` is the score of the player.
    pub(super) scores: [i32; 4],
    // NOTE: `rank` 字段已移除。之前从未被赋值（始终为 0），浪费了 4 个观测通道。
    // 现在 rank 在 obs_repr.rs 中从 scores 实时计算。
    /// Relative to `player_id`.
    pub(super) oya: u8,
    /// 55 is the theoretical max size of kawa (108 total tiles - 52 initial hands - 1 last draw = 55 max discards)
    pub(super) kawa: [ArrayVec<[Option<KawaItem>; 55]>; 4],

    pub(super) kawa_overview: [ArrayVec<[Tile; 55]>; 4],
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
    /// 加杠候选：最多 4 个（4 个碰 + 手牌各有 1 张匹配时）
    pub(super) kakan_candidates: ArrayVec<[Tile; 4]>,
    pub(super) chankan_chance: Option<()>,
    /// Track which player performed kakan when chankan occurs (for excluding gen)
    /// This is set when chankan_chance is Some(())
    pub chankan_kakan_actor: Option<u8>,
    /// The tile that was kakan'd (for chankan gen exclusion)
    pub chankan_kakan_tile: Option<u8>,
    /// If we (self.player_id) performed a kakan, record the tile until the next step.
    ///
    /// If a `Hora` happens with `target == self.player_id` before we draw from rinshan,
    /// it indicates chankan (robbed kong) and the kakan must be reverted back to a pon
    /// in our local state to keep logs replayable.
    pub(super) pending_kakan_tile: Option<u8>,
    /// Track if the last discarded tile was after a kan (for 杠上炮)
    /// This is set in dahai() when intermediate_kan is not empty
    pub(super) last_discard_was_after_kan: bool,

    /// Track which players have won (Agari).
    /// In Bloody Battle, these players are effective "out" (stopped).
    #[derivative(Default(value = "[false; 4]"))]
    pub(super) players_agari: [bool; 4],

    /// Guo Shou Hu (Temporary Furiten) flag.
    /// If a player passes a winning tile (Ron), they cannot Ron again until
    /// their hand state changes (Draw/Pon/Kan).
    #[derivative(Default(value = "false"))]
    #[pyo3(get)]
    pub temporary_furiten: bool,
    /// Fan of the last ron opportunity that was passed.
    /// Rule extension: 过手加番可胡（while temporary_furiten is true, ron is only allowed
    /// when current fan is strictly greater than this value).
    pub(super) furiten_passed_ron_fan: Option<u8>,
    /// Fan of the current ron opportunity (if any). This is populated together
    /// with `last_cans.can_ron_agari`.
    pub(super) current_ron_fan: Option<u8>,

    pub(super) at_rinshan: bool,

    /// Used for 4-kan check.
    pub(super) kans_on_board: u8,


    pub(crate) pons: ArrayVec<[u8; 4]>,
    pub(crate) minkans: ArrayVec<[u8; 4]>,
    pub(crate) ankans: ArrayVec<[u8; 4]>,

    pub has_agari: bool,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) ding_que: Option<crate::mjai::Suit>,
    pub(super) other_ding_que: [Option<crate::mjai::Suit>; 3],

    /// For shanten calc.
    pub(super) tehai_len_div3: u8,

    /// Used in single-player features to get the shanten for 3n+2.
    pub(super) has_next_shanten_discard: bool,

    /// Configurable fan rules for this game session.
    /// Set once at game start; all AgariCalculator calls use this config.
    pub fan_config: FanConfig,

    /// PERF-01: SP 表缓存。同一决策点多次调用 encode_obs（如 normal + kan_select）时
    /// 避免重复计算。手牌/牌山变化时由 invalidate_sp_cache() 清除。
    #[derivative(Default(value = "ClonableMutex::new(None)"))]
    pub(super) cached_sp: ClonableMutex<Option<super::sp_tables::SinglePlayerTables>>,
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

    /// Get the current fan configuration.
    #[getter]
    pub fn get_fan_config(&self) -> FanConfig {
        self.fan_config
    }

    /// Set the fan configuration (should be called before game start).
    #[setter]
    pub fn set_fan_config(&mut self, config: FanConfig) {
        self.fan_config = config;
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
        let sp_res = catch_unwind(AssertUnwindSafe(|| self.single_player_tables(None)));
        match sp_res {
            Ok(Ok(tables)) => {
                for candidate in tables.max_ev_table {
                    sp_tables.push('\n');
                    sp_tables.push_str(&candidate.csv_row(can_discard).join("\t"));
                }
            }
            Ok(Err(_)) => {
                // Keep debug output stable; SP tables are best-effort in brief_info.
            }
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                sp_tables.push('\n');
                sp_tables.push_str(&format!("<panic in single_player_tables: {msg}>"));
            }
        }

        format!(
            r#"player (abs): {}
oya (rel): {}
kyoku: {}
turn: {}
score (rel): {}
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
            self.last_cans,
            self.last_self_tsumo,
            self.last_kawa_tile,
            self.tiles_left,
        )
    }
}

impl PlayerState {
    /// Check if the player has cleared all tiles of their chosen Ding Que suit from hand.
    ///
    /// # Returns
    /// - `true` if a Ding Que suit has been selected AND no tiles of that suit remain in hand
    /// - `false` if no Ding Que suit has been selected (selection phase not completed)
    /// - `false` if Ding Que suit is selected but tiles of that suit still remain in hand
    ///
    /// # Note
    /// This function returning `false` when `ding_que.is_none()` is intentional:
    /// "clearing" the Ding Que suit is only meaningful after a suit has been chosen.
    /// Before selection, the concept of "完成定缺" does not apply.
    /// Includes fuuro (pons, minkans, ankans): 整手无定缺花色才算完成定缺/非花猪。
    #[must_use]
    pub fn check_ding_que_complete(&self) -> bool {
        if self.ding_que.is_none() {
            return false; // No ding_que selected, so "completion" is not applicable
        }
        !crate::ding_que::has_ding_que_tiles_in_hand(
            &self.tehai,
            &self.pons,
            &self.minkans,
            &self.ankans,
            self.ding_que,
        )
    }

    /// 返回各玩家（相对座位）的和牌状态。
    /// `[0]` 是自己，`[1..4]` 是对手（下家、对家、上家）。
    #[inline]
    #[must_use]
    pub fn players_agari(&self) -> &[bool; 4] {
        &self.players_agari
    }

    /// Count remaining ding_que suit tiles in **concealed hand (tehai)**.
    ///
    /// NOTE: 只统计手牌（tehai），不含副露。正常规则下定缺花色不能碰/杠，
    /// 因此副露中不应有定缺牌。debug_assert 用于防御性检查。
    #[must_use]
    pub fn count_ding_que_tiles(&self) -> u8 {
        if let Some(suit) = self.ding_que {
            let (start, end) = crate::ding_que::suit_range(suit);
            let tehai_count: u8 = (start..end).map(|i| self.tehai[i]).sum();
            // 防御性检查：副露中不应含定缺花色牌（规则禁止碰/杠定缺花色）。
            // 若触发，说明存在状态腐蚀或规则执行漏洞。
            debug_assert!(
                !self.pons.iter().chain(self.minkans.iter()).chain(self.ankans.iter())
                    .any(|&t| crate::ding_que::is_ding_que_tile(t as usize, self.ding_que)),
                "Fuuro contains ding_que suit tile! player={}, ding_que={:?}, pons={:?}, minkans={:?}, ankans={:?}",
                self.player_id, self.ding_que, self.pons, self.minkans, self.ankans
            );
            tehai_count
        } else {
            0 // No ding_que selected
        }
    }
}
