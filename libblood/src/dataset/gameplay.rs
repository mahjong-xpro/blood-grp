use super::{GameScore, Invisible};
use crate::mjai::Event;
use crate::state::PlayerState;
use std::array;
use std::fs::File;
use std::io;
use std::mem;

use ahash::AHashSet;
use anyhow::{Context, Result, bail};
use derivative::Derivative;
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json as json;
use tinyvec::ArrayVec;

#[pyclass]
#[derive(Derivative)]
#[derivative(Debug)]
pub struct GameplayLoader {
    #[pyo3(get)]
    version: u32,
    #[pyo3(get)]
    oracle: bool,
    #[pyo3(get)]
    player_names: Vec<String>,
    #[pyo3(get)]
    excludes: Vec<String>,
    #[pyo3(get)]
    trust_seed: bool,
    #[pyo3(get)]
    always_include_kan_select: bool,
    #[pyo3(get)]
    augmented: bool,

    #[derivative(Debug = "ignore")]
    player_names_set: AHashSet<String>,
    #[derivative(Debug = "ignore")]
    excludes_set: AHashSet<String>,
}

#[pyclass]
#[derive(Clone, Default)]
pub struct Gameplay {
    // per move
    pub obs: Vec<Array2<f32>>,
    pub invisible_obs: Vec<Array2<f32>>,
    pub actions: Vec<i64>,
    pub masks: Vec<Array1<bool>>,
    pub at_kyoku: Vec<u8>,
    pub dones: Vec<bool>,
    pub apply_gamma: Vec<bool>,
    pub at_turns: Vec<u8>,
    pub shantens: Vec<i8>,
    /// Opponent waits per move: 3 opponents × 27 tiles = 81 booleans
    pub opponent_waits: Vec<[bool; 81]>,

    // per game
    pub game_score: GameScore,
    pub player_id: u8,
    pub player_name: String,
}

struct LoaderContext<'a> {
    config: &'a GameplayLoader,
    invisibles: Option<&'a [Invisible]>,

    state: PlayerState,
    kyoku_idx: usize,

    // fields below are only used for oracle
    opponent_states: [PlayerState; 3],
    yama_idx: usize,
}

#[pymethods]
impl GameplayLoader {
    #[new]
    #[pyo3(signature = (
        version,
        *,
        oracle = true,
        player_names = None,
        excludes = None,
        trust_seed = false,
        always_include_kan_select = true,
        augmented = false,
    ))]
    fn new(
        version: u32,
        oracle: bool,
        player_names: Option<Vec<String>>,
        excludes: Option<Vec<String>>,
        trust_seed: bool,
        always_include_kan_select: bool,
        augmented: bool,
    ) -> Self {
        let player_names = player_names.unwrap_or_default();
        let player_names_set = player_names.iter().cloned().collect();
        let excludes = excludes.unwrap_or_default();
        let excludes_set = excludes.iter().cloned().collect();
        Self {
            version,
            oracle,
            player_names,
            excludes,
            trust_seed,
            always_include_kan_select,
            augmented,
            player_names_set,
            excludes_set,
        }
    }

    // Nested result is too hard to handle...
    fn load_log(&self, raw_log: &str) -> Result<Vec<Gameplay>> {
        let mut events = raw_log
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .context("failed to parse log")?;
        if self.augmented {
            events.iter_mut().for_each(Event::augment);
        }
        self.load_events(&events)
    }

    #[pyo3(name = "load_gz_log_files")]
    fn load_gz_log_files_py(&self, gzip_filenames: Vec<String>) -> Result<Vec<Vec<Gameplay>>> {
        self.load_gz_log_files(gzip_filenames)
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl GameplayLoader {
    pub fn load_gz_log_files<V, S>(&self, gzip_filenames: V) -> Result<Vec<Vec<Gameplay>>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        gzip_filenames
            .into_par_iter()
            .map(|f| {
                let filename = f.as_ref();
                let inner = || {
                    let file = File::open(filename)?;
                    let gz = GzDecoder::new(file);
                    let raw = io::read_to_string(gz)?;
                    self.load_log(&raw)
                };
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect()
    }

    pub fn load_events(&self, events: &[Event]) -> Result<Vec<Gameplay>> {
        let invisibles = self.oracle.then(|| Invisible::new(events, self.trust_seed));

        let [Event::StartGame { names, .. }, ..] = events else {
            bail!("empty or invalid game log");
        };
        // names is [String; 4] by type, but validate to catch any deserialization issues
        if names.len() != 4 {
            bail!("invalid game log: expected exactly 4 players, found {}", names.len());
        }
        let player_ids: ArrayVec<[u8; 4]> = names
            .iter()
            .enumerate()
            .filter(|&(_, name)| {
                if !self.player_names_set.is_empty() {
                    return self.player_names_set.contains(name);
                }
                if !self.excludes_set.is_empty() {
                    return !self.excludes_set.contains(name);
                }
                true
            })
            .map(|(i, _)| i as u8)
            .collect();
        
        player_ids
            .into_par_iter()
            .map(|&player_id| {
                Gameplay::load_events_by_player(self, events, player_id, invisibles.as_deref())
            })
            .collect()
    }
}

#[pymethods]
impl Gameplay {
    fn take_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.obs)
            .into_iter()
            .map(|v| PyArray2::from_owned_array(py, v))
            .collect()
    }
    fn take_invisible_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.invisible_obs)
            .into_iter()
            .map(|v| PyArray2::from_owned_array(py, v))
            .collect()
    }
    fn take_actions(&mut self) -> Vec<i64> {
        mem::take(&mut self.actions)
    }
    fn take_masks<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<bool>>> {
        mem::take(&mut self.masks)
            .into_iter()
            .map(|v| PyArray1::from_owned_array(py, v))
            .collect()
    }
    fn take_at_kyoku(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_kyoku)
    }
    fn take_dones(&mut self) -> Vec<bool> {
        mem::take(&mut self.dones)
    }
    fn take_apply_gamma(&mut self) -> Vec<bool> {
        mem::take(&mut self.apply_gamma)
    }
    fn take_at_turns(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_turns)
    }
    fn take_shantens(&mut self) -> Vec<i8> {
        mem::take(&mut self.shantens)
    }
    fn take_opponent_waits(&mut self) -> Vec<[bool; 81]> {
        mem::take(&mut self.opponent_waits)
    }

    fn take_game_score(&mut self) -> GameScore {
        mem::take(&mut self.game_score)
    }

    const fn take_player_id(&self) -> u8 {
        self.player_id
    }
}

impl Gameplay {
    fn load_events_by_player(
        config: &GameplayLoader,
        events: &[Event],
        player_id: u8,
        invisibles: Option<&[Invisible]>,
    ) -> Result<Self> {
        let game_score = GameScore::load_events(events)?;

        let mut data = Self {
            game_score,
            player_id,
            ..Default::default()
        };

        let mut ctx = LoaderContext {
            config,
            invisibles,
            state: PlayerState::new(player_id),
            kyoku_idx: 0,
            // end_state: EndState::Passive,
            opponent_states: array::from_fn(|i| PlayerState::new((player_id + i as u8 + 1) % 4)),
            yama_idx: 0,
        };

        // It is guaranteed that there are at least 4 events.
        // tsumo/dahai -> ryukyoku/hora -> end kyoku -> end game
        for wnd in events.windows(4) {
            data.extend_from_event_window(&mut ctx, wnd.try_into().unwrap())?;
        }

        data.dones = data.at_kyoku.windows(2).map(|w| w[1] > w[0]).collect();
        data.dones.push(true);

        Ok(data)
    }

    fn extend_from_event_window(
        &mut self,
        ctx: &mut LoaderContext<'_>,
        wnd: &[Event; 4],
    ) -> Result<()> {
        let cur = &wnd[0];
        let next = &wnd[1];

        match cur {
            Event::StartGame { names, .. } => {
                self.player_name.clone_from(&names[self.player_id as usize]);
            }
            Event::EndKyoku => ctx.kyoku_idx += 1,
            _ => (),
        }

        if ctx.invisibles.is_some() {
            match cur {
                Event::EndKyoku => {
                    ctx.yama_idx = 0;
                }
                Event::Tsumo { .. } => {
                    ctx.yama_idx += 1;
                }
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                }
                _ => (),
            };

            for s in &mut ctx.opponent_states {
                s.update(cur)?;
            }
        }

        let cans = ctx.state.update(cur)?;
        if !cans.can_act() {
            return Ok(());
        }

        let version = ctx.config.version;
        let mut kan_select = None;
        let label_opt = match *next {
            Event::Dahai { pai, .. } => Some(pai.as_usize()),
            Event::Pon { actor, .. } if actor == self.player_id => Some(27), // Pon action (was 41)
            Event::Daiminkan { actor, pai, .. } if actor == self.player_id => {
                if ctx.config.always_include_kan_select {
                    kan_select = Some(pai.as_usize());
                }
                Some(28) // Kan action (was 42)
            }
            Event::Kakan { pai, .. } => {
                if ctx.config.always_include_kan_select || ctx.state.kakan_candidates().len() > 1 {
                    kan_select = Some(pai.as_usize());
                }
                Some(28) // Kan action (was 42)
            }
            Event::Ankan { consumed, .. } => {
                if ctx.config.always_include_kan_select || ctx.state.ankan_candidates().len() > 1 {
                    kan_select = Some(consumed[0].as_usize());
                }
                Some(28) // Kan action (was 42)
            }
            Event::DingQue { actor, suit } if actor == self.player_id => {
                // Assert legality: DingQue should only appear when can_ding_que is true.
                if !cans.can_ding_que {
                    bail!(
                        "Dataset mismatch: DingQue labeled when can_ding_que=false.\n\
                         player: {}, kyoku: {}, turn: {}\n\
                         cur: {:?}\n\
                         wnd: {:?}\n\
                         next: {:?}\n\
                         state:\n{}",
                        self.player_id,
                        ctx.kyoku_idx,
                        ctx.state.at_turn(),
                        cur,
                        wnd,
                        next,
                        ctx.state.brief_info(),
                    );
                }
                match suit {
                    crate::mjai::Suit::Man => Some(31),
                    crate::mjai::Suit::Pin => Some(32),
                    crate::mjai::Suit::Sou => Some(33),
                }
            }

            _ => {
                let mut ret = None;

                // If the immediate next event is Hora, the current event produced a ron window
                // (multi-ron is represented as consecutive Hora events).
                //
                // IMPORTANT: only Hora events *contiguous right after cur* are reactions to `cur`.
                // A later Hora after an intervening Tsumo is a separate decision and must be
                // captured when `cur` is that Tsumo window, not here.
                let has_any_ron = matches!(wnd[1], Event::Hora { .. });
                if has_any_ron {
                    // Check if the POV is one of those who made Hora (in the contiguous Hora run).
                    for ev in &wnd[1..] {
                        match ev {
                            Event::Hora { actor, .. } if *actor == self.player_id => {
                                ret = Some(29); // Agari action (was 43)
                                break;
                            }
                            Event::Hora { .. } => {
                                // another player's ron (continue scanning contiguous Hora run)
                            }
                            Event::EndKyoku => break,
                            _ => break, // stop at the first non-Hora event (e.g. Tsumo)
                        }
                    }
                }

                if ret.is_none() {
                    // It is now proven there is no ron from the POV.
                    // Determine if POV actively passed on a reaction opportunity:
                    //
                    // 1. !has_any_ron: No one ronned. POV had pon/daiminkan/ron options
                    //    and chose none → active pass.
                    // 2. has_any_ron && can_ron_agari: Others ronned, but POV ALSO could
                    //    have ronned and actively declined → active pass (rare in blood battle).
                    // 3. has_any_ron && !can_ron_agari: Others' ron preempts POV's pon/daiminkan.
                    //    POV never got to decide → NOT an active pass, skip.
                    if cans.can_pon || cans.can_daiminkan || cans.can_ron_agari {
                        if !has_any_ron || cans.can_ron_agari {
                            ret = Some(30); // Pass action
                        }
                    }
                }

                ret
            }
        };

        if let Some(label) = label_opt {
            // If we are about to record an agari action, ensure the current state's mask allows it.
            // When this fails, it indicates either a non-replayable log (state correction not encoded)
            // or a bug in legality / mask generation.
            if label == 29 {
                let (_feature_dbg, mask_normal) = ctx.state.encode_obs(version, false);
                let (_feature_dbg2, mask_kan) = ctx.state.encode_obs(version, true);
                if !mask_normal[29] && !mask_kan[29] {
                    bail!(
                        "Mask Mismatch detected for Agari (Action 29) at window boundary.\n\
                         player: {}, kyoku: {}, turn: {}\n\
                         cur: {:?}\n\
                         wnd: {:?}\n\
                         next: {:?}\n\
                         state:\n{}",
                        self.player_id,
                        ctx.kyoku_idx,
                        ctx.state.at_turn(),
                        cur,
                        wnd,
                        next,
                        ctx.state.brief_info(),
                    );
                }
            }

            if let Some(kan) = kan_select {
                // Assert kan-select label is allowed by the kan-select mask.
                let (_feature_dbg, mask_dbg) = ctx.state.encode_obs(version, true);
                if !mask_dbg[kan] {
                    bail!(
                        "Dataset mismatch: kan_select label not allowed by mask.\n\
                         player: {}, kyoku: {}, turn: {}, kan: {}\n\
                         cur: {:?}\n\
                         wnd: {:?}\n\
                         next: {:?}\n\
                         state:\n{}",
                        self.player_id,
                        ctx.kyoku_idx,
                        ctx.state.at_turn(),
                        kan,
                        cur,
                        wnd,
                        next,
                        ctx.state.brief_info(),
                    );
                }
            }

            let ctx_msg = {
                let turn = ctx.state.at_turn();
                let state_info = ctx.state.brief_info();
                format!(
                    "Dataset mismatch: action label not allowed by mask.\n\
                     player: {}, kyoku: {}, turn: {}\n\
                     label: {}\n\
                     cur: {:?}\n\
                     wnd: {:?}\n\
                     next: {:?}\n\
                     state:\n{}",
                    self.player_id,
                    ctx.kyoku_idx,
                    turn,
                    label,
                    cur,
                    wnd,
                    next,
                    state_info,
                )
            };
            self.add_entry(ctx, false, label).with_context(|| ctx_msg.clone())?;
            if let Some(kan) = kan_select {
                let ctx_msg = {
                    let turn = ctx.state.at_turn();
                    let state_info = ctx.state.brief_info();
                    format!(
                        "Dataset mismatch: kan_select label not allowed by mask.\n\
                         player: {}, kyoku: {}, turn: {}\n\
                         kan: {}\n\
                         cur: {:?}\n\
                         wnd: {:?}\n\
                         next: {:?}\n\
                         state:\n{}",
                        self.player_id,
                        ctx.kyoku_idx,
                        turn,
                        kan,
                        cur,
                        wnd,
                        next,
                        state_info,
                    )
                };
                self.add_entry(ctx, true, kan).with_context(|| ctx_msg.clone())?;
            }
        }
        Ok(())
    }

    fn add_entry(&mut self, ctx: &LoaderContext<'_>, at_kan_select: bool, label: usize) -> Result<()> {
        let cans = ctx.state.last_cans();
        let tehai_sum: u8 = ctx.state.tehai.iter().sum();
        // Invariant: when we can discard, tehai must be 3n+2 (normally 14 tiles).
        // If violated, downstream feature generation (SP tables) can panic (ArrayVec overflow),
        // which would otherwise lose all context in Python multiprocessing.
        if cans.can_discard && tehai_sum > 14 {
            bail!(
                "Dataset mismatch: invalid hand size at discard window (tehai_sum={} > 14).\n\
                 player: {}, kyoku: {}, turn: {}, at_kan_select: {}\n\
                 label: {}, cans: {:?}\n\
                 state:\n{}",
                tehai_sum,
                self.player_id,
                ctx.kyoku_idx,
                ctx.state.at_turn(),
                at_kan_select,
                label,
                cans,
                ctx.state.brief_info(),
            );
        }

        let (feature, mask) = ctx.state.encode_obs(ctx.config.version, at_kan_select);
        // Action indices: 0-26 (discard), 27 (pon), 28 (kan), 29 (agari), 30 (pass), 31-33 (ding que)
        // Strict invariant: label must be allowed by the mask computed from the same state.
        // If this fails, it indicates a replay divergence or an event-window labeling bug.
        if label >= mask.len() || !mask[label] {
            bail!(
                "Mask mismatch: label not allowed by mask.\n\
                 player: {}, kyoku: {}, turn: {}, at_kan_select: {}\n\
                 label: {}, shanten: {}, ding_que: {:?}\n\
                 cans: {:?}\n\
                 state:\n{}",
                self.player_id,
                ctx.kyoku_idx,
                ctx.state.at_turn(),
                at_kan_select,
                label,
                ctx.state.shanten(),
                ctx.state.ding_que,
                cans,
                ctx.state.brief_info(),
            );
        }

        self.obs.push(feature);
        self.actions.push(label as i64);
        self.masks.push(mask);
        
        self.at_kyoku.push(ctx.kyoku_idx as u8);
        // Only "mainline" decisions should advance the discount horizon.
        // - Discard: 0-26 (at_kan_select=false)
        // - Kan: 28 (at_kan_select=false)
        //
        // Do NOT discount on:
        // - kan_select entries (at_kan_select=true): treat as part of the same decision
        // - pon/agari/pass/ding_que: non-advancing or terminal-like actions for value target
        let apply_gamma = !at_kan_select && (label <= 26 || label == 28);
        self.apply_gamma.push(apply_gamma);
        self.at_turns.push(ctx.state.at_turn());
        self.shantens.push(ctx.state.shanten());

        // Collect opponent waits for auxiliary learning.
        // 仅 oracle 模式下 opponent_states 被实际更新（有完整手牌信息），
        // waits() 才有意义。非 oracle 模式下 opponent_states 未更新，waits() 全为 false。
        // 此处显式区分，避免误用全零数据作为辅助损失的训练标签。
        let mut opp_waits = [false; 81];
        if ctx.invisibles.is_some() {
            for (i, s) in ctx.opponent_states.iter().enumerate() {
                for (t, &waiting) in s.waits().iter().enumerate() {
                    opp_waits[i * 27 + t] = waiting;
                }
            }
        }
        self.opponent_waits.push(opp_waits);

        if let Some(invisibles) = ctx.invisibles {
            let invisible_obs = invisibles[ctx.kyoku_idx].encode(
                &ctx.opponent_states,
                ctx.yama_idx,
                0, // Reserved parameter (previously rinshan_idx, not used in Bloody Battle Mahjong)
                ctx.config.version,
            );
            self.invisible_obs.push(invisible_obs);
        }
        Ok(())
    }
}
