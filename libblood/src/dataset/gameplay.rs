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
use serde::Deserialize;

#[derive(Deserialize)]
struct PackedChunk {
    version: u32,
    games: Vec<Vec<Event>>,
}

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

    #[pyo3(name = "load_binary_chunk")]
    fn load_binary_chunk(&self, path: String) -> Result<Vec<Gameplay>> {
        let file = File::open(&path).with_context(|| format!("failed to open packed chunk: {}", path))?;
        let reader = std::io::BufReader::new(file);
        let decoder = lz4::Decoder::new(reader)?;
        
        let chunk: PackedChunk = bincode::deserialize_from(decoder)
            .with_context(|| format!("failed to deserialize packed chunk: {}", path))?;

        if chunk.version != 1 {
            anyhow::bail!("unsupported chunk version: {}", chunk.version);
        }

        let mut games = Vec::with_capacity(chunk.games.len());
        for events in chunk.games {
             // Re-use existing load_events logic
             let invisibles = self.oracle.then(|| Invisible::new(&events, self.trust_seed));

             let [Event::StartGame { names, .. }, ..] = events.as_slice() else {
                 continue; // fail silently or log?
             };
             if names.len() != 4 { continue; }

            // Logic copied from load_events but non-parallel for now (chunk is already big)
            // Actually, we can just call self.load_events(&events) but it returns Vec<Gameplay> (4 players)
            // But here we iterate over many games.
            // Wait, load_events processes ONE game (events slice) and returns 4 Gameplays (one per player).
            // Yes.
            
            let loaded = self.load_events(&events)?;
            games.extend(loaded);
        }

        Ok(games)
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

    fn take_game_score(&mut self) -> GameScore {
        mem::take(&mut self.game_score)
    }

    const fn take_player_id(&self) -> u8 {
        self.player_id
    }

    /// Optimized: Return fully processed training samples (Tuple) directly.
    /// Returns: Vec<(obs, action, mask, steps, reward, rank)>
    fn take_batch<'py>(&mut self, py: Python<'py>) -> Vec<(Bound<'py, PyArray2<f32>>, i64, Bound<'py, PyArray1<bool>>, i64, f32, u8)> {
        let game_size = self.obs.len();
        if game_size == 0 {
            return vec![];
        }

        // 1. Calculate Rewards (Delta Points)
        let player_scores = self.game_score.scores_history.iter().map(|s| s[self.player_id as usize]).collect::<Vec<_>>();
        let final_score = self.game_score.final_scores[self.player_id as usize];
        let mut seq = Vec::with_capacity(player_scores.len() + 1);
        seq.extend_from_slice(&player_scores);
        seq.push(final_score);
        
        // Calculate raw delta points then normalize (1.0 = 10000 pts)
        let kyoku_rewards: Vec<f32> = seq.windows(2).map(|w| (w[1] - w[0]) as f32 / 10000.0).collect();

        // 2. Calculate Ranks per step
        // We need (scores_history + final_scores) for all 4 players
        let mut rank_by_player_seq = Vec::with_capacity(self.game_score.scores_history.len() + 1);
        // History ranks
        for scores in &self.game_score.scores_history {
            let mut s_with_idx: Vec<(i32, usize)> = scores.iter().enumerate().map(|(i, &s)| (s, i)).collect();
            // Sort Descending (High score first)
            s_with_idx.sort_by(|a, b| b.0.cmp(&a.0));
            let mut ranks = [0u8; 4];
            for (rank, &(_, idx)) in s_with_idx.iter().enumerate() {
                ranks[idx] = rank as u8;
            }
            rank_by_player_seq.push(ranks[self.player_id as usize]);
        }
        // Final rank
        {
            let mut s_with_idx: Vec<(i32, usize)> = self.game_score.final_scores.iter().enumerate().map(|(i, &s)| (s, i)).collect();
            s_with_idx.sort_by(|a, b| b.0.cmp(&a.0));
            // Proper resolving of ties if needed? Rust sort is stable. 
            // Python argsort kind='stable'.
            // Here just rank 0..3
            let mut final_rank = 0;
            for (rank, &(_, idx)) in s_with_idx.iter().enumerate() {
                if idx == self.player_id as usize {
                    final_rank = rank as u8;
                    break;
                }
            }
            rank_by_player_seq.push(final_rank);
        }

        // 3. Calculate Steps to Done (Reverse Pass)
        let mut steps_to_done = vec![0; game_size];
        let mut steps = 0;
        for i in (0..game_size).rev() {
            if self.dones[i] {
                steps = 0;
            } else {
                steps += self.apply_gamma[i] as i64;
            }
            steps_to_done[i] = steps;
        }

        // 4. Zip and Move Ownership
        let mut samples = Vec::with_capacity(game_size);
        
        let obs = mem::take(&mut self.obs);
        let actions = mem::take(&mut self.actions);
        let masks = mem::take(&mut self.masks);
        let at_kyoku = mem::take(&mut self.at_kyoku);
        // let shantens = mem::take(&mut self.shantens); // Not used in training tuple for now

        for i in 0..game_size {
            // Mapping AtKyoku to Reward/Rank
            let kyoku_idx = at_kyoku[i] as usize;
            let next_kyoku_idx = kyoku_idx + 1;
            
            // Safety clamp
            let safe_next_kyoku_idx = if next_kyoku_idx >= rank_by_player_seq.len() {
                rank_by_player_seq.len() - 1
            } else {
                next_kyoku_idx
            };

            let reward = if kyoku_idx < kyoku_rewards.len() {
                kyoku_rewards[kyoku_idx]
            } else {
                0.0 // Logic error fallback
            };
            
            let rank = rank_by_player_seq[safe_next_kyoku_idx];

            let sample = (
                PyArray2::from_owned_array(py, obs[i].clone()), // Clone required if Array2 is not Copy? Array2 is Clone.
                actions[i],
                PyArray1::from_owned_array(py, masks[i].clone()),
                steps_to_done[i],
                reward,
                rank,
            );
            samples.push(sample);
        }

        samples
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
        let LoaderContext {
            config,
            invisibles,
            state,
            kyoku_idx,
            opponent_states,
            yama_idx,
        } = ctx;

        let cur = &wnd[0];
        let next = &wnd[1];

        match cur {
            Event::StartGame { names, .. } => {
                self.player_name.clone_from(&names[self.player_id as usize]);
            }
            Event::EndKyoku => *kyoku_idx += 1,
            _ => (),
        }

        if invisibles.is_some() {
            match cur {
                Event::EndKyoku => {
                    *yama_idx = 0;
                }
                Event::Tsumo { .. } => {
                    *yama_idx += 1;
                }
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                }
                _ => (),
            };

            for s in opponent_states {
                s.update(cur)?;
            }
        }

        let cans = state.update(cur)?;
        if !cans.can_act() {
            return Ok(());
        }

        let mut kan_select = None;
        let label_opt = match *next {
            Event::Dahai { pai, .. } => Some(pai.as_usize()),
            Event::Pon { actor, .. } if actor == self.player_id => Some(27), // Pon action (was 41)
            Event::Daiminkan { actor, pai, .. } if actor == self.player_id => {
                if config.always_include_kan_select {
                    kan_select = Some(pai.as_usize());
                }
                Some(28) // Kan action (was 42)
            }
            Event::Kakan { pai, .. } => {
                if config.always_include_kan_select || state.kakan_candidates().len() > 1 {
                    kan_select = Some(pai.as_usize());
                }
                Some(28) // Kan action (was 42)
            }
            Event::Ankan { consumed, .. } => {
                if config.always_include_kan_select || state.ankan_candidates().len() > 1 {
                    kan_select = Some(consumed[0].as_usize());
                }
                Some(28) // Kan action (was 42)
            }
            Event::DingQue { actor, suit } if actor == self.player_id => {
                match suit {
                    crate::mjai::Suit::Man => Some(31),
                    crate::mjai::Suit::Pin => Some(32),
                    crate::mjai::Suit::Sou => Some(33),
                }
            }

            _ => {
                let mut ret = None;

                let has_any_ron = matches!(wnd[1], Event::Hora { .. });
                if has_any_ron {
                    // Check if the POV is one of those who made Hora.
                    for ev in &wnd[1..] {
                        match *ev {
                            Event::EndKyoku => break,
                            Event::Hora { actor, .. } if actor == self.player_id => {
                                ret = Some(29); // Agari action (was 43)
                                break;
                            }
                            _ => (),
                        };
                    }
                }

                if ret.is_none() {
                    // It is now proven there is no ron from the POV.
                    if (cans.can_pon || cans.can_daiminkan || cans.can_ron_agari)
                            && !has_any_ron
                    {
                        // Can pon/daiminkan/ron, but actively denied
                        // instead of being interrupted by other's ron.
                        ret = Some(30); // Pass action (was 31, originally 45)
                    }
                }

                ret
            }
        };

        if let Some(label) = label_opt {
            self.add_entry(ctx, false, label);
            if let Some(kan) = kan_select {
                self.add_entry(ctx, true, kan);
            }
        }
        Ok(())
    }

    fn add_entry(&mut self, ctx: &LoaderContext<'_>, at_kan_select: bool, label: usize) {
        let (feature, mask) = ctx.state.encode_obs(ctx.config.version, at_kan_select);
        // Action indices: 0-26 (discard), 27 (pon), 28 (kan), 29 (agari), 30 (pass), 31-33 (ding que)
        // Check mask BEFORE moving it into vector
        if label == 29 && !mask[29] {
             log::error!(
                "Mask Mismatch detected for Agari (Action 29)! Skipping sample. Player: {}, Kyoku: {}, Turn: {}, Shanten: {}, DingQue: {:?}, CanRon: {}, CanTsumo: {}, Furiten: {}, Forbidden: {:?}, Tehai: {:?}",
                self.player_id,
                ctx.kyoku_idx,
                ctx.state.at_turn(),
                ctx.state.shanten(),
                ctx.state.ding_que,
                ctx.state.last_cans().can_ron_agari,
                ctx.state.last_cans().can_tsumo_agari,
                ctx.state.temporary_furiten,
                ctx.state.forbidden_tiles,
                ctx.state.tehai
             );
             return;
        }

        self.obs.push(feature);
        self.actions.push(label as i64);
        self.masks.push(mask);
        
        self.at_kyoku.push(ctx.kyoku_idx as u8);
        // only discard and kan will discount
        self.apply_gamma.push(label <= 37);
        self.at_turns.push(ctx.state.at_turn());
        self.shantens.push(ctx.state.shanten());

        if let Some(invisibles) = ctx.invisibles {
            let invisible_obs = invisibles[ctx.kyoku_idx].encode(
                &ctx.opponent_states,
                ctx.yama_idx,
                0, // Reserved parameter (previously rinshan_idx, not used in Bloody Battle Mahjong)
                ctx.config.version,
            );
            self.invisible_obs.push(invisible_obs);
        }
    }
}
