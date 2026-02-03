use super::item::KawaItem;
use super::{PlayerState, SinglePlayerTables};
use crate::algo::sp::{Candidate, CandidateColumn};
use crate::array::Simple2DArray;
use crate::consts::{ACTION_SPACE, MAX_VERSION, obs_shape};
use crate::tile::Tile;
// use crate::{tu8, tuz}; // Unused imports
use std::num::NonZeroUsize;

use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

const MAX_NUM_TURNS: usize = 17; // aka the actual practical `MAX_TSUMOS_LEFT`

struct ObsEncoderContext<'a> {
    state: &'a PlayerState,
    arr: Simple2DArray<27, f32>,
    mask: Array1<bool>,
    idx: usize,
    at_kan_select: bool,
    version: u32,
}

#[must_use]
struct IntegerEncoder {
    n: usize,
    cap: usize,
    one_hot: bool,
    rescale: bool,
    rbf_intervals: Option<NonZeroUsize>,
}

impl IntegerEncoder {
    const fn new(n: usize, cap: usize) -> Self {
        Self {
            n,
            cap,
            one_hot: false,
            rescale: false,
            rbf_intervals: None,
        }
    }
    const fn one_hot(mut self, v: bool) -> Self {
        self.one_hot = v;
        self
    }
    const fn rescale(mut self, v: bool) -> Self {
        self.rescale = v;
        self
    }
    const fn rbf_intervals(mut self, v: usize) -> Self {
        self.rbf_intervals = NonZeroUsize::new(v);
        self
    }

    fn encode(self, ctx: &mut ObsEncoderContext<'_>) {
        let n = self.n.min(self.cap);
        match ctx.version {
            1 => {
                ctx.arr.fill_rows(ctx.idx, n, 1.);
                ctx.idx += self.cap;
            }
            2 | 3 => {
                debug_assert!(self.one_hot || self.rescale || self.rbf_intervals.is_some());

                if self.one_hot {
                    ctx.arr.fill(ctx.idx + n, 1.);
                    ctx.idx += self.cap + 1;
                }
                if self.rescale {
                    let v = n as f32 / self.cap as f32;
                    ctx.arr.fill(ctx.idx, v);
                    ctx.idx += 1;
                }

                if let Some(intervals) = self.rbf_intervals.map(|v| v.get()) {
                    debug_assert!(intervals >= 3);
                    let interval_size = self.cap as f32 / intervals as f32;
                    for i in 1..intervals {
                        let x = self.n as f32; // the original value, not the clamped
                        let mu = i as f32 * interval_size;
                        let sigma = interval_size;
                        let v = (-(x - mu).powi(2) / (2. * sigma.powi(2))).exp();
                        ctx.arr.fill(ctx.idx + i - 1, v);
                    }
                    ctx.idx += intervals - 1;
                }
            }
            4 => {
                debug_assert!(self.one_hot || self.rescale);

                if self.one_hot {
                    ctx.arr.fill(ctx.idx + n, 1.);
                    ctx.idx += self.cap + 1;
                }
                if self.rescale {
                    let v = n as f32 / self.cap as f32;
                    ctx.arr.fill(ctx.idx, v);
                    ctx.idx += 1;
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<'a> ObsEncoderContext<'a> {
    fn new(state: &'a PlayerState, version: u32, at_kan_select: bool) -> Self {
        assert!(version <= MAX_VERSION);
        let shape = obs_shape(version);
        let arr = Simple2DArray::<27, f32>::new(shape.0);
        let mask = Array1::default(ACTION_SPACE);
        Self {
            state,
            arr,
            mask,
            idx: 0,
            at_kan_select,
            version,
        }
    }

    fn encode_obs(mut self) -> (Array2<f32>, Array1<bool>) {
        let state = self.state;
        let cans = state.last_cans;
        if state.tiles_left == 56 && state.tehai.iter().sum::<u8>() == 0 {
             log::error!("Ding Que Phase (Turn 0) but Tehai is EMPTY! Player: {}. This causes deterministic AI failure (All inputs zero -> Output constant bias).", state.player_id);
        }
        state
            .tehai
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .for_each(|(tile_id, &count)| {
                let n = count as usize;
                self.arr.assign_rows(self.idx, tile_id, n, 1.);
            });
        self.idx += 4;



        for &score in &state.scores {
            let v = score.clamp(0, 100_000) as f32 / 100_000.;
            self.arr.fill(self.idx, v);
            self.idx += 1;

            match self.version {
                2 | 3 => IntegerEncoder::new(score as usize / 100, 500)
                    .rbf_intervals(10)
                    .encode(&mut self),
                4 => {
                    let v = score.clamp(0, 100_000) as f32 / 100_000.;
                    self.arr.fill(self.idx, v);
                    self.idx += 1;
                }
                _ => (),
            }
        }

        let n = state.rank as usize;
        self.arr.fill(self.idx + n, 1.);
        self.idx += 4;

        let n = state.kyoku as usize;
        match self.version {
            // for v1, this was a mistake, it actually only uses 3 channels.
            1 => self.arr.fill_rows(self.idx, n, 1.),
            2 | 3 | 4 => self.arr.fill(self.idx + n, 1.),
            _ => unreachable!(),
        }
        self.idx += 4;





        // Ding que suit (3 dimensions: one-hot for Man/Pin/Sou)
        if let Some(suit) = state.ding_que {
            match suit {
                crate::mjai::Suit::Man => self.arr.fill(self.idx, 1.),
                crate::mjai::Suit::Pin => self.arr.fill(self.idx + 1, 1.),
                crate::mjai::Suit::Sou => self.arr.fill(self.idx + 2, 1.),
            }
        }
        self.idx += 3;

        // Ding que complete status (1 dimension)
        if state.check_ding_que_complete() {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // Ding que tiles remaining (encoded with RBF)
        let ding_que_remaining = state.count_ding_que_tiles();
        IntegerEncoder::new(ding_que_remaining as usize, 13)
            .rescale(true)
            .rbf_intervals(3)
            .encode(&mut self);

        // Other players' ding que suits (3 dimensions per player × 3 players = 9 dimensions)
        for i in 0..3 {
            if let Some(suit) = state.other_ding_que[i] {
                match suit {
                    crate::mjai::Suit::Man => self.arr.fill(self.idx, 1.),
                    crate::mjai::Suit::Pin => self.arr.fill(self.idx + 1, 1.),
                    crate::mjai::Suit::Sou => self.arr.fill(self.idx + 2, 1.),
                }
            }
            self.idx += 3;
        }

        // Opponent Agari status (1 dimension per player × 3 players = 3 dimensions)
        // Crucial for Bloody Battle: AI must know who has already won (and is thus safe/out).
        for i in 0..3 {
            // self.state.players_agari includes player 0 (self).
            // We need 1, 2, 3 relative to self.
            if state.players_agari[i + 1] {
                self.arr.fill(self.idx, 1.);
            }
            self.idx += 1;
        }



        self.encode_tile_set(std::iter::empty());

        state.kawa[0]
            .iter()
            .take(6)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        // Note: encode_self_kawa uses 2 channels per item
        self.idx += (6 - state.kawa[0].len().min(6)) * 2;

        state.kawa[0]
            .iter()
            .rev()
            .take(18)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        // Note: encode_self_kawa uses 2 channels per item
        self.idx += (18 - state.kawa[0].len().min(18)) * 2;

        let max_kawa_len = state.kawa.iter().map(|k| k.len()).max().unwrap();
        if matches!(self.version, 3 | 4) {
            for (turn, kawa_item) in state.kawa[0].iter().enumerate() {
                if let Some(kawa_item) = kawa_item {
                    let sutehai = kawa_item.sutehai;
                    let tid = sutehai.tile.as_usize();
                    let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                    self.arr.assign(self.idx, tid, v);
                    if sutehai.is_tsumogiri {
                        self.arr.assign(self.idx + 1, tid, v);
                    }
                }
            }
            self.idx += 2;
        }

        for player_kawa in &state.kawa[1..] {
            player_kawa
                .iter()
                .take(6)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            // Note: encode_kawa uses 2 channels per item
            self.idx += (6 - player_kawa.len().min(6)) * 2;

            player_kawa
                .iter()
                .rev()
                .take(18)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            // Note: encode_kawa uses 2 channels per item
            self.idx += (18 - player_kawa.len().min(18)) * 2;

            match self.version {
                2 => {
                    for (turn, kawa_item) in player_kawa.iter().flatten().enumerate() {
                        let row = (turn / 6).min(2);
                        let tid = kawa_item.sutehai.tile.as_usize();
                        self.arr.assign(self.idx + row, tid, 1.);
                    }
                    self.idx += 3; // Reduced from 6 (removed tedashi encoding)
                }
                3 | 4 => {
                    for (turn, kawa_item) in player_kawa.iter().enumerate() {
                        if let Some(kawa_item) = kawa_item {
                            let sutehai = kawa_item.sutehai;
                            let tid = sutehai.tile.as_usize();
                            let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                            self.arr.assign(self.idx, tid, v);
                        }
                    }
                    self.idx += 1; // Reduced from 3 (removed tedashi encoding) -> Increased to 2 (added tsumogiri) -> Reverted to 1 (opponent tsumogiri invisible)
                }
                _ => (),
            }
        }

        let v = state.tiles_left as f32 / 56.;
        self.arr.fill(self.idx, v);
        self.idx += 1;



        // Removed doras_unseen encoding (Bloody Battle Mahjong has no dora)
        // This saves ~23 channels (IntegerEncoder with cap=23, rbf_intervals=4)

        for player_kawa_overview in &state.kawa_overview {
            self.encode_tile_set(player_kawa_overview.iter().copied());
        }

        for player_fuuro in &state.fuuro_overview {
            for f in player_fuuro {
                for tile in f {
                    let tile_id = tile.as_usize();
                    if let Some(i) = (0..4).find(|&i| self.arr.get(self.idx + i, tile_id) == 0.)
                    {
                        self.arr.assign(self.idx + i, tile_id, 1.);
                    } else {
                        // Should be unreachable (a tile kind should not appear >4 times),
                        // but guard to avoid panics on malformed state/logs.
                        log::warn!(
                            "fuuro_overview encoding overflow at idx={}, tile_id={}",
                            self.idx,
                            tile_id
                        );
                    }
                }
                self.idx += 4;
            }
            self.idx += (4 - player_fuuro.len()) * 4;
        }

        for player_ankan in &state.ankan_overview {
            for tile in player_ankan {
                let tile_id = tile.as_usize();
                if tile_id < 27 {
                    self.arr.assign(self.idx, tile_id, 1.);
                }
            }
            self.idx += 1;
        }

        if matches!(self.version, 2 | 3 | 4) {
            for (tid, count) in state.tiles_seen.iter().copied().enumerate() {
                if tid < 27 {
                    self.arr.assign(self.idx, tid, count as f32 / 4.);
                }
            }
            self.idx += 1;
            // Removed last_tedashis encoding (9 channels: 3 per player for 3 players)
            // No longer needed in Bloody Battle Mahjong
        }



        state
            .waits
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .for_each(|(t, _)| self.arr.assign(self.idx, t, 1.));
        self.idx += 1;



        let n = state.shanten as usize;
        IntegerEncoder::new(n, 6).one_hot(true).encode(&mut self);



        if self.at_kan_select {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        if cans.can_pass() {
            let tile = state
                .last_kawa_tile
                .expect("building pon/daiminkan/ron feature without any kawa tile");
                    let tile_id = tile.as_usize();

            self.arr.assign(self.idx, tile_id, 1.);

            // pass
            if !self.at_kan_select {
                self.mask[30] = true;
            } else if cans.can_daiminkan {
                self.mask[tile_id] = true;
            } else if !cans.can_ankan && !cans.can_kakan {
                // If at_kan_select is true but neither can_ankan nor can_kakan is true,
                // we should still allow pass action (fallback to normal actions)
                self.mask[30] = true;
            }
        }
        self.idx += 3;

        if cans.can_discard {
            // Only call discard_candidates() if can_discard is true (tehai is 3n+2)
            // This prevents "tehai is not 3n+2" panic
            let discard_candidates = state.discard_candidates();
            discard_candidates
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| {
                    self.arr.assign(self.idx, t, 1.);
                    // If at_kan_select is true, only set mask if we're actually in kan selection
                    // Otherwise, if can_ankan and can_kakan are both false, we should still allow discard
                    if !self.at_kan_select || (!cans.can_ankan && !cans.can_kakan) {
                        self.mask[t] = true;
                    }
                });

            state
                .keep_shanten_discards
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 1, t, 1.));
            state
                .next_shanten_discards
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 2, t, 1.));

            if state.shanten <= 1 {
                state
                    .discard_candidates_with_unconditional_tenpai()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c)
                    .for_each(|(t, _)| self.arr.assign(self.idx + 3, t, 1.));
            }

        }
        self.idx += 5;





        // Action indices: 0-26 (discard), 27 (pon), 28 (kan), 29 (agari), 30 (pass), 31-33 (ding que)
        if cans.can_pon {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[27] = true; // pon action
            } else if !cans.can_ankan && !cans.can_kakan {
                // If at_kan_select is true but neither can_ankan nor can_kakan is true,
                // we should still allow pon action (fallback to normal actions)
                self.mask[27] = true; // pon action
            }
        }
        self.idx += 1;

        if cans.can_daiminkan {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[28] = true; // kan action
            } else if !cans.can_ankan && !cans.can_kakan {
                // If at_kan_select is true but neither can_ankan nor can_kakan is true,
                // we should still allow daiminkan action (fallback to normal actions)
                self.mask[28] = true; // kan action
            }
        }
        self.idx += 1;

        if cans.can_ankan {
            for tile in &state.ankan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select {
                    self.mask[tile.as_usize()] = true; // discard tile for kan
                }
            }
            if !self.at_kan_select {
                self.mask[28] = true; // kan action
            }
        }
        self.idx += 1;

        if cans.can_kakan {
            for tile in &state.kakan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select {
                    self.mask[tile.as_usize()] = true; // discard tile for kan
                }
            }
            if !self.at_kan_select {
                self.mask[28] = true; // kan action
            }
        }
        
        // If at_kan_select is true but neither can_ankan nor can_kakan is true,
        // we should still allow discard actions (fallback to normal discard)
        if self.at_kan_select && !cans.can_ankan && !cans.can_kakan && cans.can_discard {
            let discard_candidates = state.discard_candidates();
            discard_candidates
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| {
                    self.mask[t] = true;
                });
        }
        self.idx += 1;

        if cans.can_agari() {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[29] = true; // agari action
            } else if !cans.can_ankan && !cans.can_kakan {
                // If at_kan_select is true but neither can_ankan nor can_kakan is true,
                // we should still allow agari action (fallback to normal actions)
                self.mask[29] = true; // agari action
            }
        }
        self.idx += 1;

        // Feature 31-33: Ding Que
        // 使用 can_ding_que 作为权威判断条件，而非硬编码的 tiles_left == 56 && ding_que.is_none()
        // 这确保了掩码生成与 ActionCandidate 状态一致
        if cans.can_ding_que {
            self.mask[31] = true; // Man
            self.mask[32] = true; // Pin
            self.mask[33] = true; // Sou
        }



        if self.version == 4 {
            // 定缺阶段（cans.can_ding_que=true）时，庄家可能处于“14张但不允许出牌”的中间态。
            // SP 计算器假设：
            // - can_discard=false => 3n+1（13张等待读）
            // - can_discard=true  => 3n+2（14张待弃）
            // 在定缺阶段对庄家做 SP 模拟会把“14张”等同为“3n+1”，从而在模拟摸牌后变成 15 张，
            // 导致 discard_tiles ArrayVec 溢出（backtrace: State::get_discard_tiles -> ArrayVec::push overflow）。
            //
            // 强规则：定缺阶段不计算 SP 表（语义上定缺阶段唯一动作是 DingQue）。
            if cans.can_ding_que {
                self.encode_ev(0.);
                // required tiles encoding (2*27 + 2) + sp table (3 * MAX_NUM_TURNS)
                self.idx += 2 * 27 + 2 + 3 * MAX_NUM_TURNS;
                // shape 保持一致：如果 can_discard=true（理论上不该发生于定缺阶段），也预留 best slots
                if cans.can_discard {
                    self.idx += 2; // best ev / win prob discard
                }
            } else if let Ok(SinglePlayerTables { max_ev_table }) = state.single_player_tables(None) {
                // Handle empty max_ev_table (can happen in early game or special states)
                if max_ev_table.is_empty() {
                    // Skip encoding if table is empty, just advance idx to maintain shape
                    // Skip: max_ev encoding (2), required tiles encoding, SP table encoding
                    // Note: encode_sp_table handles empty table and adds 3 * MAX_NUM_TURNS
                    // For can_discard, there are additional 2 slots for best ev/win prob discard
                    // But encode_sp_table only adds 3 * MAX_NUM_TURNS, so we need to handle the +2 separately
                    if cans.can_discard {
                        // max_ev (2) + required tiles (2 * 27) + max required tiles (2)
                        self.idx += 2 + 2 * 27 + 2;
                        // encode_sp_table will add 3 * MAX_NUM_TURNS
                        self.encode_sp_table(max_ev_table, cans.can_discard, 0.);
                        // Additional 2 for best ev/win prob discard (not handled by encode_sp_table)
                        self.idx += 2;
                    } else {
                        // max_ev (2) + required tiles (2 * 27 + 1) + first required (1)
                        self.idx += 2 + 2 * 27 + 1 + 1;
                        // encode_sp_table will add 3 * MAX_NUM_TURNS
                        self.encode_sp_table(max_ev_table, cans.can_discard, 0.);
                    }
                } else {
                    // Get the max EV from the table that maximizes EV, which should
                    // be the global max EV.
                    //
                    // `max_ev_table` is already sorted.
                    let max_ev = max_ev_table
                        .first()
                        .and_then(|c| c.exp_values.first().copied())
                        .unwrap_or_default();
                    self.encode_ev(max_ev);

                    // Encode required tiles.
                    if cans.can_discard {
                        for candidate in &max_ev_table {
                            let discard_tid = candidate.tile.as_usize();
                            for r in &candidate.required_tiles {
                                let required_tid = r.tile.as_usize();
                                if candidate.shanten_down {
                                    self.arr
                                        .assign(self.idx + 27 + discard_tid, required_tid, 1.);
                                } else {
                                    self.arr.assign(self.idx + discard_tid, required_tid, 1.);
                                }
                            }
                        }
                        self.idx += 2 * 27;

                        // Handle max required tiles
                        if let Some(max_candidate) = max_ev_table
                            .iter()
                            .max_by(|l, r| l.cmp(r, CandidateColumn::NotShantenDown))
                        {
                            let max_required_tiles_tid = max_candidate
                                .tile
                                .as_usize();
                            self.arr.assign(self.idx, max_required_tiles_tid, 1.);
                        }
                        self.idx += 2;
                    } else {
                        self.idx += 2 * 27 + 1;
                        if let Some(first_candidate) = max_ev_table.first() {
                            for r in &first_candidate.required_tiles {
                                let required_tid = r.tile.as_usize();
                                self.arr.assign(self.idx, required_tid, 1.);
                            }
                        }
                        self.idx += 1;
                    }

                    let ev_scale = if max_ev < 1. { 0. } else { 1. / max_ev };
                    self.encode_sp_table(max_ev_table, cans.can_discard, ev_scale);
                    // 业务逻辑：在 can_discard=True 时，需要额外的 2 行用于 best ev/win prob discard
                    // 这与 empty table 的情况保持一致
                    if cans.can_discard {
                        // Additional 2 for best ev/win prob discard (not handled by encode_sp_table)
                        self.idx += 2;
                    }
                }
            } else {
                // Do not silently swallow SP invariant violations: they indicate a replay/state bug
                // that can later crash with an ArrayVec overflow in SP code. Fail with a clear error.
                if let Err(e) = state.single_player_tables(None) {
                    let msg = e.to_string();
                    if msg.contains("SP invariant violation") {
                        panic!("{msg}");
                    }
                }
                // Use the minimal tsumo agari point as the max EV.
                // Note: In Bloody Battle Mahjong, there is no uradora (里宝牌)
                let min_tsumo_agari = state
                    .agari_points(cans.can_ron_agari, false, false, false, &[])
                    .map(|p| p.tsumo_total(state.is_oya()) as f32)
                    .unwrap_or_default();
                self.encode_ev(min_tsumo_agari);

                // Skip everything else.
                self.idx += 2 * 27 + 2 + 3 * MAX_NUM_TURNS;
                // 业务逻辑：在 can_discard=True 时，需要额外的 2 行用于 best ev/win prob discard
                // 这与 empty table 和 non-empty table 的情况保持一致
                if cans.can_discard {
                    // Additional 2 for best ev/win prob discard
                    self.idx += 2;
                }
            }
        }

        // 业务逻辑验证：idx 不应该超过数组行数
        // 如果超过，说明 obs_shape 的计算有问题，或者编码逻辑有误
        if self.idx > self.arr.rows() {
            panic!(
                "Observation encoding overflow: idx={} > arr.rows()={}, version={}. \
                This indicates obs_shape calculation is incorrect or encoding logic has a bug.",
                self.idx, self.arr.rows(), self.version
            );
        }
        // 如果 idx < arr.rows()，说明编码未完成或 obs_shape 计算过大
        // 但这种情况通常不会导致崩溃，只是浪费空间
        let arr = self.arr.build();
        debug_assert!(arr.iter().all(|&v| (0. ..=1.).contains(&v)));
        
        // 业务规则：mask必须至少有一个为true，否则模型无法选择动作
        // 如果mask全为false，说明can_act()返回false，但Agent仍然被调用了
        // 这通常发生在状态不一致的情况下
        let mask_count = self.mask.iter().filter(|&&m| m).count();
        if mask_count == 0 {
            // 收集详细的调试信息
            // 注意：只有在 can_discard 为 true 时才能调用 discard_candidates()
            let discard_candidates_count = if cans.can_discard {
                let discard_candidates = state.discard_candidates();
                discard_candidates.iter().filter(|&&c| c).count()
            } else {
                0
            };
            let forbidden_tiles_count = state.forbidden_tiles.iter().filter(|&&f| f).count();
            let tehai_nonzero_count = state.tehai.iter().filter(|&&c| c > 0).count();
            let ding_que_info = state.ding_que.map(|s| format!("{:?}", s)).unwrap_or_else(|| "None".to_string());
            
            panic!(
                "mask is all false: can_act()={}, can_discard={}, can_pon={}, can_kan()={}, can_agari()={}, can_pass()={}. \
                This indicates a bug: Agent was called when no actions are available. \
                State: kyoku={}, at_turn={}, tiles_left={}, tehai_sum={}, tehai_nonzero_count={}. \
                discard_candidates_count={}, forbidden_tiles_count={}, ding_que={}",
                cans.can_act(),
                cans.can_discard,
                cans.can_pon,
                cans.can_kan(),
                cans.can_agari(),
                cans.can_pass(),
                state.kyoku + 1,
                state.at_turn,
                state.tiles_left,
                state.tehai.iter().sum::<u8>(),
                tehai_nonzero_count,
                discard_candidates_count,
                forbidden_tiles_count,
                ding_que_info
            );
        }
        
        (arr, self.mask)
    }

    fn encode_ev(&mut self, value: f32) {
        let v = value.clamp(0., 100_000.) / 100_000.;
        self.arr.fill(self.idx, v);
        let v = value.clamp(0., 30_000.) / 30_000.;
        self.arr.fill(self.idx + 1, v);
        self.idx += 2;
    }

    // discard table: 3 * MAX_NUM_TURNS
    // tsumo table: 3 * MAX_NUM_TURNS
    // best ev discard: 1
    // best win prob discard: 1
    fn encode_sp_table(&mut self, candidates: Vec<Candidate>, can_discard: bool, ev_scale: f32) {
        let Some(first) = candidates
            .first()
            .filter(|c| c.tenpai_probs.first().is_some_and(|&p| p > 0.))
        else {
            // Simply do nothing when probs aren't calculated at all (when
            // shanten >= 4) or are all zero.
            self.idx += 3 * MAX_NUM_TURNS;
            return;
        };

        if can_discard {
            for candidate in candidates {
                let tid = candidate.tile.as_usize();
                for (turn, ((&tenpai_prob, &win_prob), &ev)) in candidate
                    .tenpai_probs
                    .iter()
                    .take_while(|&&p| p > 0.)
                    .zip(&candidate.win_probs)
                    .zip(&candidate.exp_values)
                    .enumerate()
                {
                    let mut idx = self.idx + turn;
                    self.arr.assign(idx, tid, tenpai_prob);
                    idx += MAX_NUM_TURNS;
                    self.arr.assign(idx, tid, win_prob);
                    idx += MAX_NUM_TURNS;
                    self.arr.assign(idx, tid, (ev * ev_scale).min(1.));
                }
            }
        } else {
            for (turn, ((&tenpai_prob, &win_prob), &ev)) in first
                .tenpai_probs
                .iter()
                .take_while(|&&p| p > 0.)
                .zip(&first.win_probs)
                .zip(&first.exp_values)
                .enumerate()
            {
                let mut idx = self.idx + turn;
                self.arr.fill(idx, tenpai_prob);
                idx += MAX_NUM_TURNS;
                self.arr.fill(idx, win_prob);
                idx += MAX_NUM_TURNS;
                self.arr.fill(idx, (ev * ev_scale).min(1.));
            }
        }
        self.idx += 3 * MAX_NUM_TURNS;
    }

    fn encode_tile_set<I>(&mut self, tiles: I)
    where
        I: IntoIterator<Item = Tile>,
    {
        let mut counts = [0; 27];
        for tile in tiles {
            let tile_id = tile.as_usize();
            if tile_id >= 27 {
                continue;
            }

            let i = &mut counts[tile_id];
            if *i >= 4 {
                // Safety check: max 4 copies of a tile
                continue;
            }
            self.arr.assign(self.idx + *i, tile_id, 1.);
            *i += 1;

        }
        self.idx += 4;
    }

    fn encode_self_kawa(&mut self, item: Option<&KawaItem>) {
        if let Some(k) = item {
            for kan in k.kan {
                // Note: In Bloody Battle Mahjong, there is no aka dora (赤牌),
                // so tiles are already normalized (no aka distinction)
                let tile_id = kan.as_usize();
                self.arr.assign(self.idx, tile_id, 1.);
            }

            let sutehai = k.sutehai;
            let tile_id = sutehai.tile.as_usize();
            self.arr.assign(self.idx + 1, tile_id, 1.);
        }
        self.idx += 2;
    }

    fn encode_kawa(&mut self, item: Option<&KawaItem>) {
        if let Some(k) = item {
            for kan in k.kan {
                let tile_id = kan.as_usize();
                self.arr.assign(self.idx, tile_id, 1.);
            }

            let sutehai = k.sutehai;
            let tile_id = sutehai.tile.as_usize();
            self.arr.assign(self.idx + 1, tile_id, 1.);
        }
        self.idx += 2;
    }
}

#[pymethods]
impl PlayerState {
    /// Returns `(obs, mask)`
    #[pyo3(name = "encode_obs")]
    fn encode_obs_py<'py>(
        &self,
        version: u32,
        at_kan_select: bool,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<bool>>) {
        let (obs, mask) = self.encode_obs(version, at_kan_select);
        let obs = PyArray2::from_owned_array(py, obs);
        let mask = PyArray1::from_owned_array(py, mask);
        (obs, mask)
    }
}

impl PlayerState {
    /// Returns `(obs, mask)`
    #[must_use]
    pub fn encode_obs(&self, version: u32, at_kan_select: bool) -> (Array2<f32>, Array1<bool>) {
        ObsEncoderContext::new(self, version, at_kan_select).encode_obs()
    }
}
