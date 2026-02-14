use super::item::KawaItem;
use super::{PlayerState, SinglePlayerTables};
use crate::algo::sp::{Candidate, CandidateColumn};
use crate::array::Simple2DArray;
use crate::consts::{ACTION_SPACE, VERSION, obs_shape, TOTAL_SCORE};
use crate::tile::Tile;
use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

// BUG-09 fix: 血战到底 2 人对局时 tsumos_left 可达 28 (56/2)。
// 旧值 14 基于"4 人始终活跃"假设，截断后半程 SP 信号。
// 与 sp/mod.rs MAX_TSUMOS_LEFT=28 对齐。
const MAX_NUM_TURNS: usize = 28;

struct ObsEncoderContext<'a> {
    state: &'a PlayerState,
    arr: Simple2DArray<27, f32>,
    mask: Array1<bool>,
    idx: usize,
    at_kan_select: bool,
}

#[must_use]
struct IntegerEncoder {
    n: usize,
    cap: usize,
    one_hot: bool,
    rescale: bool,
}

impl IntegerEncoder {
    const fn new(n: usize, cap: usize) -> Self {
        Self {
            n,
            cap,
            one_hot: false,
            rescale: false,
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

    fn encode(self, ctx: &mut ObsEncoderContext<'_>) {
        let n = self.n.min(self.cap);
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
}

impl<'a> ObsEncoderContext<'a> {
    fn new(state: &'a PlayerState, version: u32, at_kan_select: bool) -> Self {
        assert!(version == VERSION, "Only v{VERSION} is supported, got v{version}");
        let shape = obs_shape(VERSION);
        let arr = Simple2DArray::<27, f32>::new(shape.0);
        let mask = Array1::default(ACTION_SPACE);
        Self {
            state,
            arr,
            mask,
            idx: 0,
            at_kan_select,
        }
    }

    fn encode_obs(mut self) -> (Array2<f32>, Array1<bool>) {
        let state = self.state;
        let cans = state.last_cans;

        // BUG-05 fix: 定缺阶段手牌必然非空（start_kyoku 固定发 13 张）。
        // 若此处 tehai 全零，说明 encode_obs 被错误调用（如 PlayerState 未经
        // start_kyoku 初始化即被使用），属于上游根本性 Bug。
        // 改为 panic 而非 log::error，避免用全零输入静默产生垃圾决策。
        if cans.can_ding_que {
            assert!(
                state.tehai.iter().sum::<u8>() > 0,
                "BUG-05: Ding Que phase but tehai is EMPTY! \
                 Player: {}, kyoku: {}, tiles_left: {}. \
                 This indicates PlayerState was not initialized via start_kyoku().",
                state.player_id,
                state.kyoku,
                state.tiles_left,
            );
        }

        // ══════════════════════════════════════════════════════════
        //  Section 1: HAND (手牌) — 5 ch
        //  removed: suit_count (冗余, 可从 tehai 推导)
        // ══════════════════════════════════════════════════════════

        // [T] tehai one-hot count (4 ch)
        state.tehai.iter().enumerate()
            .filter(|&(_, &count)| count > 0)
            .for_each(|(tile_id, &count)| {
                self.arr.assign_rows(self.idx, tile_id, count as usize, 1.);
            });
        self.idx += 4;

        // [T] last tsumo tile (1 ch)
        if let Some(tile) = state.last_self_tsumo {
            let tid = tile.as_usize();
            if tid < 27 { self.arr.assign(self.idx, tid, 1.); }
        }
        self.idx += 1;

        // ══════════════════════════════════════════════════════════
        //  Section 2: GAME CONTEXT (场况) — 10 ch
        //  removed: score_deltas (冗余), kyoku 4→1 (单 kyoku 恒为 0)
        // ══════════════════════════════════════════════════════════

        // [S] scores (4 ch)
        for &score in &state.scores {
            let v = score.clamp(0, TOTAL_SCORE) as f32 / TOTAL_SCORE as f32;
            self.arr.fill(self.idx, v);
            self.idx += 1;
        }

        // [S] rank (4 ch): one-hot, stable-argsort semantics
        let my_score = state.scores[0];
        let my_abs_id = state.player_id as usize;
        let rank = (1..4)
            .filter(|&i| {
                let opp_score = state.scores[i];
                let opp_abs_id = (my_abs_id + i) % 4;
                opp_score > my_score || (opp_score == my_score && opp_abs_id < my_abs_id)
            })
            .count();
        self.arr.fill(self.idx + rank, 1.);
        self.idx += 4;

        // [S] kyoku (1 ch): rescaled (单 kyoku 游戏恒为 0，保留 1 ch 兼容)
        self.arr.fill(self.idx, state.kyoku as f32 / 3.);
        self.idx += 1;

        // [S] is_oya (1 ch)
        if state.oya == 0 {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // ══════════════════════════════════════════════════════════
        //  Section 3: DING QUE (定缺) — 17 ch (unchanged)
        // ══════════════════════════════════════════════════════════

        // [S] self suit (3 ch)
        if let Some(suit) = state.ding_que {
            self.arr.fill(self.idx + crate::ding_que::suit_id(suit), 1.);
        }
        self.idx += 3;

        // [S] self complete (1 ch)
        if state.check_ding_que_complete() {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // [S] self remaining (1 ch)
        IntegerEncoder::new(state.count_ding_que_tiles() as usize, 13)
            .rescale(true)
            .encode(&mut self);

        // [S] opponent suits (9 ch) — 定缺阶段置零防止 train/inference 偏移
        for i in 0..3 {
            if !cans.can_ding_que {
                if let Some(suit) = state.other_ding_que[i] {
                    self.arr.fill(self.idx + crate::ding_que::suit_id(suit), 1.);
                }
            }
            self.idx += 3;
        }

        // [S] opponent agari (3 ch)
        for i in 0..3 {
            if state.players_agari[i + 1] {
                self.arr.fill(self.idx, 1.);
            }
            self.idx += 1;
        }

        // ══════════════════════════════════════════════════════════
        //  Section 4: GAME STATE (局面) — 5 ch
        //  removed: active_players (冗余, = 4 - sum(opponent_agari))
        // ══════════════════════════════════════════════════════════

        // [S] tiles left (1 ch)
        self.arr.fill(self.idx, state.tiles_left as f32 / 56.);
        self.idx += 1;

        // [T] forbidden tiles (1 ch): per-tile furiten / ding_que map
        for (tid, &forbidden) in state.forbidden_tiles.iter().enumerate() {
            if tid < 27 && forbidden {
                self.arr.assign(self.idx, tid, 1.);
            }
        }
        self.idx += 1;

        // [S] temporary furiten (1 ch)
        if state.temporary_furiten {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // [S] at rinshan (1 ch): 杠后摸牌 (杠上开花)
        if state.at_rinshan {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // [S] kans on board (1 ch)
        self.arr.fill(self.idx, state.kans_on_board as f32 / 16.);
        self.idx += 1;

        // ══════════════════════════════════════════════════════════
        //  Section 5: SELF KAWA (自家牌河) — 38 ch
        //  removed: first-6 positional (冗余, ⊆ last-18)
        // ══════════════════════════════════════════════════════════

        // max_kawa_len: 用于指数衰减 (Section 5-6 共享)
        let max_kawa_len = state.kawa.iter().map(|k| k.len()).max().unwrap();

        // [T] last 18 positional (36 ch)
        state.kawa[0].iter().rev().take(18)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        self.idx += (18 - state.kawa[0].len().min(18)) * 2;

        // [T] exponential decay overview (2 ch: tile + tsumogiri)
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

        // ══════════════════════════════════════════════════════════
        //  Section 6: OPPONENT KAWA (对手牌河) — 111 ch
        //  removed: first-6 positional ×3 (冗余, ⊆ last-18)
        // ══════════════════════════════════════════════════════════

        for player_kawa in &state.kawa[1..] {
            // [T] last 18 positional (36 ch per opponent)
            player_kawa.iter().rev().take(18)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            self.idx += (18 - player_kawa.len().min(18)) * 2;

            // [T] exponential decay overview (1 ch per opponent)
            for (turn, kawa_item) in player_kawa.iter().enumerate() {
                if let Some(kawa_item) = kawa_item {
                    let sutehai = kawa_item.sutehai;
                    let tid = sutehai.tile.as_usize();
                    let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                    self.arr.assign(self.idx, tid, v);
                }
            }
            self.idx += 1;
        }

        // ══════════════════════════════════════════════════════════
        //  Section 7: VISIBLE TILES (可见牌) — 53 ch
        //  compressed: fuuro 4→2 ch/meld (血战无吃, 全为碰/杠同种牌)
        // ══════════════════════════════════════════════════════════

        // [T] kawa overview per player (4 ch × 4 = 16 ch)
        for player_kawa_overview in &state.kawa_overview {
            self.encode_tile_set(player_kawa_overview.iter().copied());
        }

        // [T] fuuro overview per player (2 ch × 4 melds × 4 players = 32 ch)
        // 血战无吃 — 全为碰(3张同牌)/杠(4张同牌)
        // ch 0: tile identity, ch 1: count/4 (0.75=碰, 1.0=杠)
        for player_fuuro in &state.fuuro_overview {
            for f in player_fuuro {
                if let Some(tile) = f.first() {
                    let tile_id = tile.as_usize();
                    if tile_id < 27 {
                        self.arr.assign(self.idx, tile_id, 1.);
                        self.arr.assign(self.idx + 1, tile_id, f.len() as f32 / 4.);
                    }
                }
                self.idx += 2;
            }
            self.idx += (4 - player_fuuro.len()) * 2;
        }

        // [T] ankan overview per player (1 ch × 4 = 4 ch)
        for player_ankan in &state.ankan_overview {
            for tile in player_ankan {
                let tid = tile.as_usize();
                if tid < 27 { self.arr.assign(self.idx, tid, 1.); }
            }
            self.idx += 1;
        }

        // [T] tiles seen ratio (1 ch)
        for (tid, count) in state.tiles_seen.iter().copied().enumerate() {
            if tid < 27 { self.arr.assign(self.idx, tid, count as f32 / 4.); }
        }
        self.idx += 1;

        // ══════════════════════════════════════════════════════════
        //  Section 8: DEFENSE (防守) — 9 ch
        //  removed: genbutsu (= kawa_overview[opp] ch0), fully_visible (= tiles_seen≥1.0)
        // ══════════════════════════════════════════════════════════

        // [S] opponent suit tendency (9 ch): 对手弃牌花色比例 — 推断定缺
        for i in 0..3 {
            let kawa = &state.kawa_overview[i + 1];
            let total = kawa.len() as f32;
            if total > 0. {
                let man = kawa.iter().filter(|t| t.as_usize() < 9).count() as f32;
                let pin = kawa.iter().filter(|t| (9..18).contains(&t.as_usize())).count() as f32;
                let sou = kawa.iter().filter(|t| t.as_usize() >= 18).count() as f32;
                self.arr.fill(self.idx, man / total);
                self.arr.fill(self.idx + 1, pin / total);
                self.arr.fill(self.idx + 2, sou / total);
            }
            self.idx += 3;
        }

        // ══════════════════════════════════════════════════════════
        //  Section 9: DERIVED FEATURES (导出特征) — 8 ch [NEW]
        // ══════════════════════════════════════════════════════════

        // [T] wall remaining per tile (1 ch): 壁中残留率 — 听牌质量关键
        for tid in 0..27usize {
            let remaining = 4u8.saturating_sub(state.tiles_seen[tid]);
            self.arr.assign(self.idx, tid, remaining as f32 / 4.);
        }
        self.idx += 1;

        // [S] menzen (1 ch): 門前清（无明副露：无碰、无明杠）— 门清加番
        // 暗杠不打破门清（暗杠为暗牌操作，手牌仍视为门前）。
        // 仅检查 fuuro_overview（碰/明杠），不检查 ankan_overview。
        if state.fuuro_overview[0].is_empty() {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // [S] self fuuro count (1 ch): 自家副露数 (含暗杠)
        {
            let count = state.fuuro_overview[0].len() + state.ankan_overview[0].len();
            self.arr.fill(self.idx, count as f32 / 4.);
        }
        self.idx += 1;

        // [S] at turn (1 ch): 自家摸牌巡目 — 时间压力信号
        // BUG-11 fix: 血战到底 2 人对局时 at_turn 可达 28 (56/2)，
        // 旧上限 17 基于日麻 4 人始终活跃假设。提升至 28 以保留后半程信号。
        self.arr.fill(self.idx, (state.at_turn as f32).min(28.) / 28.);
        self.idx += 1;

        // [S] acceptance count (1 ch): 听牌时有效残枚总数 — 和牌概率核心
        if state.shanten == 0 {
            let acc: u32 = state.waits.iter().enumerate()
                .filter(|&(_, &w)| w)
                .map(|(tid, _)| 4u8.saturating_sub(state.tiles_seen[tid]) as u32)
                .sum();
            self.arr.fill(self.idx, (acc as f32).min(20.) / 20.);
        }
        self.idx += 1;

        // [S] opponent fuuro count (3 ch): 对手副露数 — 手牌开放度/危险度
        for i in 1..4 {
            let count = state.fuuro_overview[i].len() + state.ankan_overview[i].len();
            self.arr.fill(self.idx, count as f32 / 4.);
            self.idx += 1;
        }

        // ══════════════════════════════════════════════════════════
        //  Section 10: HAND ANALYSIS (手牌分析) — 7 ch
        //  compressed: shanten 7→5 (>4 极少)
        // ══════════════════════════════════════════════════════════

        // [T] waits (1 ch)
        state.waits.iter().enumerate()
            .filter(|&(_, &c)| c)
            .for_each(|(t, _)| self.arr.assign(self.idx, t, 1.));
        self.idx += 1;

        // [S] shanten (5 ch): one-hot 0-4
        IntegerEncoder::new(state.shanten as usize, 4).one_hot(true).encode(&mut self);

        // [S] at kan select (1 ch)
        if self.at_kan_select { self.arr.fill(self.idx, 1.); }
        self.idx += 1;

        // ══════════════════════════════════════════════════════════
        //  Section 11: ACTION CONTEXT (动作上下文) — 11 ch (unchanged)
        // ══════════════════════════════════════════════════════════

        // [T] last kawa tile (1 ch)
        if cans.can_pass() {
            let tile = state.last_kawa_tile
                .expect("building pon/daiminkan/ron feature without any kawa tile");
            let tile_id = tile.as_usize();
            self.arr.assign(self.idx, tile_id, 1.);

            if !self.at_kan_select {
                self.mask[30] = true; // pass
            } else if cans.can_daiminkan {
                self.mask[tile_id] = true;
            } else if !cans.can_ankan && !cans.can_kakan {
                self.mask[30] = true;
            }
        }
        self.idx += 1;

        // [T] discard candidates (4 ch)
        if cans.can_discard {
            let discard_candidates = state.discard_candidates();
            discard_candidates.iter().enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| {
                    self.arr.assign(self.idx, t, 1.);
                    if !self.at_kan_select || (!cans.can_ankan && !cans.can_kakan) {
                        self.mask[t] = true;
                    }
                });
            state.keep_shanten_discards.iter().enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 1, t, 1.));
            state.next_shanten_discards.iter().enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 2, t, 1.));
            if state.shanten <= 1 {
                state.discard_candidates_with_unconditional_tenpai().iter().enumerate()
                    .filter(|&(_, &c)| c)
                    .for_each(|(t, _)| self.arr.assign(self.idx + 3, t, 1.));
            }
        }
        self.idx += 4;

        // [S] can_pon (1 ch)
        if cans.can_pon {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[27] = true;
            } else if !cans.can_ankan && !cans.can_kakan && !cans.can_daiminkan {
                self.mask[27] = true;
            }
        }
        self.idx += 1;

        // [S] can_daiminkan (1 ch)
        if cans.can_daiminkan {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[28] = true;
            } else if !cans.can_ankan && !cans.can_kakan && !cans.can_daiminkan {
                self.mask[28] = true;
            }
        }
        self.idx += 1;

        // [T] ankan candidates (1 ch)
        if cans.can_ankan {
            for tile in &state.ankan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select { self.mask[tile.as_usize()] = true; }
            }
            if !self.at_kan_select { self.mask[28] = true; }
        }
        self.idx += 1;

        // [T] kakan candidates (1 ch)
        if cans.can_kakan {
            for tile in &state.kakan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select { self.mask[tile.as_usize()] = true; }
            }
            if !self.at_kan_select { self.mask[28] = true; }
        }
        // Fallback: at_kan_select but no actual kan source
        if self.at_kan_select && !cans.can_ankan && !cans.can_kakan && !cans.can_daiminkan && cans.can_discard {
            state.discard_candidates().iter().enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| { self.mask[t] = true; });
        }
        self.idx += 1;

        // [S] can_agari (1 ch)
        if cans.can_agari() {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[29] = true;
            } else if !cans.can_ankan && !cans.can_kakan && !cans.can_daiminkan {
                self.mask[29] = true;
            }
        }
        self.idx += 1;

        // [S] current ron fan (1 ch): 当前荣和番数, rescale to [0, 1]
        if let Some(fan) = state.current_ron_fan {
            self.arr.fill(self.idx, (fan as f32).min(5.) / 5.);
        }
        self.idx += 1;

        // Ding Que mask (no feature channels, only mask)
        if cans.can_ding_que {
            self.mask[31] = true; // Man
            self.mask[32] = true; // Pin
            self.mask[33] = true; // Sou
        }

        // ══════════════════════════════════════════════════════════
        //  Section 12: SP TABLE (单人期望表) — 100 ch
        //  removed: best ev/win prob discard slots (死代码, 从未写入)
        //  reduced: MAX_NUM_TURNS 17→14
        //  所有路径统一产出 100 ch (不再有 can_discard 差异)
        // ══════════════════════════════════════════════════════════
        {
            // 强规则：定缺阶段不计算 SP 表
            if cans.can_ding_que {
                self.encode_ev(0.);
                // required tiles (2*27 + 2) + sp table (3 * MAX_NUM_TURNS)
                self.idx += 2 * 27 + 2 + 3 * MAX_NUM_TURNS;
            } else {
              // PERF-01: 只调用一次 single_player_tables，捕获 Result
              match state.single_player_tables(None) {
                Ok(SinglePlayerTables { max_ev_table }) if max_ev_table.is_empty() => {
                    if cans.can_discard {
                        self.idx += 2 + 2 * 27 + 2;
                        self.encode_sp_table(max_ev_table, cans.can_discard, 0.);
                    } else {
                        self.idx += 2 + 2 * 27 + 1 + 1;
                        self.encode_sp_table(max_ev_table, cans.can_discard, 0.);
                    }
                }
                Ok(SinglePlayerTables { max_ev_table }) => {
                    let max_ev = max_ev_table
                        .first()
                        .and_then(|c| c.exp_values.first().copied())
                        .unwrap_or_default();
                    self.encode_ev(max_ev);

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
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("SP invariant violation") {
                        panic!("{msg}");
                    }

                    let is_ron = !cans.can_tsumo_agari;
                    let active_payers = (1..4).filter(|&i| !state.players_agari[i]).count() as f32;
                    let agari_ev = state
                        .agari_points(is_ron, false, false, false, &[])
                        .map(|p| {
                            if is_ron {
                                p.ron as f32
                            } else {
                                p.tsumo_ko as f32 * active_payers
                            }
                        })
                        .unwrap_or_default();
                    self.encode_ev(agari_ev);

                    // Skip remaining: required tiles (2*27 + 2) + sp table (3 * MAX_NUM_TURNS)
                    self.idx += 2 * 27 + 2 + 3 * MAX_NUM_TURNS;
                }
              } // match
            } // else (non-ding_que)
        } // Section 12 scope

        // ── FanConfig flags (7 channels) ──
        // 每个 flag 填满一整行（27 elements）为 0.0 或 1.0。
        // 模型通过这些通道得知当前对局启用了哪些可选番型。
        for flag in state.fan_config.as_flags() {
            let v = if flag { 1.0 } else { 0.0 };
            for col in 0..27 {
                self.arr.assign(self.idx, col, v);
            }
            self.idx += 1;
        }

        // 编码一致性验证：所有路径必须恰好使用 obs_shape 声明的通道数。
        if self.idx != self.arr.rows() {
            panic!(
                "Observation encoding size mismatch: idx={} != obs_shape={}. \
                 This indicates obs_shape calculation is incorrect or encoding logic has a bug.",
                self.idx, self.arr.rows()
            );
        }
        let arr = self.arr.build();
        debug_assert!(arr.iter().all(|&v| (0. ..=1.).contains(&v)));

        let mask_count = self.mask.iter().filter(|&&m| m).count();
        if mask_count == 0 {
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

    // SP table layout: tenpai_probs(MAX_NUM_TURNS) + win_probs(MAX_NUM_TURNS) + ev(MAX_NUM_TURNS)
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
                    .take(MAX_NUM_TURNS) // safety: cap at allocated turns
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
                .take(MAX_NUM_TURNS) // safety: cap at allocated turns
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
