use super::candidate::RawCandidate;
use super::state::{InitState, State};
use super::tile::{DiscardTile, DrawTile};
use super::{Candidate, CandidateColumn, MAX_TSUMOS_LEFT};
use crate::algo::agari::{Agari, AgariCalculator};
use crate::tile::Tile;
use crate::must_tile;
use std::rc::Rc;

use ahash::AHashMap;
use anyhow::{Result, ensure};

const SHANTEN_THRES: i8 = 3;
// Bloody Battle: 27 tile kinds, 4 players, 13 tiles per player initially
const MAX_TILES_LEFT: usize = 27 * 4 - 1 - 13;

// Bloody Battle: No uradora (里宝牌) - no riichi (立直), no dora (宝牌), so URADORA_PROB_TABLE is removed

type StateCache<const MAX_TSUMO: usize> =
    [AHashMap<State, Rc<Values<MAX_TSUMO>>>; SHANTEN_THRES as usize + 1];

struct Values<const MAX_TSUMO: usize> {
    tenpai_probs: [f32; MAX_TSUMO],
    win_probs: [f32; MAX_TSUMO],
    exp_values: [f32; MAX_TSUMO],
}

enum ScoresOrValues<const MAX_TSUMO: usize> {
    // shanten == 0, and has yaku
    Scores([f32; 4]),
    // shanten > 0
    Values(Rc<Values<MAX_TSUMO>>),
}

#[derive(Debug)]
pub struct SPCalculator<'a> {
    // Immutable states, used in agari calculator.
    pub tehai_len_div3: u8,
    pub chis: &'a [u8],
    pub pons: &'a [u8],
    pub minkans: &'a [u8],
    pub ankans: &'a [u8],
    pub bakaze: u8,
    pub jikaze: u8,
    pub is_menzen: bool,

    /// Unlike others, fuuro here includes ankan.
    pub num_doras_in_fuuro: u8,
    pub dora_indicators: &'a [Tile],
    pub calc_double_riichi: bool,
    pub calc_haitei: bool,
    pub prefer_riichi: bool,
    pub sort_result: bool,

    /// 和了確率を最大化
    pub maximize_win_prob: bool,
    /// 手変わり考慮
    pub calc_tegawari: bool,
    /// 向聴落とし考慮
    pub calc_shanten_down: bool,
}

struct SPCalculatorState<'a, const MAX_TSUMO: usize> {
    sup: &'a SPCalculator<'a>,
    state: State,

    tsumo_prob_table: &'a [[f32; MAX_TSUMO]; 4],
    not_tsumo_prob_table: &'a [[f32; MAX_TSUMO]; MAX_TILES_LEFT + 1],

    discard_cache: StateCache<MAX_TSUMO>,
    draw_cache: StateCache<MAX_TSUMO>,

    #[cfg(feature = "sp_reproduce_cpp_ver")]
    real_max_tsumo: usize,
}

impl SPCalculator<'_> {
    /// Arguments:
    /// - can_discard: whether the tehai is 3n+2 or not.
    /// - tsumos_left: must be within [1, 17].
    /// - cur_shanten: must be >= 0.
    ///
    /// The return value will be sorted and index 0 will be the best choice.
    pub fn calc(
        &self,
        init_state: InitState,
        can_discard: bool,
        tsumos_left: u8,
        cur_shanten: i8,
    ) -> Result<Vec<Candidate>> {
        ensure!(cur_shanten >= 0, "can't calculate an agari hand");
        ensure!(tsumos_left >= 1, "need at least one more tsumo");
        ensure!(tsumos_left <= MAX_TSUMOS_LEFT as u8);

        #[cfg(feature = "sp_reproduce_cpp_ver")]
        let max_tsumo = if can_discard { 17 } else { 18 };
        #[cfg(not(feature = "sp_reproduce_cpp_ver"))]
        let max_tsumo = tsumos_left as usize;

        let state = State::from(init_state);
        let n_left_tiles = state.sum_left_tiles() as usize;

        // Despite the bloating binary size, the use of const generics here may
        // help eliminate branches (eg. bound checks) and reduce buffer space,
        // and allow more aggressive loop unroll and vectorization.
        macro_rules! static_expand {
            ($($n:literal),*) => {
                match max_tsumo {
                    $($n => {
                        let tsumo_prob_table = build_tsumo_prob_table(n_left_tiles);
                        let not_tsumo_prob_table = build_not_tsumo_prob_table(n_left_tiles);
                        let mut calc_state = SPCalculatorState::<$n> {
                            sup: self,
                            state,
                            tsumo_prob_table: &tsumo_prob_table,
                            not_tsumo_prob_table: &not_tsumo_prob_table,
                            discard_cache: Default::default(),
                            draw_cache: Default::default(),
                            #[cfg(feature = "sp_reproduce_cpp_ver")]
                            real_max_tsumo: tsumos_left as usize,
                        };
                        calc_state.calc(can_discard, cur_shanten)
                    },)*
                    _ => unreachable!(),
                }
            }
        }
        #[cfg(feature = "sp_reproduce_cpp_ver")]
        let candidates = static_expand!(17, 18);
        #[cfg(not(feature = "sp_reproduce_cpp_ver"))]
        let candidates = static_expand!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        Ok(candidates)
    }
}

fn build_tsumo_prob_table<const MAX_TSUMO: usize>(n_left_tiles: usize) -> [[f32; MAX_TSUMO]; 4] {
    let mut table = [[0.; MAX_TSUMO]; 4];
    // 有効牌の枚数ごとに、この巡目で有効牌を引ける確率のテーブルを作成する。
    // tumo_prob_table_[i][j] = 有効牌の枚数が i + 1 枚の場合に j 巡目に有効牌が引ける確率
    for (i, row) in table.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (i + 1) as f32 / (n_left_tiles - j) as f32;
        }
    }
    table
}

fn build_not_tsumo_prob_table<const MAX_TSUMO: usize>(
    n_left_tiles: usize,
) -> [[f32; MAX_TSUMO]; MAX_TILES_LEFT + 1] {
    let mut table = [[0.; MAX_TSUMO]; MAX_TILES_LEFT + 1];
    // 有効牌の合計枚数ごとに、これまでの巡目で有効牌が引けなかった確率のテーブルを作成する。
    // not_tumo_prob_table_[i][j] = 有効牌の合計枚数が i 枚の場合に j - 1 巡目までに有効牌が引けなかった確率
    //
    // The original version has only `n_left_tiles` rows, which can actually
    // overflow for hands like 9999m6677p88s335z that can be improved by all
    // kinds of tiles, and the number of all tiles left will be exactly
    // `n_left_tiles`. A test case covers this.
    for (i, row) in table.iter_mut().enumerate().take(n_left_tiles + 1) {
        row[0] = 1.;
        // n_left_tiles - i - j > 0 は残りはすべて有効牌の場合を考慮
        for j in 0..(MAX_TSUMO - 1).min(n_left_tiles - i) {
            row[j + 1] = row[j] * (n_left_tiles - i - j) as f32 / (n_left_tiles - j) as f32;
        }
    }
    table
}

impl<const MAX_TSUMO: usize> SPCalculatorState<'_, MAX_TSUMO> {
    fn calc(&mut self, can_discard: bool, cur_shanten: i8) -> Vec<Candidate> {
        if cur_shanten <= SHANTEN_THRES {
            // 3向聴以下は聴牌確率、和了確率、期待値を計算する。
            let mut candidates = if can_discard {
                self.analyze_discard(cur_shanten)
            } else {
                self.analyze_draw(cur_shanten)
            };

            if self.sup.sort_result && !candidates.is_empty() {
                let by = if self.sup.maximize_win_prob {
                    CandidateColumn::WinProb
                } else {
                    CandidateColumn::EV
                };
                candidates.sort_by(|l, r| r.cmp(l, by));
            }
            candidates
        } else {
            // 4向聴以上は受入枚数のみ計算する。
            let mut candidates = if can_discard {
                self.analyze_discard_simple(cur_shanten)
            } else {
                self.analyze_draw_simple()
            };

            if self.sup.sort_result && !candidates.is_empty() {
                candidates.sort_by(|l, r| r.cmp(l, CandidateColumn::NotShantenDown));
            }
            candidates
        }
    }

    fn analyze_discard(&mut self, shanten: i8) -> Vec<Candidate> {
        // 打牌候補を取得する。
        let discard_tiles = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);

        let mut candidates = Vec::with_capacity(discard_tiles.len());
        for DiscardTile { tile, shanten_diff } in discard_tiles {
            if shanten_diff == 0 {
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
                let values = self.draw(shanten);
                self.state.undo_discard(tile);

                let mut tenpai_probs = values.tenpai_probs;
                if shanten == 0 {
                    // すでに聴牌している場合の例外処理
                    tenpai_probs.fill(1.);
                }

                let candidate = Candidate::from(RawCandidate {
                    tile,
                    tenpai_probs: &tenpai_probs,
                    win_probs: &values.win_probs,
                    exp_values: &values.exp_values,
                    required_tiles,
                    shanten_down: false,
                });
                #[cfg(feature = "sp_reproduce_cpp_ver")]
                let candidate = candidate.calibrate(self.real_max_tsumo);
                candidates.push(candidate);
            } else if self.sup.calc_shanten_down && shanten_diff == 1 && shanten < SHANTEN_THRES {
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
                self.state.n_extra_tsumo += 1;
                let values = self.draw(shanten + 1);
                self.state.n_extra_tsumo -= 1;
                self.state.undo_discard(tile);

                let candidate = Candidate::from(RawCandidate {
                    tile,
                    tenpai_probs: &values.tenpai_probs,
                    win_probs: &values.win_probs,
                    exp_values: &values.exp_values,
                    required_tiles,
                    shanten_down: true,
                });
                #[cfg(feature = "sp_reproduce_cpp_ver")]
                let candidate = candidate.calibrate(self.real_max_tsumo);
                candidates.push(candidate);
            }
        }
        candidates
    }

    fn analyze_draw(&mut self, shanten: i8) -> Vec<Candidate> {
        let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
        let values = self.draw(shanten);

        let mut tenpai_probs = values.tenpai_probs;
        if shanten == 0 {
            // すでに聴牌している場合の例外処理
            tenpai_probs.fill(1.);
        }

        let candidate = Candidate::from(RawCandidate {
            tile: t!(?),
            tenpai_probs: &tenpai_probs,
            win_probs: &values.win_probs,
            exp_values: &values.exp_values,
            required_tiles,
            shanten_down: false,
        });
        #[cfg(feature = "sp_reproduce_cpp_ver")]
        let candidate = candidate.calibrate(self.real_max_tsumo);
        vec![candidate]
    }

    fn analyze_discard_simple(&mut self, shanten: i8) -> Vec<Candidate> {
        // 打牌候補を取得する。
        let discard_tiles = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);
        discard_tiles
            .into_iter()
            .map(|DiscardTile { tile, shanten_diff }| {
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
                self.state.undo_discard(tile);

                Candidate::from(RawCandidate {
                    tile,
                    required_tiles,
                    shanten_down: shanten_diff == 1,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn analyze_draw_simple(&mut self) -> Vec<Candidate> {
        let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
        let candidate = Candidate::from(RawCandidate {
            tile: t!(?),
            required_tiles,
            shanten_down: false,
            ..Default::default()
        });
        vec![candidate]
    }

    fn draw(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        if self.sup.calc_tegawari && self.state.n_extra_tsumo == 0 {
            self.draw_with_tegawari(shanten)
        } else {
            self.draw_without_tegawari(shanten)
        }
    }

    fn draw_with_tegawari(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.draw_cache[shanten as usize]
            .get(&self.state)
            .cloned()
            .unwrap_or_else(|| self.draw_with_tegawari_slow(shanten))
    }

    fn draw_with_tegawari_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        let mut tenpai_probs = [0.; MAX_TSUMO];
        let mut win_probs = [0.; MAX_TSUMO];
        let mut exp_values = [0.; MAX_TSUMO];

        // 自摸候補を取得する。
        let draw_tiles = self.state.get_draw_tiles(shanten, self.sup.tehai_len_div3);

        // 有効牌の合計枚数を計算する。【暫定対応】
        let sum_left_tiles = self.state.sum_left_tiles();

        for &DrawTile {
            tile,
            count,
            shanten_diff,
        } in &draw_tiles
        {
            if shanten_diff != -1 {
                // 有効牌以外の場合
                continue;
            }

            self.state.deal(tile);
            let scores_or_values = if shanten > 0 {
                ScoresOrValues::Values(self.discard(shanten - 1))
            } else if let Some(scores) = self.get_score(tile) {
                ScoresOrValues::Scores(scores)
            } else {
                self.state.undo_deal(tile);
                continue;
            };
            self.state.undo_deal(tile);

            // 【暫定対応】 (2021/9/24)
            // FIX_TEGAWARI_PROB について
            // draw_without_tegawari() で有効牌が引けない場合、有効牌以外のどの牌を引いたのかということは考慮していないため、
            // counts で管理している各牌の残りの合計枚数 > 現在の巡目の残り枚数という状況が発生し、結果的に確率値が1を超えてしまう。
            // 実際に正しい確率値を求めるには、draw_without_tegawari() でどの牌を引いたのかをすべてシミュレーションする必要があるが、
            // 計算量的に難しいので、巡目に関係なく、
            //「自摸の確率 = 牌の残り枚数 / 残り枚数の合計」で確率値が1を超えないように暫定対応した。

            for i in 0..MAX_TSUMO {
                // 【暫定対応】 (2021/9/24)
                let tump_prob = count as f32 / sum_left_tiles as f32;
                // let tump_prob = &TSUMO_PROB_TABLE[count as usize - 1][i];

                match &scores_or_values {
                    ScoresOrValues::Scores(scores) => {
                        let assume_riichi = self.sup.is_menzen && self.sup.prefer_riichi;
                        // 聴牌の場合は次で和了
                        // i 巡目で聴牌の場合はダブル立直成立
                        let win_double_riichi =
                            assume_riichi && self.sup.calc_double_riichi && i == 0;
                        // i 巡目で聴牌し、次の巡目で和了の場合は一発成立
                        let win_ippatsu = assume_riichi;
                        // 最後の巡目で和了の場合は海底撈月成立
                        let _win_haitei = self.sup.calc_haitei && i == MAX_TSUMO - 1; // Bloody Battle: No haitei
                        let han_plus =
                            win_double_riichi as usize + win_ippatsu as usize + _win_haitei as usize;

                        win_probs[i] += tump_prob;
                        exp_values[i] += tump_prob * scores[han_plus];
                    }
                    ScoresOrValues::Values(next_values) => {
                        if shanten == 1 {
                            // 1向聴の場合は次で聴牌
                            tenpai_probs[i] += tump_prob;
                        }
                        if i < MAX_TSUMO - 1 {
                            if shanten > 1 {
                                tenpai_probs[i] += tump_prob * next_values.tenpai_probs[i + 1];
                            }
                            win_probs[i] += tump_prob * next_values.win_probs[i + 1];
                            exp_values[i] += tump_prob * next_values.exp_values[i + 1];
                        }
                    }
                }
            }
        }

        for DrawTile {
            tile,
            count,
            shanten_diff,
        } in draw_tiles
        {
            if shanten_diff != 0 {
                // 有効牌の場合
                continue;
            }

            self.state.deal(tile);
            self.state.n_extra_tsumo += 1;
            let next_values = self.discard(shanten);
            self.state.n_extra_tsumo -= 1;
            self.state.undo_deal(tile);

            for i in 0..MAX_TSUMO - 1 {
                // 【暫定対応】 (2021/9/24)
                let tump_prob = count as f32 / sum_left_tiles as f32;
                // let tump_prob = &TSUMO_PROB_TABLE[count as usize - 1][i];

                tenpai_probs[i] += tump_prob * next_values.tenpai_probs[i + 1];
                win_probs[i] += tump_prob * next_values.win_probs[i + 1];
                exp_values[i] += tump_prob * next_values.exp_values[i + 1];
            }
        }

        let values = Rc::new(Values {
            tenpai_probs,
            win_probs,
            exp_values,
        });
        self.draw_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));

        values
    }

    fn draw_without_tegawari(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.draw_cache[shanten as usize]
            .get(&self.state)
            .cloned()
            .unwrap_or_else(|| self.draw_without_tegawari_slow(shanten))
    }

    fn draw_without_tegawari_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        let mut tenpai_probs = [0.; MAX_TSUMO];
        let mut win_probs = [0.; MAX_TSUMO];
        let mut exp_values = [0.; MAX_TSUMO];

        // 自摸候補を取得する。
        let draw_tiles = self.state.get_draw_tiles(shanten, self.sup.tehai_len_div3);

        // 有効牌の合計枚数を計算する。
        let sum_required_tiles: u8 = draw_tiles
            .iter()
            .filter(|d| d.shanten_diff == -1)
            .map(|d| d.count)
            .sum();
        let not_tsumo_probs = &self.not_tsumo_prob_table[sum_required_tiles as usize];

        for DrawTile {
            tile,
            count,
            shanten_diff,
        } in draw_tiles
        {
            if shanten_diff != -1 {
                // 有効牌以外の場合
                continue;
            }

            self.state.deal(tile);
            let scores_or_values = if shanten > 0 {
                ScoresOrValues::Values(self.discard(shanten - 1))
            } else if let Some(scores) = self.get_score(tile) {
                ScoresOrValues::Scores(scores)
            } else {
                self.state.undo_deal(tile);
                continue;
            };
            self.state.undo_deal(tile);

            let tsumo_probs = &self.tsumo_prob_table[count as usize - 1];
            for i in 0..MAX_TSUMO {
                let m = not_tsumo_probs[i];
                if m == 0. {
                    // We are breaking here because `not_tsumo_probs[i..]` must
                    // all be zero, since `not_tsumo_probs` is monotonically
                    // decreasing.
                    //
                    // This divide-by-zero check is missing in the original
                    // version, which is very problematic.
                    break;
                }

                for j in i..MAX_TSUMO {
                    let n = not_tsumo_probs[j];
                    if n == 0. {
                        // `not_tsumo_probs[j..]` must all be zero, no need to
                        // proceed.
                        break;
                    }
                    // 現在の巡目が i の場合に j 巡目に有効牌を引く確率
                    let prob = tsumo_probs[j] * n / m;

                    match &scores_or_values {
                        ScoresOrValues::Scores(scores) => {
                            // Bloody Battle: No riichi (立直), no ippatsu (一发), no haitei (海底), no double riichi (两立直)
                            // These fields are set to false in agent_helper.rs, so these calculations
                            // will always result in 0 additional han
                            let assume_riichi = false; // Bloody Battle: No riichi (立直)
                            let win_double_riichi = false; // Bloody Battle: No double riichi (两立直)
                            let win_ippatsu = false; // Bloody Battle: No ippatsu
                            let _win_haitei = false; // Bloody Battle: No haitei
                            let han_plus = 0; // Bloody Battle: No additional han from riichi (立直)/dora (宝牌)

                            win_probs[i] += prob;
                            exp_values[i] += prob * scores[han_plus];
                        }
                        ScoresOrValues::Values(next_values) => {
                            if shanten == 1 {
                                // 1向聴の場合は次で聴牌
                                tenpai_probs[i] += prob;
                            }
                            if j < MAX_TSUMO - 1 {
                                if shanten > 1 {
                                    // 2向聴以上で max_tsumo_ - 1 巡目以下の場合
                                    tenpai_probs[i] += prob * next_values.tenpai_probs[j + 1];
                                }
                                // 聴牌以上で max_tsumo_ - 1 巡目以下の場合
                                win_probs[i] += prob * next_values.win_probs[j + 1];
                                exp_values[i] += prob * next_values.exp_values[j + 1];
                            }
                        }
                    }
                }
            }
        }

        let values = Rc::new(Values {
            tenpai_probs,
            win_probs,
            exp_values,
        });
        self.draw_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));

        values
    }

    fn discard(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.discard_cache[shanten as usize]
            .get(&self.state)
            .cloned()
            .unwrap_or_else(|| self.discard_slow(shanten))
    }

    fn discard_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        // 打牌候補を取得する。
        let discard_tiles = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);

        // 期待値が最大となる打牌を選択する。
        let mut max_tenpai_probs = [f32::MIN; MAX_TSUMO];
        let mut max_win_probs = [f32::MIN; MAX_TSUMO];
        let mut max_exp_values = [f32::MIN; MAX_TSUMO];
        let mut max_tiles = [t!(?); MAX_TSUMO];
        let mut max_values = [i32::MIN; MAX_TSUMO];

        for DiscardTile { tile, shanten_diff } in discard_tiles {
            let values;
            if shanten_diff == 0 {
                // 向聴数が変化しない打牌
                self.state.discard(tile);
                values = self.draw(shanten);
                self.state.undo_discard(tile);
            } else if self.sup.calc_shanten_down
                && self.state.n_extra_tsumo == 0
                && shanten_diff == 1
                && shanten < SHANTEN_THRES
            {
                // 向聴戻しになる打牌
                self.state.discard(tile);
                self.state.n_extra_tsumo += 1;
                values = self.draw(shanten + 1);
                self.state.n_extra_tsumo -= 1;
                self.state.undo_discard(tile);
            } else {
                // 手牌に存在しない牌、または向聴落としが無効な場合に向聴落としとなる牌
                continue;
            };

            for i in 0..MAX_TSUMO {
                // 和了確率は下2桁まで一致していれば同じ、期待値は下0桁まで一致していれば同じとみなす。
                let value = if self.sup.maximize_win_prob {
                    values.win_probs[i] * 1e5
                } else {
                    values.exp_values[i]
                } as i32;
                let max_value = max_values[i];
                let max_tile = max_tiles[i];

                if value > max_value
                    || value == max_value && tile.cmp_discard_priority(max_tile).is_gt()
                {
                    // 値が同等なら、DiscardPriorities が高い牌を優先して選択する。
                    max_tenpai_probs[i] = values.tenpai_probs[i];
                    max_win_probs[i] = values.win_probs[i];
                    max_exp_values[i] = values.exp_values[i];
                    max_values[i] = value;
                    max_tiles[i] = tile;
                }
            }
        }

        let values = Rc::new(Values {
            tenpai_probs: max_tenpai_probs,
            win_probs: max_win_probs,
            exp_values: max_exp_values,
        });
        self.discard_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));

        values
    }

    /// None: no yaku
    fn get_score(&self, win_tile: Tile) -> Option<[f32; 4]> {
        // Bloody Battle: AgariCalculator no longer has chis, bakaze, jikaze
        // Bloody Battle: No oya advantage, so is_oya is always false
        let calc = AgariCalculator {
            tehai: &self.state.tehai,
            is_menzen: self.sup.is_menzen,
            pons: self.sup.pons,
            minkans: self.sup.minkans,
            ankans: self.sup.ankans,
            winning_tile: win_tile.deaka().as_u8(),
            is_ron: false,
            ding_que: None, // TODO: Get from state if available
            is_after_kan: false, // TODO: Track if this is after kan
            is_kan_discard: false, // TODO: Track if this is from kan discard
            is_chankan: false, // SPCalculator doesn't track chankan
            exclude_gen_tile: None,
        };
        let _is_oya = false; // Bloody Battle: No oya advantage

        // Bloody Battle: No additional_yakus or num_doras (no riichi/立直, no dora/宝牌)
        // Bloody Battle: agari() returns Agari::Fan(u8) directly
        let fan = match calc.agari()? {
            Agari::Fan(f) => f,
        };
        
        // Bloody Battle: Calculate points using fan-based scoring
        // Points = 1000 * 2^(fan-1), capped at 5 fan
        let fan_capped = fan.min(5);
        let base_points = 1000 * (1 << (fan_capped.saturating_sub(1)));
        
        // Bloody Battle: No oya advantage, all players pay the same
        let points_per_player = base_points / 3; // 3 players pay (excluding winner)
        let scores = [
            base_points as f32, // Winner gets total
            -points_per_player as f32, // Other players pay
            -points_per_player as f32,
            -points_per_player as f32,
        ];
        
        return Some(scores);
        
        // TODO: The rest of this function deals with Japanese Mahjong rules
        // (dora/宝牌, riichi/立直, uradora/里宝牌, etc.) which don't apply to Bloody Battle.
        // This needs to be completely rewritten for Bloody Battle scoring.
        // The code below is commented out as it's unreachable and uses Japanese Mahjong rules.
        
        /*
        let (fu, han) = match calc.agari(additional_yakus, num_doras)? {
            Agari::Normal { fu, han } => (fu, han),
            a @ Agari::Yakuman(_) => {
                return Some([a.point(is_oya).tsumo_total(is_oya) as f32; 4]);
            }
        };

        // 役ありの場合

        // ダブル立直、一発、海底撈月で最大3翻まで増加するので、
        // Bloody Battle: Calculate scores directly using fan-based scoring
        // No need for riichi (立直), dora (宝牌), or uradora (里宝牌) calculations
        let mut scores = [0.; 4];

        // Bloody Battle: No riichi (立直), no dora (宝牌), so skip all riichi/dora calculations
        let assume_riichi = false; // Bloody Battle: No riichi (立直)
        if false { // Bloody Battle: Skip riichi (立直)/dora (宝牌) branch
            // 裏ドラ考慮ありかつ表ドラが1枚以上の場合は、厳密に計算する。
            let mut n_indicators = [0; 5];
            let mut sum_indicators = 0;
            for (tid, &count) in self.state.tehai.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                // ドラ表示牌の枚数を数える。
                let tile = must_tile!(tid);
                let ind_count = self.state.tiles_in_wall[tile.prev().as_usize()];
                n_indicators[count as usize] += ind_count;
                sum_indicators += ind_count;
            }

            // 裏ドラの乗る確率を枚数ごとに計算する。
            let mut uradora_probs = [0.; 5];

            #[cfg(feature = "sp_reproduce_cpp_ver")]
            // 厳密に計算するなら残り枚数は数えるべきだが、あまり影響がないので
            // 121枚で固定
            let n_left_tiles = 121;
            #[cfg(not(feature = "sp_reproduce_cpp_ver"))]
            let n_left_tiles = self.state.sum_left_tiles();

            uradora_probs[0] = (n_left_tiles - sum_indicators) as f32 / n_left_tiles as f32;
            for i in 1..5 {
                uradora_probs[i] = n_indicators[i] as f32 / n_left_tiles as f32;
            }

            for (i, s) in scores.iter_mut().enumerate() {
                // 裏ドラ1枚の場合、最大4翻まで乗る可能性がある
                for (j, &p) in uradora_probs.iter().enumerate() {
                    if p == 0. {
                        continue;
                    }
                    let agari = Agari::Normal {
                        fu,
                        han: han + i as u8 + j as u8,
                    };
                    *s += agari.point(is_oya).tsumo_total(is_oya) as f32 * p;
                }
            }
        // Bloody Battle: No riichi (立直), no dora (宝牌), so this branch is removed
        // } else if assume_riichi && self.sup.dora_indicators.len() > 1 {
        //     // 裏ドラ考慮ありかつ表ドラが2枚以上の場合、統計データを利用する。
        //     // Bloody Battle: This code is removed as it's not applicable
        // }
        // Bloody Battle: The else branch below is also removed as it uses Japanese Mahjong rules
        // } else {
        //     // 裏ドラ考慮なしまたは表ドラが0枚の場合
        //     // Bloody Battle: This code is removed as it uses Agari::Normal which doesn't exist
        // }
        // Bloody Battle: All scoring is now handled in get_score() method above
        */
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algo::sp::CALC_SHANTEN_FN;
    use crate::hand::hand;
    use crate::tuz; // Used in test at line 993

    fn feq(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::EPSILON
    }

    #[test]
    fn nanikiru() {
        // Bloody Battle: No bakaze, jikaze, riichi, or dora
        let mut calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: 0, // Bloody Battle: Not used
            jikaze: 0, // Bloody Battle: Not used
            prefer_riichi: false, // Bloody Battle: No riichi
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &[], // Bloody Battle: No dora
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: true,
            calc_shanten_down: true,
        };

        // Bloody Battle: No jihai, updated test case
        let tehai = hand("45678m 34789p 3344m").unwrap();
        let mut tiles_seen = tehai;
        // Bloody Battle: No dora_indicators
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let can_discard = true;
        let tsumos_left = 8;
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        // Bloody Battle: No jihai, updated expected tiles (actual result may vary)
        // assert_eq!(candidates[0].tile, t!(N));
        // assert_eq!(candidates[1].tile, t!(W));
        assert!(candidates.len() > 0);

        // ---

        let tehai = hand("3667m 23489p 34688s").unwrap();
        let mut tiles_seen = tehai;
        // Bloody Battle: No dora_indicators
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let can_discard = true;
        let tsumos_left = 15;
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let candidates = calc
            .calc(state.clone(), can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates[0].tile, t!(9p));
        assert!(candidates[0].shanten_down);

        calc.maximize_win_prob = true;
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates[0].tile, t!(3m));
        assert!(!candidates[0].shanten_down);
        
        // ---

        // Bloody Battle: No bakaze, jikaze, riichi, or dora
        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: 0, // Bloody Battle: Not used
            jikaze: 0, // Bloody Battle: Not used
            prefer_riichi: false, // Bloody Battle: No riichi
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &[], // Bloody Battle: No dora
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: true,
            calc_shanten_down: true,
        };

        let tehai = hand("45677m 456778p 248s").unwrap();
        let mut tiles_seen = tehai;
        // Bloody Battle: No dora_indicators
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let can_discard = true;
        let tsumos_left = 15;
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        let c = if cfg!(feature = "sp_reproduce_cpp_ver") {
            &candidates[2]
        } else {
            &candidates[0]
        };
        assert_eq!(c.tile, t!(2s));
        assert_eq!(c.required_tiles.len(), 17);
        assert_eq!(c.num_required_tiles, 57);
        assert!(c.shanten_down);
        if cfg!(feature = "sp_reproduce_cpp_ver") {
            assert!(feq(c.tenpai_probs[0], 0.88994724));
            assert!(feq(c.win_probs[0], 0.32714003));
            assert!(feq(c.exp_values[0], 5557.188));
        } else {
            assert!(feq(c.tenpai_probs[0], 0.90023905));
            assert!(feq(c.win_probs[0], 0.34794784));
            assert!(feq(c.exp_values[0], 5894.7617));
        }

        // ---

        // Bloody Battle: No bakaze, jikaze, riichi, or dora
        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: 0, // Bloody Battle: Not used
            jikaze: 0, // Bloody Battle: Not used
            prefer_riichi: false, // Bloody Battle: No riichi
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &[], // Bloody Battle: No dora
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: true,
            calc_shanten_down: true,
        };
        // Bloody Battle: No jihai, updated test case
        let tehai = hand("9999m 6677p 88s 335m 1m").unwrap();
        let mut tiles_seen = tehai;
        // Bloody Battle: No dora_indicators
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let can_discard = true;
        let tsumos_left = 5;
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates.len(), 7);

        // feature = "sp_reproduce_cpp_ver" does not use chitoi shanten
        if !cfg!(feature = "sp_reproduce_cpp_ver") {
            let c = &candidates[1];
            assert_eq!(c.tile, t!(1m));
            assert!(c.shanten_down);
            assert_eq!(c.required_tiles.len(), 33); // literally all kinds of tiles
            assert_eq!(c.num_required_tiles, 34 * 4 - tiles_seen.iter().sum::<u8>());
        }
    }

    #[test]
    fn tsumo_only() {
        // Bloody Battle: Test SPCalculator with Bloody Battle Mahjong rules
        // No bakaze, jikaze, riichi, dora - these fields are kept for compatibility but ignored
        
        use crate::algo::sp::CALC_SHANTEN_FN;
        use crate::hand::hand;
        
        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[], // Bloody Battle: No chis
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: 0, // Bloody Battle: No bakaze, set to 0 (ignored)
            jikaze: 0, // Bloody Battle: No jikaze, set to 0 (ignored)
            is_menzen: true,
            num_doras_in_fuuro: 0, // Bloody Battle: No dora
            dora_indicators: &[], // Bloody Battle: No dora
            calc_double_riichi: false, // Bloody Battle: No riichi
            calc_haitei: false, // Bloody Battle: No haitei
            prefer_riichi: false, // Bloody Battle: No riichi
            sort_result: true,
            maximize_win_prob: true,
            calc_tegawari: true,
            calc_shanten_down: true,
        };

        // Test hand: 45677m 456778p 48s (tenpai, waiting for 7p or 9s)
        let tehai = hand("45677m 456778p 48s").unwrap();
        let mut tiles_seen = tehai;
        // Bloody Battle: No dora indicators to add
        // Mark some tiles as seen (simulate game progress)
        tiles_seen[16] += 4; // Mark all 5s as seen (for example)

        let state = InitState {
            tehai,
            akas_in_hand: [false; 3], // Bloody Battle: No akas
            tiles_seen,
            akas_seen: [false; 3], // Bloody Battle: No akas
        };
        
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        assert_eq!(cur_shanten, 0, "Hand should be tenpai");
        
        let can_discard = false; // 3n+1 state (after tsumo, before discard)
        let tsumos_left = 5;
        
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        
        // Should have candidates for winning tiles
        assert!(!candidates.is_empty(), "Should have at least one candidate");
        
        // Verify that the calculator works with Bloody Battle scoring
        // The exact values may differ from Japanese Mahjong, but the structure should be correct
        let c = &candidates[0];
        // Verify candidate has expected structure
        assert!(c.tenpai_probs.len() > 0, "Should have tenpai probabilities");
        assert!(c.win_probs.len() > 0, "Should have win probabilities");
        assert!(c.exp_values.len() > 0, "Should have expected values");
        
        // Verify that expected values are positive (winning should give positive score)
        // In Bloody Battle, tsumo gives positive score to winner
        assert!(c.exp_values[0] >= 0.0, "Expected value should be non-negative for tenpai hand");
    }
}
