use super::board::{Board, BoardState, Poll};
use super::result::GameResult;
use crate::agent::BatchAgent;
use crate::consts::INITIAL_SCORE;
use crate::hand::tehai_to_strings;
use crate::mjai::EventExt;
use std::time::Duration;
use std::{array, mem};

use anyhow::{Result, ensure};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::prelude::*;

pub struct BatchGame {
    pub length: u8,
    pub init_scores: [i32; 4],
    pub disable_progress_bar: bool,
}

#[derive(Clone, Copy, Default)]
pub struct Index {
    /// For `Game` to find a specific `Agent` (game -> agent).
    pub agent_idx: usize,
    /// For `Agent` to find a specific player ID (agent -> game).
    pub player_id_idx: usize,
}

#[derive(Default)]
struct Game {
    length: u8,
    seed: (u64, u64),
    indexes: [Index; 4],

    oracle_obs_versions: [Option<u32>; 4],
    invisible_state_cache: [Option<Array2<f32>>; 4],

    last_reactions: [EventExt; 4], // cached for poll phase

    board: BoardState,
    kyoku: u8,
    scores: [i32; 4],
    game_log: Vec<Vec<EventExt>>,
    /// 和牌顺序（从最后一局继承），用于同分排名。
    agari_order: Vec<u8>,

    kyoku_started: bool,
    ended: bool,
}

impl Game {
    /// Returns iff any player in the game can act or the game has ended.
    fn poll(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<()> {
        if self.ended {
            return Ok(());
        }

        if !self.kyoku_started {
            // Bloody Battle "game" is modeled as a fixed number of kyoku (hands).
            // `BoardState` already contains the end condition within a kyoku:
            // it ends when 3 players have agari or the wall is exhausted.
            //
            // So at this layer we must NOT apply riichi-style "all-last / 30k / extra rounds" rules.
            if self.kyoku >= self.length {
                self.ended = true;
                return Ok(());
            }

            let mut next_board = Board {
                kyoku: self.kyoku,
                scores: self.scores,
                ..Default::default()
            };
            next_board.init_from_seed(self.seed);
            self.board = next_board.into_state();
            self.kyoku_started = true;
        }

        let reactions = mem::take(&mut self.last_reactions);
        let poll = self.board.poll(reactions)?;
        match poll {
            Poll::InGame => {
                let ctx = self.board.agent_context();

                // 血战到底规则：胡牌优先于碰/杠，但优先级在 board.rs step() 中处理。
                // 这里让所有能行动的玩家都获得 set_scene，确保如果荣和玩家放弃，
                // 碰/杠玩家仍有机会行动。
                for (player_id, state) in ctx.player_states.iter().enumerate() {
                    let needs_reaction = if self.board.is_ding_que_phase() {
                        !self.board.ding_que_selected(player_id)
                    } else {
                        // FIX: 已和牌玩家不需要反应。
                        // Hora 是 announce 事件，last_cans 不会被重置，can_act() 可能仍为 true。
                        // 若不排除已和牌玩家，agent 可能返回非法动作（如 Dahai），
                        // 而 board.rs 跳过已和牌玩家的 validate_reaction，导致状态腐蚀。
                        state.last_cans().can_act() && !state.has_agari
                    };

                    if !needs_reaction {
                        continue;
                    }

                    let invisible_state = self.oracle_obs_versions[player_id]
                        .map(|ver| self.board.encode_oracle_obs(player_id as u8, ver));
                    self.invisible_state_cache[player_id].clone_from(&invisible_state);

                    let idx = self.indexes[player_id];
                    agents[idx.agent_idx].set_scene(
                        idx.player_id_idx,
                        ctx.log,
                        state,
                        invisible_state,
                    )?;
                }
            }
            Poll::End => {
                self.kyoku_started = false;

                let ctx = self.board.agent_context();
                for idx in &self.indexes {
                    agents[idx.agent_idx].end_kyoku(idx.player_id_idx, Some(ctx.log))?;
                }

                let kyoku_result = self.board.end();
                self.scores = kyoku_result.scores;
                // 保存最后一局的和牌顺序用于最终排名
                self.agari_order = kyoku_result.agari_order;

                let logs = self.board.take_log();
                self.game_log.push(logs);

                let has_tobi = self.scores.iter().any(|&s| s < 0);
                if has_tobi {
                    self.ended = true;
                    return Ok(());
                }

                // 血战到底无连庄规则，直接进入下一局
                self.kyoku += 1;
                return self.poll(agents);
            }
        };

        Ok(())
    }

    fn commit(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<Option<GameResult>> {
        if self.ended {

            let names = array::from_fn(|i| agents[self.indexes[i].agent_idx].name());
            let ctx = self.board.agent_context();
            let final_tehais = ctx
                .player_states
                .iter()
                .map(|s| tehai_to_strings(&s.tehai))
                .collect::<Vec<_>>();
            let game_result = GameResult {
                names,
                scores: self.scores,
                seed: self.seed,
                game_log: mem::take(&mut self.game_log),
                final_tehais: Some(final_tehais),
                agari_order: mem::take(&mut self.agari_order),
            };

            for idx in &self.indexes {
                agents[idx.agent_idx].end_game(idx.player_id_idx, &game_result)?;
            }
            return Ok(Some(game_result));
        }

        let ctx = self.board.agent_context();

        // 血战到底规则：胡牌优先于碰/杠。优先级由 board.rs step() 在处理反应时强制执行。
        // 这里让所有能行动的玩家都提交反应，如果荣和玩家放弃，碰/杠玩家仍可行动。
        for (player_id, state) in ctx.player_states.iter().enumerate() {
            let needs_reaction = if self.board.is_ding_que_phase() {
                !self.board.ding_que_selected(player_id)
            } else {
                // FIX: 同 poll() 中的修复，已和牌玩家不需要提交反应。
                state.last_cans().can_act() && !state.has_agari
            };

            if !needs_reaction {
                continue;
            }

            let invisible_state = self.invisible_state_cache[player_id].take();

            let idx = self.indexes[player_id];
            self.last_reactions[player_id] = agents[idx.agent_idx].get_reaction(
                idx.player_id_idx,
                ctx.log,
                state,
                invisible_state,
            )?;
        }

        Ok(None)
    }
}

impl BatchGame {
    pub const fn standard_game(disable_progress_bar: bool) -> Self {
        Self {
            // Bloody Battle: one kyoku (one deal) per game/episode.
            length: 1,
            init_scores: [INITIAL_SCORE; 4],
            disable_progress_bar,
        }
    }

    pub fn run(
        &self,
        agents: &mut [Box<dyn BatchAgent>],
        indexes: &[[Index; 4]],
        seeds: &[(u64, u64)],
    ) -> Result<Vec<GameResult>> {
        ensure!(!agents.is_empty());
        ensure!(!indexes.is_empty());
        ensure!(
            indexes.len() == seeds.len(),
            "expected `indexes.len() == seeds.len()`, got {} and {}",
            indexes.len(),
            seeds.len(),
        );

        let mut games = indexes
            .iter()
            .zip(seeds)
            .enumerate()
            .map(|(game_idx, (idxs, &seed))| {
                let mut oracle_obs_versions = [None; 4];
                for (i, idx) in idxs.iter().enumerate() {
                    agents[idx.agent_idx].start_game(idx.player_id_idx)?;
                    oracle_obs_versions[i] = agents[idx.agent_idx].oracle_obs_version();
                }

                let game = Box::new(Game {
                    length: self.length,
                    seed,
                    indexes: *idxs,
                    scores: self.init_scores,
                    oracle_obs_versions,
                    ..Default::default()
                });
                Ok((game_idx, game))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut game_results = vec![GameResult::default(); games.len()];
        let mut to_remove = vec![];
        let mut cycles = 0;
        let mut actions = 0;

        let bar = if self.disable_progress_bar {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(games.len() as u64)
        };
        const TEMPLATE: &str =
            "{spinner:.cyan} {msg}\n[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
        let style = ProgressStyle::with_template(TEMPLATE)?
            .tick_chars(".oO°Oo*")
            .progress_chars("#-");
        bar.set_style(style);
        bar.enable_steady_tick(Duration::from_millis(150));

        while !games.is_empty() {
            cycles += 1;
            
            for (_, game) in &mut games {
                game.poll(agents)?;
            }

            for (idx_for_rm, (game_idx, game)) in games.iter_mut().enumerate() {
                if let Some(game_result) = game.commit(agents)? {
                    game_results[*game_idx] = game_result;
                    to_remove.push(idx_for_rm);
                }
            }

            for idx_for_rm in to_remove.drain(..).rev() {
                games.swap_remove(idx_for_rm);
                bar.inc(1);
            }

            actions += games.len();

            let secs = bar.elapsed().as_secs_f64();
            bar.set_message(format!(
                "cycles: {cycles} ({:.3} cycle/s), actions: {actions} ({:.3} action/s)",
                cycles as f64 / secs,
                actions as f64 / secs,
            ));
        }
        bar.abandon();

        Ok(game_results)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::Tsumogiri;

    #[test]
    fn tsumogiri() {
        let g = BatchGame::standard_game(true);
        let mut agents = [
            Box::new(Tsumogiri::new_batched(&[0, 1, 2, 3]).unwrap()) as _,
            Box::new(Tsumogiri::new_batched(&[3, 2, 1, 0]).unwrap()) as _,
        ];
        let indexes = &[
            [
                Index {
                    agent_idx: 0,
                    player_id_idx: 0,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 1,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 1,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 0,
                },
            ],
            [
                Index {
                    agent_idx: 1,
                    player_id_idx: 3,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 2,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 2,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 3,
                },
            ],
        ];

        g.run(&mut agents, indexes, &[(1009, 0), (1021, 0)])
            .unwrap();
    }
}
