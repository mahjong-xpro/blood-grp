use super::PlayerState;
use crate::mjai::Event;
use crate::tile::Tile;

use anyhow::{Result, bail, ensure};
use pyo3::prelude::*;
use serde::Serialize;

#[pyclass]
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct ActionCandidate {
    #[pyo3(get)]
    pub can_discard: bool,
    #[pyo3(get)]
    pub can_pon: bool,
    #[pyo3(get)]
    pub can_daiminkan: bool,
    #[pyo3(get)]
    pub can_kakan: bool,
    #[pyo3(get)]
    pub can_ankan: bool,
    #[pyo3(get)]
    pub can_tsumo_agari: bool,
    #[pyo3(get)]
    pub can_ron_agari: bool,
    #[pyo3(get)]
    pub can_ding_que: bool,
    #[pyo3(get)]
    pub target_actor: u8,
}

#[pymethods]
impl ActionCandidate {
    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_kan(&self) -> bool {
        self.can_daiminkan || self.can_kakan || self.can_ankan
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_agari(&self) -> bool {
        self.can_tsumo_agari || self.can_ron_agari
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_pass(&self) -> bool {
        self.can_pon || self.can_daiminkan || self.can_ron_agari
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_act(&self) -> bool {
        self.can_discard
            || self.can_pon
            || self.can_kan()
            || self.can_agari()
            || self.can_ding_que
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl PlayerState {
    /// Check if `action` is a valid reaction to the current state.
    pub fn validate_reaction(&self, action: &Event) -> Result<()> {
        // 防御性检查：已和牌的玩家不应再做任何动作
        ensure!(
            !self.has_agari,
            "player {} has already won and cannot act",
            self.player_id,
        );

        let cans = self.last_cans;

        match action {
            Event::None => {
                // In mjai, `none` is used as "no reaction".
                // For safety, only allow it when:
                // - the player truly cannot act, OR
                // - the player can legally pass on an interrupt (pon/daiminkan/ron window).
                //
                // This prevents silent deadlocks / invalid logs where the player should act
                // (discard, ding_que, etc.) but returned `none`.
                ensure!(
                    !cans.can_act() || cans.can_pass(),
                    "cannot pass (none) when an action is required: {cans:?}"
                );
                return Ok(());
            }
            _ => (),
        };

        if let Some(actor) = action.actor() {
            ensure!(
                actor == self.player_id,
                "actor is {actor}, not self ({})",
                self.player_id,
            );
        } else {
            bail!("action does not have actor and is not ryukyoku");
        }

        match *action {
            Event::Dahai { pai, tsumogiri, .. } => {
                ensure!(cans.can_discard, "cannot discard");
                self.ensure_tiles_in_hand(&[pai])?;
                
                // 基础规则：定缺完成前不允许打牌（本项目选择严格执行，避免死锁/无效日志）
                ensure!(self.ding_que.is_some(), "cannot discard before ding_que is selected");

                // 基础规则（按你确认的语义）：
                // - 只要手牌中还有缺门花色的牌，就必须优先打出缺门花色的牌
                // - 如果手牌中没有缺门花色的牌，则不再对出牌花色做额外限制（由其它规则决定）
                let tile_id = pai.as_usize();

                // 碰后禁打规则：碰了某张牌后，本轮不能立即打出同种牌
                // forbidden_tiles 由 update.rs 的 pon() 设置
                ensure!(
                    !self.forbidden_tiles[tile_id],
                    "cannot discard forbidden tile {pai:?} (e.g. same tile just pon'd)"
                );

                ensure!(
                    crate::ding_que::discard_allowed(tile_id, &self.tehai, self.ding_que),
                    "ding_que discard rule violated: {pai:?} (ding_que: {:?})",
                    self.ding_que
                );
                
                if tsumogiri {
                    if let Some(tile) = self.last_self_tsumo {
                        ensure!(tile == pai, "cannot tsumogiri");
                    } else {
                        bail!("tsumogiri but the player has not dealt any tile yet");
                    }
                }
            }

            Event::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                ensure!(target != actor, "pon from itself");
                ensure!(
                    target == cans.target_actor,
                    "pon target {target} does not match the actual discarder {}",
                    cans.target_actor,
                );
                ensure!(
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "pon target is not the last kawa tile",
                );
                ensure!(cans.can_pon, "cannot pon");
                // consumed 必须全部与 pai 同种（碰的 2 张必须和被碰的牌相同）
                ensure!(
                    consumed.iter().all(|&t| t == pai),
                    "pon consumed tiles {:?} do not match pai {pai}",
                    consumed,
                );
                self.ensure_tiles_in_hand(&consumed)?;
            }

            Event::Daiminkan {
                actor,
                target,
                pai,
                consumed,
                ..
            } => {
                ensure!(target != actor, "daiminkan from itself");
                ensure!(
                    target == cans.target_actor,
                    "daiminkan target {target} does not match the actual discarder {}",
                    cans.target_actor,
                );
                ensure!(
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "daiminkan target is not the last kawa tile",
                );
                ensure!(cans.can_daiminkan, "cannot daiminkan");
                // consumed 必须全部与 pai 同种（大明杠的 3 张必须和被杠的牌相同）
                ensure!(
                    consumed.iter().all(|&t| t == pai),
                    "daiminkan consumed tiles {:?} do not match pai {pai}",
                    consumed,
                );
                self.ensure_tiles_in_hand(&consumed)?;
            }
            Event::Kakan { pai, .. } => {
                ensure!(cans.can_kakan, "cannot kakan");
                ensure!(
                    self.kakan_candidates.contains(&pai),
                    "cannot kakan {pai}",
                );
                self.ensure_tiles_in_hand(&[pai])?;
            }
            Event::Ankan { consumed, .. } => {
                ensure!(cans.can_ankan, "cannot ankan");
                let tile = consumed[0];
                ensure!(self.ankan_candidates.contains(&tile), "cannot ankan {tile}");
                // consumed 4 张必须全部相同（暗杠）
                ensure!(
                    consumed.iter().all(|&t| t == tile),
                    "ankan consumed tiles {:?} are not all the same",
                    consumed,
                );
                self.ensure_tiles_in_hand(&consumed)?;
            }

            Event::Hora { target, .. } => {
                if target == self.player_id {
                    ensure!(cans.can_tsumo_agari, "cannot tsumo agari");
                } else {
                    ensure!(cans.can_ron_agari, "cannot ron agari");
                    ensure!(
                        target == cans.target_actor,
                        "ron target {target} does not match the actual discarder/kakan actor {}",
                        cans.target_actor,
                    );
                }
            }

            Event::DingQue { actor, suit: _ } => {
                ensure!(actor == self.player_id, "ding_que from others");
                ensure!(cans.can_ding_que, "cannot ding_que");
            }

            // Event::None 已在上方第一个 match 处理并 early return，此处不可达。
            // 保留通配符以捕获未预期的事件类型。
            _ => bail!("unexpected action {action:?}"),
        };

        Ok(())
    }

    fn ensure_tiles_in_hand(&self, tiles: &[Tile]) -> Result<()> {
        // 正确验证数量：统计需要消耗的每种牌数量，再与手牌对比
        let mut needed = [0u8; 27];
        for &tile in tiles {
            needed[tile.as_usize()] += 1;
        }
        for (tid, &count) in needed.iter().enumerate() {
            if count > 0 {
                ensure!(
                    self.tehai[tid] >= count,
                    "not enough tiles in hand: need {} of tile id {}, but only have {}",
                    count,
                    tid,
                    self.tehai[tid]
                );
            }
        }
        Ok(())
    }
}
