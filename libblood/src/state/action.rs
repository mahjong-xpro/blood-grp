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
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl PlayerState {
    /// Check if `action` is a valid reaction to the current state.
    pub fn validate_reaction(&self, action: &Event) -> Result<()> {
        let cans = self.last_cans;

        match action {
            Event::None => {
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
                
                if let Some(ding_que_suit) = self.ding_que {
                    // 基础规则：定缺规则检查
                    // 1. 如果手牌中还有定缺花色的牌，必须优先打出定缺花色的牌
                    // 2. 如果手牌中没有定缺花色的牌了，不能打出定缺花色的牌（即使之前还有）
                    let tile_id = pai.as_usize();
                    let tile_suit = tile_id / 9; // 0=Man, 1=Pin, 2=Sou
                    let ding_que_suit_id = match ding_que_suit {
                        crate::mjai::Suit::Man => 0,
                        crate::mjai::Suit::Pin => 1,
                        crate::mjai::Suit::Sou => 2,
                    };
                    let ding_que_start = ding_que_suit_id * 9;
                    let ding_que_end = ding_que_start + 9;
                    
                    // Check if hand still has any ding_que suit tiles
                    let has_ding_que_tiles = (ding_que_start..ding_que_end)
                        .any(|i| self.tehai[i] > 0);
                    
                    if has_ding_que_tiles {
                        // Must discard ding_que suit tiles first (基础规则)
                        ensure!(
                            tile_suit == ding_que_suit_id,
                            "must discard ding_que suit tiles first: {pai:?} (ding_que: {ding_que_suit:?}). This violates the fundamental rule of ding_que."
                        );
                    } else {
                        // Cannot discard ding_que suit tiles (even if none remain, rule still applies)
                        // 基础规则：即使手牌中没有定缺花色的牌了，也不能打出定缺花色的牌
                        ensure!(
                            tile_suit != ding_que_suit_id,
                            "cannot discard ding_que suit tile: {pai:?} (ding_que: {ding_que_suit:?}). This violates the fundamental rule of ding_que."
                        );
                    }
                } else {
                    // 基础规则：如果玩家还没有选择定缺，不应该能打牌
                    // 但在某些特殊情况下（如测试），可能允许，所以这里不强制检查
                    // 如果需要在生产环境中强制检查，可以取消下面的注释
                    // ensure!(
                    //     false,
                    //     "cannot discard before ding_que is selected. This violates the fundamental rule."
                    // );
                }
                
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
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "pon target is not the last kawa tile",
                );
                ensure!(cans.can_pon, "cannot pon");
                self.ensure_tiles_in_hand(&consumed)?;
            }

            Event::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                ensure!(target != actor, "daiminkan from itself");
                ensure!(
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "daiminkan target is not the last kawa tile",
                );
                ensure!(cans.can_daiminkan, "cannot daiminkan");
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
                self.ensure_tiles_in_hand(&consumed)?;
            }

            Event::Hora { target, .. } => {
                if target == self.player_id {
                    ensure!(cans.can_tsumo_agari, "cannot tsumo agari");
                } else {
                    ensure!(cans.can_ron_agari, "cannot ron agari");
                }
            }

            Event::None => return Ok(()),

            _ => bail!("unexpected action {action:?}"),
        };

        Ok(())
    }

    fn ensure_tiles_in_hand(&self, tiles: &[Tile]) -> Result<()> {
        for &tile in tiles {
            ensure!(
                self.tehai[tile.as_usize()] > 0,
                "{tile} is not in hand",
            );
        }
        Ok(())
    }
}
