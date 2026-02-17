use super::{ActionCandidate, PlayerState};
use crate::tile::Tile;

use pyo3::prelude::*;

#[pymethods]
impl PlayerState {
    #[getter]
    #[inline]
    #[must_use]
    pub const fn player_id(&self) -> u8 {
        self.player_id
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyoku(&self) -> u8 {
        self.kyoku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn is_oya(&self) -> bool {
        self.oya == 0
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tehai(&self) -> [u8; 27] {
        self.tehai
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn pons(&self) -> &[u8] {
        &self.pons
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn minkans(&self) -> &[u8] {
        &self.minkans
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn ankans(&self) -> &[u8] {
        &self.ankans
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_turn(&self) -> u8 {
        self.at_turn
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn shanten(&self) -> i8 {
        self.shanten
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn waits(&self) -> [bool; 27] {
        self.waits
    }

    #[inline]
    #[pyo3(name = "last_self_tsumo")]
    fn last_self_tsumo_py(&self) -> Option<String> {
        self.last_self_tsumo.map(|t| t.to_string())
    }
    #[inline]
    #[pyo3(name = "last_kawa_tile")]
    fn last_kawa_tile_py(&self) -> Option<String> {
        self.last_kawa_tile.map(|t| t.to_string())
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn last_cans(&self) -> ActionCandidate {
        self.last_cans
    }

    /// 清除 last_cans 中所有行动标志。
    /// 用于 board.rs 在 Hora 事件全部处理完毕后清除残留的行动标志。
    ///
    /// 背景：Hora 是 announce 事件，PlayerState::update 不会重置 last_cans
    /// （为支持多家荣和时连续 broadcast Hora）。但所有 Hora 在同一个 step()
    /// 内完成后，Dahai/Kakan 窗口残留的 can_pon / can_daiminkan / can_ron_agari
    /// 等标志必须清除，否则 poll() 会误认为仍有玩家可行动。
    ///
    /// 安全性：
    /// - agari_points() 在 handle_hora() 内调用，早于此方法
    /// - furiten 检测在 broadcast(Hora) → update() 时完成，也早于此方法
    pub fn clear_action_candidates(&mut self) {
        self.last_cans = ActionCandidate::default();
        self.ankan_candidates.clear();
        self.kakan_candidates.clear();
    }

    #[inline]
    #[pyo3(name = "ankan_candidates")]
    fn ankan_candidates_py(&self) -> Vec<String> {
        self.ankan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }
    #[inline]
    #[pyo3(name = "kakan_candidates")]
    fn kakan_candidates_py(&self) -> Vec<String> {
        self.kakan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }


}

impl PlayerState {
    #[inline]
    #[must_use]
    pub const fn last_self_tsumo(&self) -> Option<Tile> {
        self.last_self_tsumo
    }
    #[inline]
    #[must_use]
    pub const fn last_kawa_tile(&self) -> Option<Tile> {
        self.last_kawa_tile
    }

    #[inline]
    #[must_use]
    pub fn ankan_candidates(&self) -> &[Tile] {
        &self.ankan_candidates
    }
    #[inline]
    #[must_use]
    pub fn kakan_candidates(&self) -> &[Tile] {
        &self.kakan_candidates
    }
}
