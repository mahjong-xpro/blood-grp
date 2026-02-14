use crate::py_helper::add_submodule;

use pyo3::prelude::*;

/// Only version 4 is supported. v1-v3 have been removed (legacy from Japanese Mahjong migration).
pub const VERSION: u32 = 4;

/// Initial score per player (血战到底零和：4×INITIAL_SCORE = TOTAL_SCORE).
/// 60000 起步可避免飞人时出现负分。
pub const INITIAL_SCORE: i32 = 60_000;
/// Total points in the game (zero-sum). Must equal 4 * INITIAL_SCORE.
pub const TOTAL_SCORE: i32 = 4 * INITIAL_SCORE;

pub const ACTION_SPACE: usize = 27 // discard (27 tile kinds)
                              + 1  // pon
                              + 1  // kan (decide)
                              + 1  // agari
                              + 1  // pass
                              + 3; // ding que (Man, Pin, Sou)
// = 34

#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    match version {
        // 423 = 精简血战到底特征编码 + 可配置番型标志 + BUG-09 修复 (SP 巡数 14→28)
        // 删除冗余: suit_count, score_deltas, active_players, genbutsu, fully_visible,
        //   kawa first-6 (×4 players), SP dead code
        // 压缩: fuuro 4→2 ch/meld, kyoku 4→1, shanten 7→5
        // 新增: wall_remaining, menzen, self_fuuro, at_turn, acceptance, opp_fuuro(×3)
        // 新增: fan_config flags ×7
        // BUG-09 fix: SP turns 14→28 (+42 ch)，覆盖血战到底 2 人对局全巡程
        4 => (423, 27),
        _ => panic!("Unsupported version: only v4 is supported"),
    }
}

#[pyfunction]
#[inline]
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
    match version {
        4 => (121, 27),
        _ => panic!("Unsupported version: only v4 is supported"),
    }
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "consts")?;
    m.add_function(wrap_pyfunction!(obs_shape, &m)?)?;
    m.add_function(wrap_pyfunction!(oracle_obs_shape, &m)?)?;
    m.add("VERSION", VERSION)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("INITIAL_SCORE", INITIAL_SCORE)?;
    m.add("TOTAL_SCORE", TOTAL_SCORE)?;
    m.add_class::<crate::algo::agari::FanConfig>()?;
    add_submodule(py, prefix, super_mod, &m)
}
