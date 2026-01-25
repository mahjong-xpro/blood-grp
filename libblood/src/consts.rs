use crate::py_helper::add_submodule;

use pyo3::prelude::*;

pub const MAX_VERSION: u32 = 4;

// Bloody Battle Mahjong: ACTION_SPACE
pub const ACTION_SPACE: usize = 27 // discard (27 tile kinds)
                              + 1  // pon
                              + 1  // kan (decide)
                              + 1  // agari
                              + 1  // ryukyoku
                              + 1; // pass
// = 32 (no riichi, no chi)
// Bloody Battle Mahjong: GRP_SIZE = [kyoku, [score[i] / 10000]] = 1 + 4 = 5
pub const GRP_SIZE: usize = 5;

#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    // Bloody Battle: 27 tile kinds (no jihai, no red 5s)
    // Calculated dimensions based on encode_obs function in obs_repr.rs
    match version {
        1 => (938, 27), // Calculated: 938 rows × 27 tile kinds
        2 => (944, 27), // Calculated: 944 rows × 27 tile kinds
        3 => (936, 27), // Calculated: 936 rows × 27 tile kinds
        4 => (998, 27), // Calculated: 998 rows × 27 tile kinds (includes SP table)
        _ => unreachable!(),
    }
}

#[pyfunction]
#[inline]
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
    // Bloody Battle: 27 tile kinds (no jihai, no red 5s)
    // Calculated dimensions based on encode_oracle_obs function in board.rs and invisible.rs
    match version {
        1 => (128, 27), // Calculated: 128 rows × 27 tile kinds
        2 | 3 | 4 => (134, 27), // Calculated: 134 rows × 27 tile kinds
        _ => unreachable!(),
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
    m.add("MAX_VERSION", MAX_VERSION)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("GRP_SIZE", GRP_SIZE)?;
    add_submodule(py, prefix, super_mod, &m)
}
