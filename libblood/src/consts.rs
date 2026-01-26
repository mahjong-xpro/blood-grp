use crate::py_helper::add_submodule;

use pyo3::prelude::*;

pub const MAX_VERSION: u32 = 4;

pub const ACTION_SPACE: usize = 27 // discard (27 tile kinds)
                              + 1  // pon
                              + 1  // kan (decide)
                              + 1  // agari
                              + 1  // ryukyoku
                              + 1; // pass
// = 32


#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    // Calculated dimensions based on encode_obs function in obs_repr.rs
    // Ding que encoding added:
    //   v1: +26 (3+1+13+9)
    //   v2|3: +16 (3+1+3+9)
    //   v4: +14 (3+1+1+9)
    match version {
        1 => (964, 27), // 938 + 26 = 964 (ding que: 3+1+13+9)
        2 => (960, 27), // 944 + 16 = 960 (ding que: 3+1+3+9)
        3 => (952, 27), // 936 + 16 = 952 (ding que: 3+1+3+9)
        4 => (1466, 27), // 修复后：移除死特征 (-6) + 移除Ryukyoku (-1) + 添加Agari (+3) = 1470 - 4 = 1466
        _ => panic!("Unsupported version"),
    }
}

#[pyfunction]
#[inline]
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
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
    add_submodule(py, prefix, super_mod, &m)
}
