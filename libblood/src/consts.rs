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
// = 32 (no riichi/立直, no chi/吃)
// Bloody Battle Mahjong: GRP_SIZE = [kyoku, [score[i] / 10000], [agari[i]], [ding_que[i]]] = 1 + 4 + 4 + 4 = 13
// agari[i] = 1.0 if player i has agari, 0.0 otherwise
// ding_que[i] = 0.0 for Man, 0.5 for Pin, 1.0 for Sou (normalized)
pub const GRP_SIZE: usize = 13;

#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    // Bloody Battle: 27 tile kinds (no jihai, no red 5s)
    // Calculated dimensions based on encode_obs function in obs_repr.rs
    // Ding que encoding added:
    //   v1: +26 (3+1+13+9)
    //   v2|3: +16 (3+1+3+9)
    //   v4: +14 (3+1+1+9)
    match version {
        1 => (964, 27), // 938 + 26 = 964 (ding que: 3+1+13+9)
        2 => (960, 27), // 944 + 16 = 960 (ding que: 3+1+3+9)
        3 => (952, 27), // 936 + 16 = 952 (ding que: 3+1+3+9)
        4 => (1100, 27), // 临时设置为 1100 行以覆盖所有编码路径
                         // 问题：idx 一直在增加（1024 → 1026 → 1038 → 1040/1048/1050 → 1052）
                         // 说明不同的编码路径需要不同的行数，可能存在编码逻辑不一致
                         // TODO: 需要深入分析编码逻辑，找出所有编码路径的最大行数需求
                         // 并优化编码逻辑以减少空间使用，而不是简单地增加 obs_shape 的值
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
