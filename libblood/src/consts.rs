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
// GRP_SIZE = [kyoku, [score[i] / 10000], [agari[i]], [ding_que[i]]] = 1 + 4 + 4 + 4 = 13
// agari[i] = 1.0 if player i has agari, 0.0 otherwise
// ding_que[i] = 0.0 for Man, 0.5 for Pin, 1.0 for Sou (normalized)
pub const GRP_SIZE: usize = 13;

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
        4 => (1468, 27), // 修复后：考虑最大情况（所有位置都有 item，且 can_discard=true）
                         // 理论计算：
                         //   - SP table 编码之前（最大情况）: 875 + 48 (self_kawa) + 432 (other_kawa) = 1355 行
                         //   - SP table 编码（最大路径）: 111 行
                         //   - can_discard=true 时的额外 2 行（best ev/win prob discard）: 2 行
                         //   - 理论最大总计: 1355 + 111 + 2 = 1468 行
                         // 修复内容：
                         //   - 修复了 encode_self_kawa 和 encode_kawa 的补偿逻辑
                         //   - 补偿逻辑现在考虑最大情况（所有位置都有 item）
                         //   - self_kawa: (6-len) * (4+2) = (6-len) * 6
                         //   - other_kawa: (6-len) * (8+6) = (6-len) * 14
                         //   - 添加了 can_discard=true 时的额外 2 行
                         // 注意：实际运行中，不是所有位置都有 item，所以实际行数会小于 1468
                         // 但使用最大情况可以确保不会溢出
        _ => unreachable!(),
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
    m.add("GRP_SIZE", GRP_SIZE)?;
    add_submodule(py, prefix, super_mod, &m)
}
