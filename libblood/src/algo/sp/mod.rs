//! Rust port of nekobean's C++ implementation of his single-player mahjong
//! calculator. Some of the original comments are included.
//!
//! Source: <https://github.com/nekobean/mahjong-cpp>
//!
//! Major differences compared to the C++ version:
//! - Whenever shanten calculation is involved, all types of shanten will be
//!   considered (using `shanten::calc_all`). In the original version, you can
//!   only choose one of normal and chitoi (血战到底无国士).
//! - The actual number of tiles left is calculated and used, while the original
//!   version uses a fixed value of 121.
//! - `max_tsumo` is set to the actual value, instead of the hardcoded 17 or 18
//!   in the original version. Not only does this reduce the amount of
//!   calculations, but more importantly, I think this is the theoretically
//!   correct way to calculate, since we keep track of the actual `tiles_seen`
//!   on board so we can have the accurate denominator when building the
//!   `tsumo_prob_table`.
//!
//! Other improvements:
//! - More aggressive compile-time optimizations.
//!
//! To reproduce the behavior of the original C++ version, set feature
//! `sp_reproduce_cpp_ver`.

mod calc;
mod candidate;
mod state;
mod tile;

pub use calc::SPCalculator;
pub use candidate::{Candidate, CandidateColumn};
pub use state::InitState;
pub use tile::RequiredTile;

#[cfg(feature = "sp_reproduce_cpp_ver")]
pub const MAX_TSUMOS_LEFT: usize = 18;
/// In practice, the max number of tsumos left should be 17, since the first
/// tsumo of oya is mandatory.
#[cfg(not(feature = "sp_reproduce_cpp_ver"))]
pub const MAX_TSUMOS_LEFT: usize = 17;

#[cfg(feature = "sp_reproduce_cpp_ver")]
fn calc_normal_wrapper(tiles: &[u8; 27], len_div3: u8, ding_que: Option<crate::mjai::Suit>) -> i8 {
    // FIX: 旧代码忽略 ding_que，直接用 calc_normal(tiles, len_div3)，导致：
    // 1. 定缺牌被当作有效牌，向听被严重低估
    // 2. 未计算七对子路径
    // 修复后使用 calc_all 正确处理定缺罚分和七对子。
    // NOTE: reproduce_cpp_ver 模式主要用于与 C++ 版本对比验证，
    //       但忽略定缺会使 SP 值在血战到底下完全不可靠。
    super::shanten::calc_all(tiles, len_div3, ding_que)
}

#[cfg(feature = "sp_reproduce_cpp_ver")]
const CALC_SHANTEN_FN: fn(&[u8; 27], u8, Option<crate::mjai::Suit>) -> i8 = calc_normal_wrapper;
#[cfg(not(feature = "sp_reproduce_cpp_ver"))]
const CALC_SHANTEN_FN: fn(&[u8; 27], u8, Option<crate::mjai::Suit>) -> i8 = super::shanten::calc_all;
