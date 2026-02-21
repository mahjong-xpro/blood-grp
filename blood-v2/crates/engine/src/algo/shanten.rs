/// Shanten calculation.
///
/// Currently uses the recursive approach from hand.rs.
/// TODO: Replace with precomputed SUHAI_TABLE (1.9M entries) via build.rs
/// for ~100x speedup needed by SP table.
pub use crate::hand::{calc_shanten, is_complete, waiting_tiles};
