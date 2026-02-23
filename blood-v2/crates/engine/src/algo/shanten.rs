/// Shanten calculation with thread-local memoization cache.
///
/// Uses FxHashMap (identity/FNV-based hasher) for faster lookups on small
/// fixed-size keys like (HandCounts, usize).
///
/// Cache lifecycle: call `clear_shanten_cache()` at the start of each
/// SP Table computation to avoid stale entries across game steps.

use crate::hand::HandCounts;

use std::cell::RefCell;
use rustc_hash::FxHashMap;

thread_local! {
    static SHANTEN_CACHE: RefCell<FxHashMap<(HandCounts, usize), i8>> =
        RefCell::new(FxHashMap::with_capacity_and_hasher(1024, Default::default()));
}

/// Clear the thread-local shanten cache.
/// Call once per SP Table computation (before iterating discard candidates).
#[inline]
pub fn clear_shanten_cache() {
    SHANTEN_CACHE.with(|c| c.borrow_mut().clear());
}

/// Compute shanten with memoization.
/// Returns -1 for complete, 0 for tenpai, 1 for iishanten, etc.
#[inline]
pub fn calc_shanten(hand: &HandCounts, num_melds: usize) -> i8 {
    SHANTEN_CACHE.with(|cache| {
        *cache.borrow_mut()
            .entry((*hand, num_melds))
            .or_insert_with(|| crate::hand::calc_shanten(hand, num_melds))
    })
}

// Re-export waiting_tiles and is_complete from hand.rs (unchanged)
pub use crate::hand::{waiting_tiles, is_complete};
