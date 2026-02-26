use pyo3::prelude::*;

mod env;
mod ismce_py;
mod opponent;

#[pymodule]
#[pyo3(name = "_engine")]
fn blood_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<env::RustMahjongEnv>()?;
    m.add_function(wrap_pyfunction!(ismce_py::ismce_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(ismce_py::ismce_evaluate_full, m)?)?;
    m.add_function(wrap_pyfunction!(ismce_py::ismce_evaluate_informed, m)?)?;
    m.add_function(wrap_pyfunction!(ismce_py::ismce_danger, m)?)?;
    m.add("NUM_TILE_TYPES", engine::consts::NUM_TILE_TYPES)?;
    m.add("NUM_PLAYERS", engine::consts::NUM_PLAYERS)?;
    m.add("ACTION_SPACE", engine::consts::ACTION_SPACE)?;
    m.add("NUM_STUDENT_CHANNELS", engine::consts::NUM_STUDENT_CHANNELS)?;
    m.add("NUM_ORACLE_CHANNELS", engine::consts::NUM_ORACLE_CHANNELS)?;
    m.add("NUM_ORACLE_EXTRA_CHANNELS", engine::consts::NUM_ORACLE_EXTRA_CHANNELS)?;
    m.add("INITIAL_SCORE", engine::consts::INITIAL_SCORE)?;
    m.add("REWARD_NORM", engine::consts::REWARD_NORM)?;
    m.add("MAX_FAN", engine::consts::MAX_FAN)?;
    m.add("MAX_TURNS", engine::consts::MAX_TURNS)?;
    // Observation channel offsets (for RTPA, etc.)
    m.add("CH_HAND_BASE", engine::consts::CH_HAND_BASE)?;
    m.add("CH_HAND_COUNT", engine::consts::CH_HAND_COUNT)?;
    m.add("CH_GAME_CONTEXT_BASE", engine::consts::CH_GAME_CONTEXT_BASE)?;
    m.add("CH_TURN_PROGRESS", engine::consts::CH_TURN_PROGRESS)?;
    m.add("CH_DING_QUE_BASE", engine::consts::CH_DING_QUE_BASE)?;
    m.add("CH_OPP_DING_QUE_BASE", engine::consts::CH_OPP_DING_QUE_BASE)?;
    m.add("CH_OPP_AGARI_BASE", engine::consts::CH_OPP_AGARI_BASE)?;
    m.add("CH_WALL_REMAINING", engine::consts::CH_WALL_REMAINING)?;
    m.add("CH_SELF_KAWA_BASE", engine::consts::CH_SELF_KAWA_BASE)?;
    m.add("CH_OPP_KAWA_BASE", engine::consts::CH_OPP_KAWA_BASE)?;
    m.add("CH_OPP_KAWA_STRIDE", engine::consts::CH_OPP_KAWA_STRIDE)?;
    m.add("CH_VISIBLE_TILES_BASE", engine::consts::CH_VISIBLE_TILES_BASE)?;
    m.add("CH_OPP_KAWA_OVERVIEW_BASE", engine::consts::CH_OPP_KAWA_OVERVIEW_BASE)?;
    m.add("CH_OPP_SUIT_RATIO_BASE", engine::consts::CH_OPP_SUIT_RATIO_BASE)?;
    m.add("CH_TILES_REMAINING", engine::consts::CH_TILES_REMAINING)?;
    m.add("CH_SELF_MENZEN", engine::consts::CH_SELF_MENZEN)?;
    m.add("CH_SELF_MELDS", engine::consts::CH_SELF_MELDS)?;
    m.add("CH_OPP_MELD_BASE", engine::consts::CH_OPP_MELD_BASE)?;
    m.add("CH_OPP_TERMINAL_RATIO_BASE", engine::consts::CH_OPP_TERMINAL_RATIO_BASE)?;
    m.add("CH_SELF_DISCARD_COUNT", engine::consts::CH_SELF_DISCARD_COUNT)?;
    m.add("CH_HAND_ANALYSIS_BASE", engine::consts::CH_HAND_ANALYSIS_BASE)?;
    m.add("CH_SHANTEN_BASE", engine::consts::CH_SHANTEN_BASE)?;
    m.add("CH_ACTION_CONTEXT_BASE", engine::consts::CH_ACTION_CONTEXT_BASE)?;
    m.add("CH_SP_TABLE_BASE", engine::consts::CH_SP_TABLE_BASE)?;
    m.add("CH_FAN_CONFIG_BASE", engine::consts::CH_FAN_CONFIG_BASE)?;
    m.add("CH_OPP_HAND_INFO_BASE", engine::consts::CH_OPP_HAND_INFO_BASE)?;
    m.add("CH_GENBUTSU_BASE", engine::consts::CH_GENBUTSU_BASE)?;
    Ok(())
}
