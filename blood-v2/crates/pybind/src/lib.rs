use pyo3::prelude::*;

mod env;
mod ismce_py;
mod opponent;

#[pymodule]
#[pyo3(name = "_engine")]
fn blood_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<env::RustMahjongEnv>()?;
    m.add_function(wrap_pyfunction!(ismce_py::ismce_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(ismce_py::ismce_danger, m)?)?;
    m.add("NUM_TILE_TYPES", engine::consts::NUM_TILE_TYPES)?;
    m.add("NUM_PLAYERS", engine::consts::NUM_PLAYERS)?;
    m.add("ACTION_SPACE", engine::consts::ACTION_SPACE)?;
    m.add("NUM_STUDENT_CHANNELS", engine::consts::NUM_STUDENT_CHANNELS)?;
    m.add("NUM_ORACLE_CHANNELS", engine::consts::NUM_ORACLE_CHANNELS)?;
    Ok(())
}
