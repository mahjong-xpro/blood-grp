mod gameplay;
mod grp;
mod invisible;

use crate::py_helper::add_submodule;
pub use gameplay::{Gameplay, GameplayLoader};
// Re-export GameScore from grp module
pub use grp::GameScore;
pub use invisible::Invisible;

use pyo3::prelude::*;

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    m.add_class::<Gameplay>()?;
    m.add_class::<GameplayLoader>()?;
    m.add_class::<GameScore>()?;
    add_submodule(py, prefix, super_mod, &m)
}
