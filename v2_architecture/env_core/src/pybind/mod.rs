use pyo3::prelude::*;
use crate::core::batched_env::BatchedEnv;

#[pyclass(unsendable)]
struct PyBatchedEnv {
    inner: BatchedEnv,
}

#[pymethods]
impl PyBatchedEnv {
    #[new]
    fn new(num_envs: usize) -> Self {
        Self {
            inner: BatchedEnv::new(num_envs),
        }
    }

    fn reset(&mut self, indices: Option<Vec<usize>>) {
        match indices {
            Some(ref idxs) => self.inner.reset(Some(idxs.as_slice())),
            None => self.inner.reset(None),
        }
    }

    fn step(&mut self, actions: Vec<i32>) -> Vec<bool> {
        self.inner.step(&actions)
    }

    fn num_envs(&self) -> usize {
        self.inner.num_envs
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn env_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchedEnv>()?;
    Ok(())
}
