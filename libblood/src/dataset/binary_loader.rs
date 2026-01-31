use crate::dataset::Gameplay;
use crate::mjai::Event;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use serde::Deserialize;
use pyo3::prelude::*;

#[derive(Deserialize)]
struct PackedChunk {
    version: u32,
    games: Vec<Vec<Event>>,
}

#[pyclass]
pub struct BinaryLoader;

#[pymethods]
impl BinaryLoader {
    #[staticmethod]
    fn load_chunk(path: String) -> anyhow::Result<Vec<Gameplay>> {
        let file = File::open(&path).with_context(|| format!("failed to open packed chunk: {}", path))?;
        let reader = BufReader::new(file);
        let decoder = lz4::Decoder::new(reader)?;
        
        let chunk: PackedChunk = bincode::deserialize_from(decoder)
            .with_context(|| format!("failed to deserialize packed chunk: {}", path))?;

        if chunk.version != 1 {
            anyhow::bail!("unsupported chunk version: {}", chunk.version);
        }

        let mut games = Vec::with_capacity(chunk.games.len());
        for events in chunk.games {
            // Re-use existing load_events logic
            // Note: Data parallelism is less critical here since we already have the events parsed?
            // Wait, load_events does the heavy state simulation.
            // We SHOULD parallelize this part using Rayon if Possible.
            // But this function is called from Python DataLoader worker which is already parallelized (num_workers=8).
            // So single-threaded here is fine, but we can do par_iter if we want extra speed.
            // Let's stick to sequential for simplicity inside this function, 
            // relying on Process-level parallelism.
            let game = Gameplay::load_events(&events)?;
            games.push(game);
        }

        Ok(games)
    }
}
