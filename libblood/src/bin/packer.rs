use anyhow::{Context, Result};
use clap::Parser;
use libblood::dataset::GameScore; // Re-use existing structs if possible, or define new ones
use libblood::mjai::Event;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write, Read};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 5000)]
    batch_size: usize,
}

#[derive(Serialize, Deserialize)]
struct PackedChunk {
    version: u32,
    games: Vec<Vec<Event>>, // Store raw events for now, or processed Game struct?
                            // Events are flexible.
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    log::info!("Starting Packer Service");
    log::info!("Input: {:?}", args.input);
    log::info!("Output: {:?}", args.output);

    fs::create_dir_all(&args.input)?;
    fs::create_dir_all(&args.output)?;

    let mut buffer: Vec<Vec<Event>> = Vec::with_capacity(args.batch_size);
    let mut files_to_delete: Vec<PathBuf> = Vec::with_capacity(args.batch_size);

    loop {
        let entries = fs::read_dir(&args.input)?;
        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map_or(false, |ext| ext == "gz") 
                && !p.extension().map_or(false, |ext| ext == "tmp")
            })
            .collect();

        // Sort to ensure deterministic order if needed, or mostly chronological
        files.sort();

        if files.is_empty() {
            if buffer.is_empty() {
                sleep(Duration::from_secs(1));
                continue;
            } else {
                // If we have some data but no new files, maybe flush after timeout?
                // For now, just wait.
                sleep(Duration::from_secs(1));
                continue;
            }
        }

        for path in files {
            match process_file(&path) {
                Ok(events) => {
                    buffer.push(events);
                    files_to_delete.push(path);
                }
                Err(e) => {
                    log::error!("Failed to process {:?}: {}", path, e);
                    // Move to broken? Or delete?
                    // fs::rename(&path, args.input.join("broken").join(path.file_name().unwrap()))?;
                }
            }

            if buffer.len() >= args.batch_size {
                flush_buffer(&args.output, &mut buffer)?;
                cleanup_files(&files_to_delete)?;
                files_to_delete.clear();
            }
        }
        
        // Optional: Flush partial buffer if too old?
    }
}

fn process_file(path: &Path) -> Result<Vec<Event>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder = GzDecoder::new(reader);
    let events: Vec<Event> = serde_json::from_reader(decoder)?;
    Ok(events)
}

fn flush_buffer(output_dir: &Path, buffer: &mut Vec<Vec<Event>>) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let filename = format!("chunk_{}.bin.lz4", timestamp);
    let path = output_dir.join(&filename);
    let temp_path = output_dir.join(format!("{}.tmp", filename));

    log::info!("Flushing {} games to {:?}", buffer.len(), path);

    let file = File::create(&temp_path)?;
    let mut writer = BufWriter::new(file);
    
    // LZ4 Compression
    let mut encoder = lz4::EncoderBuilder::new()
        .level(4)
        .build(writer)?;

    // Bincode Serialization
    let chunk = PackedChunk {
        version: 1,
        games: std::mem::take(buffer),
    };
    
    bincode::serialize_into(&mut encoder, &chunk)?;
    
    let (_output, result) = encoder.finish();
    result?;

    fs::rename(temp_path, path)?;

    Ok(())
}

fn cleanup_files(files: &[PathBuf]) -> Result<()> {
    for path in files {
        fs::remove_file(path).unwrap_or_else(|e| log::warn!("Failed to delete {:?}: {}", path, e));
    }
    Ok(())
}
