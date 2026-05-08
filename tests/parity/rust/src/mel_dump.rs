//! Debug bin: dump our pure-Rust Mel-fbank + CMVN features for a WAV.
//!
//! Output JSON shape mirrors `tests/parity/python/feat_dump.py` so the
//! mel-feature scorer can diff them element-wise.

use std::path::PathBuf;

use clap::Parser;
use firered_vad::Vad;
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
  #[arg(long)]
  wav: PathBuf,
  #[arg(long)]
  out: PathBuf,
}

#[derive(Serialize)]
struct Output {
  wav_path: String,
  n_frames: usize,
  n_mel_bins: usize,
  cmvn_features: Vec<f32>, // post-CMVN, flat row-major [T * 80]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();

  let mut reader = hound::WavReader::open(&args.wav)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 || spec.channels != 1 {
    return Err(format!(
      "expected 16 kHz mono, got {} Hz {}-channel",
      spec.sample_rate, spec.channels
    )
    .into());
  }
  let pcm: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => reader
      .samples::<i16>()
      .map(|s| s.map(|s| s as f32 / 32_768.0))
      .collect::<Result<_, _>>()?,
    hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
  };

  let mut vad = Vad::bundled()?;
  let features = vad._debug_extract_mel_features(&pcm);
  let n_mel_bins = 80usize;
  let n_frames = features.len() / n_mel_bins;

  let output = Output {
    wav_path: args.wav.canonicalize()?.to_string_lossy().into_owned(),
    n_frames,
    n_mel_bins,
    cmvn_features: features,
  };

  let f = std::fs::File::create(&args.out)?;
  serde_json::to_writer(f, &output)?;

  println!("rust: {} frames x {} bins -> {}", n_frames, n_mel_bins, args.out.display());
  Ok(())
}
