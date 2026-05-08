//! Run our `firered_vad::Vad` on a 16 kHz mono WAV and dump per-frame
//! results as JSON in the same shape as `tests/parity/python/run.py`.

use std::{path::PathBuf, time::Duration};

use clap::Parser;
use firered_vad::{Vad, VadOptions};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "firered-vad-parity")]
struct Args {
  #[arg(long)]
  wav: PathBuf,
  #[arg(long)]
  out: PathBuf,

  #[arg(long, default_value_t = 5)]
  smooth_window_size: u32,
  #[arg(long, default_value_t = 0.5)]
  speech_threshold: f32,
  #[arg(long, default_value_t = 5)]
  pad_start_frame: u32,
  #[arg(long, default_value_t = 8)]
  min_speech_frame: u32,
  #[arg(long, default_value_t = 2000)]
  max_speech_frame: u32,
  #[arg(long, default_value_t = 20)]
  min_silence_frame: u32,
}

#[derive(Serialize)]
struct Frame {
  frame_index: u64,
  raw_prob: f32,
  smoothed_prob: f32,
  is_speech: bool,
  is_speech_start: bool,
  is_speech_end: bool,
  speech_start_frame: Option<u64>,
  speech_end_frame: Option<u64>,
}

#[derive(Serialize)]
struct Config {
  smooth_window_size: u32,
  speech_threshold: f32,
  pad_start_frame: u32,
  min_speech_frame: u32,
  max_speech_frame: u32,
  min_silence_frame: u32,
}

#[derive(Serialize)]
struct Output {
  wav_path: String,
  sample_rate: u32,
  n_samples: u64,
  duration_s: f64,
  n_frames: u64,
  config: Config,
  frames: Vec<Frame>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();

  // Load WAV (16 kHz mono int16 → f32 in [-1, 1]).
  let mut reader = hound::WavReader::open(&args.wav)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    return Err(format!("expected 16 kHz, got {} Hz", spec.sample_rate).into());
  }
  if spec.channels != 1 {
    return Err(format!("expected mono, got {} channels", spec.channels).into());
  }
  let pcm: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => reader
      .samples::<i16>()
      .map(|s| s.map(|s| s as f32 / 32_768.0))
      .collect::<Result<_, _>>()?,
    hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
  };

  // Build the same options as the Python runner.
  let frame_to_dur = |frames: u32| Duration::from_millis(u64::from(frames) * 10);
  let options = VadOptions::new()
    .with_smooth_window_size(args.smooth_window_size)
    .with_speech_threshold(args.speech_threshold)
    .with_pad_start(frame_to_dur(args.pad_start_frame))
    .with_min_speech_duration(frame_to_dur(args.min_speech_frame))
    .with_max_speech_duration(frame_to_dur(args.max_speech_frame))
    .with_min_silence_duration(frame_to_dur(args.min_silence_frame));

  let mut vad = Vad::bundled_with(options)?;

  // Process the entire WAV in one push to mirror Python's detect_chunk.
  // We discard segments returned via push_samples (we only care about
  // per-frame data for parity comparison).
  let _ = vad.push_samples(&pcm)?;

  let mut frames = Vec::with_capacity(vad.recent_frames().len());
  for f in vad.recent_frames() {
    frames.push(Frame {
      frame_index: f.frame_index(),
      raw_prob: f.raw_prob(),
      smoothed_prob: f.smoothed_prob(),
      is_speech: f.is_speech(),
      is_speech_start: f.is_speech_start(),
      is_speech_end: f.is_speech_end(),
      speech_start_frame: f.speech_start_frame(),
      speech_end_frame: f.speech_end_frame(),
    });
  }

  let n_frames = frames.len() as u64;
  let starts = frames.iter().filter(|f| f.is_speech_start).count();
  let ends = frames.iter().filter(|f| f.is_speech_end).count();

  let output = Output {
    wav_path: args.wav.canonicalize()?.to_string_lossy().into_owned(),
    sample_rate: spec.sample_rate,
    n_samples: pcm.len() as u64,
    duration_s: (pcm.len() as f64 / spec.sample_rate as f64 * 1000.0).round() / 1000.0,
    n_frames,
    config: Config {
      smooth_window_size: args.smooth_window_size,
      speech_threshold: args.speech_threshold,
      pad_start_frame: args.pad_start_frame,
      min_speech_frame: args.min_speech_frame,
      max_speech_frame: args.max_speech_frame,
      min_silence_frame: args.min_silence_frame,
    },
    frames,
  };

  let f = std::fs::File::create(&args.out)?;
  serde_json::to_writer(f, &output)?;

  println!(
    "rust:   {} -> {}  ({} frames, {} starts, {} ends)",
    args.wav.display(),
    args.out.display(),
    n_frames,
    starts,
    ends
  );
  Ok(())
}
