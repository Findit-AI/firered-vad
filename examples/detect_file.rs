//! Reads a 16 kHz mono WAV (int16 or float32) and prints every closed
//! speech segment. Designed as the prototype for "feed Whisper" use cases.
//!
//! Usage:
//!     cargo run --example detect_file -- path/to/audio.wav

use firered_vad::{Vad, VadEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = std::env::args()
    .nth(1)
    .ok_or("usage: detect_file <path.wav>")?;
  let mut reader = hound::WavReader::open(&path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    return Err(
      format!(
        "expected 16 kHz mono WAV; got {} Hz {}-channel",
        spec.sample_rate, spec.channels
      )
      .into(),
    );
  }
  if spec.channels != 1 {
    return Err("expected mono".into());
  }

  let pcm: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => reader
      .samples::<i16>()
      .map(|s| s.map(|s| s as f32 / 32_768.0))
      .collect::<Result<_, _>>()?,
    hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
  };

  let mut vad = Vad::bundled()?;
  for chunk in pcm.chunks(16_000) {
    vad.push_samples(chunk)?;
    while let Some(event) = vad.poll_event() {
      if let VadEvent::SegmentClosed(s) = event {
        println!(
          "{:.3}s..{:.3}s  ({:>6} samples)",
          s.start().as_secs_f32(),
          s.end().as_secs_f32(),
          s.sample_count()
        );
      }
    }
  }
  vad.finish()?;
  while let Some(event) = vad.poll_event() {
    if let VadEvent::SegmentClosed(s) = event {
      println!(
        "{:.3}s..{:.3}s  ({:>6} samples)  (trailing)",
        s.start().as_secs_f32(),
        s.end().as_secs_f32(),
        s.sample_count()
      );
    }
  }
  Ok(())
}
