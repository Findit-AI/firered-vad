//! Reads a 16 kHz mono WAV (int16 or float32) and prints every closed
//! speech segment. Designed as the prototype for "feed Whisper" use cases.
//!
//! Usage:
//!     cargo run --example detect_file -- path/to/audio.wav

use firered_vad::Vad;

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
    let mut chunk: &[f32] = chunk;
    while let Some(segment) = vad.push_samples(chunk)? {
      println!(
        "{:.3}s..{:.3}s  ({:>6} samples)",
        segment.start().as_secs_f32(),
        segment.end().as_secs_f32(),
        segment.sample_count()
      );
      chunk = &[]; // drain remaining buffered segments before pushing the next chunk
    }
  }
  if let Some(segment) = vad.finish()? {
    println!(
      "{:.3}s..{:.3}s  ({:>6} samples)  (trailing)",
      segment.start().as_secs_f32(),
      segment.end().as_secs_f32(),
      segment.sample_count()
    );
  }
  Ok(())
}
