//! Throughput benchmark: time `Vad::push_samples` on a real WAV under two
//! modes — single-shot (whole audio in one push) and streaming
//! (100 ms chunks). Useful for measuring perf-tuning impact.
//!
//! Usage:
//!     cargo run --release --example benchmark -- <path/to/16k-mono.wav>

use std::time::Instant;

use firered_vad::Vad;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = std::env::args()
    .nth(1)
    .ok_or("usage: benchmark <path-to-16k-mono.wav>")?;
  let mut reader = hound::WavReader::open(&path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 || spec.channels != 1 {
    return Err(format!(
      "expected 16 kHz mono; got {} Hz {}-channel",
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
  let duration_s = pcm.len() as f64 / 16_000.0;
  println!("file: {} ({:.2} s, {} samples)", path, duration_s, pcm.len());
  println!();

  // ── Mode A: single-shot ────────────────────────────────────────
  {
    let mut vad = Vad::bundled()?;
    let start = Instant::now();
    let _ = vad.push_samples(&pcm)?;
    let _ = vad.finish()?;
    let elapsed = start.elapsed();
    let segments_count = vad.recent_frames().iter().filter(|f| f.is_speech_end()).count();
    println!(
      "single-shot push: {:.3} s wall  ({:.2}× realtime, {} segments)",
      elapsed.as_secs_f64(),
      duration_s / elapsed.as_secs_f64(),
      segments_count,
    );
  }

  // ── Mode B: 100ms streaming chunks ────────────────────────────
  {
    let mut vad = Vad::bundled()?;
    let chunk = 1_600usize; // 100 ms at 16 kHz
    let start = Instant::now();
    for chunk_pcm in pcm.chunks(chunk) {
      let mut chunk_pcm: &[f32] = chunk_pcm;
      while let Some(_segment) = vad.push_samples(chunk_pcm)? {
        chunk_pcm = &[];
      }
    }
    let _ = vad.finish()?;
    let elapsed = start.elapsed();
    println!(
      "100ms streaming: {:.3} s wall  ({:.2}× realtime)",
      elapsed.as_secs_f64(),
      duration_s / elapsed.as_secs_f64(),
    );
  }

  // ── Mode C: 10ms streaming chunks (worst-case overhead) ───────
  {
    let mut vad = Vad::bundled()?;
    let chunk = 160usize; // 10 ms at 16 kHz
    let start = Instant::now();
    for chunk_pcm in pcm.chunks(chunk) {
      let mut chunk_pcm: &[f32] = chunk_pcm;
      while let Some(_segment) = vad.push_samples(chunk_pcm)? {
        chunk_pcm = &[];
      }
    }
    let _ = vad.finish()?;
    let elapsed = start.elapsed();
    println!(
      "10ms streaming:  {:.3} s wall  ({:.2}× realtime)",
      elapsed.as_secs_f64(),
      duration_s / elapsed.as_secs_f64(),
    );
  }

  // ── Mode D: feature extraction only (no ONNX) ────────────────
  // Times only the Mel-fbank + CMVN pipeline. Lets us see the
  // SIMD-vs-scalar delta on the feature stage in isolation, since
  // end-to-end throughput is dominated by ONNX inference and would
  // otherwise mask any feature-extraction speedup.
  #[cfg(feature = "_debug-features")]
  {
    let mut vad = Vad::bundled()?;
    let start = Instant::now();
    let _ = vad._debug_extract_mel_features(&pcm);
    let elapsed = start.elapsed();
    let n_frames = (pcm.len() - 240) / 160;
    println!(
      "feat extract only: {:.3} s wall  ({:.2}× realtime, {} frames, {:.1} ns/frame)",
      elapsed.as_secs_f64(),
      duration_s / elapsed.as_secs_f64(),
      n_frames,
      elapsed.as_nanos() as f64 / n_frames as f64,
    );
  }

  Ok(())
}
