//! End-to-end black-box tests for the public `firered-vad` API.
//!
//! These tests deliberately avoid the crate-internal modules and instead
//! drive the engine the same way callers do: construct a `Vad`, push
//! PCM, drain segments. They run against the bundled ONNX model and
//! CMVN, so they're gated on the `bundled` feature.

#![cfg(feature = "bundled")]

use firered_vad::{SpeechSegment, Vad};

const SAMPLE_RATE_HZ: u32 = 16_000;

fn synthetic_speech_like(duration_secs: f32) -> Vec<f32> {
  use std::num::Wrapping;
  let n = (duration_secs * SAMPLE_RATE_HZ as f32) as usize;
  let mut buf = Vec::with_capacity(n);
  // Use a simple LCG for pseudo-random noise plus formant sinusoids.
  // This is more likely to trigger speech classification than pure sinusoids.
  let mut rng: Wrapping<u32> = Wrapping(12345);
  const A: u32 = 1103515245;
  const C: u32 = 12345;

  let formants = [200.0f32, 700.0, 1_500.0, 2_500.0, 3_500.0];
  for i in 0..n {
    rng = rng * Wrapping(A) + Wrapping(C);
    let noise = ((rng.0 / 65536 % 2048) as f32 / 1024.0 - 1.0) * 0.3;

    let t = i as f32 / SAMPLE_RATE_HZ as f32;
    let mut sample = noise;
    for f in formants {
      sample += 0.1 * (core::f32::consts::TAU * f * t).sin();
    }
    buf.push(sample.clamp(-1.0, 1.0));
  }
  buf
}

#[test]
fn bundled_constructs() {
  let _ = Vad::bundled().expect("bundled constructs");
}

#[test]
fn pure_silence_produces_no_segments() {
  let mut vad = Vad::bundled().expect("bundled");
  vad
    .push_samples(&vec![0.0; (SAMPLE_RATE_HZ * 2) as usize])
    .expect("push silence");
  let mut segments = 0usize;
  while vad.push_samples(&[]).expect("drain").is_some() {
    segments += 1;
  }
  if vad.finish().expect("finish").is_some() {
    segments += 1;
  }
  while vad.push_samples(&[]).expect("drain after finish").is_some() {
    segments += 1;
  }
  assert_eq!(segments, 0);
}

#[test]
fn synthetic_speech_then_silence_emits_at_least_one_segment() {
  let mut vad = Vad::bundled().expect("bundled");

  let mut pcm = synthetic_speech_like(1.5); // 1.5 s of "speech"
  pcm.extend(vec![0.0; SAMPLE_RATE_HZ as usize]); // 1 s of silence

  let mut segments = Vec::new();
  if let Some(s) = vad.push_samples(&pcm).expect("push samples") {
    segments.push(s);
  }
  while let Some(s) = vad.push_samples(&[]).expect("drain") {
    segments.push(s);
  }
  if let Some(s) = vad.finish().expect("finish") {
    segments.push(s);
  }
  while let Some(s) = vad.push_samples(&[]).expect("drain after finish") {
    segments.push(s);
  }

  assert!(!segments.is_empty(), "expected at least 1 closed segment");
  for s in &segments {
    assert!(s.start_sample() < s.end_sample());
    assert!(s.end_sample() <= pcm.len() as u64);
  }
}

#[test]
fn pushing_samples_in_arbitrary_chunks_yields_identical_event_stream() {
  let pcm = {
    let mut p = synthetic_speech_like(0.5);
    p.extend(vec![0.0; (SAMPLE_RATE_HZ as f32 * 0.5) as usize]);
    p
  };

  let collect = |chunk_size: usize| -> Vec<SpeechSegment> {
    let mut vad = Vad::bundled().expect("bundled");
    for chunk in pcm.chunks(chunk_size) {
      vad.push_samples(chunk).expect("push");
      // drain any segments produced by this chunk
      while vad.push_samples(&[]).expect("drain").is_some() {}
    }
    let mut out = Vec::new();
    while let Some(s) = vad.push_samples(&[]).expect("drain before finish") {
      out.push(s);
    }
    if let Some(s) = vad.finish().expect("finish") {
      out.push(s);
    }
    while let Some(s) = vad.push_samples(&[]).expect("drain after finish") {
      out.push(s);
    }
    out
  };

  let baseline = collect(SAMPLE_RATE_HZ as usize); // one big push
  for chunk in [1usize, 160, 320, 1024, 4_000, 16_000] {
    assert_eq!(
      collect(chunk),
      baseline,
      "segment stream must be deterministic across chunkings (chunk_size={chunk})"
    );
  }
}

#[test]
fn reset_returns_engine_to_freshly_constructed_state() {
  let mut vad = Vad::bundled().expect("bundled");
  vad.push_samples(&vec![0.0; 1_600]).expect("push 100ms");
  vad.reset();

  assert_eq!(vad.frame_count(), 0);
  assert_eq!(vad.pending_segments(), 0);
  assert_eq!(vad.pending_samples(), 0);
  assert!(!vad.is_active());
  assert!(!vad.is_finished());
}
