//! End-to-end `Vad::push_samples` benchmark, including ONNX inference.
//! Mirrors the realistic deployment shape (streaming chunks fed to the
//! engine + segments emitted) rather than the contrived single-shot
//! "feed every sample at once" mode.
//!
//! Build / run:
//!     cargo bench --bench push_samples
//!
//! Three chunkings are exercised:
//!   - 10 ms (160 samples) — worst-case ONNX call rate; one inference per frame
//!   - 100 ms (1 600 samples) — typical real-time cadence
//!   - 1 s (16 000 samples) — batched-inference shape, good MACs amortization
//!
//! For each chunking we report frames/sec throughput in the criterion
//! report. The numbers are dominated by ONNX inference; feature
//! extraction is < 5% of wall time at this granularity.

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use firered_vad::Vad;

fn synth_pcm(seconds: f32) -> Vec<f32> {
  let n = (seconds * 16_000.0) as usize;
  let mut buf = Vec::with_capacity(n);
  let formants = [200.0f32, 700.0, 1_500.0, 2_500.0, 3_500.0];
  let mut state: u32 = 0xC0FFEEu32;
  for i in 0..n {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let noise = ((state >> 8) & 0x7FF) as i32 - 1024;
    let t = i as f32 / 16_000.0;
    let mut s = 0.0f32;
    for f in formants {
      s += (core::f32::consts::TAU * f * t).sin();
    }
    buf.push(0.05 * s + (noise as f32) * 1e-5);
  }
  buf
}

fn bench(c: &mut Criterion) {
  // 10 seconds is a reasonable bench window — long enough to amortize
  // setup (FFT plan, ONNX session warm-up) but short enough that one
  // criterion sample completes in ~0.1–1 s.
  let pcm = synth_pcm(10.0);
  let n_frames = ((pcm.len().saturating_sub(240)) / 160) as u64;

  let mut group = c.benchmark_group("vad_push_samples");
  group.throughput(Throughput::Elements(n_frames));

  // (chunk_samples, label).
  const CHUNKS: &[(usize, &str)] = &[(160, "10ms"), (1_600, "100ms"), (16_000, "1s")];

  for &(chunk, label) in CHUNKS {
    group.bench_with_input(BenchmarkId::from_parameter(label), &chunk, |b, &chunk| {
      let mut vad = Vad::bundled().expect("bundled");
      b.iter(|| {
        // Reset state per iteration so each measurement starts from a
        // clean cache. Without this the smoothing window / state
        // machine would carry over and bias later chunks.
        vad.reset();
        for piece in pcm.chunks(chunk) {
          let mut piece: &[f32] = piece;
          while let Some(_seg) = black_box(vad.push_samples(black_box(piece))).expect("push") {
            piece = &[];
          }
        }
        let _ = vad.finish().expect("finish");
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
