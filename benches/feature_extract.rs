//! Feature-extraction-only benchmark: times the whole Mel-fbank +
//! CMVN pipeline (DC remove → pre-emphasis → Povey window → FFT →
//! power spectrum → mel-bank dot products → CMVN), excluding ONNX
//! inference. This is where SIMD speedups land before being masked
//! by the model's wall-time.
//!
//! Build / run:
//!     cargo bench --bench feature_extract --features _debug-features
//!
//! Compare scalar vs NEON by running twice:
//!     cargo bench --bench feature_extract --features _debug-features
//!     RUSTFLAGS='--cfg firered_vad_force_scalar' \
//!       cargo bench --bench feature_extract --features _debug-features
//!
//! Criterion's `--save-baseline` / `--baseline` flags pair with this
//! to produce a side-by-side regression table.

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use firered_vad::Vad;

/// Generate `seconds` of synthetic audio at 16 kHz that exercises the
/// whole mel-bank (broad-spectrum sum-of-formants + low-amplitude
/// noise so every bin has non-trivial energy).
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
  // 100ms / 1s / 10s windows. 100 ms = 10 frames, 1 s = 100 frames,
  // 10 s = 1000 frames. The criterion report's per-iteration time
  // divided by the frame count gives a per-frame extraction cost
  // independent of the FFT plan + window setup amortization.
  const SECONDS: &[f32] = &[0.1, 1.0, 10.0];

  let mut group = c.benchmark_group("mel_fbank_extract");

  for &secs in SECONDS {
    let pcm = synth_pcm(secs);
    let n_frames = ((pcm.len().saturating_sub(240)) / 160) as u64;

    // Throughput in frames-per-iter so the report shows realtime
    // multiples directly (1 frame = 10 ms of audio).
    group.throughput(Throughput::Elements(n_frames));

    group.bench_with_input(
      BenchmarkId::from_parameter(format!("{secs:.1}s")),
      &pcm,
      |b, pcm| {
        let mut vad = Vad::bundled().expect("bundled");
        b.iter(|| {
          let _ = black_box(vad._debug_extract_mel_features(black_box(pcm)));
        });
      },
    );
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
