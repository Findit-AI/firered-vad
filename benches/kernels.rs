//! Per-kernel microbench: compares scalar vs SIMD for each Mel-fbank
//! inner-loop kernel. Lets us read the SIMD speedup directly from
//! adjacent lines in the criterion report instead of running
//! `cargo bench` twice with different `RUSTFLAGS`.
//!
//! Build / run:
//!     cargo bench --bench kernels --features _bench-internals
//!
//! On aarch64 every `BenchmarkId::new("scalar", N)` line gets a
//! matching `BenchmarkId::new("neon", N)` line. On x86_64, additional
//! `sse4.1`, `avx2`, and `avx512f` lines appear for whichever
//! features the host CPU advertises.

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use firered_vad::__bench_internals::{
  FFT_BINS, FRAME_LENGTH_SAMPLES, NUM_MEL_BINS, scalar,
};
#[cfg(target_arch = "aarch64")]
use firered_vad::__bench_internals::neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use firered_vad::__bench_internals::{x86_avx2, x86_avx512, x86_sse41};

/// Deterministic LCG fill so we don't measure cache-friendly uniform data.
fn fill_pseudo_random(buf: &mut [f32], seed: u32) {
  let mut state = seed;
  for v in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *v = (state as i32 as f32) / i32::MAX as f32;
  }
}

// ── pcm_scale_extend ────────────────────────────────────────────────
// Scalar-only — the dispatcher routes straight to scalar (no NEON
// path exposed). Kept here so we keep an eye on the kernel's
// per-element cost as a regression guard. See
// `src/features/arch/neon.rs` for the rationale.
fn bench_pcm_scale_extend(c: &mut Criterion) {
  const SIZES: &[usize] = &[160, 1_600, 16_000];

  let mut group = c.benchmark_group("pcm_scale_extend");
  for &n in SIZES {
    let mut src = vec![0.0f32; n];
    let mut dst = vec![0.0f32; n];
    fill_pseudo_random(&mut src, 0x1111);

    group.throughput(Throughput::Elements(n as u64));

    group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_| {
      b.iter(|| scalar::pcm_scale_extend(black_box(&src), black_box(&mut dst)));
    });
  }
  group.finish();
}

// ── dc_remove ───────────────────────────────────────────────────────
fn bench_dc_remove(c: &mut Criterion) {
  let mut group = c.benchmark_group("dc_remove");
  let mut window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
  let mut out = vec![0.0f32; FRAME_LENGTH_SAMPLES];
  fill_pseudo_random(&mut window, 0x2222);

  group.throughput(Throughput::Elements(FRAME_LENGTH_SAMPLES as u64));

  group.bench_function(BenchmarkId::new("scalar", FRAME_LENGTH_SAMPLES), |b| {
    b.iter(|| scalar::dc_remove(black_box(&window), black_box(&mut out)));
  });

  #[cfg(target_arch = "aarch64")]
  group.bench_function(BenchmarkId::new("neon", FRAME_LENGTH_SAMPLES), |b| {
    b.iter(|| unsafe { neon::dc_remove(black_box(&window), black_box(&mut out)) });
  });
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::arch::is_x86_feature_detected!("sse4.1") {
      group.bench_function(BenchmarkId::new("sse4.1", FRAME_LENGTH_SAMPLES), |b| {
        b.iter(|| unsafe { x86_sse41::dc_remove(black_box(&window), black_box(&mut out)) });
      });
    }
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
      group.bench_function(BenchmarkId::new("avx2", FRAME_LENGTH_SAMPLES), |b| {
        b.iter(|| unsafe { x86_avx2::dc_remove(black_box(&window), black_box(&mut out)) });
      });
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
      group.bench_function(BenchmarkId::new("avx512f", FRAME_LENGTH_SAMPLES), |b| {
        b.iter(|| unsafe { x86_avx512::dc_remove(black_box(&window), black_box(&mut out)) });
      });
    }
  }
  group.finish();
}

// ── pre_emphasis ────────────────────────────────────────────────────
// Sequential by definition (data dependency on `x[i-1]`); no SIMD path.
fn bench_pre_emphasis(c: &mut Criterion) {
  let mut group = c.benchmark_group("pre_emphasis");
  let baseline = {
    let mut buf = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    fill_pseudo_random(&mut buf, 0x3333);
    buf
  };

  group.throughput(Throughput::Elements(FRAME_LENGTH_SAMPLES as u64));

  group.bench_function(BenchmarkId::new("scalar", FRAME_LENGTH_SAMPLES), |b| {
    b.iter_batched(
      || baseline.clone(),
      |mut samples| scalar::pre_emphasis(black_box(&mut samples)),
      criterion::BatchSize::SmallInput,
    );
  });
  group.finish();
}

// ── window_apply ────────────────────────────────────────────────────
fn bench_window_apply(c: &mut Criterion) {
  let mut group = c.benchmark_group("window_apply");
  let baseline = {
    let mut buf = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    fill_pseudo_random(&mut buf, 0x4444);
    buf
  };
  let mut window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
  fill_pseudo_random(&mut window, 0x5555);

  group.throughput(Throughput::Elements(FRAME_LENGTH_SAMPLES as u64));

  group.bench_function(BenchmarkId::new("scalar", FRAME_LENGTH_SAMPLES), |b| {
    b.iter_batched(
      || baseline.clone(),
      |mut samples| scalar::window_apply(black_box(&mut samples), black_box(&window)),
      criterion::BatchSize::SmallInput,
    );
  });

  #[cfg(target_arch = "aarch64")]
  group.bench_function(BenchmarkId::new("neon", FRAME_LENGTH_SAMPLES), |b| {
    b.iter_batched(
      || baseline.clone(),
      |mut samples| unsafe {
        neon::window_apply(black_box(&mut samples), black_box(&window))
      },
      criterion::BatchSize::SmallInput,
    );
  });
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::arch::is_x86_feature_detected!("sse4.1") {
      group.bench_function(BenchmarkId::new("sse4.1", FRAME_LENGTH_SAMPLES), |b| {
        b.iter_batched(
          || baseline.clone(),
          |mut samples| unsafe {
            x86_sse41::window_apply(black_box(&mut samples), black_box(&window))
          },
          criterion::BatchSize::SmallInput,
        );
      });
    }
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
      group.bench_function(BenchmarkId::new("avx2", FRAME_LENGTH_SAMPLES), |b| {
        b.iter_batched(
          || baseline.clone(),
          |mut samples| unsafe {
            x86_avx2::window_apply(black_box(&mut samples), black_box(&window))
          },
          criterion::BatchSize::SmallInput,
        );
      });
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
      group.bench_function(BenchmarkId::new("avx512f", FRAME_LENGTH_SAMPLES), |b| {
        b.iter_batched(
          || baseline.clone(),
          |mut samples| unsafe {
            x86_avx512::window_apply(black_box(&mut samples), black_box(&window))
          },
          criterion::BatchSize::SmallInput,
        );
      });
    }
  }
  group.finish();
}

// ── power_spectrum ──────────────────────────────────────────────────
fn bench_power_spectrum(c: &mut Criterion) {
  let mut group = c.benchmark_group("power_spectrum");
  let mut complex = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0); FFT_BINS];
  for (i, c) in complex.iter_mut().enumerate() {
    let v = (i as f32) * 0.0123;
    c.re = v.sin();
    c.im = v.cos();
  }
  let mut out = vec![0.0f32; FFT_BINS];

  group.throughput(Throughput::Elements(FFT_BINS as u64));

  group.bench_function(BenchmarkId::new("scalar", FFT_BINS), |b| {
    b.iter(|| scalar::power_spectrum(black_box(&complex), black_box(&mut out)));
  });

  #[cfg(target_arch = "aarch64")]
  group.bench_function(BenchmarkId::new("neon", FFT_BINS), |b| {
    b.iter(|| unsafe { neon::power_spectrum(black_box(&complex), black_box(&mut out)) });
  });
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::arch::is_x86_feature_detected!("sse4.1") {
      group.bench_function(BenchmarkId::new("sse4.1", FFT_BINS), |b| {
        b.iter(|| unsafe { x86_sse41::power_spectrum(black_box(&complex), black_box(&mut out)) });
      });
    }
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
      group.bench_function(BenchmarkId::new("avx2", FFT_BINS), |b| {
        b.iter(|| unsafe { x86_avx2::power_spectrum(black_box(&complex), black_box(&mut out)) });
      });
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
      group.bench_function(BenchmarkId::new("avx512f", FFT_BINS), |b| {
        b.iter(|| unsafe {
          x86_avx512::power_spectrum(black_box(&complex), black_box(&mut out))
        });
      });
    }
  }
  group.finish();
}

// ── cmvn_apply ──────────────────────────────────────────────────────
fn bench_cmvn_apply(c: &mut Criterion) {
  let mut group = c.benchmark_group("cmvn_apply");
  let baseline = {
    let mut buf = vec![0.0f32; NUM_MEL_BINS];
    fill_pseudo_random(&mut buf, 0x6666);
    buf
  };
  let mut means = vec![0.0f32; NUM_MEL_BINS];
  let mut istd = vec![0.0f32; NUM_MEL_BINS];
  fill_pseudo_random(&mut means, 0x7777);
  fill_pseudo_random(&mut istd, 0x8888);

  group.throughput(Throughput::Elements(NUM_MEL_BINS as u64));

  group.bench_function(BenchmarkId::new("scalar", NUM_MEL_BINS), |b| {
    b.iter_batched(
      || baseline.clone(),
      |mut feature| scalar::cmvn_apply(black_box(&mut feature), black_box(&means), black_box(&istd)),
      criterion::BatchSize::SmallInput,
    );
  });

  #[cfg(target_arch = "aarch64")]
  group.bench_function(BenchmarkId::new("neon", NUM_MEL_BINS), |b| {
    b.iter_batched(
      || baseline.clone(),
      |mut feature| unsafe {
        neon::cmvn_apply(black_box(&mut feature), black_box(&means), black_box(&istd))
      },
      criterion::BatchSize::SmallInput,
    );
  });
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::arch::is_x86_feature_detected!("sse4.1") {
      group.bench_function(BenchmarkId::new("sse4.1", NUM_MEL_BINS), |b| {
        b.iter_batched(
          || baseline.clone(),
          |mut feature| unsafe {
            x86_sse41::cmvn_apply(black_box(&mut feature), black_box(&means), black_box(&istd))
          },
          criterion::BatchSize::SmallInput,
        );
      });
    }
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
      group.bench_function(BenchmarkId::new("avx2", NUM_MEL_BINS), |b| {
        b.iter_batched(
          || baseline.clone(),
          |mut feature| unsafe {
            x86_avx2::cmvn_apply(black_box(&mut feature), black_box(&means), black_box(&istd))
          },
          criterion::BatchSize::SmallInput,
        );
      });
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
      group.bench_function(BenchmarkId::new("avx512f", NUM_MEL_BINS), |b| {
        b.iter_batched(
          || baseline.clone(),
          |mut feature| unsafe {
            x86_avx512::cmvn_apply(black_box(&mut feature), black_box(&means), black_box(&istd))
          },
          criterion::BatchSize::SmallInput,
        );
      });
    }
  }
  group.finish();
}

// ── mel_dot_log ─────────────────────────────────────────────────────
// Mel filters in our config typically have 5–25 weights; pick a few
// representative widths.
fn bench_mel_dot_log(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[5, 12, 25];

  let mut group = c.benchmark_group("mel_dot_log");

  for &w in WIDTHS {
    let mut power_slice = vec![0.0f32; w];
    let mut weights = vec![0.0f32; w];
    fill_pseudo_random(&mut power_slice, 0x9999);
    fill_pseudo_random(&mut weights, 0xAAAA);
    // Make weights non-negative so mel_dot_log returns finite values.
    for v in &mut weights {
      *v = v.abs();
    }

    group.throughput(Throughput::Elements(w as u64));

    group.bench_function(BenchmarkId::new("scalar", w), |b| {
      b.iter(|| scalar::mel_dot_log(black_box(&power_slice), black_box(&weights)));
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function(BenchmarkId::new("neon", w), |b| {
      b.iter(|| unsafe { neon::mel_dot_log(black_box(&power_slice), black_box(&weights)) });
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      if std::arch::is_x86_feature_detected!("sse4.1") {
        group.bench_function(BenchmarkId::new("sse4.1", w), |b| {
          b.iter(|| unsafe {
            x86_sse41::mel_dot_log(black_box(&power_slice), black_box(&weights))
          });
        });
      }
      if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
      {
        group.bench_function(BenchmarkId::new("avx2", w), |b| {
          b.iter(|| unsafe {
            x86_avx2::mel_dot_log(black_box(&power_slice), black_box(&weights))
          });
        });
      }
      if std::arch::is_x86_feature_detected!("avx512f") {
        group.bench_function(BenchmarkId::new("avx512f", w), |b| {
          b.iter(|| unsafe {
            x86_avx512::mel_dot_log(black_box(&power_slice), black_box(&weights))
          });
        });
      }
    }
  }
  group.finish();
}

criterion_group!(
  benches,
  bench_pcm_scale_extend,
  bench_dc_remove,
  bench_pre_emphasis,
  bench_window_apply,
  bench_power_spectrum,
  bench_cmvn_apply,
  bench_mel_dot_log,
);
criterion_main!(benches);
