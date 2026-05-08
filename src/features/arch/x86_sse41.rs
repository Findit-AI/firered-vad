//! x86_64 SSE4.1 kernels for the Mel-fbank inner loops.
//!
//! Selected by the dispatcher in [`super::super::dispatch`] when
//! AVX-512 and AVX2 are unavailable (or disabled via the
//! `firered_vad_disable_*` cfg flags) but SSE4.1 is. The kernels
//! carry `#[target_feature(enable = "sse4.1")]` so their intrinsics
//! execute in an explicitly SSE4.1-enabled context.
//!
//! # Pipeline (4 f32 lanes per instruction)
//!
//! Same vector width as NEON (`__m128` carries 4 f32). No FMA — uses
//! `_mm_mul_ps` + `_mm_add_ps` pairs. For float-32 ops at this width,
//! SSE4.1 is functionally equivalent to NEON; we follow the same
//! tail-handler pattern for any leftover elements not covered by full
//! 4-lane blocks.
//!
//! # Numerical contract
//!
//! Bit-exact reordering of float adds means SSE output may differ
//! from the scalar reference by one ULP at most when associativity
//! reorders fold differently. The parity harness verifies the full
//! pipeline against upstream Python's `kaldi-native-fbank` — every
//! arch backend must clear `--prob-tol 5e-3`.
//!
//! # Verification status (2026-05-08)
//!
//! Cross-compile-checked via `cargo check --target x86_64-apple-darwin`.
//! Bit-exact parity is **unverified** until the parity harness runs on
//! x86_64 hardware (CI gap tracked at the project level). Treat as
//! best-effort until that runner exists.

#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::features::{FRAME_LENGTH_SAMPLES, NUM_MEL_BINS, scalar};

// `pcm_scale_extend` deliberately has no SSE4.1 path — same rationale as
// the NEON omission in `super::neon`: the kernel is one FP mul per
// element which LLVM auto-vectorizes inline into the caller, and the
// `#[target_feature]` annotation on a hand-rolled wrapper would add a
// function-call boundary that wipes out any SIMD gain.

/// DC removal: compute mean, store `window[i] - mean` into `out`.
///
/// # Safety
///
/// SSE4.1 must be available. `window.len() == out.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(out.len(), FRAME_LENGTH_SAMPLES);
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut acc = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= n {
      let v = _mm_loadu_ps(window.as_ptr().add(i));
      acc = _mm_add_ps(acc, v);
      i += 4;
    }
    // Horizontal reduce 4 lanes → scalar via two `_mm_hadd_ps` from SSE3.
    let acc = _mm_hadd_ps(acc, acc);
    let acc = _mm_hadd_ps(acc, acc);
    let mut sum = _mm_cvtss_f32(acc);
    for k in i..n {
      sum += *window.get_unchecked(k);
    }
    let mean = sum / FRAME_LENGTH_SAMPLES as f32;
    let mean_v = _mm_set1_ps(mean);

    let mut i = 0usize;
    while i + 4 <= n {
      let v = _mm_loadu_ps(window.as_ptr().add(i));
      let r = _mm_sub_ps(v, mean_v);
      _mm_storeu_ps(out.as_mut_ptr().add(i), r);
      i += 4;
    }
    for k in i..n {
      *out.get_unchecked_mut(k) = *window.get_unchecked(k) - mean;
    }
  }
}

/// Element-wise multiply `samples[i] *= window[i]`. 4 lanes / iter.
///
/// # Safety
///
/// SSE4.1 must be available. `samples.len() == window.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
  debug_assert_eq!(samples.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut i = 0usize;
    while i + 4 <= n {
      let s = _mm_loadu_ps(samples.as_ptr().add(i));
      let w = _mm_loadu_ps(window.as_ptr().add(i));
      _mm_storeu_ps(samples.as_mut_ptr().add(i), _mm_mul_ps(s, w));
      i += 4;
    }
    // FRAME_LENGTH_SAMPLES = 400 is a multiple of 4 — no tail.
    debug_assert_eq!(i, n);
  }
}

/// Power spectrum: `out[i] = re*re + im*im` over the non-redundant
/// FFT half. The Complex<f32> source has interleaved layout
/// (re0, im0, re1, im1, …); we load two adjacent 128-bit chunks and
/// shuffle to de-interleave RE/IM into separate vectors.
///
/// # Safety
///
/// SSE4.1 must be available. `complex.len() == out.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn power_spectrum(
  complex: &[rustfft::num_complex::Complex<f32>],
  out: &mut [f32],
) {
  debug_assert_eq!(complex.len(), out.len());
  let n = complex.len();
  unsafe {
    let mut i = 0usize;
    let base = complex.as_ptr() as *const f32;
    while i + 4 <= n {
      // Load 8 floats (4 complex pairs) as two 128-bit chunks.
      let a = _mm_loadu_ps(base.add(i * 2));     // re0 im0 re1 im1
      let b = _mm_loadu_ps(base.add(i * 2 + 4)); // re2 im2 re3 im3
      // Even-indexed lanes → re; odd-indexed lanes → im.
      // _mm_shuffle_ps mask layout: (b3 b2 a3 a2) for low half-pair-pick.
      // 0b10_00_10_00: pick a[0], a[2], b[0], b[2] → re0 re1 re2 re3
      // 0b11_01_11_01: pick a[1], a[3], b[1], b[3] → im0 im1 im2 im3
      let re = _mm_shuffle_ps::<0b10_00_10_00>(a, b);
      let im = _mm_shuffle_ps::<0b11_01_11_01>(a, b);
      let p = _mm_add_ps(_mm_mul_ps(re, re), _mm_mul_ps(im, im));
      _mm_storeu_ps(out.as_mut_ptr().add(i), p);
      i += 4;
    }
    if i < n {
      scalar::power_spectrum(&complex[i..], &mut out[i..]);
    }
  }
}

/// CMVN: `feature[d] = (feature[d] - means[d]) * istd[d]` over
/// `NUM_MEL_BINS` (= 80, multiple of 4 → no tail).
///
/// # Safety
///
/// SSE4.1 must be available. All slices have length `NUM_MEL_BINS`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
  debug_assert_eq!(feature.len(), NUM_MEL_BINS);
  debug_assert_eq!(means.len(), NUM_MEL_BINS);
  debug_assert_eq!(istd.len(), NUM_MEL_BINS);
  let n = NUM_MEL_BINS;
  unsafe {
    let mut i = 0usize;
    while i + 4 <= n {
      let f = _mm_loadu_ps(feature.as_ptr().add(i));
      let m = _mm_loadu_ps(means.as_ptr().add(i));
      let s = _mm_loadu_ps(istd.as_ptr().add(i));
      let r = _mm_mul_ps(_mm_sub_ps(f, m), s);
      _mm_storeu_ps(feature.as_mut_ptr().add(i), r);
      i += 4;
    }
    debug_assert_eq!(i, n);
  }
}

/// Sparse dot product + `max(LOG_FLOOR).ln()` over one Mel filter's
/// weights.
///
/// # Safety
///
/// SSE4.1 must be available. `power_slice.len() == weights.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
  debug_assert_eq!(power_slice.len(), weights.len());
  let n = power_slice.len();
  let mut energy = 0.0f32;
  unsafe {
    let mut i = 0usize;
    let mut acc = _mm_setzero_ps();
    while i + 4 <= n {
      let p = _mm_loadu_ps(power_slice.as_ptr().add(i));
      let w = _mm_loadu_ps(weights.as_ptr().add(i));
      acc = _mm_add_ps(acc, _mm_mul_ps(p, w));
      i += 4;
    }
    // Horizontal reduce 4 lanes → scalar.
    let acc = _mm_hadd_ps(acc, acc);
    let acc = _mm_hadd_ps(acc, acc);
    energy += _mm_cvtss_f32(acc);
    for k in i..n {
      energy += *power_slice.get_unchecked(k) * *weights.get_unchecked(k);
    }
  }
  energy.max(crate::features::LOG_FLOOR).ln()
}
