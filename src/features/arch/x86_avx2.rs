//! x86_64 AVX2 + FMA kernels for the Mel-fbank inner loops.
//!
//! Selected by the dispatcher in [`super::super::dispatch`] when
//! AVX-512 is unavailable but AVX2 + FMA are. The kernels carry
//! `#[target_feature(enable = "avx2,fma")]` — AVX2 brings the 256-bit
//! integer/permute ops we need for the power-spectrum de-interleave,
//! and FMA gives the fused multiply-add used by `mel_dot_log` and
//! the bulk of the AVX-era SIMD speedups.
//!
//! # Pipeline (8 f32 lanes per instruction)
//!
//! Vector width doubles vs SSE4.1: `__m256` carries 8 f32. The
//! kernel structure is otherwise identical — same scalar-tail
//! handler for any leftover elements not covered by full 8-lane
//! blocks.
//!
//! # Numerical contract
//!
//! Bit-exact reordering of float adds means AVX2 output may differ
//! from the scalar reference by one ULP at most when associativity
//! reorders fold differently. Additionally, FMA in `mel_dot_log`
//! retains intermediate precision differently from sequential
//! `mul + add`; the upstream parity harness clears `--prob-tol 5e-3`
//! and absorbs that drift.
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

// `pcm_scale_extend` deliberately has no AVX2 path — same rationale as
// the NEON / SSE4.1 omissions: a single-FMul-per-element kernel that
// LLVM auto-vectorizes inline into the caller, with no headroom left
// for a `#[target_feature]`-gated function call to pay for itself.

/// DC removal: compute mean, store `window[i] - mean` into `out`.
///
/// # Safety
///
/// AVX2 must be available. `window.len() == out.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(out.len(), FRAME_LENGTH_SAMPLES);
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
      let v = _mm256_loadu_ps(window.as_ptr().add(i));
      acc = _mm256_add_ps(acc, v);
      i += 8;
    }
    // Horizontal reduce 8 lanes → scalar: fold 256→128 then hadd twice.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let mut sum = _mm_cvtss_f32(s);
    for k in i..n {
      sum += *window.get_unchecked(k);
    }
    let mean = sum / FRAME_LENGTH_SAMPLES as f32;
    let mean_v = _mm256_set1_ps(mean);

    let mut i = 0usize;
    while i + 8 <= n {
      let v = _mm256_loadu_ps(window.as_ptr().add(i));
      _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_sub_ps(v, mean_v));
      i += 8;
    }
    // FRAME_LENGTH_SAMPLES = 400 is a multiple of 8 — no tail.
    debug_assert_eq!(i, n);
  }
}

/// Element-wise multiply `samples[i] *= window[i]`. 8 lanes / iter.
///
/// # Safety
///
/// AVX2 must be available. `samples.len() == window.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
  debug_assert_eq!(samples.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut i = 0usize;
    while i + 8 <= n {
      let s = _mm256_loadu_ps(samples.as_ptr().add(i));
      let w = _mm256_loadu_ps(window.as_ptr().add(i));
      _mm256_storeu_ps(samples.as_mut_ptr().add(i), _mm256_mul_ps(s, w));
      i += 8;
    }
    debug_assert_eq!(i, n);
  }
}

/// Power spectrum: `out[i] = re*re + im*im` over the non-redundant FFT
/// half. Mirrors the AVX-512 design — de-interleave first via two
/// `_mm256_shuffle_ps` (within-128-bit-lane), then FMA `re² + im²` in
/// a single fused instruction. The final cross-lane
/// `_mm256_permute4x64_pd::<0xD8>` (interpreting as f64 lanes) reorders
/// the within-lane shuffle output `[p0 p1 p4 p5 p2 p3 p6 p7]` back to
/// `[p0..p7]`.
///
/// # Safety
///
/// AVX2 + FMA must be available. `complex.len() == out.len()`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn power_spectrum(
  complex: &[rustfft::num_complex::Complex<f32>],
  out: &mut [f32],
) {
  debug_assert_eq!(complex.len(), out.len());
  let n = complex.len();
  unsafe {
    let mut i = 0usize;
    let base = complex.as_ptr() as *const f32;
    while i + 8 <= n {
      // 16 floats = 8 complex pairs.
      let a = _mm256_loadu_ps(base.add(i * 2));
      let b = _mm256_loadu_ps(base.add(i * 2 + 8));
      // De-interleave (within each 128-bit lane). `shuffle_ps` mask
      // `0b10_00_10_00` picks even f32 lanes from (a, b); `0b11_01_11_01`
      // picks odd lanes. Output is in the within-lane interleaved order
      // (re0, re1, re4, re5, re2, re3, re6, re7) — fixed below.
      let re = _mm256_shuffle_ps::<0b10_00_10_00>(a, b);
      let im = _mm256_shuffle_ps::<0b11_01_11_01>(a, b);
      // Single-rounding fused multiply-add: `re*re + im*im`.
      let p_swizzled = _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));
      // Reorder 64-bit pairs (0,1,2,3) → (0,2,1,3) to land [p0..p7].
      let permuted = _mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(_mm256_castps_pd(
        p_swizzled,
      )));
      _mm256_storeu_ps(out.as_mut_ptr().add(i), permuted);
      i += 8;
    }
    if i < n {
      scalar::power_spectrum(&complex[i..], &mut out[i..]);
    }
  }
}

/// CMVN: `feature[d] = (feature[d] - means[d]) * istd[d]` over
/// `NUM_MEL_BINS` (= 80, multiple of 8 → no tail).
///
/// # Safety
///
/// AVX2 must be available. All slices have length `NUM_MEL_BINS`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
  debug_assert_eq!(feature.len(), NUM_MEL_BINS);
  debug_assert_eq!(means.len(), NUM_MEL_BINS);
  debug_assert_eq!(istd.len(), NUM_MEL_BINS);
  let n = NUM_MEL_BINS;
  unsafe {
    let mut i = 0usize;
    while i + 8 <= n {
      let f = _mm256_loadu_ps(feature.as_ptr().add(i));
      let m = _mm256_loadu_ps(means.as_ptr().add(i));
      let s = _mm256_loadu_ps(istd.as_ptr().add(i));
      let r = _mm256_mul_ps(_mm256_sub_ps(f, m), s);
      _mm256_storeu_ps(feature.as_mut_ptr().add(i), r);
      i += 8;
    }
    debug_assert_eq!(i, n);
  }
}

/// Sparse dot product + `max(LOG_FLOOR).ln()` over one Mel filter's
/// weights. FMA lane: `acc = fmadd(p, w, acc)` retains the 256-bit
/// accumulator without an intermediate rounding step.
///
/// Most Mel filters carry 5–25 weights, so the SIMD loop is a single
/// pass at most — the scalar tail handler does the bulk of the work
/// for narrow filters.
///
/// # Safety
///
/// AVX2 + FMA must be available. `power_slice.len() == weights.len()`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
  debug_assert_eq!(power_slice.len(), weights.len());
  let n = power_slice.len();
  let mut energy = 0.0f32;
  unsafe {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
      let p = _mm256_loadu_ps(power_slice.as_ptr().add(i));
      let w = _mm256_loadu_ps(weights.as_ptr().add(i));
      acc = _mm256_fmadd_ps(p, w, acc);
      i += 8;
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    energy += _mm_cvtss_f32(s);
    for k in i..n {
      energy += *power_slice.get_unchecked(k) * *weights.get_unchecked(k);
    }
  }
  energy.max(crate::features::LOG_FLOOR).ln()
}
