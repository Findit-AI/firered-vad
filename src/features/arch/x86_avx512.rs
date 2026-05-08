//! x86_64 AVX-512F kernels for the Mel-fbank inner loops.
//!
//! Selected by the dispatcher in [`super::super::dispatch`] when
//! the host advertises `avx512f`. The kernels carry
//! `#[target_feature(enable = "avx512f")]` — that single flag covers
//! every intrinsic this module uses (FMA, cross-lane permute, full
//! horizontal reduce).
//!
//! # Pipeline (16 f32 lanes per instruction)
//!
//! Vector width doubles vs AVX2: `__m512` carries 16 f32. The kernel
//! structure mirrors the AVX2 backend with two simplifications: AVX-512F
//! ships `_mm512_reduce_add_ps` (no manual hadd cascade), and
//! `_mm512_permutex2var_ps` is a single cross-lane shuffle that
//! de-interleaves `Complex<f32>` → (re, im) without the AVX2 hadd-then-
//! permute trick.
//!
//! `avx512f` has been required-stable since Rust 1.89 (2025); the
//! intrinsics module sits in stable `core::arch::x86_64` and needs no
//! nightly feature gate.
//!
//! # Numerical contract
//!
//! Bit-exact reordering of float adds means AVX-512 output may differ
//! from the scalar reference by one ULP at most when associativity
//! reorders fold differently. FMA in `mel_dot_log` retains intermediate
//! precision differently from sequential `mul + add`; the upstream
//! parity harness clears `--prob-tol 5e-3` and absorbs that drift.
//!
//! # Verification status (2026-05-08)
//!
//! Cross-compile-checked via `cargo check --target x86_64-apple-darwin`.
//! Bit-exact parity is **unverified** until the parity harness runs on
//! AVX-512-capable x86_64 hardware (CI gap tracked at the project level).

#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::features::{FRAME_LENGTH_SAMPLES, NUM_MEL_BINS, scalar};

// `pcm_scale_extend` deliberately has no AVX-512 path — same rationale
// as the NEON / SSE4.1 / AVX2 omissions: a single-FMul-per-element
// kernel that LLVM auto-vectorizes into the caller, with no headroom
// left for a `#[target_feature]`-gated function call to pay for itself.

/// DC removal: compute mean, store `window[i] - mean` into `out`.
///
/// # Safety
///
/// AVX-512F must be available. `window.len() == out.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(out.len(), FRAME_LENGTH_SAMPLES);
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
      let v = _mm512_loadu_ps(window.as_ptr().add(i));
      acc = _mm512_add_ps(acc, v);
      i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for k in i..n {
      sum += *window.get_unchecked(k);
    }
    let mean = sum / FRAME_LENGTH_SAMPLES as f32;
    let mean_v = _mm512_set1_ps(mean);

    let mut i = 0usize;
    while i + 16 <= n {
      let v = _mm512_loadu_ps(window.as_ptr().add(i));
      _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_sub_ps(v, mean_v));
      i += 16;
    }
    // FRAME_LENGTH_SAMPLES = 400 = 25 * 16 → no tail.
    debug_assert_eq!(i, n);
  }
}

/// Element-wise multiply `samples[i] *= window[i]`. 16 lanes / iter.
///
/// # Safety
///
/// AVX-512F must be available. `samples.len() == window.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
  debug_assert_eq!(samples.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut i = 0usize;
    while i + 16 <= n {
      let s = _mm512_loadu_ps(samples.as_ptr().add(i));
      let w = _mm512_loadu_ps(window.as_ptr().add(i));
      _mm512_storeu_ps(samples.as_mut_ptr().add(i), _mm512_mul_ps(s, w));
      i += 16;
    }
    debug_assert_eq!(i, n);
  }
}

/// Power spectrum: `out[i] = re*re + im*im` over the non-redundant
/// FFT half. Loads two 512-bit chunks (32 floats = 16 complex), then
/// `_mm512_permutex2var_ps` extracts the 16 even-indexed lanes (re)
/// and the 16 odd-indexed lanes (im) in one shuffle each. FMA folds
/// `im*im + re*re` without an intermediate rounding step.
///
/// # Safety
///
/// AVX-512F must be available. `complex.len() == out.len()`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn power_spectrum(
  complex: &[rustfft::num_complex::Complex<f32>],
  out: &mut [f32],
) {
  debug_assert_eq!(complex.len(), out.len());
  let n = complex.len();
  unsafe {
    // Static index vectors for the de-interleave.
    let idx_re = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let idx_im = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    let mut i = 0usize;
    let base = complex.as_ptr() as *const f32;
    while i + 16 <= n {
      let a = _mm512_loadu_ps(base.add(i * 2));
      let b = _mm512_loadu_ps(base.add(i * 2 + 16));
      let re = _mm512_permutex2var_ps(a, idx_re, b);
      let im = _mm512_permutex2var_ps(a, idx_im, b);
      let p = _mm512_fmadd_ps(im, im, _mm512_mul_ps(re, re));
      _mm512_storeu_ps(out.as_mut_ptr().add(i), p);
      i += 16;
    }
    if i < n {
      scalar::power_spectrum(&complex[i..], &mut out[i..]);
    }
  }
}

/// CMVN: `feature[d] = (feature[d] - means[d]) * istd[d]` over
/// `NUM_MEL_BINS` (= 80 = 5 * 16 → no tail).
///
/// # Safety
///
/// AVX-512F must be available. All slices have length `NUM_MEL_BINS`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
  debug_assert_eq!(feature.len(), NUM_MEL_BINS);
  debug_assert_eq!(means.len(), NUM_MEL_BINS);
  debug_assert_eq!(istd.len(), NUM_MEL_BINS);
  let n = NUM_MEL_BINS;
  unsafe {
    let mut i = 0usize;
    while i + 16 <= n {
      let f = _mm512_loadu_ps(feature.as_ptr().add(i));
      let m = _mm512_loadu_ps(means.as_ptr().add(i));
      let s = _mm512_loadu_ps(istd.as_ptr().add(i));
      let r = _mm512_mul_ps(_mm512_sub_ps(f, m), s);
      _mm512_storeu_ps(feature.as_mut_ptr().add(i), r);
      i += 16;
    }
    debug_assert_eq!(i, n);
  }
}

/// Sparse dot product + `max(LOG_FLOOR).ln()` over one Mel filter's
/// weights. FMA lane: `acc = fmadd(p, w, acc)`; `_mm512_reduce_add_ps`
/// is the native horizontal reduction.
///
/// Most Mel filters carry 5–25 weights, so the 16-wide SIMD loop fires
/// at most once per filter — the scalar tail handler does the bulk of
/// the work for narrow filters.
///
/// # Safety
///
/// AVX-512F must be available. `power_slice.len() == weights.len()`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
  debug_assert_eq!(power_slice.len(), weights.len());
  let n = power_slice.len();
  let mut energy = 0.0f32;
  unsafe {
    let mut i = 0usize;
    let mut acc = _mm512_setzero_ps();
    while i + 16 <= n {
      let p = _mm512_loadu_ps(power_slice.as_ptr().add(i));
      let w = _mm512_loadu_ps(weights.as_ptr().add(i));
      acc = _mm512_fmadd_ps(p, w, acc);
      i += 16;
    }
    energy += _mm512_reduce_add_ps(acc);
    for k in i..n {
      energy += *power_slice.get_unchecked(k) * *weights.get_unchecked(k);
    }
  }
  energy.max(crate::features::LOG_FLOOR).ln()
}
