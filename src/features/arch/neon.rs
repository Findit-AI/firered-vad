//! aarch64 NEON kernels for the Mel-fbank inner loops.
//!
//! Selected by the dispatcher in [`super::super::dispatch`] when
//! `is_aarch64_feature_detected!("neon")` returns true (runtime,
//! std-gated) or `cfg!(target_feature = "neon")` evaluates true
//! (compile-time, no-std). The kernels carry
//! `#[target_feature(enable = "neon")]` so their intrinsics execute in
//! an explicitly NEON-enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! # Numerical contract
//!
//! Bit-exact reordering of float adds means SIMD output may differ
//! from the scalar reference by a single ULP at most when associativity
//! reorders fold differently. The parity harness exercises the full
//! pipeline against upstream Python's `kaldi-native-fbank`; both
//! scalar and NEON paths must pass `--prob-tol 5e-3` (in practice both
//! sit at ~3e-6).
//!
//! # Pipeline (4 f32 lanes per instruction)
//!
//! NEON's `float32x4_t` carries four lanes; every kernel here processes
//! `len / 4 * 4` elements in vector blocks then dispatches the remaining
//! `len % 4` elements through the matching scalar helper from
//! `super::super::scalar`.

#![allow(clippy::missing_safety_doc)] // each fn has explicit `# Safety` in its doc.
// `core::arch::aarch64::*` intrinsics are `unsafe fn` in the standard
// library; this module exists precisely to wrap them. The crate's
// blanket `deny(unsafe_code)` in `lib.rs` is opted out of here only.
#![allow(unsafe_code)]

use core::arch::aarch64::*;

use crate::features::{FRAME_LENGTH_SAMPLES, INT16_SCALE_VEC, NUM_MEL_BINS, scalar};

/// Scale `pcm` by `INT16_SCALE` and store into `out`. 4 lanes / iter.
///
/// # Safety
///
/// NEON must be available. `pcm.len() == out.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn pcm_scale_extend(pcm: &[f32], out: &mut [f32]) {
  debug_assert_eq!(pcm.len(), out.len());
  let n = pcm.len();
  let mut i = 0usize;
  unsafe {
    let scale = vdupq_n_f32(INT16_SCALE_VEC);
    while i + 4 <= n {
      let v = vld1q_f32(pcm.as_ptr().add(i));
      let scaled = vmulq_f32(v, scale);
      vst1q_f32(out.as_mut_ptr().add(i), scaled);
      i += 4;
    }
  }
  if i < n {
    scalar::pcm_scale_extend(&pcm[i..], &mut out[i..]);
  }
}

/// DC removal: compute mean, store `window[i] - mean` into `out`.
/// 4 lanes / iter for both the sum reduction and the subtract loop.
///
/// # Safety
///
/// NEON must be available. `window.len() == out.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(out.len(), FRAME_LENGTH_SAMPLES);

  // FRAME_LENGTH_SAMPLES = 400 = 100 * 4, so the vector loop covers
  // the whole frame and there is no scalar remainder for sum or sub.
  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= n {
      let v = vld1q_f32(window.as_ptr().add(i));
      acc = vaddq_f32(acc, v);
      i += 4;
    }
    // Horizontal reduce 4 lanes → scalar.
    let mut sum = vaddvq_f32(acc);
    if i < n {
      for k in i..n {
        sum += *window.get_unchecked(k);
      }
    }
    let mean = sum / FRAME_LENGTH_SAMPLES as f32;
    let mean_v = vdupq_n_f32(mean);

    let mut i = 0usize;
    while i + 4 <= n {
      let v = vld1q_f32(window.as_ptr().add(i));
      let r = vsubq_f32(v, mean_v);
      vst1q_f32(out.as_mut_ptr().add(i), r);
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
/// NEON must be available. `samples.len() == window.len() == FRAME_LENGTH_SAMPLES`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
  debug_assert_eq!(samples.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);

  let n = FRAME_LENGTH_SAMPLES;
  unsafe {
    let mut i = 0usize;
    while i + 4 <= n {
      let s = vld1q_f32(samples.as_ptr().add(i));
      let w = vld1q_f32(window.as_ptr().add(i));
      vst1q_f32(samples.as_mut_ptr().add(i), vmulq_f32(s, w));
      i += 4;
    }
    // FRAME_LENGTH_SAMPLES = 400 is a multiple of 4 — no tail.
    debug_assert_eq!(i, n);
  }
}

/// Power spectrum: `out[i] = re*re + im*im` over the non-redundant
/// FFT half. The Complex<f32> source has interleaved layout
/// (re0, im0, re1, im1, …); we use `vld2q_f32` to load 4 reals + 4 imags
/// into separate registers, then a fused multiply-add.
///
/// # Safety
///
/// NEON must be available. `complex.len() == out.len()`.
#[inline]
#[target_feature(enable = "neon")]
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
      // vld2q_f32 deinterleaves 8 contiguous f32 → (re_x4, im_x4).
      let pair = vld2q_f32(base.add(i * 2));
      let re = pair.0;
      let im = pair.1;
      // re*re + im*im via FMA.
      let p = vfmaq_f32(vmulq_f32(re, re), im, im);
      vst1q_f32(out.as_mut_ptr().add(i), p);
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
/// NEON must be available. All slices have length `NUM_MEL_BINS`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
  debug_assert_eq!(feature.len(), NUM_MEL_BINS);
  debug_assert_eq!(means.len(), NUM_MEL_BINS);
  debug_assert_eq!(istd.len(), NUM_MEL_BINS);

  let n = NUM_MEL_BINS;
  unsafe {
    let mut i = 0usize;
    while i + 4 <= n {
      let f = vld1q_f32(feature.as_ptr().add(i));
      let m = vld1q_f32(means.as_ptr().add(i));
      let s = vld1q_f32(istd.as_ptr().add(i));
      let r = vmulq_f32(vsubq_f32(f, m), s);
      vst1q_f32(feature.as_mut_ptr().add(i), r);
      i += 4;
    }
    debug_assert_eq!(i, n);
  }
}

/// Sparse dot product + log floor over one Mel filter's weights. The
/// inner length is small (5-25 typical); we vectorize the bulk and
/// scalar-tail the remainder.
///
/// # Safety
///
/// NEON must be available. `power_slice.len() == weights.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
  debug_assert_eq!(power_slice.len(), weights.len());
  let n = power_slice.len();
  let mut energy = 0.0f32;
  unsafe {
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= n {
      let p = vld1q_f32(power_slice.as_ptr().add(i));
      let w = vld1q_f32(weights.as_ptr().add(i));
      acc = vfmaq_f32(acc, p, w);
      i += 4;
    }
    energy += vaddvq_f32(acc);
    for k in i..n {
      energy += *power_slice.get_unchecked(k) * *weights.get_unchecked(k);
    }
  }
  energy.max(crate::features::LOG_FLOOR).ln()
}
