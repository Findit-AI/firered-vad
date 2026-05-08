//! Scalar reference implementations of every Mel-fbank inner-loop kernel.
//!
//! Each function here is the byte-exact reference. The SIMD kernels in
//! `super::arch` are required to produce numerically equal outputs for
//! every input the parity harness exercises (within the float-reorder
//! ULP noise that LLVM's auto-vectorized scalar already permits — see
//! `tests/parity/`).
//!
//! Layout matches `colconv-be-tier10b/src/row/scalar/*` — one file per
//! domain, each kernel `pub(super) fn`, no `unsafe`.

use super::{FRAME_LENGTH_SAMPLES, INT16_SCALE, LOG_FLOOR, NUM_MEL_BINS, PRE_EMPHASIS};

/// Scale `pcm` (`[-1.0, 1.0]`) by `INT16_SCALE` and append into `out`.
/// `out` must already have capacity for `pcm.len()` more elements; the
/// caller has just `resize`d it to the desired final length.
///
/// Non-finite samples (NaN / ±Inf) are mapped to `0.0` before scaling
/// — without this, a single NaN would propagate through the
/// DC-removal mean, the FFT, the mel filterbank, and the CMVN step,
/// then corrupt the DFSMN cache for the entire stream's remaining
/// inferences. The cost is one `is_finite()` branch per sample, which
/// LLVM lowers to a small mask + select on aarch64 / x86 alike.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn pcm_scale_extend(pcm: &[f32], out: &mut [f32]) {
  debug_assert_eq!(pcm.len(), out.len());
  for (dst, &src) in out.iter_mut().zip(pcm.iter()) {
    *dst = if src.is_finite() {
      src * INT16_SCALE
    } else {
      0.0
    };
  }
}

/// Sum `window`, divide by length to get mean, write `window[i] - mean`
/// into `out`. Used as the DC-removal step. Length is fixed at
/// `FRAME_LENGTH_SAMPLES`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn dc_remove(window: &[f32], out: &mut [f32]) {
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(out.len(), FRAME_LENGTH_SAMPLES);
  let mean: f32 = window.iter().copied().sum::<f32>() / FRAME_LENGTH_SAMPLES as f32;
  for (o, &w) in out.iter_mut().zip(window.iter()) {
    *o = w - mean;
  }
}

/// Pre-emphasis: `x[i] -= 0.97 * x[i-1]` for `i = N-1..1`, then
/// `x[0] -= 0.97 * x[0]`. Strictly sequential (data dependency on
/// `x[i-1]`), so SIMD does not apply — there is no arch override and
/// `dispatch.rs` calls this function directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn pre_emphasis(samples: &mut [f32]) {
  debug_assert_eq!(samples.len(), FRAME_LENGTH_SAMPLES);
  for i in (1..FRAME_LENGTH_SAMPLES).rev() {
    samples[i] -= PRE_EMPHASIS * samples[i - 1];
  }
  samples[0] -= PRE_EMPHASIS * samples[0];
}

/// Element-wise multiply: `samples[i] *= window[i]` for the whole
/// 400-sample frame.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn window_apply(samples: &mut [f32], window: &[f32]) {
  debug_assert_eq!(samples.len(), FRAME_LENGTH_SAMPLES);
  debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
  for (s, &w) in samples.iter_mut().zip(window.iter()) {
    *s *= w;
  }
}

/// Power spectrum: `out[i] = re*re + im*im` over the non-redundant FFT
/// half. `complex` is the real/imaginary interleave produced by
/// `rustfft` (one `Complex<f32>` per FFT bin).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn power_spectrum(complex: &[rustfft::num_complex::Complex<f32>], out: &mut [f32]) {
  debug_assert_eq!(complex.len(), out.len());
  for (p, c) in out.iter_mut().zip(complex.iter()) {
    *p = c.re * c.re + c.im * c.im;
  }
}

/// CMVN: `feature[d] = (feature[d] - means[d]) * istd[d]` for the
/// 80-bin Mel feature vector.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
  debug_assert_eq!(feature.len(), NUM_MEL_BINS);
  debug_assert_eq!(means.len(), NUM_MEL_BINS);
  debug_assert_eq!(istd.len(), NUM_MEL_BINS);
  for ((f, &m), &i) in feature.iter_mut().zip(means.iter()).zip(istd.iter()) {
    *f = (*f - m) * i;
  }
}

/// Sparse dot product: `sum(power[start_bin + j] * weights[j])` then
/// `max(LOG_FLOOR).ln()`. Done once per Mel filter (80 filters per
/// frame). Each filter's `weights` slice is short (typically 5–25
/// elements) so the SIMD opportunity is in the aggregate, not the
/// inner loop alone.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
  debug_assert_eq!(power_slice.len(), weights.len());
  let mut energy = 0.0f32;
  for (b, w) in power_slice.iter().zip(weights.iter()) {
    energy += b * w;
  }
  energy.max(LOG_FLOOR).ln()
}
