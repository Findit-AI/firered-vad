//! Runtime dispatchers for the Mel-fbank inner-loop kernels.
//!
//! Each function here picks the best available backend for the host
//! architecture, falling back to the scalar reference in `super::scalar`
//! when no SIMD path applies. This mirrors the
//! `colconv-be-tier10b/src/row/dispatch/` pattern: every kernel has a
//! dispatcher that is called from the orchestrating `extract` and
//! `push_pcm` functions in `super::mod`.
//!
//! `firered_vad_force_scalar` cfg (set via
//! `RUSTFLAGS='--cfg firered_vad_force_scalar'`) bypasses the SIMD
//! cascade so CI / coverage runs exercise the scalar baseline.

// SIMD intrinsics are `unsafe fn` in `core::arch`; the dispatcher is
// the thin call-site that gates them behind the runtime feature check.
// The crate-wide `deny(unsafe_code)` in `lib.rs` is opted out of here.
#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use super::arch;
use super::scalar;

// ---- runtime feature detection ----

/// NEON availability on aarch64. Uses runtime CPU feature detection
/// (cached after first call) on hosted targets — `firered-vad`
/// always links the `std` library, so the no-std compile-time
/// fallback `colconv-be-tier10b` ships isn't needed here. Mirrors
/// the `*_available()` pattern from `colconv` so a no-std fork is
/// a one-line change later.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn neon_available() -> bool {
  if cfg!(firered_vad_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

// ---- dispatchers ----

/// Scale + extend PCM into the output region. Vectorizes via NEON
/// when available; falls back to scalar otherwise.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn pcm_scale_extend(pcm: &[f32], out: &mut [f32]) {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      unsafe { arch::neon::pcm_scale_extend(pcm, out) };
      return;
    }
  }
  scalar::pcm_scale_extend(pcm, out);
}

/// DC removal: subtract the frame mean from every sample.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn dc_remove(window: &[f32], out: &mut [f32]) {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      unsafe { arch::neon::dc_remove(window, out) };
      return;
    }
  }
  scalar::dc_remove(window, out);
}

/// Pre-emphasis. Sequential by definition (`x[i] -= 0.97 * x[i-1]`)
/// → no SIMD, scalar always.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn pre_emphasis(samples: &mut [f32]) {
  scalar::pre_emphasis(samples);
}

/// Element-wise multiply samples by the Povey window.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn window_apply(samples: &mut [f32], window: &[f32]) {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      unsafe { arch::neon::window_apply(samples, window) };
      return;
    }
  }
  scalar::window_apply(samples, window);
}

/// Power spectrum `|X|^2` over the non-redundant FFT half.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn power_spectrum(complex: &[rustfft::num_complex::Complex<f32>], out: &mut [f32]) {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      unsafe { arch::neon::power_spectrum(complex, out) };
      return;
    }
  }
  scalar::power_spectrum(complex, out);
}

/// CMVN normalize one 80-bin Mel feature in place.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      unsafe { arch::neon::cmvn_apply(feature, means, istd) };
      return;
    }
  }
  scalar::cmvn_apply(feature, means, istd);
}

/// Mel filter sparse dot product + `max(LOG_FLOOR).ln()`. Returns the
/// log energy for one filter.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      return unsafe { arch::neon::mel_dot_log(power_slice, weights) };
    }
  }
  scalar::mel_dot_log(power_slice, weights)
}
