//! Pure-Rust Kaldi-compatible Mel filterbank + CMVN preprocessing.
//!
//! All public types here are `pub(crate)` — feature extraction is an
//! implementation detail of [`crate::Vad`].
//!
//! # Module layout (mirrors `colconv-be-tier10b/src/row/`)
//!
//! - [`scalar`] — reference implementations of every inner-loop kernel.
//!   Always compiled, used as the baseline and the SIMD remainder
//!   (tail) handler.
//! - [`arch`] — architecture-specific SIMD kernels gated on
//!   `target_arch`. Today only `aarch64::neon` is implemented; x86_64
//!   and wasm32 fall through to scalar via the dispatcher and are easy
//!   drop-in additions.
//! - [`dispatch`] — runtime selection helpers + `*_available()`
//!   feature-detection wrappers. Called from `MelFilterbank::extract`
//!   and `FeatureExtractor::push_pcm` once per inner-loop kernel.
//!
//! Setting `RUSTFLAGS='--cfg firered_vad_force_scalar'` short-circuits
//! every `*_available()` to `false` so CI / parity runs can exercise
//! the scalar baseline on machines that would otherwise pick NEON.

mod arch;
mod dispatch;
mod scalar;

// Bench-only thunks: `benches/kernels.rs` calls scalar and NEON
// kernels directly so the criterion microbench can compare each path
// without going through the dispatcher's runtime feature check. The
// real kernels are `pub(crate)` and can't be re-exported as `pub`,
// so we wrap them in thin `pub fn` thunks under this feature flag.
// The `_bench-internals` cargo feature gates the entire module.
#[cfg(feature = "_bench-internals")]
#[doc(hidden)]
pub mod __bench_internals {
  //! Internal — do not depend on this. The shape of these wrappers
  //! is not part of the crate's public API and may change in any
  //! release without notice.

  // The crate-internal constants are `pub(crate)` and can't be re-
  // exported. Mirror their values here so benches can drive the
  // kernels with the right buffer sizes.
  pub const NUM_MEL_BINS: usize = super::NUM_MEL_BINS;
  pub const FRAME_LENGTH_SAMPLES: usize = super::FRAME_LENGTH_SAMPLES;
  pub const FFT_SIZE: usize = super::FFT_SIZE;
  pub const FFT_BINS: usize = super::FFT_BINS;
  pub const PRE_EMPHASIS: f32 = super::PRE_EMPHASIS;
  pub const LOG_FLOOR: f32 = super::LOG_FLOOR;
  pub const INT16_SCALE: f32 = super::INT16_SCALE;

  /// Scalar kernel thunks. Each delegates to the matching
  /// `pub(crate) fn` in [`super::scalar`].
  pub mod scalar {
    use super::super::scalar as inner;

    #[inline]
    pub fn pcm_scale_extend(pcm: &[f32], out: &mut [f32]) {
      inner::pcm_scale_extend(pcm, out);
    }
    #[inline]
    pub fn dc_remove(window: &[f32], out: &mut [f32]) {
      inner::dc_remove(window, out);
    }
    #[inline]
    pub fn pre_emphasis(samples: &mut [f32]) {
      inner::pre_emphasis(samples);
    }
    #[inline]
    pub fn window_apply(samples: &mut [f32], window: &[f32]) {
      inner::window_apply(samples, window);
    }
    #[inline]
    pub fn power_spectrum(
      complex: &[rustfft::num_complex::Complex<f32>],
      out: &mut [f32],
    ) {
      inner::power_spectrum(complex, out);
    }
    #[inline]
    pub fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
      inner::cmvn_apply(feature, means, istd);
    }
    #[inline]
    pub fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
      inner::mel_dot_log(power_slice, weights)
    }
  }

  /// aarch64 NEON kernel thunks. Each delegates to the matching
  /// `pub(crate) unsafe fn` in [`super::arch::neon`].
  ///
  /// # Safety
  /// Caller must ensure NEON is available on the host (`is_aarch64_feature_detected!("neon")`).
  #[cfg(target_arch = "aarch64")]
  #[allow(unsafe_code)]
  pub mod neon {
    // `pcm_scale_extend` has no NEON variant — see the rationale in
    // `super::super::arch::neon`. Scalar wins for that kernel.

    use super::super::arch::neon as inner;

    #[inline]
    pub unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
      unsafe { inner::dc_remove(window, out) }
    }
    #[inline]
    pub unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
      unsafe { inner::window_apply(samples, window) }
    }
    #[inline]
    pub unsafe fn power_spectrum(
      complex: &[rustfft::num_complex::Complex<f32>],
      out: &mut [f32],
    ) {
      unsafe { inner::power_spectrum(complex, out) }
    }
    #[inline]
    pub unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
      unsafe { inner::cmvn_apply(feature, means, istd) }
    }
    #[inline]
    pub unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
      unsafe { inner::mel_dot_log(power_slice, weights) }
    }
  }

  /// x86_64 SSE4.1 kernel thunks.
  ///
  /// # Safety
  /// Caller must ensure SSE4.1 is available on the host
  /// (`is_x86_feature_detected!("sse4.1")`).
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  #[allow(unsafe_code)]
  pub mod x86_sse41 {
    use super::super::arch::x86_sse41 as inner;

    #[inline]
    pub unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
      unsafe { inner::dc_remove(window, out) }
    }
    #[inline]
    pub unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
      unsafe { inner::window_apply(samples, window) }
    }
    #[inline]
    pub unsafe fn power_spectrum(
      complex: &[rustfft::num_complex::Complex<f32>],
      out: &mut [f32],
    ) {
      unsafe { inner::power_spectrum(complex, out) }
    }
    #[inline]
    pub unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
      unsafe { inner::cmvn_apply(feature, means, istd) }
    }
    #[inline]
    pub unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
      unsafe { inner::mel_dot_log(power_slice, weights) }
    }
  }

  /// x86_64 AVX2 + FMA kernel thunks.
  ///
  /// # Safety
  /// Caller must ensure AVX2 + FMA are available on the host
  /// (`is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`).
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  #[allow(unsafe_code)]
  pub mod x86_avx2 {
    use super::super::arch::x86_avx2 as inner;

    #[inline]
    pub unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
      unsafe { inner::dc_remove(window, out) }
    }
    #[inline]
    pub unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
      unsafe { inner::window_apply(samples, window) }
    }
    #[inline]
    pub unsafe fn power_spectrum(
      complex: &[rustfft::num_complex::Complex<f32>],
      out: &mut [f32],
    ) {
      unsafe { inner::power_spectrum(complex, out) }
    }
    #[inline]
    pub unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
      unsafe { inner::cmvn_apply(feature, means, istd) }
    }
    #[inline]
    pub unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
      unsafe { inner::mel_dot_log(power_slice, weights) }
    }
  }

  /// x86_64 AVX-512F kernel thunks.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F is available on the host
  /// (`is_x86_feature_detected!("avx512f")`).
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  #[allow(unsafe_code)]
  pub mod x86_avx512 {
    use super::super::arch::x86_avx512 as inner;

    #[inline]
    pub unsafe fn dc_remove(window: &[f32], out: &mut [f32]) {
      unsafe { inner::dc_remove(window, out) }
    }
    #[inline]
    pub unsafe fn window_apply(samples: &mut [f32], window: &[f32]) {
      unsafe { inner::window_apply(samples, window) }
    }
    #[inline]
    pub unsafe fn power_spectrum(
      complex: &[rustfft::num_complex::Complex<f32>],
      out: &mut [f32],
    ) {
      unsafe { inner::power_spectrum(complex, out) }
    }
    #[inline]
    pub unsafe fn cmvn_apply(feature: &mut [f32], means: &[f32], istd: &[f32]) {
      unsafe { inner::cmvn_apply(feature, means, istd) }
    }
    #[inline]
    pub unsafe fn mel_dot_log(power_slice: &[f32], weights: &[f32]) -> f32 {
      unsafe { inner::mel_dot_log(power_slice, weights) }
    }
  }
}

use crate::error::{Error, Result};

/// Number of Mel filterbank bins the model expects.
pub(crate) const NUM_MEL_BINS: usize = 80;

/// Sample rate the model expects.
pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;

/// Number of samples in a 25 ms analysis window.
pub(crate) const FRAME_LENGTH_SAMPLES: usize = 400;

/// Number of samples between successive 10 ms frame starts.
pub(crate) const FRAME_SHIFT_SAMPLES: usize = 160;

/// FFT length used for the mel filterbank (next power of 2 ≥ FRAME_LENGTH_SAMPLES).
pub(crate) const FFT_SIZE: usize = 512;

/// Number of unique non-redundant FFT bins (`FFT_SIZE / 2 + 1`).
pub(crate) const FFT_BINS: usize = FFT_SIZE / 2 + 1;

/// Pre-emphasis coefficient (Kaldi default; upstream keeps the default).
pub(crate) const PRE_EMPHASIS: f32 = 0.97;

/// Mel-bin range (low_freq=20, high_freq=Nyquist for 16 kHz).
const MEL_LOW_FREQ_HZ: f32 = 20.0;
const MEL_HIGH_FREQ_HZ: f32 = 8_000.0;

/// Floor for the log of bin energies, matching `kaldi-native-fbank`'s
/// `std::numeric_limits<float>::epsilon()` (see `feature-fbank.cc`'s
/// `FbankComputer::Compute`). Note: differs from Kaldi-proper's
/// hand-tuned 1e-20 — `kaldi-native-fbank` rolled it back to f32::EPSILON,
/// which is what upstream FireRedVAD's pipeline actually uses.
pub(crate) const LOG_FLOOR: f32 = f32::EPSILON;

/// Scale factor applied to incoming PCM before feature extraction.
///
/// Upstream Python reads WAVs as `int16` and feeds raw int16-range
/// values to `kaldi_native_fbank`. We accept f32 in `[-1.0, 1.0]` from
/// callers and multiply by this constant on the way in to keep the
/// downstream filterbank values numerically identical to upstream.
pub(crate) const INT16_SCALE: f32 = 32_768.0;

/// One sparse triangular Mel filter, addressed by `start_bin` and `weights`.
#[derive(Debug, Clone)]
struct MelFilter {
  start_bin: usize,
  weights: Vec<f32>,
}

/// Pure-Rust Kaldi-compatible Mel filterbank.
///
/// Configuration is hard-coded to match upstream FireRedVAD exactly:
/// 16 kHz, 25 ms / 10 ms windows, 80 mel bins, Povey window,
/// pre-emphasis 0.97, DC removal on, snip_edges=true, log floor f32::EPSILON.
pub(crate) struct MelFilterbank {
  fft: rustfft::algorithm::Radix4<f32>,
  fft_buf: Vec<rustfft::num_complex::Complex<f32>>,
  povey_window: Vec<f32>,
  filters: Vec<MelFilter>,
  /// Persistent scratch for the DC-removed / pre-emphasized / windowed
  /// 25 ms frame. Length `FRAME_LENGTH_SAMPLES`. Replaces a 1.6 KB
  /// stack array that was previously re-zeroed every call.
  samples_scratch: Vec<f32>,
  /// Persistent scratch for the power spectrum (`|X|^2` over the
  /// non-redundant FFT half). Length `FFT_BINS`. Replaces a 1 KB
  /// stack array that was previously re-zeroed every call.
  power_scratch: Vec<f32>,
}

impl std::fmt::Debug for MelFilterbank {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("MelFilterbank")
      .field("fft_buf_len", &self.fft_buf.len())
      .field("povey_window_len", &self.povey_window.len())
      .field("filters_len", &self.filters.len())
      .finish()
  }
}

/// Cepstral Mean and Variance Normalization stats parsed from a Kaldi
/// `.ark` file. The 80-dim means and inverse-std-variances are applied
/// to each Mel-fbank feature vector before it is fed to the model.
#[derive(Debug, Clone)]
pub(crate) struct Cmvn {
  /// Private — mutate only via `from_ark_bytes` / `from_components`.
  /// Direct construction would let callers smuggle non-finite stats
  /// into the feature pipeline; the constructors validate.
  means: Vec<f32>,
  inverse_std_variances: Vec<f32>,
}

impl Cmvn {
  /// Parse a Kaldi binary double-matrix `.ark`.
  ///
  /// Format expected:
  ///
  /// ```text
  /// \0B            (2 bytes, magic)
  /// "DM "          (3 bytes, double-matrix marker — note trailing space)
  /// \x04 + i32_le  (5 bytes, rows)
  /// \x04 + i32_le  (5 bytes, cols)
  /// rows*cols*8 bytes f64 LE
  /// ```
  ///
  /// `rows` must be 2 (sums and sum-squares). `cols` must be `NUM_MEL_BINS + 1`
  /// (80 stat columns plus a count column at index 80). `count` lives at
  /// `data[0][NUM_MEL_BINS]`. Each mean is `sums[d] / count`; each
  /// inverse-std-variance is `1 / sqrt(max(1e-20, sum_sq[d]/count - mean[d]^2))`.
  pub(crate) fn from_ark_bytes(bytes: &[u8]) -> Result<Self> {
    let mut i: usize = 0;
    let need = |i: usize, n: usize| -> Result<()> {
      if bytes.len() < i + n {
        Err(Error::InvalidCmvn {
          reason: "truncated header",
        })
      } else {
        Ok(())
      }
    };

    need(i, 2)?;
    if &bytes[i..i + 2] != b"\x00B" {
      return Err(Error::InvalidCmvn {
        reason: "missing \\0B magic",
      });
    }
    i += 2;

    need(i, 3)?;
    if &bytes[i..i + 3] != b"DM " {
      return Err(Error::InvalidCmvn {
        reason: "expected double-matrix marker 'DM '",
      });
    }
    i += 3;

    need(i, 1)?;
    if bytes[i] != 4 {
      return Err(Error::InvalidCmvn {
        reason: "expected 4-byte int32 size token before rows",
      });
    }
    i += 1;
    need(i, 4)?;
    let rows = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    i += 4;
    if rows != 2 {
      return Err(Error::InvalidCmvn {
        reason: "expected exactly 2 rows (sums, sum_sqs)",
      });
    }

    need(i, 1)?;
    if bytes[i] != 4 {
      return Err(Error::InvalidCmvn {
        reason: "expected 4-byte int32 size token before cols",
      });
    }
    i += 1;
    need(i, 4)?;
    let cols = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    i += 4;
    if cols != (NUM_MEL_BINS as i32) + 1 {
      return Err(Error::InvalidCmvn {
        reason: "expected NUM_MEL_BINS + 1 columns",
      });
    }

    let total = (rows as usize) * (cols as usize) * 8;
    need(i, total)?;
    let mut data = Vec::with_capacity((rows as usize) * (cols as usize));
    let mut p = i;
    for _ in 0..(rows as usize) * (cols as usize) {
      let chunk = [
        bytes[p],
        bytes[p + 1],
        bytes[p + 2],
        bytes[p + 3],
        bytes[p + 4],
        bytes[p + 5],
        bytes[p + 6],
        bytes[p + 7],
      ];
      data.push(f64::from_le_bytes(chunk));
      p += 8;
    }

    let count = data[NUM_MEL_BINS]; // first row, last column
    if !(count.is_finite() && count >= 1.0) {
      return Err(Error::InvalidCmvn {
        reason: "non-positive CMVN count",
      });
    }

    let mut means = Vec::with_capacity(NUM_MEL_BINS);
    let mut inverse_std_variances = Vec::with_capacity(NUM_MEL_BINS);
    let row_stride = cols as usize;
    for d in 0..NUM_MEL_BINS {
      let sum = data[d];
      let sum_sq = data[row_stride + d];
      // Reject non-finite stats up front so we can't silently produce
      // NaN means / istds that would poison every feature vector
      // downstream. (audit fix L-006.)
      if !sum.is_finite() || !sum_sq.is_finite() {
        return Err(Error::InvalidCmvn {
          reason: "non-finite stat (NaN or Inf) in CMVN matrix",
        });
      }
      let mean = sum / count;
      let mut var = sum_sq / count - mean * mean;
      if var < 1e-20 {
        var = 1e-20;
      }
      let istd = 1.0 / var.sqrt();
      let mean_f32 = mean as f32;
      let istd_f32 = istd as f32;
      if !mean_f32.is_finite() || !istd_f32.is_finite() {
        return Err(Error::InvalidCmvn {
          reason: "non-finite mean or inverse-std after normalization",
        });
      }
      means.push(mean_f32);
      inverse_std_variances.push(istd_f32);
    }

    Ok(Self {
      means,
      inverse_std_variances,
    })
  }

  /// Construct directly from already-validated mean and
  /// inverse-std-variance vectors. Used only by tests; the public
  /// constructor is `from_ark_bytes`. Both vectors must have length
  /// `NUM_MEL_BINS`. The caller is responsible for ensuring all values
  /// are finite.
  #[cfg(test)]
  pub(crate) fn from_components(means: Vec<f32>, inverse_std_variances: Vec<f32>) -> Self {
    debug_assert_eq!(means.len(), NUM_MEL_BINS);
    debug_assert_eq!(inverse_std_variances.len(), NUM_MEL_BINS);
    Self {
      means,
      inverse_std_variances,
    }
  }

  /// Apply CMVN in place to one 80-dim feature vector. Dispatches to
  /// the SIMD backend on aarch64; scalar otherwise.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn apply(&self, feature: &mut [f32]) {
    dispatch::cmvn_apply(feature, &self.means, &self.inverse_std_variances);
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn hz_to_mel(hz: f32) -> f32 {
  // Kaldi/HTK convention: 1127 * ln(1 + f/700)
  1127.0 * (1.0 + hz / 700.0).ln()
}

/// Centre frequency of the `i`-th non-redundant FFT bin in Hz.
#[cfg_attr(not(tarpaulin), inline(always))]
fn fft_bin_hz(i: usize) -> f32 {
  (i as f32) * (SAMPLE_RATE_HZ as f32) / (FFT_SIZE as f32)
}

/// Povey window. Computed entirely in `f64` (Kaldi's `cos` and `pow` are
/// `double`-precision in `feature-window.cc::GetWindow`), then cast to
/// `f32` for storage. Doing this in `f32` from the start would round at
/// every step and accumulate ~1 ULP per element vs the C++ reference.
fn build_povey_window() -> Vec<f32> {
  let n = FRAME_LENGTH_SAMPLES;
  let a: f64 = std::f64::consts::TAU / ((n - 1) as f64);
  (0..n)
    .map(|i| {
      let i_fl = i as f64;
      ((0.5 - 0.5 * (a * i_fl).cos()).powf(0.85)) as f32
    })
    .collect()
}

/// Mel-bank construction matching `kaldi-native-fbank`'s
/// `MelBanks::InitKaldiMelBanks` precisely:
///
/// - The (`num_bins` + 2) anchors are spaced **linearly in mel space**
///   between `mel(low_freq)` and `mel(high_freq)`.
/// - Each FFT bin's center frequency is converted to mel, and the
///   triangular weight is computed in **mel space** as
///   `(mel - left_mel) / (center_mel - left_mel)` (or the symmetric
///   right-side formula).
/// - The boundary test is strict (`mel > left_mel && mel < right_mel`).
/// - The FFT-bin loop runs over `0..num_fft_bins` where
///   `num_fft_bins = FFT_SIZE / 2 = 256`. The Nyquist bin (index 256)
///   is **excluded** to match Kaldi.
///
/// Earlier drafts of this file computed the triangle in Hz space, which
/// gave subtly different filter weights for every bin (Hz-space triangle
/// vs mel-space triangle differ because mel(f) is non-linear).
fn build_mel_filters() -> Vec<MelFilter> {
  let mel_low = hz_to_mel(MEL_LOW_FREQ_HZ);
  let mel_high = hz_to_mel(MEL_HIGH_FREQ_HZ);
  let mel_freq_delta = (mel_high - mel_low) / (NUM_MEL_BINS as f32 + 1.0);

  // Kaldi iterates over FFT_SIZE/2 bins (Nyquist excluded).
  let num_fft_bins = FFT_SIZE / 2;

  let mut filters = Vec::with_capacity(NUM_MEL_BINS);
  for b in 0..NUM_MEL_BINS {
    let left_mel = mel_low + (b as f32) * mel_freq_delta;
    let center_mel = mel_low + (b as f32 + 1.0) * mel_freq_delta;
    let right_mel = mel_low + (b as f32 + 2.0) * mel_freq_delta;

    let mut start_bin = num_fft_bins;
    let mut end_bin: usize = 0;
    let mut found_any = false;
    for i in 0..num_fft_bins {
      let freq = fft_bin_hz(i);
      let mel = hz_to_mel(freq);
      if mel > left_mel && mel < right_mel {
        if !found_any {
          start_bin = i;
          found_any = true;
        }
        end_bin = i;
      }
    }

    let mut weights = Vec::new();
    if found_any {
      weights.reserve(end_bin - start_bin + 1);
      for i in start_bin..=end_bin {
        let freq = fft_bin_hz(i);
        let mel = hz_to_mel(freq);
        let w = if mel <= center_mel {
          (mel - left_mel) / (center_mel - left_mel)
        } else {
          (right_mel - mel) / (right_mel - center_mel)
        };
        weights.push(w);
      }
    } else {
      start_bin = 0;
    }

    filters.push(MelFilter { start_bin, weights });
  }

  filters
}

impl MelFilterbank {
  pub(crate) fn new() -> Self {
    use rustfft::FftDirection;
    Self {
      fft: rustfft::algorithm::Radix4::<f32>::new(FFT_SIZE, FftDirection::Forward),
      fft_buf: vec![rustfft::num_complex::Complex::new(0.0, 0.0); FFT_SIZE],
      povey_window: build_povey_window(),
      filters: build_mel_filters(),
      samples_scratch: vec![0.0; FRAME_LENGTH_SAMPLES],
      power_scratch: vec![0.0; FFT_BINS],
    }
  }

  /// Extract one 80-dim log-Mel feature from a 25 ms window of int16-range
  /// samples. The caller's slice is **not** mutated; intermediate
  /// transformations live in `samples_scratch` and `power_scratch`.
  ///
  /// Each inner-loop step delegates to [`dispatch`], which picks the
  /// best available backend (NEON when on aarch64, scalar otherwise).
  pub(crate) fn extract(&mut self, window: &[f32], out: &mut [f32]) {
    debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
    debug_assert_eq!(out.len(), NUM_MEL_BINS);

    // 1. DC removal into samples_scratch.
    dispatch::dc_remove(window, &mut self.samples_scratch);

    // 2. Pre-emphasis (sequential, scalar always).
    dispatch::pre_emphasis(&mut self.samples_scratch);

    // 3. Apply Povey window.
    dispatch::window_apply(&mut self.samples_scratch, &self.povey_window);

    // 4. Copy windowed samples into fft_buf real parts; zero out the
    //    [400..512] tail and all imaginary parts. Then FFT in place.
    for (buf, &s) in self.fft_buf.iter_mut().zip(self.samples_scratch.iter()) {
      buf.re = s;
      buf.im = 0.0;
    }
    for buf in &mut self.fft_buf[FRAME_LENGTH_SAMPLES..] {
      buf.re = 0.0;
      buf.im = 0.0;
    }
    use rustfft::Fft;
    self.fft.process(&mut self.fft_buf);

    // 5. Power spectrum (|X|^2) into power_scratch for the non-redundant half.
    dispatch::power_spectrum(&self.fft_buf[..FFT_BINS], &mut self.power_scratch);

    // 6. Mel filterbank dot products → log.
    for (out_val, f) in out.iter_mut().zip(self.filters.iter()) {
      let bins = &self.power_scratch[f.start_bin..f.start_bin + f.weights.len()];
      *out_val = dispatch::mel_dot_log(bins, &f.weights);
    }
  }
}

/// Streaming feature extractor: buffers PCM, emits one 80-dim Mel-fbank
/// feature vector per consumed 10 ms frame.
#[derive(Debug)]
pub(crate) struct FeatureExtractor {
  fbank: MelFilterbank,
  cmvn: Cmvn,
  /// Up to `FRAME_LENGTH_SAMPLES` of pending int16-range samples.
  pcm_tail: Vec<f32>,
  /// Reusable scratch for one 80-dim feature vector.
  feature_scratch: Vec<f32>,
}

impl FeatureExtractor {
  /// Construct from raw CMVN bytes (Kaldi binary `.ark` format).
  pub(crate) fn new(cmvn_bytes: &[u8]) -> Result<Self> {
    Ok(Self {
      fbank: MelFilterbank::new(),
      cmvn: Cmvn::from_ark_bytes(cmvn_bytes)?,
      pcm_tail: Vec::with_capacity(FRAME_LENGTH_SAMPLES),
      feature_scratch: vec![0.0; NUM_MEL_BINS],
    })
  }

  /// Reset all streaming state. Cmvn / fbank / scratch buffers stay allocated.
  pub(crate) fn reset(&mut self) {
    self.pcm_tail.clear();
  }

  /// Append PCM in `[-1.0, 1.0]` range. Internally rescaled to int16-range
  /// to match upstream's input domain. Dispatches to a SIMD-vectorized
  /// scale-and-extend on aarch64; scalar otherwise.
  pub(crate) fn push_pcm(&mut self, pcm: &[f32]) {
    let offset = self.pcm_tail.len();
    self.pcm_tail.resize(offset + pcm.len(), 0.0);
    dispatch::pcm_scale_extend(pcm, &mut self.pcm_tail[offset..]);
  }

  /// Number of pending int16-range samples in the tail buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn pending_samples(&self) -> usize {
    self.pcm_tail.len()
  }

  /// True if the tail buffer holds at least one full 25 ms window.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn has_full_window(&self) -> bool {
    self.pcm_tail.len() >= FRAME_LENGTH_SAMPLES
  }

  /// Consume one 25 ms window from the head of the tail and write its
  /// CMVN-normalized 80-dim feature into `out`. Drops the leading
  /// `FRAME_SHIFT_SAMPLES` (10 ms) of the tail so successive calls
  /// produce overlapping 25 ms / 10 ms-hop frames.
  pub(crate) fn extract_one(&mut self, out: &mut [f32]) {
    debug_assert_eq!(out.len(), NUM_MEL_BINS);
    debug_assert!(self.has_full_window());

    // Pass the 25 ms head of pcm_tail directly to the fbank — `extract`
    // copies into its own `samples_scratch`, so the slice doesn't need to
    // be a separate scratch buffer.
    self.fbank.extract(
      &self.pcm_tail[..FRAME_LENGTH_SAMPLES],
      &mut self.feature_scratch,
    );
    self.cmvn.apply(&mut self.feature_scratch);
    out.copy_from_slice(&self.feature_scratch);

    // Drop the oldest 10 ms (frame shift) so the next call sees the next
    // 25 ms window aligned at +10 ms.
    self.pcm_tail.drain(..FRAME_SHIFT_SAMPLES);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The bundled CMVN file is the most reliable parity reference.
  const BUNDLED_CMVN: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/cmvn.ark"));

  #[test]
  fn parses_bundled_cmvn_into_80_means_and_istds() {
    let cmvn = Cmvn::from_ark_bytes(BUNDLED_CMVN).expect("parse cmvn");
    // Apply on a sentinel feature: with the bundled stats the first
    // bin's mean is in the log-mel-energy range, so a feature value
    // equal to that mean should map to ≈ 0.
    let mut probe = vec![0.0f32; NUM_MEL_BINS];
    let baseline = probe.clone();
    cmvn.apply(&mut probe);
    // The output must differ from the input (CMVN actually applied).
    assert_ne!(probe, baseline);
    // Every entry must be finite (no NaN / Inf leaks).
    for v in &probe {
      assert!(v.is_finite(), "non-finite CMVN output: {v}");
    }
  }

  #[test]
  fn rejects_truncated_input() {
    let bytes = b"\x00BDM ";
    assert!(matches!(
      Cmvn::from_ark_bytes(bytes),
      Err(Error::InvalidCmvn { .. })
    ));
  }

  #[test]
  fn rejects_wrong_magic() {
    let mut bytes = b"\x00BDM \x04\x02\x00\x00\x00\x04\x51\x00\x00\x00".to_vec();
    bytes[0] = 0xFF;
    assert!(matches!(
      Cmvn::from_ark_bytes(&bytes),
      Err(Error::InvalidCmvn { reason: r }) if r.contains("magic")
    ));
  }

  #[test]
  fn apply_subtracts_mean_and_divides_by_std() {
    let cmvn = Cmvn::from_components(vec![1.0; NUM_MEL_BINS], vec![2.0; NUM_MEL_BINS]);
    let mut feature = vec![3.0f32; NUM_MEL_BINS];
    cmvn.apply(&mut feature);
    for value in &feature {
      assert!((*value - 4.0).abs() < f32::EPSILON);
    }
  }

  #[test]
  fn povey_window_endpoints_are_zero_and_centre_is_one() {
    let w = build_povey_window();
    assert_eq!(w.len(), FRAME_LENGTH_SAMPLES);
    assert!(w[0].abs() < 1e-6);
    assert!(w[FRAME_LENGTH_SAMPLES - 1].abs() < 1e-6);
    let centre = (FRAME_LENGTH_SAMPLES - 1) / 2;
    assert!(
      (w[centre] - 1.0).abs() < 1e-3,
      "centre weight = {}",
      w[centre]
    );
  }

  #[test]
  fn mel_filters_cover_the_target_frequency_range() {
    let filters = build_mel_filters();
    assert_eq!(filters.len(), NUM_MEL_BINS);

    // The first filter should start at a low FFT bin.
    let first_centre_hz = fft_bin_hz(filters[0].start_bin + filters[0].weights.len() / 2);
    assert!(first_centre_hz > MEL_LOW_FREQ_HZ);
    assert!(first_centre_hz < 200.0);

    // The last filter should reach close to Nyquist.
    let last = &filters[NUM_MEL_BINS - 1];
    let last_max_bin = last.start_bin + last.weights.len();
    assert!(fft_bin_hz(last_max_bin) > 7_000.0);
  }

  #[test]
  fn mel_filterbank_silence_produces_log_floor_features() {
    let mut bank = MelFilterbank::new();
    let window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    let mut out = vec![0.0f32; NUM_MEL_BINS];
    bank.extract(&window, &mut out);
    let log_floor = LOG_FLOOR.ln();
    for v in &out {
      assert!(
        (*v - log_floor).abs() < 1e-3,
        "expected log_floor, got {}",
        v
      );
    }
  }

  #[test]
  fn mel_filterbank_responds_to_a_pure_tone() {
    let mut bank = MelFilterbank::new();
    let mut window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    // 1 kHz sinusoid at int16-range amplitude.
    let f = 1_000.0f32;
    let amp = 8_000.0f32;
    for (n, w) in window.iter_mut().enumerate() {
      *w = amp * (std::f32::consts::TAU * f * (n as f32) / SAMPLE_RATE_HZ as f32).sin();
    }
    let mut out = vec![0.0f32; NUM_MEL_BINS];
    bank.extract(&window, &mut out);

    // The peak Mel bin should sit somewhere in the lower half of the bank
    // (mel index for 1 kHz is ~28 with these parameters).
    let max_bin = (0..NUM_MEL_BINS)
      .max_by(|a, b| out[*a].partial_cmp(&out[*b]).unwrap())
      .unwrap();
    assert!((20..40).contains(&max_bin), "peak Mel bin = {max_bin}");
  }

  const BUNDLED_CMVN_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/cmvn.ark"));

  #[test]
  fn feature_extractor_buffers_partial_frames() {
    let mut fx = FeatureExtractor::new(BUNDLED_CMVN_BYTES).expect("init");
    fx.push_pcm(&vec![0.0; 100]);
    assert!(!fx.has_full_window());
    assert_eq!(fx.pending_samples(), 100);

    fx.push_pcm(&vec![0.0; 300]);
    assert!(fx.has_full_window());
    assert_eq!(fx.pending_samples(), 400);

    let mut out = vec![0.0; NUM_MEL_BINS];
    fx.extract_one(&mut out);
    // After consuming one frame, 240 samples (15 ms overlap) remain.
    assert_eq!(fx.pending_samples(), 240);
  }

  #[test]
  fn feature_extractor_emits_consistent_features_for_silence() {
    let mut fx = FeatureExtractor::new(BUNDLED_CMVN_BYTES).expect("init");
    fx.push_pcm(&vec![0.0; FRAME_LENGTH_SAMPLES + 3 * FRAME_SHIFT_SAMPLES]);

    let mut a = vec![0.0; NUM_MEL_BINS];
    let mut b = vec![0.0; NUM_MEL_BINS];
    fx.extract_one(&mut a);
    fx.extract_one(&mut b);
    assert_eq!(
      a, b,
      "two consecutive silence frames must produce identical features"
    );
  }

  #[test]
  fn feature_extractor_reset_clears_pending() {
    let mut fx = FeatureExtractor::new(BUNDLED_CMVN_BYTES).expect("init");
    fx.push_pcm(&vec![0.0; 100]);
    fx.reset();
    assert_eq!(fx.pending_samples(), 0);
    assert!(!fx.has_full_window());
  }
}
