//! Pure-Rust Kaldi-compatible Mel filterbank + CMVN preprocessing.
//!
//! All public types here are `pub(crate)` — feature extraction is an
//! implementation detail of [`crate::Vad`].

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
const PRE_EMPHASIS: f32 = 0.97;

/// Mel-bin range (low_freq=20, high_freq=Nyquist for 16 kHz).
const MEL_LOW_FREQ_HZ: f32 = 20.0;
const MEL_HIGH_FREQ_HZ: f32 = 8_000.0;

/// Floor for the log of bin energies (Kaldi `log_floor`).
const LOG_FLOOR: f32 = 1e-20;

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
/// pre-emphasis 0.97, DC removal on, snip_edges=true, log floor 1e-20.
pub(crate) struct MelFilterbank {
  fft: rustfft::algorithm::Radix4<f32>,
  fft_buf: Vec<rustfft::num_complex::Complex<f32>>,
  povey_window: Vec<f32>,
  filters: Vec<MelFilter>,
}

/// Cepstral Mean and Variance Normalization stats parsed from a Kaldi
/// `.ark` file. The 80-dim means and inverse-std-variances are applied
/// to each Mel-fbank feature vector before it is fed to the model.
#[derive(Debug, Clone)]
pub(crate) struct Cmvn {
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
        Err(Error::InvalidCmvn { reason: "truncated header" })
      } else {
        Ok(())
      }
    };

    need(i, 2)?;
    if &bytes[i..i + 2] != b"\x00B" {
      return Err(Error::InvalidCmvn { reason: "missing \\0B magic" });
    }
    i += 2;

    need(i, 3)?;
    if &bytes[i..i + 3] != b"DM " {
      return Err(Error::InvalidCmvn { reason: "expected double-matrix marker 'DM '" });
    }
    i += 3;

    need(i, 1)?;
    if bytes[i] != 4 {
      return Err(Error::InvalidCmvn { reason: "expected 4-byte int32 size token before rows" });
    }
    i += 1;
    need(i, 4)?;
    let rows = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    i += 4;
    if rows != 2 {
      return Err(Error::InvalidCmvn { reason: "expected exactly 2 rows (sums, sum_sqs)" });
    }

    need(i, 1)?;
    if bytes[i] != 4 {
      return Err(Error::InvalidCmvn { reason: "expected 4-byte int32 size token before cols" });
    }
    i += 1;
    need(i, 4)?;
    let cols = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    i += 4;
    if cols != (NUM_MEL_BINS as i32) + 1 {
      return Err(Error::InvalidCmvn { reason: "expected NUM_MEL_BINS + 1 columns" });
    }

    let total = (rows as usize) * (cols as usize) * 8;
    need(i, total)?;
    let mut data = Vec::with_capacity((rows as usize) * (cols as usize));
    let mut p = i;
    for _ in 0..(rows as usize) * (cols as usize) {
      let chunk = [
        bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3],
        bytes[p + 4], bytes[p + 5], bytes[p + 6], bytes[p + 7],
      ];
      data.push(f64::from_le_bytes(chunk));
      p += 8;
    }

    let count = data[NUM_MEL_BINS]; // first row, last column
    if !(count.is_finite() && count >= 1.0) {
      return Err(Error::InvalidCmvn { reason: "non-positive CMVN count" });
    }

    let mut means = Vec::with_capacity(NUM_MEL_BINS);
    let mut inverse_std_variances = Vec::with_capacity(NUM_MEL_BINS);
    let row_stride = cols as usize;
    for d in 0..NUM_MEL_BINS {
      let sum = data[d];
      let sum_sq = data[row_stride + d];
      let mean = sum / count;
      let mut var = sum_sq / count - mean * mean;
      if var < 1e-20 {
        var = 1e-20;
      }
      let istd = 1.0 / var.sqrt();
      means.push(mean as f32);
      inverse_std_variances.push(istd as f32);
    }

    Ok(Self { means, inverse_std_variances })
  }

  /// Apply CMVN in place to one 80-dim feature vector.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn apply(&self, feature: &mut [f32]) {
    debug_assert_eq!(feature.len(), NUM_MEL_BINS);
    for d in 0..NUM_MEL_BINS {
      feature[d] = (feature[d] - self.means[d]) * self.inverse_std_variances[d];
    }
  }

  pub(crate) fn means(&self) -> &[f32] {
    &self.means
  }

  pub(crate) fn inverse_std_variances(&self) -> &[f32] {
    &self.inverse_std_variances
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn hz_to_mel(hz: f32) -> f32 {
  // Kaldi/HTK convention: 1127 * ln(1 + f/700)
  1127.0 * (1.0 + hz / 700.0).ln()
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn mel_to_hz(mel: f32) -> f32 {
  700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Centre frequency of the `i`-th non-redundant FFT bin in Hz.
#[cfg_attr(not(tarpaulin), inline(always))]
fn fft_bin_hz(i: usize) -> f32 {
  (i as f32) * (SAMPLE_RATE_HZ as f32) / (FFT_SIZE as f32)
}

fn build_povey_window() -> Vec<f32> {
  let n = FRAME_LENGTH_SAMPLES;
  let a = std::f32::consts::TAU / ((n - 1) as f32);
  (0..n)
    .map(|i| (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85))
    .collect()
}

fn build_mel_filters() -> Vec<MelFilter> {
  let mel_low = hz_to_mel(MEL_LOW_FREQ_HZ);
  let mel_high = hz_to_mel(MEL_HIGH_FREQ_HZ);
  let mel_step = (mel_high - mel_low) / (NUM_MEL_BINS as f32 + 1.0);

  // The (NUM_MEL_BINS + 2) Mel-frequency anchor points spanning the band:
  // `points[b]` is the left edge of filter b-1, the centre of filter b, and
  // the right edge of filter b+1 (for the matching `b` indices).
  let mut hz_points = Vec::with_capacity(NUM_MEL_BINS + 2);
  for k in 0..(NUM_MEL_BINS + 2) {
    hz_points.push(mel_to_hz(mel_low + (k as f32) * mel_step));
  }

  let mut filters = Vec::with_capacity(NUM_MEL_BINS);
  for b in 0..NUM_MEL_BINS {
    let left = hz_points[b];
    let centre = hz_points[b + 1];
    let right = hz_points[b + 2];

    // Find the FFT bin index range that overlaps [left, right].
    let mut start_bin = FFT_BINS;
    let mut end_bin = 0;
    for i in 0..FFT_BINS {
      let f = fft_bin_hz(i);
      if f > left && f < right {
        if i < start_bin {
          start_bin = i;
        }
        end_bin = i;
      }
    }

    let mut weights = Vec::new();
    if start_bin <= end_bin {
      weights.reserve(end_bin - start_bin + 1);
      for i in start_bin..=end_bin {
        let f = fft_bin_hz(i);
        let w = if f <= centre {
          (f - left) / (centre - left)
        } else {
          (right - f) / (right - centre)
        };
        weights.push(w.max(0.0));
      }
    } else {
      // Filter band falls between FFT bins (very narrow band at low Mel
      // indices); leave it empty — Kaldi behaves the same.
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
    }
  }

  /// Extract one 80-dim log-Mel feature from a 25 ms window of int16-range
  /// samples. The input is mutated in place (DC removal, pre-emphasis,
  /// windowing happen on a copy inside the FFT buffer; the caller's slice
  /// is **not** mutated).
  pub(crate) fn extract(&mut self, window: &[f32], out: &mut [f32]) {
    debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
    debug_assert_eq!(out.len(), NUM_MEL_BINS);

    // 1. Copy + remove DC offset.
    let mean: f32 = window.iter().copied().sum::<f32>() / FRAME_LENGTH_SAMPLES as f32;
    let mut samples: [f32; FRAME_LENGTH_SAMPLES] = [0.0; FRAME_LENGTH_SAMPLES];
    for i in 0..FRAME_LENGTH_SAMPLES {
      samples[i] = window[i] - mean;
    }

    // 2. Pre-emphasis: x[i] -= 0.97 * x[i-1] for i = N-1..1; then x[0] -= 0.97 * x[0].
    for i in (1..FRAME_LENGTH_SAMPLES).rev() {
      samples[i] -= PRE_EMPHASIS * samples[i - 1];
    }
    samples[0] -= PRE_EMPHASIS * samples[0];

    // 3. Window with Povey.
    for i in 0..FRAME_LENGTH_SAMPLES {
      samples[i] *= self.povey_window[i];
    }

    // 4. Zero-pad to FFT_SIZE and run the radix-2 FFT.
    for i in 0..FFT_SIZE {
      let re = if i < FRAME_LENGTH_SAMPLES { samples[i] } else { 0.0 };
      self.fft_buf[i].re = re;
      self.fft_buf[i].im = 0.0;
    }
    use rustfft::Fft;
    self.fft.process(&mut self.fft_buf);

    // 5. Power spectrum (|X|^2) for the non-redundant half.
    let mut power: [f32; FFT_BINS] = [0.0; FFT_BINS];
    for i in 0..FFT_BINS {
      let c = self.fft_buf[i];
      power[i] = c.re * c.re + c.im * c.im;
    }

    // 6. Mel filterbank → log.
    for b in 0..NUM_MEL_BINS {
      let f = &self.filters[b];
      let mut energy = 0.0f32;
      for (j, w) in f.weights.iter().enumerate() {
        energy += power[f.start_bin + j] * *w;
      }
      out[b] = energy.max(LOG_FLOOR).ln();
    }
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
    assert_eq!(cmvn.means().len(), NUM_MEL_BINS);
    assert_eq!(cmvn.inverse_std_variances().len(), NUM_MEL_BINS);
    // Means should be roughly in log-mel-energy range; pin the first one so
    // future regressions in parsing immediately surface.
    let first_mean = cmvn.means()[0];
    assert!(first_mean > 5.0 && first_mean < 20.0, "first mean = {first_mean}");
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
    let cmvn = Cmvn {
      means: vec![1.0; NUM_MEL_BINS],
      inverse_std_variances: vec![2.0; NUM_MEL_BINS],
    };
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
    assert!((w[centre] - 1.0).abs() < 1e-3, "centre weight = {}", w[centre]);
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
      assert!((*v - log_floor).abs() < 1e-3, "expected log_floor, got {}", v);
    }
  }

  #[test]
  fn mel_filterbank_responds_to_a_pure_tone() {
    let mut bank = MelFilterbank::new();
    let mut window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    // 1 kHz sinusoid at int16-range amplitude.
    let f = 1_000.0f32;
    let amp = 8_000.0f32;
    for n in 0..FRAME_LENGTH_SAMPLES {
      window[n] = amp * (std::f32::consts::TAU * f * (n as f32) / SAMPLE_RATE_HZ as f32).sin();
    }
    let mut out = vec![0.0f32; NUM_MEL_BINS];
    bank.extract(&window, &mut out);

    // The peak Mel bin should sit somewhere in the lower half of the bank
    // (mel index for 1 kHz is ~28 with these parameters).
    let max_bin = (0..NUM_MEL_BINS).max_by(|a, b| out[*a].partial_cmp(&out[*b]).unwrap()).unwrap();
    assert!((20..40).contains(&max_bin), "peak Mel bin = {max_bin}");
  }
}
