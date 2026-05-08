//! Pure-Rust Kaldi-compatible Mel filterbank + CMVN preprocessing.
//!
//! All public types here are `pub(crate)` — feature extraction is an
//! implementation detail of [`crate::Vad`].

use crate::error::{Error, Result};

/// Number of Mel filterbank bins the model expects.
pub(crate) const NUM_MEL_BINS: usize = 80;

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
}
