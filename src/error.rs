//! Error type for the `firered-vad` crate.

use std::path::PathBuf;

/// Errors returned by the `firered-vad` crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// Failed to load an ONNX model from disk.
  #[error("failed to load model from {path}: {source}")]
  LoadModel {
    /// Path that was being loaded.
    path: PathBuf,
    /// Underlying ONNX Runtime error.
    source: ort::Error,
  },

  /// An ONNX Runtime error not specific to model loading.
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// Failed to read a CMVN file from disk.
  #[error("failed to read CMVN file from {path}: {source}")]
  LoadCmvn {
    /// Path that was being loaded.
    path: PathBuf,
    /// Underlying I/O error.
    source: std::io::Error,
  },

  /// The CMVN bytes were not in the expected Kaldi binary format.
  #[error("invalid CMVN format: {reason}")]
  InvalidCmvn {
    /// Human-readable reason describing what failed to parse.
    reason: &'static str,
  },

  /// The caller pushed PCM at a sample rate the model does not support.
  #[error("input PCM sample rate is unsupported (model is fixed at {expected} Hz)")]
  UnsupportedSampleRate {
    /// The sample rate the model expects (always 16_000 for FireRedVAD).
    expected: u32,
  },

  /// An ONNX output tensor had an unexpected shape.
  #[error("ONNX output {tensor} had unexpected shape {shape:?}")]
  UnexpectedOutputShape {
    /// The output tensor name (e.g. `"probs"`, `"caches_out"`).
    tensor: &'static str,
    /// The actual shape returned by the ONNX runtime.
    shape: Vec<i64>,
  },

  /// A speech-threshold value was outside the valid `[0, 1]` range and could not be sanitized.
  #[error("invalid speech threshold {value} (must be in [0, 1])")]
  InvalidSpeechThreshold {
    /// The offending value.
    value: f32,
  },
}

/// Convenience alias for `Result<T, firered_vad::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn unsupported_sample_rate_displays_expected_value() {
    let err = Error::UnsupportedSampleRate { expected: 16_000 };
    assert_eq!(
      err.to_string(),
      "input PCM sample rate is unsupported (model is fixed at 16000 Hz)"
    );
  }

  #[test]
  fn invalid_cmvn_carries_static_reason() {
    let err = Error::InvalidCmvn {
      reason: "missing magic",
    };
    assert!(err.to_string().contains("missing magic"));
  }

  #[test]
  fn unexpected_output_shape_renders_shape() {
    let err = Error::UnexpectedOutputShape {
      tensor: "probs",
      shape: vec![1, 2, 3],
    };
    assert!(err.to_string().contains("probs"));
    assert!(err.to_string().contains("[1, 2, 3]"));
  }
}
