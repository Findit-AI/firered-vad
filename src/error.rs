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

  /// An ONNX output tensor had an unexpected shape.
  #[error("ONNX output {tensor} had unexpected shape {shape:?}")]
  UnexpectedOutputShape {
    /// The output tensor name (e.g. `"probs"`, `"caches_out"`).
    tensor: &'static str,
    /// The actual shape returned by the ONNX runtime.
    shape: Vec<i64>,
  },
}

/// Convenience alias for `Result<T, firered_vad::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
  use super::*;

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
