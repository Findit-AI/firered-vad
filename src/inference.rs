//! ONNX Runtime wrapper for the FireRedVAD streaming model.

use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::{
  error::{Error, Result},
  features::NUM_MEL_BINS,
  options::SessionOptions,
};

const FEAT_NAME: &str = "feat";
const CACHES_IN_NAME: &str = "caches_in";
const PROBS_NAME: &str = "probs";
const CACHES_OUT_NAME: &str = "caches_out";

/// Number of cache slots (8 DFSMN blocks).
pub(crate) const CACHE_BLOCKS: usize = 8;
/// Cache channel dimension.
pub(crate) const CACHE_CHANNELS: usize = 128;
/// Cache time dimension.
pub(crate) const CACHE_TIME: usize = 19;
/// Total cache f32 count: `8 * 1 * 128 * 19 = 19_456`.
pub(crate) const CACHE_TOTAL: usize = CACHE_BLOCKS * CACHE_CHANNELS * CACHE_TIME;

/// Wraps the ONNX session + reusable scratch buffers + the per-stream caches.
pub(crate) struct OrtRunner {
  inner: OrtSession,
  caches: Vec<f32>,
  feat_scratch: Vec<f32>,
  prob_scratch: Vec<f32>,
}

impl OrtRunner {
  /// Construct from in-memory model bytes.
  pub(crate) fn from_memory(model: &[u8], opts: &SessionOptions) -> Result<Self> {
    let session = OrtSession::builder()?
      .with_optimization_level(opts.optimization_level())
      .map_err(ort::Error::from)?
      .commit_from_memory(model)?;
    Ok(Self::from_ort_session(session))
  }

  /// Construct from an ONNX file on disk.
  pub(crate) fn from_file(path: impl AsRef<Path>, opts: &SessionOptions) -> Result<Self> {
    let path = path.as_ref();
    let session = OrtSession::builder()?
      .with_optimization_level(opts.optimization_level())
      .map_err(ort::Error::from)?
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self::from_ort_session(session))
  }

  /// Wrap an externally-built `ort::Session`. Caller is responsible for
  /// matching the model contract (`feat` + `caches_in` → `probs` +
  /// `caches_out`); the contract is asserted on first inference.
  pub(crate) fn from_ort_session(inner: OrtSession) -> Self {
    // Pre-size the scratch buffers for a realistic streaming batch
    // (10 frames × 80 mel bins ≈ 100 ms of audio at 10 ms hop). This
    // moves the first-frame allocation into construction. Buffers
    // grow if larger pushes arrive; the pre-size is a hint, not a cap.
    const PRESIZED_FRAMES: usize = 10;
    Self {
      inner,
      caches: vec![0.0f32; CACHE_TOTAL],
      feat_scratch: Vec::with_capacity(PRESIZED_FRAMES * NUM_MEL_BINS),
      prob_scratch: Vec::with_capacity(PRESIZED_FRAMES),
    }
  }

  /// Reset the per-stream cache to zero AND drop any in-flight
  /// scratch state.
  ///
  /// Clearing `feat_scratch` and `prob_scratch` matters when `reset()`
  /// is called mid-batch — i.e. between `push_feature` calls and
  /// `infer`. Without this, stale features from the previous stream
  /// would be processed by the next `infer()` call and corrupt the
  /// new stream's first inference.
  pub(crate) fn reset(&mut self) {
    self.caches.fill(0.0);
    self.feat_scratch.clear();
    self.prob_scratch.clear();
  }

  /// Number of frames currently buffered in `feat_scratch`. Always a
  /// multiple of `NUM_MEL_BINS`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn pending_feature_frames(&self) -> usize {
    self.feat_scratch.len() / NUM_MEL_BINS
  }

  /// Append one 80-dim feature into the input scratch.
  pub(crate) fn push_feature(&mut self, feature: &[f32]) {
    debug_assert_eq!(feature.len(), NUM_MEL_BINS);
    self.feat_scratch.extend_from_slice(feature);
  }

  /// Run the model on every buffered feature frame at once. Updates the
  /// cache in place. Returns a slice of `T` raw probabilities into
  /// `prob_scratch`. Empty if no features are buffered.
  pub(crate) fn infer(&mut self) -> Result<&[f32]> {
    let n = self.pending_feature_frames();
    self.prob_scratch.clear();
    if n == 0 {
      return Ok(&self.prob_scratch);
    }

    let outputs = self.inner.run(ort::inputs![
      FEAT_NAME => TensorRef::from_array_view((
        [1usize, n, NUM_MEL_BINS],
        self.feat_scratch.as_slice(),
      ))?,
      CACHES_IN_NAME => TensorRef::from_array_view((
        [CACHE_BLOCKS, 1usize, CACHE_CHANNELS, CACHE_TIME],
        self.caches.as_slice(),
      ))?,
    ])?;

    let (probs_shape, probs_data) = outputs[PROBS_NAME].try_extract_tensor::<f32>()?;
    validate_shape(PROBS_NAME, probs_shape.as_ref(), &[1, n as i64, 1])?;

    let (caches_shape, caches_data) = outputs[CACHES_OUT_NAME].try_extract_tensor::<f32>()?;
    validate_shape(
      CACHES_OUT_NAME,
      caches_shape.as_ref(),
      &[
        CACHE_BLOCKS as i64,
        1,
        CACHE_CHANNELS as i64,
        CACHE_TIME as i64,
      ],
    )?;

    // Clamp model output into [0, 1]. The bundled FireRedVAD model is
    // sigmoid-terminated and produces values in range, but custom
    // models passed via `from_ort_session` could violate that. Clamp
    // here so the postprocessor's threshold comparison and the
    // smoothing accumulator never see garbage. NaN propagates as 0.0
    // (mirrors `options::sanitize_probability`).
    self.prob_scratch.reserve(probs_data.len());
    for &p in probs_data {
      let clamped = if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 };
      self.prob_scratch.push(clamped);
    }
    self.caches.copy_from_slice(caches_data);

    self.feat_scratch.clear();

    Ok(&self.prob_scratch)
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn validate_shape(tensor: &'static str, actual: &[i64], expected: &[i64]) -> Result<()> {
  if actual == expected {
    Ok(())
  } else {
    Err(Error::UnexpectedOutputShape {
      tensor,
      shape: actual.to_vec(),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const BUNDLED_MODEL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/fireredvad_stream_vad_with_cache.onnx"
  ));

  #[test]
  fn infer_with_no_pending_features_returns_empty_slice() {
    let mut runner = OrtRunner::from_memory(BUNDLED_MODEL, &SessionOptions::default())
      .expect("load bundled model");
    let probs = runner.infer().expect("infer");
    assert!(probs.is_empty());
  }

  #[test]
  fn infer_with_one_silence_frame_returns_one_prob() {
    let mut runner = OrtRunner::from_memory(BUNDLED_MODEL, &SessionOptions::default())
      .expect("load bundled model");
    let silence = vec![-15.0f32; NUM_MEL_BINS]; // approximate post-CMVN silence value
    runner.push_feature(&silence);
    let probs = runner.infer().expect("infer").to_vec();
    assert_eq!(probs.len(), 1);
    assert!(probs[0] >= 0.0 && probs[0] <= 1.0);
  }

  #[test]
  fn infer_advances_internal_cache() {
    let mut runner = OrtRunner::from_memory(BUNDLED_MODEL, &SessionOptions::default())
      .expect("load bundled model");
    let silence = vec![0.0f32; NUM_MEL_BINS];
    let initial = runner.caches.clone();
    runner.push_feature(&silence);
    runner.infer().expect("infer");
    assert_ne!(
      initial, runner.caches,
      "caches should change after one inference"
    );
  }

  #[test]
  fn reset_zeroes_caches_without_clearing_feat_scratch() {
    let mut runner = OrtRunner::from_memory(BUNDLED_MODEL, &SessionOptions::default())
      .expect("load bundled model");
    let silence = vec![0.0f32; NUM_MEL_BINS];
    runner.push_feature(&silence);
    runner.infer().expect("infer");
    runner.reset();
    assert!(runner.caches.iter().all(|v| *v == 0.0));
  }
}
