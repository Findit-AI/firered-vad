//! The Sans-I/O `Vad` engine.

use std::{collections::VecDeque, path::Path};

use crate::{
  detector::Postprocessor,
  error::Result,
  event::SpeechSegment,
  features::{FeatureExtractor, NUM_MEL_BINS},
  inference::OrtRunner,
  options::VadOptions,
};

/// Bundled FireRedVAD streaming ONNX (Apache-2.0; see `THIRD_PARTY_NOTICES.md`).
#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub const BUNDLED_MODEL: &[u8] = include_bytes!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/models/fireredvad_stream_vad_with_cache.onnx"
));

/// Bundled CMVN stats (Apache-2.0).
#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub const BUNDLED_CMVN: &[u8] =
  include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/cmvn.ark"));

/// Streaming Voice Activity Detector for the FireRedVAD model.
///
/// `Vad` is a Sans-I/O state machine: callers push 16 kHz f32 PCM in
/// `[-1.0, 1.0]` via [`Self::push_samples`], which returns the next
/// available closed [`SpeechSegment`] (or `None`). See the crate-level
/// docs for the canonical streaming loop.
pub struct Vad {
  runner: OrtRunner,
  features: FeatureExtractor,
  detector: Postprocessor,
  pending_segments: VecDeque<SpeechSegment>,
  feature_scratch: Vec<f32>,
  finished: bool,
}

impl Vad {
  // ── Construction ─────────────────────────────────────────────────────

  /// Construct from the bundled ONNX model + CMVN with default `VadOptions`.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn bundled() -> Result<Self> {
    Self::bundled_with(VadOptions::default())
  }

  /// Construct from the bundled artifacts with custom `VadOptions`.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn bundled_with(options: VadOptions) -> Result<Self> {
    Self::from_memory_with_cmvn(BUNDLED_MODEL, BUNDLED_CMVN, options)
  }

  /// Construct from in-memory model bytes + bundled CMVN with default options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn from_memory(model: &[u8]) -> Result<Self> {
    Self::from_memory_with(model, VadOptions::default())
  }

  /// Construct from in-memory model bytes + bundled CMVN with custom options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn from_memory_with(model: &[u8], options: VadOptions) -> Result<Self> {
    Self::from_memory_with_cmvn(model, BUNDLED_CMVN, options)
  }

  /// Construct from an ONNX file on disk + bundled CMVN with default options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn from_file(model: impl AsRef<Path>) -> Result<Self> {
    Self::from_file_with(model, VadOptions::default())
  }

  /// Construct from an ONNX file + bundled CMVN with custom options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn from_file_with(model: impl AsRef<Path>, options: VadOptions) -> Result<Self> {
    let runner = OrtRunner::from_file(model, options.session_options())?;
    Self::wrap(runner, BUNDLED_CMVN, options)
  }

  /// Construct with explicit model + CMVN bytes.
  pub fn from_memory_with_cmvn(model: &[u8], cmvn: &[u8], options: VadOptions) -> Result<Self> {
    let runner = OrtRunner::from_memory(model, options.session_options())?;
    Self::wrap(runner, cmvn, options)
  }

  /// Construct with explicit model file + CMVN file paths.
  pub fn from_file_with_cmvn(
    model: impl AsRef<Path>,
    cmvn: impl AsRef<Path>,
    options: VadOptions,
  ) -> Result<Self> {
    let runner = OrtRunner::from_file(model, options.session_options())?;
    let cmvn_bytes =
      std::fs::read(cmvn.as_ref()).map_err(|source| crate::error::Error::LoadCmvn {
        path: cmvn.as_ref().to_path_buf(),
        source,
      })?;
    Self::wrap(runner, &cmvn_bytes, options)
  }

  /// Wrap an externally built `ort::Session`. The session must implement
  /// the FireRedVAD streaming model contract.
  pub fn from_ort_session(
    session: ort::session::Session,
    cmvn: &[u8],
    options: VadOptions,
  ) -> Result<Self> {
    let runner = OrtRunner::from_ort_session(session);
    Self::wrap(runner, cmvn, options)
  }

  fn wrap(runner: OrtRunner, cmvn: &[u8], options: VadOptions) -> Result<Self> {
    let features = FeatureExtractor::new(cmvn)?;
    let detector = Postprocessor::new(options.clone());
    Ok(Self {
      runner,
      features,
      detector,
      pending_segments: VecDeque::new(),
      feature_scratch: vec![0.0; NUM_MEL_BINS],
      finished: false,
    })
  }

  // ── Sans-I/O surface ─────────────────────────────────────────────────

  /// Feed 16 kHz f32 PCM and return the next available closed segment.
  ///
  /// Returns `Ok(Some(segment))` when a segment is ready, `Ok(None)` when
  /// none is available yet. Pass an empty slice (`&[]`) to drain buffered
  /// segments without processing new PCM — useful when a single push
  /// closes more than one segment (rare but possible at force-split).
  pub fn push_samples(&mut self, pcm: &[f32]) -> Result<Option<SpeechSegment>> {
    if !pcm.is_empty() {
      self.features.push_pcm(pcm);
      while self.features.has_full_window() {
        self.features.extract_one(&mut self.feature_scratch);
        self.runner.push_feature(&self.feature_scratch);
      }
      if self.runner.pending_feature_frames() > 0 {
        let probs: Vec<f32> = self.runner.infer()?.to_vec();
        for prob in probs {
          if let Some(segment) = self.detector.push_probability(prob) {
            self.pending_segments.push_back(segment);
          }
        }
      }
    }
    Ok(self.pending_segments.pop_front())
  }

  /// Mark end-of-stream. Returns the trailing segment if one was open, or
  /// `None` when the stream ended in silence.
  ///
  /// Call `push_samples(&[])` after `finish` to drain any additionally
  /// buffered segments in the rare multi-segment case.
  pub fn finish(&mut self) -> Result<Option<SpeechSegment>> {
    self.finished = true;
    if let Some(segment) = self.detector.finish_active() {
      self.pending_segments.push_back(segment);
    }
    Ok(self.pending_segments.pop_front())
  }

  /// Reset all per-stream state (caches, smoothing, state machine, queue,
  /// frame counters). Re-uses the underlying `ort::Session`.
  pub fn reset(&mut self) {
    self.runner.reset();
    self.features.reset();
    self.detector.reset();
    self.pending_segments.clear();
    self.finished = false;
  }

  // ── Inspection ───────────────────────────────────────────────────────

  /// Currently active options.
  pub const fn options(&self) -> &VadOptions {
    self.detector_options()
  }

  // Tiny helper so `options()` stays `const fn` — VecDeque::is_empty is
  // const since 1.71 but we need a `&VadOptions` borrow that the borrow
  // checker can prove without surfacing the private detector field.
  const fn detector_options(&self) -> &VadOptions {
    self.detector.options_const()
  }

  /// Replace the options at runtime. In-flight detector state is preserved.
  pub fn set_options(&mut self, options: VadOptions) {
    self.detector.set_options(options);
  }

  /// Total number of 10 ms frames consumed since the last reset.
  pub const fn frame_count(&self) -> u64 {
    self.detector.frame_count()
  }

  /// Number of int16-range PCM samples buffered awaiting the next frame.
  pub fn pending_samples(&self) -> usize {
    self.features.pending_samples()
  }

  /// Whether the postprocessor is currently inside a SPEECH or POSSIBLE_SILENCE state.
  pub fn is_active(&self) -> bool {
    self.detector.is_active()
  }

  /// Whether [`Self::finish`] has been called.
  pub const fn is_finished(&self) -> bool {
    self.finished
  }

  /// Number of buffered segments awaiting drain via `push_samples(&[])`.
  pub fn pending_segments(&self) -> usize {
    self.pending_segments.len()
  }
}

#[cfg(test)]
mod tests {
  #[allow(unused_imports)]
  use super::*;

  #[cfg(feature = "bundled")]
  #[test]
  fn bundled_constructs_with_defaults() {
    let _ = Vad::bundled().expect("bundled constructs");
  }

  #[cfg(feature = "bundled")]
  #[test]
  fn one_second_of_silence_emits_no_segment() {
    let mut vad = Vad::bundled().expect("bundled constructs");
    vad.push_samples(&vec![0.0; 16_000]).expect("push silence");
    let mut segments = 0usize;
    while vad.push_samples(&[]).expect("drain").is_some() {
      segments += 1;
    }
    assert_eq!(segments, 0);
    assert!(!vad.is_active());
  }

  #[cfg(feature = "bundled")]
  #[test]
  fn reset_clears_segment_queue_and_frame_counter() {
    let mut vad = Vad::bundled().expect("bundled");
    vad.push_samples(&vec![0.0; 1_600]).expect("push 100ms");
    vad.reset();
    assert_eq!(vad.frame_count(), 0);
    assert_eq!(vad.pending_segments(), 0);
    assert_eq!(vad.pending_samples(), 0);
    assert!(!vad.is_finished());
  }

  #[cfg(feature = "bundled")]
  #[test]
  fn finish_marks_finished_and_flushes_no_segment_when_idle() {
    let mut vad = Vad::bundled().expect("bundled");
    let result = vad.finish().expect("finish");
    assert!(vad.is_finished());
    assert!(result.is_none());
    let mut segments = 0usize;
    while vad.push_samples(&[]).expect("drain").is_some() {
      segments += 1;
    }
    assert_eq!(segments, 0);
  }
}
