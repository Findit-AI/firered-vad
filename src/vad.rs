//! The Sans-I/O `Vad` engine.

use std::collections::VecDeque;
use std::path::Path;

use crate::detector::Postprocessor;
use crate::error::Result;
use crate::event::VadEvent;
use crate::features::{FeatureExtractor, NUM_MEL_BINS};
use crate::inference::OrtRunner;
use crate::options::VadOptions;

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
pub const BUNDLED_CMVN: &[u8] = include_bytes!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/models/cmvn.ark"
));

/// Streaming Voice Activity Detector for the FireRedVAD model.
///
/// `Vad` is a Sans-I/O state machine: callers push 16 kHz f32 PCM in
/// `[-1.0, 1.0]` via [`Self::push_samples`] and pull
/// [`VadEvent`]s via [`Self::poll_event`]. See the crate-level docs
/// for the canonical streaming loop.
pub struct Vad {
  runner: OrtRunner,
  features: FeatureExtractor,
  detector: Postprocessor,
  events: VecDeque<VadEvent>,
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
    let cmvn_bytes = std::fs::read(cmvn.as_ref()).map_err(|source| {
      crate::error::Error::LoadCmvn { path: cmvn.as_ref().to_path_buf(), source }
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
      events: VecDeque::new(),
      feature_scratch: vec![0.0; NUM_MEL_BINS],
      finished: false,
    })
  }

  // ── Sans-I/O surface ─────────────────────────────────────────────────

  /// Push 16 kHz f32 PCM. Newly produced events are queued for `poll_event`.
  pub fn push_samples(&mut self, pcm: &[f32]) -> Result<()> {
    self.features.push_pcm(pcm);
    while self.features.has_full_window() {
      self.features.extract_one(&mut self.feature_scratch);
      self.runner.push_feature(&self.feature_scratch);
    }
    if self.runner.pending_feature_frames() == 0 {
      return Ok(());
    }
    let probs: Vec<f32> = self.runner.infer()?.to_vec();
    for prob in probs {
      let (frame_result, segment) = self.detector.push_probability(prob);
      self.events.push_back(VadEvent::Frame(frame_result));
      if let Some(s) = segment {
        self.events.push_back(VadEvent::SegmentClosed(s));
      }
    }
    Ok(())
  }

  /// Mark end-of-stream. Closes any currently open segment.
  pub fn finish(&mut self) -> Result<()> {
    self.finished = true;
    if let Some(segment) = self.detector.finish_active() {
      self.events.push_back(VadEvent::SegmentClosed(segment));
    }
    Ok(())
  }

  /// Pull the next queued event; `None` once the queue is empty.
  pub fn poll_event(&mut self) -> Option<VadEvent> {
    self.events.pop_front()
  }

  /// Drain the queue through a closure (thin convenience over `poll_event`).
  pub fn drain_events<F>(&mut self, mut f: F)
  where
    F: FnMut(VadEvent),
  {
    while let Some(event) = self.events.pop_front() {
      f(event);
    }
  }

  /// Reset all per-stream state (caches, smoothing, state machine, queue,
  /// frame counters). Re-uses the underlying `ort::Session`.
  pub fn reset(&mut self) {
    self.runner.reset();
    self.features.reset();
    self.detector.reset();
    self.events.clear();
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
  pub fn frame_count(&self) -> u64 {
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

  /// Number of events in the queue awaiting `poll_event`.
  pub fn pending_events(&self) -> usize {
    self.events.len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::event::VadEvent;

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
    vad.drain_events(|ev| {
      if matches!(ev, VadEvent::SegmentClosed(_)) {
        segments += 1;
      }
    });
    assert_eq!(segments, 0);
    assert!(!vad.is_active());
  }

  #[cfg(feature = "bundled")]
  #[test]
  fn reset_clears_event_queue_and_frame_counter() {
    let mut vad = Vad::bundled().expect("bundled");
    vad.push_samples(&vec![0.0; 1_600]).expect("push 100ms");
    vad.reset();
    assert_eq!(vad.frame_count(), 0);
    assert_eq!(vad.pending_events(), 0);
    assert_eq!(vad.pending_samples(), 0);
    assert!(!vad.is_finished());
  }

  #[cfg(feature = "bundled")]
  #[test]
  fn finish_marks_finished_and_flushes_no_segment_when_idle() {
    let mut vad = Vad::bundled().expect("bundled");
    vad.finish().expect("finish");
    assert!(vad.is_finished());
    let mut segments = 0usize;
    vad.drain_events(|ev| {
      if matches!(ev, VadEvent::SegmentClosed(_)) {
        segments += 1;
      }
    });
    assert_eq!(segments, 0);
  }

  #[cfg(feature = "bundled")]
  #[test]
  fn push_samples_emits_one_frame_event_per_full_10ms_frame() {
    let mut vad = Vad::bundled().expect("bundled");
    // Need 25 ms (400 samples) to produce the FIRST frame; subsequent
    // frames need only 10 ms (160 samples) each. 5*160 + 240 = 1040 samples.
    vad.push_samples(&vec![0.0; 1040]).expect("push samples");
    let mut frames = 0usize;
    vad.drain_events(|ev| {
      if matches!(ev, VadEvent::Frame(_)) {
        frames += 1;
      }
    });
    assert_eq!(frames, 5);
  }
}
