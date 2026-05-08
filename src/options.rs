//! Configuration types for `firered-vad`.
//!
//! `VadOptions` controls postprocessor behavior; `SessionOptions` controls
//! the underlying ONNX Runtime session. `GraphOptimizationLevel` is
//! re-exported from `ort` so callers share vocabulary with everyone else
//! using the runtime directly.

use core::time::Duration;

pub use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
mod graph_optimization_level {
  use serde::*;

  use super::GraphOptimizationLevel;

  #[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
  )]
  #[serde(rename_all = "snake_case")]
  enum OptimizationLevel {
    Disable,
    Level1,
    Level2,
    #[default]
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptimizationLevel {
    #[inline]
    fn from(value: GraphOptimizationLevel) -> Self {
      match value {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptimizationLevel> for GraphOptimizationLevel {
    #[inline]
    fn from(value: OptimizationLevel) -> Self {
      match value {
        OptimizationLevel::Disable => Self::Disable,
        OptimizationLevel::Level1 => Self::Level1,
        OptimizationLevel::Level2 => Self::Level2,
        OptimizationLevel::Level3 => Self::Level3,
        OptimizationLevel::All => Self::All,
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn serialize<S>(level: &GraphOptimizationLevel, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptimizationLevel::from(*level).serialize(serializer)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn deserialize<'de, D>(deserializer: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptimizationLevel::deserialize(deserializer).map(Into::into)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default() -> GraphOptimizationLevel {
    GraphOptimizationLevel::Disable
  }
}

/// Options for constructing the ONNX session.
///
/// This stays small: deployment-specific knobs (intra-thread count,
/// inter-thread count, execution providers) belong one layer up and
/// should be applied to a manually built `ort::Session` passed into
/// [`crate::Vad::from_ort_session`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SessionOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "graph_optimization_level::default",
      with = "graph_optimization_level"
    )
  )]
  optimization_level: GraphOptimizationLevel,
}

impl Default for SessionOptions {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl SessionOptions {
  /// Create a new `SessionOptions` with default values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      optimization_level: GraphOptimizationLevel::Level3,
    }
  }

  /// Returns the graph optimization level used when constructing the ONNX session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }

  /// Set the graph optimization level (`&mut Self` for chaining).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = level;
    self
  }

  /// Builder variant of [`Self::set_optimization_level`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }
}

/// Frame shift in milliseconds for the FireRedVAD model (10 ms).
pub(crate) const FRAME_SHIFT_MS: u128 = 10;

/// Convert a `Duration` to whole 10-ms frames. Saturates at `u32::MAX`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn duration_to_frames(d: Duration) -> u32 {
  let frames = d.as_millis() / FRAME_SHIFT_MS;
  if frames > u32::MAX as u128 {
    u32::MAX
  } else {
    frames as u32
  }
}

/// Clamp a probability into `[0, 1]`, mapping non-finite values to 0.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn sanitize_probability(value: f32) -> f32 {
  if value.is_finite() {
    value.clamp(0.0, 1.0)
  } else {
    0.0
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_smooth_window_size() -> u32 {
  5
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_speech_threshold() -> f32 {
  0.5
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_pad_start() -> Duration {
  Duration::from_millis(50)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_speech_duration() -> Duration {
  Duration::from_millis(80)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_silence_duration() -> Duration {
  Duration::from_millis(200)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_max_speech_duration() -> Option<Duration> {
  Some(Duration::from_secs(20))
}

/// Configuration for turning streaming probabilities into speech segments.
///
/// Defaults reproduce upstream Python's `FireRedStreamVadConfig` exactly.
/// The four upstream "mode" presets are not exposed as an enum — see
/// the crate-level docs for the recipe values you can apply via the
/// `with_*` builders directly.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VadOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_smooth_window_size"))]
  smooth_window_size: u32,

  #[cfg_attr(feature = "serde", serde(default = "default_speech_threshold"))]
  speech_threshold: f32,

  #[cfg_attr(
    feature = "serde",
    serde(default = "default_pad_start", with = "humantime_serde")
  )]
  pad_start: Duration,

  #[cfg_attr(
    feature = "serde",
    serde(default = "default_min_speech_duration", with = "humantime_serde")
  )]
  min_speech_duration: Duration,

  #[cfg_attr(
    feature = "serde",
    serde(default = "default_min_silence_duration", with = "humantime_serde")
  )]
  min_silence_duration: Duration,

  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_max_speech_duration",
      with = "humantime_serde::option"
    )
  )]
  max_speech_duration: Option<Duration>,

  #[cfg_attr(feature = "serde", serde(default))]
  session_options: SessionOptions,
}

impl Default for VadOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl VadOptions {
  /// Create `VadOptions` with upstream `FireRedStreamVadConfig` defaults.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      smooth_window_size: default_smooth_window_size(),
      speech_threshold: default_speech_threshold(),
      pad_start: default_pad_start(),
      min_speech_duration: default_min_speech_duration(),
      min_silence_duration: default_min_silence_duration(),
      max_speech_duration: default_max_speech_duration(),
      session_options: SessionOptions::new(),
    }
  }

  /// Smoothing-window size in frames (10 ms each).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn smooth_window_size(&self) -> u32 {
    self.smooth_window_size
  }

  /// Set the smoothing-window size; returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_smooth_window_size(&mut self, size: u32) -> &mut Self {
    self.smooth_window_size = size;
    self
  }

  /// Builder variant of [`Self::set_smooth_window_size`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_smooth_window_size(mut self, size: u32) -> Self {
    self.smooth_window_size = size;
    self
  }

  /// Threshold above which a smoothed probability counts as speech.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn speech_threshold(&self) -> f32 {
    self.speech_threshold
  }

  /// Set the speech threshold; values are clamped into `[0, 1]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_speech_threshold(&mut self, t: f32) -> &mut Self {
    self.speech_threshold = sanitize_probability(t);
    self
  }

  /// Builder variant of [`Self::set_speech_threshold`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_speech_threshold(mut self, t: f32) -> Self {
    self.speech_threshold = sanitize_probability(t);
    self
  }

  /// Padding extending the start of every emitted speech segment backward.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pad_start(&self) -> Duration {
    self.pad_start
  }

  /// Set `pad_start`; returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_pad_start(&mut self, d: Duration) -> &mut Self {
    self.pad_start = d;
    self
  }

  /// Builder variant of [`Self::set_pad_start`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_pad_start(mut self, d: Duration) -> Self {
    self.pad_start = d;
    self
  }

  /// Minimum speech duration before a `POSSIBLE_SPEECH` run promotes to `SPEECH`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_speech_duration(&self) -> Duration {
    self.min_speech_duration
  }

  /// Set the minimum speech duration; returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_speech_duration(&mut self, d: Duration) -> &mut Self {
    self.min_speech_duration = d;
    self
  }

  /// Builder variant of [`Self::set_min_speech_duration`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_speech_duration(mut self, d: Duration) -> Self {
    self.min_speech_duration = d;
    self
  }

  /// Maximum speech duration before a force-split (None disables force-split).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_speech_duration(&self) -> Option<Duration> {
    self.max_speech_duration
  }

  /// Set the maximum speech duration; returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_speech_duration(&mut self, d: Duration) -> &mut Self {
    self.max_speech_duration = Some(d);
    self
  }

  /// Builder variant of [`Self::set_max_speech_duration`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_speech_duration(mut self, d: Duration) -> Self {
    self.max_speech_duration = Some(d);
    self
  }

  /// Disable max-speech force-splitting.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clear_max_speech_duration(mut self) -> Self {
    self.max_speech_duration = None;
    self
  }

  /// Minimum silence duration required to close an open speech segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_silence_duration(&self) -> Duration {
    self.min_silence_duration
  }

  /// Set the minimum silence duration; returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_silence_duration(&mut self, d: Duration) -> &mut Self {
    self.min_silence_duration = d;
    self
  }

  /// Builder variant of [`Self::set_min_silence_duration`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_silence_duration(mut self, d: Duration) -> Self {
    self.min_silence_duration = d;
    self
  }

  /// The session options used when constructing the ONNX runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn session_options(&self) -> &SessionOptions {
    &self.session_options
  }

  /// Set the `SessionOptions`; returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_session_options(&mut self, opts: SessionOptions) -> &mut Self {
    self.session_options = opts;
    self
  }

  /// Builder variant of [`Self::set_session_options`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_session_options(mut self, opts: SessionOptions) -> Self {
    self.session_options = opts;
    self
  }

  // ── Sample-domain conversions used by the postprocessor ───────────
  /// Smoothing-window size in frames.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn smooth_window_size_frames(&self) -> u32 {
    self.smooth_window_size
  }

  /// Pad-start in frames; clamped to be at least `smooth_window_size`,
  /// matching upstream `__init__`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn pad_start_frames(&self) -> u32 {
    let raw = duration_to_frames(self.pad_start);
    raw.max(self.smooth_window_size)
  }

  /// `min_speech_duration` in frames.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn min_speech_frames(&self) -> u32 {
    duration_to_frames(self.min_speech_duration)
  }

  /// `min_silence_duration` in frames.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn min_silence_frames(&self) -> u32 {
    duration_to_frames(self.min_silence_duration)
  }

  /// `max_speech_duration` in frames, if force-split is enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn max_speech_frames(&self) -> Option<u32> {
    self.max_speech_duration.map(duration_to_frames)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn session_options_default_optimizes_at_level_3() {
    let opts = SessionOptions::default();
    assert!(matches!(
      opts.optimization_level(),
      GraphOptimizationLevel::Level3
    ));
  }

  #[test]
  fn session_options_with_optimization_level_overrides() {
    let opts = SessionOptions::new().with_optimization_level(GraphOptimizationLevel::Level1);
    assert!(matches!(
      opts.optimization_level(),
      GraphOptimizationLevel::Level1
    ));
  }

  #[test]
  fn vad_options_default_match_upstream_firered_stream_vad_config() {
    let opts = VadOptions::default();
    assert_eq!(opts.smooth_window_size(), 5);
    assert!((opts.speech_threshold() - 0.5).abs() < f32::EPSILON);
    assert_eq!(opts.pad_start(), Duration::from_millis(50));
    assert_eq!(opts.min_speech_duration(), Duration::from_millis(80));
    assert_eq!(opts.max_speech_duration(), Some(Duration::from_secs(20)));
    assert_eq!(opts.min_silence_duration(), Duration::from_millis(200));
  }

  #[test]
  fn vad_options_speech_threshold_clamps_into_unit_interval() {
    let mut opts = VadOptions::new();
    opts.set_speech_threshold(2.5);
    assert!((opts.speech_threshold() - 1.0).abs() < f32::EPSILON);
    opts.set_speech_threshold(-0.3);
    assert!((opts.speech_threshold() - 0.0).abs() < f32::EPSILON);
    opts.set_speech_threshold(f32::NAN);
    assert!((opts.speech_threshold() - 0.0).abs() < f32::EPSILON);
  }

  #[test]
  fn vad_options_clear_max_speech_duration_disables_force_split() {
    let opts = VadOptions::new()
      .with_max_speech_duration(Duration::from_secs(5))
      .clear_max_speech_duration();
    assert_eq!(opts.max_speech_duration(), None);
    assert_eq!(opts.max_speech_frames(), None);
  }

  #[test]
  fn pad_start_frames_is_clamped_to_smooth_window_size() {
    let opts = VadOptions::new()
      .with_smooth_window_size(8)
      .with_pad_start(Duration::from_millis(30)); // 3 frames
    assert_eq!(opts.pad_start_frames(), 8);
  }

  #[test]
  fn duration_to_frames_truncates_partial_frames() {
    assert_eq!(duration_to_frames(Duration::from_millis(15)), 1);
    assert_eq!(duration_to_frames(Duration::from_millis(20)), 2);
    assert_eq!(duration_to_frames(Duration::ZERO), 0);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn vad_options_round_trip_through_humantime_serde() {
    let opts = VadOptions::new()
      .with_min_silence_duration(Duration::from_millis(250))
      .with_max_speech_duration(Duration::from_secs(15));
    let serialized = serde_json::to_string(&opts).expect("serialize");
    assert!(serialized.contains("250ms"));
    assert!(serialized.contains("15s"));
    let restored: VadOptions = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(restored.min_silence_duration(), opts.min_silence_duration());
    assert_eq!(restored.max_speech_duration(), opts.max_speech_duration());
  }
}
