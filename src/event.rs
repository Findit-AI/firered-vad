//! Public value types: [`SpeechSegment`] (the primary output) and
//! [`FrameResult`] (per-frame inspection data, available via
//! [`crate::Vad::recent_frames`]).

use core::{ops::Range, time::Duration};

/// One closed continuous human-speech window on the stream timeline.
///
/// Slice the original PCM with [`Self::range_usize`] to recover the
/// audio that triggered this segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
  start_sample: u64,
  end_sample: u64,
}

impl SpeechSegment {
  /// The sample rate every emitted segment is referenced against.
  pub const SAMPLE_RATE_HZ: u32 = 16_000;

  /// Construct a segment from absolute sample indices.
  ///
  /// `end_sample` is exclusive (the first sample *after* the segment).
  /// Public so it's easy to construct in tests; the `Vad` engine is the
  /// only producer in normal use.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(start_sample: u64, end_sample: u64) -> Self {
    Self {
      start_sample,
      end_sample,
    }
  }

  /// Absolute sample index where the segment starts (inclusive).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// Absolute sample index where the segment ends (exclusive).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Number of samples spanned by this segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_count(&self) -> u64 {
    self.end_sample.saturating_sub(self.start_sample)
  }

  /// Start time of the segment as a `Duration`.
  pub fn start(&self) -> Duration {
    Duration::from_secs_f64(self.start_sample as f64 / Self::SAMPLE_RATE_HZ as f64)
  }

  /// End time of the segment as a `Duration`.
  pub fn end(&self) -> Duration {
    Duration::from_secs_f64(self.end_sample as f64 / Self::SAMPLE_RATE_HZ as f64)
  }

  /// Duration of the segment.
  pub fn duration(&self) -> Duration {
    Duration::from_secs_f64(self.sample_count() as f64 / Self::SAMPLE_RATE_HZ as f64)
  }

  /// `Range<u64>` covering the segment, useful for arithmetic on absolute timelines.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn range(&self) -> Range<u64> {
    self.start_sample..self.end_sample
  }

  /// `Range<usize>` for slicing a `&[f32]` PCM buffer.
  ///
  /// On 64-bit targets this is identity; on 32-bit targets segment indices
  /// above `u32::MAX` saturate. Audio streams long enough to hit that
  /// limit (~37 hours at 16 kHz) are not a v1 concern.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn range_usize(&self) -> Range<usize> {
    (self.start_sample as usize)..(self.end_sample as usize)
  }
}

/// Per-frame view of the streaming detector's internal state.
///
/// One `FrameResult` is produced for every 10 ms frame consumed by
/// [`crate::Vad::push_samples`]. The slice from the most recent
/// non-empty `push_samples` call is accessible via
/// [`crate::Vad::recent_frames`].
///
/// Frame indices are 0-based; upstream Python uses 1-based indices and
/// we shift on construction. `speech_start_frame` and
/// `speech_end_frame` are the 0-based frame indices of the most-recent
/// segment opening and closing seen so far (`None` until a segment has
/// actually opened).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameResult {
  frame_index: u64,
  raw_prob: f32,
  smoothed_prob: f32,
  is_speech: bool,
  is_speech_start: bool,
  is_speech_end: bool,
  speech_start_frame: Option<u64>,
  speech_end_frame: Option<u64>,
}

impl FrameResult {
  /// Frame shift in samples (`160` at 16 kHz, i.e. 10 ms).
  pub const FRAME_SHIFT_SAMPLES: u32 = 160;
  /// Sample rate in Hz (always `16_000`).
  pub const SAMPLE_RATE_HZ: u32 = 16_000;

  /// Construct a `FrameResult`. Public so it is easy to assemble in
  /// tests; the `Vad` engine is the normal producer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    frame_index: u64,
    raw_prob: f32,
    smoothed_prob: f32,
    is_speech: bool,
    is_speech_start: bool,
    is_speech_end: bool,
    speech_start_frame: Option<u64>,
    speech_end_frame: Option<u64>,
  ) -> Self {
    Self {
      frame_index,
      raw_prob,
      smoothed_prob,
      is_speech,
      is_speech_start,
      is_speech_end,
      speech_start_frame,
      speech_end_frame,
    }
  }

  /// 0-based frame index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn frame_index(&self) -> u64 {
    self.frame_index
  }

  /// Raw sigmoid output from the model for this frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn raw_prob(&self) -> f32 {
    self.raw_prob
  }

  /// Trailing moving-average of `raw_prob` over the configured smooth window.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn smoothed_prob(&self) -> f32 {
    self.smoothed_prob
  }

  /// Whether `smoothed_prob` exceeds the speech threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_speech(&self) -> bool {
    self.is_speech
  }

  /// Whether this frame opened a new segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_speech_start(&self) -> bool {
    self.is_speech_start
  }

  /// Whether this frame closed an open segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_speech_end(&self) -> bool {
    self.is_speech_end
  }

  /// 0-based frame index of the most recent segment opening, if any.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn speech_start_frame(&self) -> Option<u64> {
    self.speech_start_frame
  }

  /// 0-based frame index of the most recent segment closing, if any.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn speech_end_frame(&self) -> Option<u64> {
    self.speech_end_frame
  }

  /// Timestamp at the *start* of this frame.
  pub fn timestamp(&self) -> Duration {
    let samples = self.frame_index * Self::FRAME_SHIFT_SAMPLES as u64;
    Duration::from_secs_f64(samples as f64 / Self::SAMPLE_RATE_HZ as f64)
  }

  /// If this frame closes a segment, return it; otherwise `None`.
  pub fn closed_segment(&self) -> Option<SpeechSegment> {
    if !self.is_speech_end {
      return None;
    }
    let start = self.speech_start_frame? * Self::FRAME_SHIFT_SAMPLES as u64;
    let end = self.speech_end_frame? * Self::FRAME_SHIFT_SAMPLES as u64;
    Some(SpeechSegment::new(start, end))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_count_is_end_minus_start() {
    let s = SpeechSegment::new(160, 1600);
    assert_eq!(s.sample_count(), 1440);
  }

  #[test]
  fn timestamps_round_trip_through_sample_rate() {
    let s = SpeechSegment::new(16_000, 32_000);
    assert_eq!(s.start(), Duration::from_secs(1));
    assert_eq!(s.end(), Duration::from_secs(2));
    assert_eq!(s.duration(), Duration::from_secs(1));
  }

  #[test]
  fn range_usize_slices_pcm_directly() {
    let pcm = [0.0f32; 2_000];
    let s = SpeechSegment::new(160, 320);
    let slice = &pcm[s.range_usize()];
    assert_eq!(slice.len(), 160);
  }

  #[test]
  fn empty_segment_has_zero_sample_count() {
    let s = SpeechSegment::new(100, 100);
    assert_eq!(s.sample_count(), 0);
    assert!(s.range().is_empty());
  }

  #[test]
  fn frame_result_closed_segment_is_some_only_when_is_speech_end() {
    let result = FrameResult::new(20, 0.9, 0.85, true, false, true, Some(2), Some(20));
    let segment = result.closed_segment().expect("segment closes");
    assert_eq!(segment.start_sample(), 2 * 160);
    assert_eq!(segment.end_sample(), 20 * 160);

    let mid = FrameResult::new(15, 0.8, 0.75, true, false, false, Some(2), None);
    assert!(mid.closed_segment().is_none());
  }

  #[test]
  fn frame_result_timestamp_uses_frame_shift_samples() {
    let result = FrameResult::new(100, 0.0, 0.0, false, false, false, None, None);
    assert_eq!(result.timestamp(), Duration::from_millis(1_000));
  }
}
