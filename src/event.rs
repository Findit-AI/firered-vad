//! Public event types emitted by [`crate::Vad`].

use core::ops::Range;
use core::time::Duration;

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
    Self { start_sample, end_sample }
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
}
