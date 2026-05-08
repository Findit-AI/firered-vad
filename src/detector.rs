//! Streaming postprocessor: turns raw frame probabilities into per-frame
//! decisions and closed [`SpeechSegment`]s.
//!
//! This is a bit-for-bit port of upstream Python's
//! `StreamVadPostprocessor`. Frame indices are 0-based on the way out
//! (upstream is 1-based; we shift on construction).

use std::collections::VecDeque;

use crate::event::{FrameResult, SpeechSegment};
use crate::options::VadOptions;

#[derive(Debug, Clone, Copy, PartialEq)]
enum VadState {
  Silence,
  PossibleSpeech,
  Speech,
  PossibleSilence,
}

/// Streaming probability postprocessor.
#[derive(Debug)]
pub(crate) struct Postprocessor {
  options: VadOptions,
  smooth_window: VecDeque<f32>,
  smooth_window_sum: f64,
  state: VadState,
  /// 1-based count of consecutive speech frames in the current run.
  speech_cnt: u32,
  /// 1-based count of consecutive silence frames in the current run.
  silence_cnt: u32,
  /// True while a force-split has just fired and the next frame must be
  /// flagged as a new speech start regardless of state-machine evolution.
  hit_max_speech: bool,
  /// 0-based frame index of the most recent segment opening (`None` once
  /// the segment closes).
  last_speech_start_frame: Option<u64>,
  /// 0-based frame index of the most recent segment closing.
  last_speech_end_frame: Option<u64>,
  /// 1-based frame counter (mirrors upstream's `self.frame_cnt`).
  frame_cnt_1based: u64,
}

impl Postprocessor {
  pub(crate) fn new(options: VadOptions) -> Self {
    let smooth_window_size = options.smooth_window_size_frames().max(1) as usize;
    Self {
      options,
      smooth_window: VecDeque::with_capacity(smooth_window_size),
      smooth_window_sum: 0.0,
      state: VadState::Silence,
      speech_cnt: 0,
      silence_cnt: 0,
      hit_max_speech: false,
      last_speech_start_frame: None,
      last_speech_end_frame: None,
      frame_cnt_1based: 0,
    }
  }

  /// Replace options. Existing in-flight state is preserved (matches the
  /// "set_options at runtime" use case where you're tuning thresholds).
  pub(crate) fn set_options(&mut self, options: VadOptions) {
    self.options = options;
  }

  pub(crate) fn options(&self) -> &VadOptions {
    &self.options
  }

  pub(crate) fn reset(&mut self) {
    self.smooth_window.clear();
    self.smooth_window_sum = 0.0;
    self.state = VadState::Silence;
    self.speech_cnt = 0;
    self.silence_cnt = 0;
    self.hit_max_speech = false;
    self.last_speech_start_frame = None;
    self.last_speech_end_frame = None;
    self.frame_cnt_1based = 0;
  }

  pub(crate) fn is_active(&self) -> bool {
    matches!(self.state, VadState::Speech | VadState::PossibleSilence)
  }

  fn smooth(&mut self, raw: f32) -> f32 {
    let size = self.options.smooth_window_size_frames().max(1) as usize;
    if size <= 1 {
      return raw;
    }
    self.smooth_window.push_back(raw);
    self.smooth_window_sum += raw as f64;
    while self.smooth_window.len() > size {
      let dropped = self.smooth_window.pop_front().unwrap_or(0.0);
      self.smooth_window_sum -= dropped as f64;
    }
    (self.smooth_window_sum / self.smooth_window.len() as f64) as f32
  }

  /// Helper: compute the padded segment-start frame, clamped per upstream.
  ///
  /// `speech_cnt` is the 1-based count of speech frames in the current run
  /// at the moment of promotion; the formula reproduces upstream's
  /// `max(1, frame_cnt - speech_cnt + 1 - pad_start_frame, last_end + 1)`,
  /// then shifts to 0-based.
  fn padded_speech_start(&self) -> u64 {
    let pad = self.options.pad_start_frames() as u64;
    let raw = self
      .frame_cnt_1based
      .saturating_sub(self.speech_cnt as u64)
      .saturating_add(1)
      .saturating_sub(pad);
    let lower = self.last_speech_end_frame.map(|e| e + 1).unwrap_or(0);
    let one_based = raw.max(1).max(lower + 1);
    one_based.saturating_sub(1)
  }
}

impl Postprocessor {
  /// Push one raw frame probability. Returns the per-frame view and
  /// `Some(SpeechSegment)` if this frame closed a segment.
  pub(crate) fn push_probability(&mut self, raw_prob: f32) -> (FrameResult, Option<SpeechSegment>) {
    self.frame_cnt_1based += 1;
    let smoothed = self.smooth(raw_prob);
    let is_speech = smoothed >= self.options.speech_threshold();

    let mut is_speech_start = false;
    let mut is_speech_end = false;
    let mut start_frame: Option<u64> = self.last_speech_start_frame;
    let mut end_frame: Option<u64> = None;

    // hit_max_speech re-arms a fresh segment-start on this frame.
    if self.hit_max_speech {
      is_speech_start = true;
      let new_start = self.frame_cnt_1based.saturating_sub(1); // 0-based current frame
      self.last_speech_start_frame = Some(new_start);
      start_frame = Some(new_start);
      self.hit_max_speech = false;
    }

    let max_speech = self.options.max_speech_frames();

    match self.state {
      VadState::Silence => {
        if is_speech {
          self.state = VadState::PossibleSpeech;
          self.speech_cnt += 1;
        } else {
          self.silence_cnt += 1;
          self.speech_cnt = 0;
        }
      }
      VadState::PossibleSpeech => {
        if is_speech {
          self.speech_cnt += 1;
          if self.speech_cnt >= self.options.min_speech_frames() {
            self.state = VadState::Speech;
            is_speech_start = true;
            let new_start = self.padded_speech_start();
            self.last_speech_start_frame = Some(new_start);
            start_frame = Some(new_start);
            self.silence_cnt = 0;
          }
        } else {
          self.state = VadState::Silence;
          self.silence_cnt = 1;
          self.speech_cnt = 0;
        }
      }
      VadState::Speech => {
        self.speech_cnt += 1;
        if is_speech {
          self.silence_cnt = 0;
          if let Some(max) = max_speech {
            if self.speech_cnt >= max {
              // Force-split: mark this frame as a segment end and re-arm.
              self.hit_max_speech = true;
              self.speech_cnt = 0;
              is_speech_end = true;
              let close = self.frame_cnt_1based.saturating_sub(1);
              end_frame = Some(close);
              start_frame = self.last_speech_start_frame;
              self.last_speech_end_frame = Some(close);
              self.last_speech_start_frame = None;
            }
          }
        } else {
          self.state = VadState::PossibleSilence;
          self.silence_cnt += 1;
        }
      }
      VadState::PossibleSilence => {
        self.speech_cnt += 1;
        if is_speech {
          self.state = VadState::Speech;
          self.silence_cnt = 0;
          if let Some(max) = max_speech {
            if self.speech_cnt >= max {
              self.hit_max_speech = true;
              self.speech_cnt = 0;
              is_speech_end = true;
              let close = self.frame_cnt_1based.saturating_sub(1);
              end_frame = Some(close);
              start_frame = self.last_speech_start_frame;
              self.last_speech_end_frame = Some(close);
              self.last_speech_start_frame = None;
            }
          }
        } else {
          self.silence_cnt += 1;
          if self.silence_cnt >= self.options.min_silence_frames() {
            self.state = VadState::Silence;
            is_speech_end = true;
            let close = self.frame_cnt_1based.saturating_sub(1);
            end_frame = Some(close);
            start_frame = self.last_speech_start_frame;
            self.last_speech_end_frame = Some(close);
            self.last_speech_start_frame = None;
            self.speech_cnt = 0;
          }
        }
      }
    }

    let frame_index_0based = self.frame_cnt_1based - 1;
    let result = FrameResult::new(
      frame_index_0based,
      raw_prob,
      smoothed,
      is_speech,
      is_speech_start,
      is_speech_end,
      start_frame,
      end_frame,
    );
    let segment = result.closed_segment();
    (result, segment)
  }

  /// EOF: if a segment is currently open, close it at the current frame
  /// and emit. Returns `None` when no open segment exists.
  pub(crate) fn finish_active(&mut self) -> Option<SpeechSegment> {
    if !self.is_active() {
      return None;
    }
    let start = self.last_speech_start_frame.take()?;
    let end = self.frame_cnt_1based; // exclusive 0-based end == 1-based current frame count
    self.state = VadState::Silence;
    self.last_speech_end_frame = Some(end.saturating_sub(1));
    self.speech_cnt = 0;
    self.silence_cnt = 0;
    Some(SpeechSegment::new(
      start * (FrameResult::FRAME_SHIFT_SAMPLES as u64),
      end * (FrameResult::FRAME_SHIFT_SAMPLES as u64),
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::time::Duration;

  fn opts() -> VadOptions {
    // Permissive defaults useful for unit testing: smaller min_speech /
    // min_silence so segments close quickly inside a hand-tuned trace.
    VadOptions::new()
      .with_smooth_window_size(1)
      .with_speech_threshold(0.5)
      .with_min_speech_duration(Duration::from_millis(30))   // 3 frames
      .with_min_silence_duration(Duration::from_millis(30))  // 3 frames
      .with_pad_start(Duration::from_millis(10))             // 1 frame, clamped to smooth_window_size=1
      .clear_max_speech_duration()
  }

  fn drive(post: &mut Postprocessor, probs: &[f32]) -> Vec<SpeechSegment> {
    let mut out = Vec::new();
    for &p in probs {
      let (_, seg) = post.push_probability(p);
      if let Some(s) = seg {
        out.push(s);
      }
    }
    out
  }

  #[test]
  fn silence_alone_yields_no_segments() {
    let mut p = Postprocessor::new(opts());
    let segs = drive(&mut p, &vec![0.0; 50]);
    assert!(segs.is_empty());
    assert!(!p.is_active());
  }

  #[test]
  fn min_speech_must_be_reached_before_speech_state() {
    let mut p = Postprocessor::new(opts());
    // 2 speech frames — below min_speech_frames=3 — must NOT enter SPEECH.
    drive(&mut p, &[0.9, 0.9, 0.0, 0.0, 0.0, 0.0]);
    assert!(!p.is_active());
  }

  #[test]
  fn three_speech_frames_then_silence_closes_one_segment() {
    let mut p = Postprocessor::new(opts());
    let mut probs = vec![0.9; 3];                 // promote to SPEECH on frame 3 (1-based)
    probs.extend(vec![0.9; 5]);                   // hold SPEECH
    probs.extend(vec![0.0; 4]);                   // POSSIBLE_SILENCE then close after 3 silence frames
    let segs = drive(&mut p, &probs);
    assert_eq!(segs.len(), 1);
    assert!(segs[0].sample_count() > 0);
  }

  #[test]
  fn finish_flushes_open_segment() {
    let mut p = Postprocessor::new(opts());
    let probs = vec![0.9; 10];                    // open and stay in SPEECH
    drive(&mut p, &probs);
    assert!(p.is_active());
    let segment = p.finish_active().expect("trailing segment");
    assert!(segment.sample_count() > 0);
    assert!(!p.is_active());
  }

  #[test]
  fn max_speech_force_split_produces_two_segments_back_to_back() {
    let mut p = Postprocessor::new(
      opts().with_max_speech_duration(Duration::from_millis(50)), // 5 frames
    );
    // 12 speech frames + 5 silence: should fire force-split at speech_cnt=5,
    // immediately reopen on the next frame, then close via min_silence.
    let mut probs = vec![0.9; 12];
    probs.extend(vec![0.0; 5]);
    let segs = drive(&mut p, &probs);
    assert!(segs.len() >= 2, "expected at least 2 segments; got {}", segs.len());
  }

  #[test]
  fn smoothing_window_dampens_isolated_high_probs() {
    let opts = opts().with_smooth_window_size(5).with_speech_threshold(0.5);
    let mut p = Postprocessor::new(opts);
    // 4 zeros + 1 high prob: smoothed = 0.18 → no_speech.
    drive(&mut p, &[0.0, 0.0, 0.0, 0.0, 0.9]);
    assert!(!p.is_active());
  }

  #[test]
  fn frame_indexing_is_zero_based() {
    let mut p = Postprocessor::new(opts());
    let (first, _) = p.push_probability(0.0);
    assert_eq!(first.frame_index(), 0);
    let (second, _) = p.push_probability(0.0);
    assert_eq!(second.frame_index(), 1);
  }
}
