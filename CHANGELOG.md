# Changelog

## UNRELEASED

## 0.1.0 — 2026-05-08

Initial release.

- Sans-I/O streaming `Vad` engine: `push_samples` returns the next closed
  `SpeechSegment` (or `None`); `finish` returns the trailing segment if any.
  The drain-via-empty-push idiom handles the rare multi-segment-per-push case.
- Bit-for-bit port of upstream Python's `StreamVadPostprocessor`:
  trailing-mean smoothing, 4-state machine
  (SILENCE / POSSIBLE_SPEECH / SPEECH / POSSIBLE_SILENCE),
  `hit_max_speech` re-arm on force-split,
  `last_speech_end_frame` clamping for `pad_start`.
- Pure-Rust Kaldi-compatible Mel-filterbank + CMVN preprocessing.
  No `dyn` dispatch (concrete `rustfft::algorithm::Radix4<f32>`).
- ONNX Runtime via `ort` 2.0.0-rc.12, contract pinned to
  `feat[1, T, 80] + caches_in[8, 1, 128, 19] -> probs[1, T, 1] + caches_out`.
- `bundled` feature (default) embeds the FireRedVAD streaming ONNX
  model and CMVN stats (Apache-2.0; see `THIRD_PARTY_NOTICES.md`).
- Optional `serde` feature mirrors silero's per-field
  `humantime-serde` idiom.
- Voice-activity classification: the bundled streaming model treats speech and
  singing as positive (segments emitted) and pure instrumental music as negative
  (no segment). The 3-class AED model is non-streaming upstream and is out of
  scope for v1.
