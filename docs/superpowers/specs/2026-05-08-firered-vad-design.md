# `firered-vad` design — 2026-05-08

A Rust crate exposing FireRedVAD as a streaming Voice Activity Detector. Inputs 16 kHz f32 PCM and emits `SpeechSegment { start, end }` ranges suitable for slicing audio into Whisper. Bit-for-bit parity with FireRedVAD's upstream Python `FireRedStreamVad` + `StreamVadPostprocessor`.

The crate is a sibling to [`silero`](https://github.com/uqio/silero) by the same author and follows the same coding style — but the public API is **not** silero-compatible, because FireRedVAD is a true streaming model and silero is not.

---

## 1. Motivation & scope

### 1.1 Why a new crate

The downstream pipeline feeds Whisper. Whisper transcribes best when given continuous human-speech windows separated by silence. The existing `silero` crate works, but the silero model itself is **not streaming** — it requires fixed 32 ms chunks with externally-managed RNN state, and silence between speech segments has to be reconstructed by post-processing thresholds. FireRedVAD is a true streaming VAD: it natively consumes 10 ms frames and outputs continuous speech regions via a 4-state machine. Wrapping it gives us:

- Lower-latency boundary decisions (10 ms granularity vs silero's 32 ms).
- A built-in postprocessor that already handles smoothing, mode presets, and segment boundary detection — no need to layer hysteresis on top.
- Better quality on noisy multi-speaker audio (per upstream benchmarks).

### 1.2 In scope (v1)

- Streaming VAD over 16 kHz f32 PCM in `[-1.0, 1.0]`.
- Bit-for-bit parity with upstream `StreamVadPostprocessor`: smoothing, threshold, padding, 4-state machine, mode presets, max-speech force-split.
- Pure-Rust Mel-filterbank + CMVN preprocessing (Kaldi-compatible).
- ONNX inference via `ort` (the same crate `silero` uses).
- Bundled model + CMVN behind a `bundled` Cargo feature (default-on).
- Standard sibling artifacts: `examples/`, `tests/integration_test.rs`, `tests/parity/` harness, `.github/workflows/ci.yml`, README, CHANGELOG, LICENSE-MIT, LICENSE-APACHE, THIRD_PARTY_NOTICES.

### 1.3 Out of scope (v1)

- Offline / full-audio detection (`detect_full`). Streaming is sufficient; offline is a thin wrapper if anyone asks.
- Audio Event Detection (AED) — separate model, separate concerns.
- 8 kHz support — model is 16 kHz only.
- Built-in resampling — caller's job.
- Multi-stream batched inference — model fixes batch=1.
- `no_std` support.
- A silero-style `SpeechSegmenter` adapter — explicitly rejected; FireRedVAD's own postprocessor is sufficient.

---

## 2. Replacing the current scaffold

The existing `firered-vad/` is the untouched `al8n/template-rs` template (package name `template-rs`, version `0.0.0`, edition 2021, MSRV 1.73). The implementation step replaces it wholesale with a silero-shaped sibling:

| Item | Action |
| --- | --- |
| `Cargo.toml` | Replace: `name = "firered-vad"`, `version = "0.1.0"`, `edition = "2024"`, `rust-version = "1.85"`, `license = "MIT OR Apache-2.0"` |
| `build.rs` | Replace with silero's tarpaulin-detection-only version |
| `rustfmt.toml` | Replace with silero's (100-col, 2-space indent) |
| `.github/workflows/ci.yml` | Replace with silero's CI matrix (rustfmt / clippy / build / test on Linux+macOS+Windows + tarpaulin coverage on Linux nightly). Drop the template's sanitizers/Miri/Loom |
| `.github/workflows/loc.yml`, `dependabot.yml`, `FUNDING.yml` | Drop |
| `ci/` (sanitizer.sh, miri_*.sh) | Drop |
| `src/` | Replace (modules: `lib.rs`, `vad.rs`, `features.rs`, `inference.rs`, `detector.rs`, `options.rs`, `event.rs`, `error.rs`) |
| `tests/foo.rs`, `examples/foo.rs` | Replace with new integration tests + parity harness + examples |
| `benches/foo.rs` | Drop for v1 (no perf-tuning targets) |
| `models/` (new) | Vendor `fireredvad_stream_vad_with_cache.onnx` (~2.3 MB) and `cmvn.ark` (~1.3 KB) from upstream's `pretrained_models/onnx_models/` |
| `THIRD_PARTY_NOTICES.md` (new) | Apache-2.0 attribution for the bundled model + CMVN |
| `README.md`, `CHANGELOG.md` | Rewrite. Keep `LICENSE-MIT`, `LICENSE-APACHE` (already present, dual-license matches silero) |
| `.codecov.yml`, `.gitignore` | Replace with silero's |

---

## 3. Public API

### 3.1 Coding conventions (project-wide)

Every public type follows the no-public-fields rule:

- All struct/enum-variant fields are private.
- Each piece of data is paired with: a `pub const fn field(&self)` getter, a `pub fn set_field(&mut self, value)` setter, and a `pub const fn with_field(self, value) -> Self` builder. `const fn` whenever the body permits.
- Same for `enum` payloads — prefer tuple variants with accessor methods over braced variants with public fields.

### 3.2 `lib.rs` re-exports

```rust
#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod detector;
mod error;
mod event;
mod features;
mod inference;
mod options;
mod vad;

pub use error::{Error, Result};
pub use event::{FrameResult, SpeechSegment, VadEvent};
pub use options::{SessionOptions, VadOptions};
pub use vad::Vad;
// `GraphOptimizationLevel` is re-exported from `ort` rather than defined locally —
// matches silero's pattern of treating it as a foreign type and bridging to serde
// via a private mirror enum (see options.rs).
pub use ort::session::builder::GraphOptimizationLevel;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub use vad::{BUNDLED_CMVN, BUNDLED_MODEL};
```

### 3.3 `Vad` — the only public state machine

```rust
pub struct Vad { /* private */ }

impl Vad {
    // ── Construction ──────────────────────────────────────────────────
    #[cfg(feature = "bundled")] pub fn bundled() -> Result<Self>;
    #[cfg(feature = "bundled")] pub fn bundled_with(options: VadOptions) -> Result<Self>;

    #[cfg(feature = "bundled")] pub fn from_memory(model: &[u8]) -> Result<Self>;
    #[cfg(feature = "bundled")] pub fn from_memory_with(model: &[u8], options: VadOptions) -> Result<Self>;

    #[cfg(feature = "bundled")] pub fn from_file<P: AsRef<Path>>(model: P) -> Result<Self>;
    #[cfg(feature = "bundled")] pub fn from_file_with<P: AsRef<Path>>(model: P, options: VadOptions) -> Result<Self>;

    pub fn from_memory_with_cmvn(model: &[u8], cmvn: &[u8], options: VadOptions) -> Result<Self>;
    pub fn from_file_with_cmvn<P: AsRef<Path>>(model: P, cmvn: P, options: VadOptions) -> Result<Self>;

    pub fn from_ort_session(session: ort::Session, cmvn: &[u8], options: VadOptions) -> Result<Self>;

    // ── Sans-I/O surface ───────────────────────────────────────────────
    pub fn push_samples(&mut self, pcm: &[f32]) -> Result<()>;
    pub fn finish(&mut self) -> Result<()>;
    pub fn poll_event(&mut self) -> Option<VadEvent>;
    pub fn drain_events<F>(&mut self, f: F) where F: FnMut(VadEvent);
    pub fn reset(&mut self);

    // ── Inspection ─────────────────────────────────────────────────────
    pub const fn options(&self) -> &VadOptions;
    pub fn set_options(&mut self, options: VadOptions);
    pub const fn frame_count(&self) -> u64;
    pub const fn pending_samples(&self) -> usize;
    pub const fn is_active(&self) -> bool;
    pub const fn is_finished(&self) -> bool;
    pub const fn pending_events(&self) -> usize;
}
```

### 3.4 `VadEvent`, `SpeechSegment`, `FrameResult`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    Frame(FrameResult),
    SegmentClosed(SpeechSegment),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment { /* private fields */ }

impl SpeechSegment {
    pub const SAMPLE_RATE_HZ: u32 = 16_000;

    pub const fn start_sample(&self) -> u64;
    pub const fn end_sample(&self) -> u64;          // exclusive
    pub const fn sample_count(&self) -> u64;
    pub fn start(&self) -> Duration;
    pub fn end(&self) -> Duration;
    pub fn duration(&self) -> Duration;
    pub fn range(&self) -> Range<u64>;
    pub fn range_usize(&self) -> Range<usize>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameResult { /* private fields */ }

impl FrameResult {
    pub const FRAME_SHIFT_SAMPLES: u32 = 160;
    pub const SAMPLE_RATE_HZ: u32 = 16_000;

    pub const fn frame_index(&self) -> u64;          // 0-based (we shift upstream's 1-based)
    pub const fn raw_prob(&self) -> f32;
    pub const fn smoothed_prob(&self) -> f32;
    pub const fn is_speech(&self) -> bool;
    pub const fn is_speech_start(&self) -> bool;
    pub const fn is_speech_end(&self) -> bool;
    pub const fn speech_start_frame(&self) -> Option<u64>;
    pub const fn speech_end_frame(&self) -> Option<u64>;
    pub fn timestamp(&self) -> Duration;
    pub fn closed_segment(&self) -> Option<SpeechSegment>;   // Some only when is_speech_end
}
```

`SpeechSegment` boundaries are absolute sample indices on the stream timeline. `range_usize()` slices the user's PCM `Vec<f32>` directly: `&pcm[segment.range_usize()]` is the human-speech window for that segment.

### 3.5 `VadOptions` and `SessionOptions`

There is no `Mode` enum and no preset machinery. Upstream's "mode 0..3" presets just set three numeric fields (`speech_threshold`, `min_speech_frame`, `min_silence_frame`); we leave callers to set those via the `with_*` / `set_*` builders directly. Documentation will list the four upstream presets as suggested values for callers who want to mirror them.

`GraphOptimizationLevel` is re-exported from `ort::session::builder` rather than redefined locally. This matches silero's pattern (foreign type, no `Default`/`Serialize`/`Deserialize` derive on the upstream enum, so we bridge serde via a private mirror enum and a `with = "..."` attribute).

The serde idiom is silero's: derive `Serialize` / `Deserialize` cfg-gated, use `serde(default = "default_field_name", with = "humantime_serde")` per Duration field and `with = "humantime_serde::option"` per `Option<Duration>` field, and supply per-field `const fn default_*()` functions to back the `serde(default = ...)` attributes (so they round-trip through `{}`).

```rust
use core::time::Duration;
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ── private serde bridge for the foreign GraphOptimizationLevel enum ──
#[cfg(feature = "serde")]
mod graph_optimization_level {
    use super::GraphOptimizationLevel;
    use serde::*;

    #[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum OptimizationLevel { Disable, Level1, Level2, #[default] Level3, All }

    impl From<GraphOptimizationLevel> for OptimizationLevel { /* one-arm-per-variant */ }
    impl From<OptimizationLevel> for GraphOptimizationLevel { /* one-arm-per-variant */ }

    pub fn serialize<S: Serializer>(l: &GraphOptimizationLevel, s: S) -> Result<S::Ok, S::Error>;
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<GraphOptimizationLevel, D::Error>;
    pub const fn default() -> GraphOptimizationLevel { GraphOptimizationLevel::Disable }
}

// ── const-fn defaults backing serde(default = "...") attributes ──
const fn default_smooth_window_size() -> u32              { 5 }
const fn default_speech_threshold() -> f32                { 0.5 }
const fn default_pad_start() -> Duration                  { Duration::from_millis(50) }
const fn default_min_speech_duration() -> Duration        { Duration::from_millis(80) }
const fn default_min_silence_duration() -> Duration       { Duration::from_millis(200) }
const fn default_max_speech_duration() -> Option<Duration> { Some(Duration::from_secs(20)) }

// ── SessionOptions ──
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SessionOptions {
    #[cfg_attr(feature = "serde",
        serde(default = "graph_optimization_level::default", with = "graph_optimization_level"))]
    optimization_level: GraphOptimizationLevel,
}

impl Default for SessionOptions { fn default() -> Self { Self::new() } }

impl SessionOptions {
    pub const fn new() -> Self;
    pub const fn optimization_level(&self) -> GraphOptimizationLevel;
    pub const fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self;
    pub const fn with_optimization_level(self, level: GraphOptimizationLevel) -> Self;
}

// ── VadOptions ──
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VadOptions {
    #[cfg_attr(feature = "serde", serde(default = "default_smooth_window_size"))]
    smooth_window_size: u32,

    #[cfg_attr(feature = "serde", serde(default = "default_speech_threshold"))]
    speech_threshold: f32,

    #[cfg_attr(feature = "serde",
        serde(default = "default_pad_start", with = "humantime_serde"))]
    pad_start: Duration,

    #[cfg_attr(feature = "serde",
        serde(default = "default_min_speech_duration", with = "humantime_serde"))]
    min_speech_duration: Duration,

    #[cfg_attr(feature = "serde",
        serde(default = "default_min_silence_duration", with = "humantime_serde"))]
    min_silence_duration: Duration,

    #[cfg_attr(feature = "serde",
        serde(default = "default_max_speech_duration", with = "humantime_serde::option"))]
    max_speech_duration: Option<Duration>,

    #[cfg_attr(feature = "serde", serde(default))]
    session_options: SessionOptions,
}

impl Default for VadOptions { fn default() -> Self { Self::new() } }

impl VadOptions {
    pub const fn new() -> Self;       // matches upstream FireRedStreamVadConfig defaults

    // For every field, the trio: getter (const fn), with_* (const fn, takes mut self),
    // and set_* (const fn, takes &mut self, returns &mut Self for chaining).
    pub const fn smooth_window_size(&self) -> u32;
    pub const fn with_smooth_window_size(mut self, size: u32) -> Self;       // calls set_smooth_window_size internally
    pub const fn set_smooth_window_size(&mut self, size: u32) -> &mut Self;

    pub const fn speech_threshold(&self) -> f32;
    pub const fn with_speech_threshold(mut self, t: f32) -> Self;
    pub const fn set_speech_threshold(&mut self, t: f32) -> &mut Self;       // clamped to [0,1] via sanitize_probability

    pub const fn pad_start(&self) -> Duration;
    pub const fn with_pad_start(mut self, d: Duration) -> Self;
    pub const fn set_pad_start(&mut self, d: Duration) -> &mut Self;

    pub const fn min_speech_duration(&self) -> Duration;
    pub const fn with_min_speech_duration(mut self, d: Duration) -> Self;
    pub const fn set_min_speech_duration(&mut self, d: Duration) -> &mut Self;

    pub const fn max_speech_duration(&self) -> Option<Duration>;
    pub const fn with_max_speech_duration(mut self, d: Duration) -> Self;
    pub const fn set_max_speech_duration(&mut self, d: Duration) -> &mut Self;
    pub const fn clear_max_speech_duration(mut self) -> Self;                // disables force-split

    pub const fn min_silence_duration(&self) -> Duration;
    pub const fn with_min_silence_duration(mut self, d: Duration) -> Self;
    pub const fn set_min_silence_duration(&mut self, d: Duration) -> &mut Self;

    pub const fn session_options(&self) -> &SessionOptions;
    pub const fn with_session_options(mut self, opts: SessionOptions) -> Self;
    pub const fn set_session_options(&mut self, opts: SessionOptions) -> &mut Self;

    // Internal sample-domain getters used by the postprocessor (private; match silero's pattern).
    pub(crate) const fn smooth_window_size_frames(&self) -> u32;
    pub(crate) fn pad_start_frames(&self) -> u32;
    pub(crate) fn min_speech_frames(&self) -> u32;
    pub(crate) fn min_silence_frames(&self) -> u32;
    pub(crate) fn max_speech_frames(&self) -> Option<u32>;
}
```

Helper:

```rust
pub(crate) const fn duration_to_frames(d: Duration) -> u32 {
    // 10 ms = 1 frame  →  frames = ms / 10  =  (sec * 1000 + nanos / 1_000_000) / 10
    let ms = d.as_millis();
    let frames = ms / 10;
    if frames > u32::MAX as u128 { u32::MAX } else { frames as u32 }
}

const fn sanitize_probability(value: f32) -> f32 {
    // Same as silero's helper: NaN/inf → 0.0, otherwise clamp to [0,1].
    if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.0 }
}
```

Hot accessors (`*_frames`, getters, setters) are decorated with `#[cfg_attr(not(tarpaulin), inline(always))]` to match silero's perf-hint pattern (the `tarpaulin` cfg disables inline-always under coverage so line counts attribute correctly).

#### 3.5.1 Defaults (match upstream `FireRedStreamVadConfig`)

| Field | Upstream | Ours |
| --- | --- | --- |
| `smooth_window_size` | 5 | 5 |
| `speech_threshold` | 0.5 | 0.5 |
| `pad_start_frame` | 5 | `Duration::from_millis(50)` (5 frames) |
| `min_speech_frame` | 8 | `Duration::from_millis(80)` |
| `max_speech_frame` | 2000 | `Some(Duration::from_secs(20))` |
| `min_silence_frame` | 20 | `Duration::from_millis(200)` |

#### 3.5.2 Suggested preset values (no enum, just numbers callers can apply)

Documentation will list upstream's four `set_mode` presets as copy-paste recipes:

```rust
// "Very permissive" (upstream mode 0)
let opts = VadOptions::new()
    .with_speech_threshold(0.3)
    .with_min_speech_duration(Duration::from_millis(80))
    .with_min_silence_duration(Duration::from_millis(200));

// "Permissive" (upstream mode 1) — threshold 0.5, min_speech 100 ms, min_silence 150 ms
// "Aggressive" (upstream mode 2) — threshold 0.7, min_speech 150 ms, min_silence 100 ms
// "Very aggressive" (upstream mode 3) — threshold 0.9, min_speech 200 ms, min_silence 50 ms
```

---

## 4. Internal architecture

### 4.1 Module layout

```
src/
  lib.rs       crate-level re-exports + docs
  vad.rs       pub struct Vad — the public state machine
  features.rs  pub(crate) MelFilterbank + Cmvn + FeatureExtractor (Kaldi-compatible, pure Rust, no dyn dispatch)
  inference.rs pub(crate) OrtRunner — wraps ort::Session, runs single inference, scratch buffers
  detector.rs  pub(crate) Postprocessor — smoothing window + 4-state machine
  options.rs   pub VadOptions + pub SessionOptions; re-exports ort::session::builder::GraphOptimizationLevel
  event.rs     pub VadEvent, pub FrameResult, pub SpeechSegment
  error.rs     pub Error, pub Result
```

### 4.2 `vad.rs` — `Vad`

```rust
pub struct Vad {
    runner: OrtRunner,
    features: FeatureExtractor,
    detector: Postprocessor,
    options: VadOptions,
    events: VecDeque<VadEvent>,
    frame_count: u64,
    finished: bool,
}
```

`push_samples` does:

1. Append PCM to `features.pcm_tail`.
2. While ≥ 400 samples are available, extract a 25 ms window into a feature vector `[80]`. Drop the oldest 160 samples (10 ms hop) from `pcm_tail`.
3. Once at least one feature vector is extracted from this batch, batch them as `[1, T, 80]` and call `runner.infer(features, &mut detector_caches)`. T is the number of features extracted from this `push_samples` call.
4. For each prob in `[1, T, 1]`:
   1. `detector.push_probability(prob, frame_index)` → `(FrameResult, Option<SpeechSegment>)`.
   2. `events.push_back(VadEvent::Frame(frame_result))`.
   3. If `Some(segment)`, `events.push_back(VadEvent::SegmentClosed(segment))`.
5. `frame_count += T`.

`finish` sets `finished = true`, calls `detector.finish_active() -> Option<SpeechSegment>` to flush any open segment, queues that as a final `VadEvent::SegmentClosed`. Does NOT pad a partial 25 ms window (Kaldi `snip_edges=true` semantics).

`poll_event` is `events.pop_front()`. `drain_events` is the obvious loop. `reset` clears caches, smoothing window, state machine state, frame counters, event queue, sets `finished = false`.

### 4.3 `inference.rs` — `OrtRunner`

```rust
pub(crate) struct OrtRunner {
    session: ort::Session,
    feat_scratch: Vec<f32>,    // [1, T, 80] flattened
    cache_scratch: Vec<f32>,   // [8, 1, 128, 19] flattened — owned by Vad's StreamState contextually,
                               //   but for v1 we let OrtRunner hold the active caches directly to
                               //   simplify the single-stream case.
    prob_scratch: Vec<f32>,    // [1, T, 1] flattened
}
```

Loaders (`from_bytes`, `from_file`, `from_ort_session`) parameterize on `SessionOptions`. Single inference method:

```rust
pub(crate) fn infer(&mut self, features: &[f32]) -> Result<&[f32]>;
```

It packs `features` into `feat_scratch`, runs `ort::Session::run()` binding `feat`, `caches_in`, expecting `probs`, `caches_out`. Updates `cache_scratch` in place, returns the prob slice. The crate-internal `caches` Vec lives inside `OrtRunner` (single-stream) — the `StreamState` concept from earlier drafts is folded into `OrtRunner` since we're not exposing multi-stream.

### 4.4 `features.rs` — `FeatureExtractor`

Pure-Rust port of Kaldi's online log-Mel filterbank, hard-coded to upstream's configuration:

| Parameter | Value |
| --- | --- |
| `samp_freq` | 16000 Hz |
| `frame_length_ms` | 25 (= 400 samples) |
| `frame_shift_ms` | 10 (= 160 samples) |
| `num_mel_bins` | 80 |
| `dither` | 0.0 |
| `snip_edges` | true |
| `pre_emph_coeff` | 0.97 |
| `remove_dc_offset` | true |
| `window_type` | Povey (raised cosine, exponent 0.85) |
| `round_to_power_of_two` | true (FFT size = 512) |
| `low_freq` | 20 Hz |
| `high_freq` | Nyquist (8000 Hz) |
| `use_energy` | false |
| `use_log_fbank` | true |
| `use_power` | true |

```rust
pub(crate) struct FeatureExtractor {
    fft: Radix2<f32>,                    // concrete; no dyn dispatch
    fft_buf: Vec<Complex<f32>>,          // capacity 512
    window: Vec<f32>,                    // len 400, precomputed Povey
    pre_emph: f32,
    mel_filters: Vec<MelFilter>,         // 80 sparse triangular filters
    cmvn: Cmvn,
    pcm_tail: Vec<f32>,                  // 0..400 samples; capacity 400
    log_floor: f32,                      // 1e-20 floor on power
}

pub(crate) struct Cmvn {
    means: Vec<f32>,                     // len 80
    inverse_std_variances: Vec<f32>,     // len 80
}

pub(crate) struct MelFilter {
    start_bin: usize,
    weights: Vec<f32>,                   // sparse weights from start_bin
}

impl FeatureExtractor {
    pub(crate) fn new(cmvn_bytes: &[u8]) -> Result<Self>;
    pub(crate) fn pcm_tail_mut(&mut self) -> &mut Vec<f32>;
    pub(crate) fn pending_samples(&self) -> usize;
    pub(crate) fn extract_one(&mut self, out: &mut [f32; 80]) -> Result<()>;   // consumes one frame from pcm_tail
    pub(crate) fn reset(&mut self);
}
```

PCM is normalized to int16-range internally (multiply by 32768.0) before feature extraction, matching upstream's `sf.read(audio, dtype="int16")` path. Public API stays in `[-1.0, 1.0]`.

`Cmvn` parses Kaldi `.ark` format: 2-row stats matrix, computes means and inverse std-variances per upstream's formula:

```
mean[d]     = sum[d] / count
variance[d] = sum_sq[d] / count - mean[d]^2          (clamped >= 1e-20)
istd[d]     = 1.0 / sqrt(variance[d])
```

### 4.5 `detector.rs` — `Postprocessor`

Bit-for-bit port of upstream `StreamVadPostprocessor`. All seven feature areas preserved.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
enum VadState {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

pub(crate) struct Postprocessor {
    options: VadOptions,
    smooth_window: VecDeque<f32>,
    smooth_window_sum: f64,
    state: VadState,
    speech_cnt: u32,
    silence_cnt: u32,
    hit_max_speech: bool,
    last_speech_start_frame: Option<u64>,
    last_speech_end_frame: Option<u64>,
    frame_cnt: u64,
}

impl Postprocessor {
    pub(crate) fn new(options: VadOptions) -> Self;
    pub(crate) fn reset(&mut self);
    pub(crate) fn set_options(&mut self, options: VadOptions);

    /// Push one raw probability. Returns the per-frame result and, if this
    /// frame finalizes a segment, the SpeechSegment.
    pub(crate) fn push_probability(&mut self, raw_prob: f32) -> (FrameResult, Option<SpeechSegment>);

    /// EOF: if currently in SPEECH or POSSIBLE_SILENCE, close at the current
    /// frame count and emit. Returns None if no open segment.
    pub(crate) fn finish_active(&mut self) -> Option<SpeechSegment>;

    pub(crate) fn is_active(&self) -> bool;     // SPEECH | POSSIBLE_SILENCE
}
```

### 4.6 Per-frame result construction

Upstream returns 1-based frame indices for `frame_idx`, `speech_start_frame`, `speech_end_frame`. We shift to 0-based on construction:

- `frame_index = frame_cnt` (0-based; we increment AFTER processing, so the frame currently being processed is `frame_cnt`).
- `speech_start_frame = upstream_speech_start_frame - 1`.
- `speech_end_frame = upstream_speech_end_frame - 1`.

`SpeechSegment` derived from a `FrameResult` with `is_speech_end == true`:

- `start_sample = speech_start_frame.unwrap() * 160`.
- `end_sample = speech_end_frame.unwrap() * 160` (exclusive — first sample after the segment).

The user slices PCM as `&pcm[segment.range_usize()]` to get the human-speech window for that segment.

**Note on trailing silence:** upstream's state machine fires `is_speech_end` on the frame where `silence_cnt` first reaches `min_silence_frame`. Counting from the first silence frame after speech, that's the *N*-th silence frame in a run (where *N* = `min_silence_frame`). The segment therefore includes `min_silence_frame - 1` frames of trailing silence. At default `min_silence_frame = 20`, that's 190 ms of trailing silence per segment. This is upstream's behavior and we preserve it for parity. Callers who want tight trimming can post-process the segment range themselves.

### 4.7 Probability flow (end-to-end)

```
push_samples(&[f32])
  └─▶ append to features.pcm_tail
  └─▶ while pcm_tail.len() >= 400:
        ├─▶ features.extract_one(&mut feat_buf)        (consumes 1 frame from pcm_tail; drops 160)
        └─▶ append feat_buf to runner.feat_scratch
  └─▶ if any features extracted:
        ├─▶ runner.infer(features) → &[f32] of T probs
        └─▶ for each prob:
              ├─▶ detector.push_probability(prob) → (FrameResult, Option<SpeechSegment>)
              ├─▶ events.push_back(VadEvent::Frame(...))
              └─▶ if Some(segment): events.push_back(VadEvent::SegmentClosed(segment))
  └─▶ frame_count += T

finish()
  └─▶ finished = true
  └─▶ if let Some(s) = detector.finish_active(): events.push_back(VadEvent::SegmentClosed(s))

poll_event() → events.pop_front()
reset() → wipe everything; finished = false
```

---

## 5. Bundled artifacts & licensing

- `models/fireredvad_stream_vad_with_cache.onnx` — copied from `FireRedVAD/pretrained_models/onnx_models/`. ~2.3 MB. ONNX I/O contract (verified by running the model with `onnxruntime` for `T ∈ {1, 5, 10}`):
  - Inputs: `feat: [1, T, 80] f32` (T dynamic), `caches_in: [8, 1, 128, 19] f32`.
  - Outputs: `probs: [1, T, 1] f32`, `caches_out: [8, 1, 128, 19] f32`. The dim-3 name in the ONNX schema is `Concatcaches_out_dim_3` and *appears* dynamic, but in practice the model slices internally so the output dim is always 19 — no slicing is required in our code; we just feed `caches_out` directly back as `caches_in` on the next call.
  - Opset 18. Batch dim is fixed at 1 — no multi-stream batched inference.
- `models/cmvn.ark` — copied from the same upstream directory. ~1.3 KB.
- Re-exported as `BUNDLED_MODEL`, `BUNDLED_CMVN` byte slices behind the `bundled` feature (default-on).

`THIRD_PARTY_NOTICES.md` declares: bundled assets are derived from the FireRedVAD project by Xiaohongshu (Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu), Apache-2.0 licensed, with full license-text reproduction and upstream URL.

Crate license stays dual `MIT OR Apache-2.0` — the Apache-2.0 attribution is for the bundled artifacts, not the crate code.

---

## 6. Cargo features & dependencies

```toml
[features]
default  = ["bundled"]
bundled  = []
serde    = ["dep:serde", "dep:humantime-serde"]

# ONNX execution provider passthroughs (mirroring silero):
coreml    = ["ort/coreml"]
directml  = ["ort/directml"]
cuda      = ["ort/cuda"]
rocm      = ["ort/rocm"]
tensorrt  = ["ort/tensorrt"]
openvino  = ["ort/openvino"]

[dependencies]
ort        = "2.0.0-rc.12"
thiserror  = "2"
rustfft    = "6"

serde            = { version = "1", optional = true, features = ["derive"] }
humantime-serde  = { version = "1", optional = true }

[dev-dependencies]
hound       = "3"
serde_json  = "1"
```

`rustfft` is the only new dependency vs silero (silero takes raw PCM; we need FFT for Mel-fbank). Concrete type `rustfft::algorithm::Radix2<f32>` is used directly — no `Arc<dyn Fft<f32>>`.

---

## 7. Error model

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load model from {path}: {source}")]
    LoadModel { path: PathBuf, source: ort::Error },

    #[error(transparent)]
    Ort(#[from] ort::Error),

    #[error("failed to read CMVN file from {path}: {source}")]
    LoadCmvn { path: PathBuf, source: std::io::Error },

    #[error("invalid CMVN format: {reason}")]
    InvalidCmvn { reason: &'static str },

    #[error("input PCM sample rate is unsupported (model is fixed at {expected} Hz)")]
    UnsupportedSampleRate { expected: u32 },

    #[error("ONNX output {tensor} had unexpected shape {shape:?}")]
    UnexpectedOutputShape { tensor: &'static str, shape: Vec<i64> },

    #[error("invalid speech threshold {value} (must be in [0, 1])")]
    InvalidSpeechThreshold { value: f32 },
}

pub type Result<T> = std::result::Result<T, Error>;
```

`ort::Error` is exposed transparently via `#[from]`. Model-specific failures get explicit variants for actionable callers.

---

## 8. Testing strategy

### 8.1 Unit tests (per module)

- **`features.rs`** — deterministic Mel-fbank + CMVN against precomputed reference vectors (extracted offline from upstream Python on a fixed PCM input). Tolerance ≤ 1e-3 per coefficient. Tests cover: empty input, single full window, partial window dropped at end (snip_edges), CMVN normalization, FFT/window correctness on synthetic sinusoids.
- **`detector.rs`** — probability-sequence-driven tests covering each state-machine transition (SILENCE → POSSIBLE_SPEECH → SPEECH → POSSIBLE_SILENCE → SILENCE), `min_speech_frame` enforcement (single-speech-frame must NOT promote), `max_speech_frame` force-split (correctly sets `hit_max_speech` and re-arms `is_speech_start` on the next frame), mode-preset application, smoothing-window correctness (size ≤ 1 passthrough, FIFO eviction, mean-of-window).
- **`inference.rs`** — round-trip with bundled model on a synthetic `[1, T, 80]` feature tensor; checks output shape and that `caches_out` differs from `caches_in` (inference actually advanced state).
- **`vad.rs`** — full-pipeline checks: silence stays silent, sustained synthetic noise produces a segment, push/poll/finish/reset transitions, event ordering (Frame events precede the SegmentClosed they finalize on the same probability).

### 8.2 Integration tests (`tests/integration_test.rs`)

- Bundled `Vad` constructs without error.
- 1 second of zero PCM yields no `SegmentClosed` events; `is_active() == false` afterward.
- 1 second of synthetic speech-like signal (band-limited noise around speech formants) yields ≥ 1 `SegmentClosed`.
- `push_samples` accepts arbitrary chunk sizes (1 sample, 160, 320, 1024, 16000) and produces an *identical* event stream regardless of chunking — pinned via deterministic hash of the event list.
- `reset()` returns the engine to a state indistinguishable from a freshly constructed one (event queue empty, frame_count 0, is_active false).

### 8.3 Parity harness (`tests/parity/`)

Mirrors silero's `tests/parity/` layout; not part of `cargo test`. Manual invocation via `tests/parity/run.sh`:

```
tests/parity/
  README.md            how to run, expected outputs, IoU-tolerance rationale
  fixtures/            short multi-speaker WAVs (sourced from upstream's wav/ dir)
  python/
    requirements.txt   fireredvad + numpy
    run.py             FireRedStreamVad over each fixture; dumps per-frame
                         {raw_prob, smoothed_prob, is_speech, is_speech_*, speech_start_frame, speech_end_frame}
                         + segment timestamps as JSON
  rust/
    Cargo.toml         standalone bin depending on the parent crate via path
    src/main.rs        Vad over the same fixtures; dumps the same JSON shape
  scorer.py            diff JSON outputs; reports per-frame deltas + segment IoU
  run.sh               orchestrates Python + Rust runs and scorer
```

Pinned tolerances:

- `|raw_prob_python - raw_prob_rust| < 1e-3` per frame.
- `is_speech` agreement = 100% per frame.
- Segment IoU ≥ 0.99 on every fixture.

Failures are blocking before any release.

---

## 9. Examples

- `examples/streaming.rs` — synthesizes 5 s of alternating noise/silence, drives `Vad`, prints emitted segments in the canonical Sans-I/O loop.
- `examples/detect_file.rs` — reads a 16 kHz mono int16 WAV via `hound`, normalizes to f32 in `[-1.0, 1.0]`, drives `Vad`, prints `start..end` of each `SegmentClosed`. Prototype of the "feed Whisper" use case minus the Whisper call.

---

## 10. CI (`.github/workflows/ci.yml`)

Mirrors silero's CI exactly (drops the template's sanitizers/Miri/Loom):

- `rustfmt` (Ubuntu) — format check.
- `clippy` (Ubuntu, macOS, Windows) — `cargo clippy --all-features -- -D warnings`.
- `build` (Ubuntu, macOS, Windows) — three matrix shards: `cargo build`, `cargo build --no-default-features`, `cargo build --all-features`.
- `test` (Ubuntu, macOS, Windows) — `cargo test --all-features`.
- `coverage` (Ubuntu, nightly) — tarpaulin with `--cfg tarpaulin`, codecov upload.
- Triggers: push (excluding docs), PR, scheduled monthly.
- Env: `RUSTFLAGS=-Dwarnings`, `RUST_BACKTRACE=1`, `CARGO_TERM_COLOR=always`.

---

## 11. Reference: upstream feature parity table

| Upstream feature | Source | Our port |
| --- | --- | --- |
| `smooth_window_size` (default 5) — moving avg over raw probs → `smoothed_prob` | `stream_vad_postprocessor.py:77-86` | `VadOptions::smooth_window_size` + `Postprocessor.smooth_window: VecDeque<f32>` |
| `speech_threshold` (default 0.5) — applied to **smoothed** prob → `is_speech` | `stream_vad_postprocessor.py:88-89` | `VadOptions::speech_threshold` (clamped `[0,1]`) |
| `pad_start_frame` (default 5; clamped `>= smooth_window_size`) | `stream_vad_postprocessor.py:36-38, 112-114` | `VadOptions::pad_start: Duration`; clamp preserved |
| `min_speech_frame` (default 8) — accumulate before SPEECH promotion | `stream_vad_postprocessor.py:106-115` | `VadOptions::min_speech_duration` |
| `max_speech_frame` (default 2000) — force-split via `hit_max_speech` re-arm | `stream_vad_postprocessor.py:122-150` | `VadOptions::max_speech_duration: Option<Duration>`; identical re-arm semantics |
| `min_silence_frame` (default 20) — silence threshold to close in POSSIBLE_SILENCE | `stream_vad_postprocessor.py:138-161` | `VadOptions::min_silence_duration` |
| 4-state machine (SILENCE / POSSIBLE_SPEECH / SPEECH / POSSIBLE_SILENCE) | `stream_vad_postprocessor.py:91-163` | private `enum VadState` in `detector.rs`; transitions translated 1:1 |
| `set_mode(0..3)` presets | `stream_vad.py:142-161` | Documented as recipes (see §3.5.2) — callers apply via direct `VadOptions::with_*` calls. No `Mode` enum exposed |
| Per-frame result fields | `stream_vad_postprocessor.py:8-17` | `pub struct FrameResult` with const-fn accessors; **frame index 0-based** |
| `hit_max_speech` re-arm flag | `stream_vad_postprocessor.py:92-96` | preserved as `Postprocessor.hit_max_speech: bool` |
| `last_speech_start_frame` / `last_speech_end_frame` clamping for `pad_start` | `stream_vad_postprocessor.py:54-55, 112-114, 131-133, 159-160` | preserved as `Option<u64>` slots in `Postprocessor` |

The **only** upstream knob not exposed: `chunk_max_frame` (offline-only batching control; we're streaming-only).

---

## 12. Rationale & alternatives considered

- **Clean-room implementation vs depending on `wavekat-vad`** — `wavekat-vad` already wraps FireRedVAD in Rust, but in a different idiom. We chose clean-room to keep the crate as a true sibling of `silero` in coding style.
- **Single Sans-I/O `Vad` engine vs three-type split (silero pattern)** — initially proposed silero's three-type split (`Session` / `StreamState` / `SpeechDetector`) for testability. Reverted to single-engine after user feedback: the model fixes batch=1 (no multi-stream), and Sans-I/O makes the API easier to drive from non-Rust hosts (FFI, async runtimes).
- **Closure callbacks vs Sans-I/O `poll_event`** — closures complicate borrow semantics across multi-state-machine pipelines and don't compose with async drivers. Sans-I/O matches modern Rust patterns (quinn, h2, prost-driven gRPC). `drain_events(F)` is provided as a thin convenience for callers that do want a closure.
- **`merge_silence_duration` knob** — speculatively added in an earlier draft; dropped because silence between speech runs IS the segmentation boundary the caller wants. Adding cross-silence merging would conflate "this speech run" with "this conversation" and is the caller's job if needed.
- **Heap-allocated buffers vs stack arrays** — even small fixed-size buffers (`[f32; 400]`) are heap-allocated as `Vec<f32>` to keep async-runtime stacks well under their 64 KB–256 KB ceilings. Penalty is one allocation per `Vad` (lifetime-amortized).
- **`Duration` for time-valued config vs raw frame counts** — `Duration` is the silero idiom; `humantime-serde` round-trips human-readable strings under the optional `serde` feature. Conversion to frames is centralized at the `Postprocessor` boundary.
- **No `Mode` enum** — earlier drafts exposed a `Mode { VeryPermissive, Permissive, Aggressive, VeryAggressive }` enum that overlaid three numeric fields. Dropped because the presets are just three-field assignments — wrapping them in an enum complicates the API (extra `with_mode` method, "mode is applied not stored" caveats, parity edge cases) for no benefit. Callers configure `speech_threshold`, `min_speech_duration`, and `min_silence_duration` directly; documentation lists the four upstream presets verbatim as copy-paste recipes.
- **Re-export `ort::session::builder::GraphOptimizationLevel` instead of redefining** — silero treats it as a foreign type and bridges to serde via a private mirror enum. We do the same: callers get the same vocabulary as everyone else using `ort` directly, and we don't have to maintain a parallel enum that drifts from upstream.

---

## 13. Open questions

None. All design questions are resolved. Implementation can proceed.
