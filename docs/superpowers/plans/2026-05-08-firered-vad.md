# firered-vad Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Rust crate `firered-vad` that wraps the FireRedVAD streaming model and emits continuous human-speech `SpeechSegment` ranges suitable for slicing into Whisper.

**Architecture:** A single Sans-I/O state-machine type (`Vad`) that combines a private `OrtRunner` (ONNX Runtime via `ort`), a private `FeatureExtractor` (pure-Rust Kaldi-compatible Mel-fbank + CMVN), and a private `Postprocessor` (bit-for-bit port of upstream Python `StreamVadPostprocessor`). Callers push f32 PCM at 16 kHz and pull `VadEvent`s. No closures, no dyn dispatch, no `Mode` enum.

**Tech Stack:** Rust 2024 edition, MSRV 1.85, `ort` 2.0.0-rc.12 for ONNX Runtime, `rustfft` 6 (concrete `Radix2<f32>`), `thiserror` 2, optional `serde` + `humantime-serde` matching silero's serde idiom.

**Spec:** `docs/superpowers/specs/2026-05-08-firered-vad-design.md` (committed at `9065b3b`).

**Reference crate:** `/Users/user/Develop/findit-studio/silero` — sibling project by the same author. We mirror its coding style, file layout conventions, and `ort` usage patterns. Read silero's `src/session.rs`, `src/options.rs`, `src/error.rs`, and `Cargo.toml` whenever you need a concrete pattern reference.

**Upstream Python:** `/Users/user/Develop/findit-studio/FireRedVAD` — read `fireredvad/core/stream_vad_postprocessor.py`, `fireredvad/stream_vad.py`, `fireredvad/core/audio_feat.py`, `fireredvad/core/constants.py` whenever the spec references upstream behavior.

---

## File structure (created or modified)

| File | Action | Responsibility |
| --- | --- | --- |
| `Cargo.toml` | Replace | Package metadata, deps, features (replaces template-rs config) |
| `build.rs` | Replace | Tarpaulin cfg-detection only |
| `rustfmt.toml` | Replace | 100-col, 2-space, silero rules |
| `.gitignore` | Replace | Standard Rust + parity-harness artifacts |
| `.codecov.yml` | Replace | Coverage config matching silero |
| `README.md` | Replace | User-facing docs, usage example |
| `CHANGELOG.md` | Replace | 0.1.0 entry |
| `THIRD_PARTY_NOTICES.md` | Create | Apache-2.0 attribution for bundled artifacts |
| `models/fireredvad_stream_vad_with_cache.onnx` | Create (vendor) | ONNX model bytes, ~2.3 MB |
| `models/cmvn.ark` | Create (vendor) | Kaldi CMVN stats, ~1.3 KB |
| `src/lib.rs` | Replace | Module decls + re-exports + crate docs |
| `src/error.rs` | Create | `Error` + `Result` |
| `src/event.rs` | Create | `VadEvent`, `SpeechSegment`, `FrameResult` |
| `src/options.rs` | Create | `VadOptions`, `SessionOptions`, serde bridges |
| `src/features.rs` | Create | `Cmvn`, `MelFilterbank`, `FeatureExtractor` |
| `src/inference.rs` | Create | `OrtRunner` ONNX wrapper |
| `src/detector.rs` | Create | `Postprocessor` (4-state machine + smoothing) |
| `src/vad.rs` | Create | `Vad` (Sans-I/O orchestrator) |
| `tests/integration_test.rs` | Create | End-to-end black-box tests |
| `examples/streaming.rs` | Create | Synthetic streaming demo |
| `examples/detect_file.rs` | Create | WAV-file demo |
| `.github/workflows/ci.yml` | Replace | Silero-style CI matrix |
| `tests/foo.rs`, `examples/foo.rs`, `benches/foo.rs` | Delete | Template stubs |
| `benches/` | Delete | No v1 perf targets |
| `ci/` | Delete | Template sanitizers/Miri scripts |
| `.github/workflows/loc.yml` | Delete | LoC tracker not in scope |
| `.github/dependabot.yml` | Delete | Template bot config |
| `.github/FUNDING.yml` | Delete | Template author's sponsorship |

---

## Pre-flight reading

Before starting Task 1, read these once for shared context:

- `docs/superpowers/specs/2026-05-08-firered-vad-design.md` — the full design.
- `/Users/user/Develop/findit-studio/silero/src/options.rs` — silero's serde idiom and builder pattern.
- `/Users/user/Develop/findit-studio/silero/src/session.rs` — silero's `ort` 2.0.0-rc.12 usage (`OrtSession::builder`, `TensorRef::from_array_view`, `try_extract_tensor`).
- `/Users/user/Develop/findit-studio/silero/Cargo.toml` — feature flags, MSRV, lint config.
- `/Users/user/Develop/findit-studio/FireRedVAD/fireredvad/core/stream_vad_postprocessor.py` — the postprocessor logic we're porting bit-for-bit.

---

## Task 1: Wipe the template scaffold

**Files:**
- Delete: `src/lib.rs`, `tests/foo.rs`, `examples/foo.rs`, `benches/foo.rs`, `ci/sanitizer.sh`, `ci/miri_tb.sh`, `ci/miri_sb.sh`, `.github/workflows/ci.yml`, `.github/workflows/loc.yml`, `.github/dependabot.yml`, `.github/FUNDING.yml`, `README.md`, `CHANGELOG.md`
- Delete (directories, after they're empty): `benches/`, `ci/`

- [ ] **Step 1: Verify we're starting from a clean working tree**

Run: `git status`
Expected: `nothing to commit, working tree clean` on branch `0.1.0`.

- [ ] **Step 2: Delete template source/test/example/bench files**

```bash
rm src/lib.rs tests/foo.rs examples/foo.rs benches/foo.rs
rmdir benches
```

- [ ] **Step 3: Delete template CI scripts and workflows**

```bash
rm ci/sanitizer.sh ci/miri_tb.sh ci/miri_sb.sh
rmdir ci
rm .github/workflows/ci.yml .github/workflows/loc.yml
rm .github/dependabot.yml .github/FUNDING.yml
```

- [ ] **Step 4: Delete template README and CHANGELOG**

```bash
rm README.md CHANGELOG.md
```

- [ ] **Step 5: Verify git sees the deletions**

Run: `git status`
Expected: a list of `deleted:` entries for every file removed in steps 2-4.

- [ ] **Step 6: Stage and commit the wipe**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: wipe template-rs scaffold

Removes the al8n/template-rs scaffolding so the next commits can
introduce firered-vad-shaped sources matching the silero sibling crate.
No application code yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: New Cargo.toml + build.rs + rustfmt.toml + .gitignore + .codecov.yml

**Files:**
- Modify: `Cargo.toml`
- Modify: `build.rs`
- Modify: `rustfmt.toml`
- Modify: `.gitignore`
- Modify: `.codecov.yml`

- [ ] **Step 1: Write the new Cargo.toml**

Write `Cargo.toml`:

```toml
[package]
name = "firered-vad"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT OR Apache-2.0"
description = "Streaming Voice Activity Detection wrapping the FireRedVAD model via ONNX Runtime"
repository = "https://github.com/uqio/firered-vad"
documentation = "https://docs.rs/firered-vad"
readme = "README.md"
keywords = ["vad", "voice-activity-detection", "fireredvad", "speech", "audio"]
categories = ["multimedia::audio", "science"]

[features]
default = ["bundled"]
bundled = []
serde = ["dep:serde", "dep:humantime-serde"]
coreml = ["ort/coreml"]
directml = ["ort/directml"]
cuda = ["ort/cuda"]
rocm = ["ort/rocm"]
tensorrt = ["ort/tensorrt"]
openvino = ["ort/openvino"]

[dependencies]
ort = { version = "=2.0.0-rc.12", default-features = false, features = ["std", "ndarray"] }
thiserror = "2"
rustfft = "6"

serde = { version = "1", optional = true, features = ["derive"] }
humantime-serde = { version = "1", optional = true }

[dev-dependencies]
hound = "3"
serde_json = "1"

[lints.rust]
unsafe_code = "forbid"
missing_docs = "deny"
rust_2018_idioms = "warn"
single_use_lifetimes = "warn"
unexpected_cfgs = { level = "warn", check-cfg = ["cfg(tarpaulin)"] }

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

- [ ] **Step 2: Write the new build.rs**

Write `build.rs`:

```rust
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_TARPAULIN").is_some()
        || std::env::var_os("CARGO_TARPAULIN").is_some()
        || std::env::var_os("CARGO_CFG_TARPAULIN").is_some()
    {
        println!("cargo:rustc-cfg=tarpaulin");
    }
}
```

- [ ] **Step 3: Write the new rustfmt.toml**

Write `rustfmt.toml`:

```toml
edition = "2024"
max_width = 100
hard_tabs = false
tab_spaces = 2
newline_style = "Unix"
use_field_init_shorthand = true
use_try_shorthand = true
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true
reorder_modules = true
merge_derives = true
force_explicit_abi = true
fn_params_layout = "Tall"
```

- [ ] **Step 4: Write the new .gitignore**

Write `.gitignore`:

```gitignore
/target
Cargo.lock
.DS_Store
.idea/
.vscode/
.claude/
tests/parity/out/
tests/parity/python/.venv/
tests/parity/python/uv.lock
*.pyc
__pycache__/
```

- [ ] **Step 5: Write the new .codecov.yml**

Write `.codecov.yml`:

```yaml
coverage:
  status:
    project:
      default:
        target: auto
        threshold: 1%
    patch:
      default:
        target: auto
        threshold: 1%

ignore:
  - "benches/"
  - "examples/"
  - "tests/"
  - "build.rs"

codecov:
  require_ci_to_pass: true
  notify:
    require_ci_to_pass: true

comment:
  layout: "reach,diff,flags,tree"
  behavior: default
  require_changes: false
  require_base: false
  require_head: false
```

- [ ] **Step 6: Verify the package metadata parses**

Run: `cargo metadata --no-deps --format-version=1 > /dev/null`
Expected: exit code 0, no output.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml build.rs rustfmt.toml .gitignore .codecov.yml
git commit -m "$(cat <<'EOF'
chore: rewrite Cargo.toml, build.rs, rustfmt, gitignore, codecov

Switches from al8n/template-rs config to a silero-shaped sibling:
edition 2024, MSRV 1.85, dual MIT/Apache-2.0, ort 2.0.0-rc.12 with
ndarray feature, rustfft for the pure-Rust Mel-fbank, thiserror 2.
Optional serde and humantime-serde mirror silero's idiom.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Vendor the ONNX model and CMVN

**Files:**
- Create: `models/fireredvad_stream_vad_with_cache.onnx`
- Create: `models/cmvn.ark`

- [ ] **Step 1: Create the models/ directory**

```bash
mkdir -p models
```

- [ ] **Step 2: Copy the ONNX model from upstream**

```bash
cp /Users/user/Develop/findit-studio/FireRedVAD/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx \
   models/fireredvad_stream_vad_with_cache.onnx
```

- [ ] **Step 3: Copy the CMVN stats from upstream**

```bash
cp /Users/user/Develop/findit-studio/FireRedVAD/pretrained_models/onnx_models/cmvn.ark \
   models/cmvn.ark
```

- [ ] **Step 4: Verify file sizes match**

Run: `ls -la models/`
Expected: `fireredvad_stream_vad_with_cache.onnx` ≈ 2.3 MB; `cmvn.ark` ≈ 1311 bytes.

- [ ] **Step 5: Commit (add binary files separately so git diff stays readable)**

```bash
git add models/fireredvad_stream_vad_with_cache.onnx models/cmvn.ark
git commit -m "$(cat <<'EOF'
chore: vendor FireRedVAD streaming ONNX + CMVN stats

Bundled assets are derived from
https://github.com/FireRedTeam/FireRedVAD (Xiaohongshu, Apache-2.0).
Attribution lives in the next commit's THIRD_PARTY_NOTICES.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: THIRD_PARTY_NOTICES.md

**Files:**
- Create: `THIRD_PARTY_NOTICES.md`

- [ ] **Step 1: Write the preamble half of THIRD_PARTY_NOTICES.md**

Write `THIRD_PARTY_NOTICES.md` with **only** this preamble content (the license body is appended in Step 2):

```markdown
# Third-Party Notices

This crate vendors and embeds artifacts derived from third-party projects.
Each entry below names the artifact, its upstream source, and the license
that governs its use.

## FireRedVAD

- **Artifacts:** `models/fireredvad_stream_vad_with_cache.onnx`,
  `models/cmvn.ark`.
- **Upstream:** https://github.com/FireRedTeam/FireRedVAD
- **Authors / copyright:** Xiaohongshu — Kaituo Xu, Wenpeng Li, Kai Huang,
  Kun Liu.
- **License:** Apache License, Version 2.0.

The full text of the Apache 2.0 license, as published by the FireRedVAD
project, is reproduced below. The Rust source code in this crate
(`src/`, `tests/`, `examples/`) is the original work of the crate
authors and is licensed under MIT OR Apache-2.0 — see `LICENSE-MIT`
and `LICENSE-APACHE` for details.

---
```

- [ ] **Step 2: Append the verbatim Apache-2.0 license body from upstream**

Run:

```bash
{
  cat THIRD_PARTY_NOTICES.md
  printf '\n```\n'
  cat /Users/user/Develop/findit-studio/FireRedVAD/LICENSE
  printf '\n```\n'
} > THIRD_PARTY_NOTICES.md.tmp && mv THIRD_PARTY_NOTICES.md.tmp THIRD_PARTY_NOTICES.md
```

This appends `\n```\n<contents of upstream LICENSE>\n```\n` to the preamble, producing a single self-contained Markdown file with the full Apache 2.0 license body fenced as a code block.

- [ ] **Step 3: Verify the file contains the full license**

Run: `grep -c "Apache License" THIRD_PARTY_NOTICES.md`
Expected: at least `2` (one in the preamble, at least one in the embedded license body).

Run: `grep -c "Licensed under the Apache License, Version 2.0" THIRD_PARTY_NOTICES.md`
Expected: at least `1` (the canonical attribution line embedded by upstream).

- [ ] **Step 4: Commit**

```bash
git add THIRD_PARTY_NOTICES.md
git commit -m "$(cat <<'EOF'
docs: add THIRD_PARTY_NOTICES with Apache-2.0 attribution

Apache-2.0 attribution for the bundled FireRedVAD ONNX model and CMVN
stats, with author/copyright/upstream-URL identification and full
license-text reproduction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: src/error.rs

**Files:**
- Create: `src/error.rs`

- [ ] **Step 1: Create the file with the error enum**

Write `src/error.rs`:

```rust
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

  /// The caller pushed PCM at a sample rate the model does not support.
  #[error("input PCM sample rate is unsupported (model is fixed at {expected} Hz)")]
  UnsupportedSampleRate {
    /// The sample rate the model expects (always 16_000 for FireRedVAD).
    expected: u32,
  },

  /// An ONNX output tensor had an unexpected shape.
  #[error("ONNX output {tensor} had unexpected shape {shape:?}")]
  UnexpectedOutputShape {
    /// The output tensor name (e.g. `"probs"`, `"caches_out"`).
    tensor: &'static str,
    /// The actual shape returned by the ONNX runtime.
    shape: Vec<i64>,
  },

  /// A speech-threshold value was outside the valid `[0, 1]` range and could not be sanitized.
  #[error("invalid speech threshold {value} (must be in [0, 1])")]
  InvalidSpeechThreshold {
    /// The offending value.
    value: f32,
  },
}

/// Convenience alias for `Result<T, firered_vad::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn unsupported_sample_rate_displays_expected_value() {
    let err = Error::UnsupportedSampleRate { expected: 16_000 };
    assert_eq!(
      err.to_string(),
      "input PCM sample rate is unsupported (model is fixed at 16000 Hz)"
    );
  }

  #[test]
  fn invalid_cmvn_carries_static_reason() {
    let err = Error::InvalidCmvn { reason: "missing magic" };
    assert!(err.to_string().contains("missing magic"));
  }

  #[test]
  fn unexpected_output_shape_renders_shape() {
    let err = Error::UnexpectedOutputShape { tensor: "probs", shape: vec![1, 2, 3] };
    assert!(err.to_string().contains("probs"));
    assert!(err.to_string().contains("[1, 2, 3]"));
  }
}
```

- [ ] **Step 2: Verify the file compiles in isolation (lib.rs doesn't exist yet, so use `--lib` + a stub)**

Skip this step — `cargo check` will run after Task 9 once `lib.rs` is in place. Move on.

- [ ] **Step 3: Commit**

```bash
git add src/error.rs
git commit -m "$(cat <<'EOF'
feat(error): add Error enum and Result alias

ort::Error is exposed transparently via #[from]; model-specific failures
get explicit variants for callers that want to act on them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: src/options.rs — `SessionOptions` + serde bridge for `GraphOptimizationLevel`

**Files:**
- Create: `src/options.rs`

- [ ] **Step 1: Write the file scaffold + the `GraphOptimizationLevel` serde bridge**

Write `src/options.rs`:

```rust
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
  use super::GraphOptimizationLevel;
  use serde::*;

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
/// should be applied to a manually built [`ort::Session`] passed into
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

  /// Builder variant of [`set_optimization_level`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }
}

// VadOptions follows in Task 7.
#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn session_options_default_optimizes_at_level_3() {
    let opts = SessionOptions::default();
    assert!(matches!(opts.optimization_level(), GraphOptimizationLevel::Level3));
  }

  #[test]
  fn session_options_with_optimization_level_overrides() {
    let opts = SessionOptions::new().with_optimization_level(GraphOptimizationLevel::Level1);
    assert!(matches!(opts.optimization_level(), GraphOptimizationLevel::Level1));
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/options.rs
git commit -m "$(cat <<'EOF'
feat(options): add SessionOptions + GraphOptimizationLevel serde bridge

Re-exports ort::session::builder::GraphOptimizationLevel and bridges it
to serde via a private mirror enum, matching silero's pattern. Tests
pin the default optimization level (Level3) and the with_* setter.
VadOptions arrives in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: src/options.rs — `VadOptions`

**Files:**
- Modify: `src/options.rs`

- [ ] **Step 1: Append the helpers + `VadOptions` definition**

Append the following to `src/options.rs` (after the line `// VadOptions follows in Task 7.` and before the `#[cfg(test)]` block):

```rust
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

  /// Builder variant of [`set_smooth_window_size`].
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

  /// Builder variant of [`set_speech_threshold`].
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

  /// Builder variant of [`set_pad_start`].
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

  /// Builder variant of [`set_min_speech_duration`].
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

  /// Builder variant of [`set_max_speech_duration`].
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

  /// Builder variant of [`set_min_silence_duration`].
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

  /// Builder variant of [`set_session_options`].
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
```

- [ ] **Step 2: Append targeted unit tests inside the existing `tests` module**

Append inside `mod tests { ... }` (after the existing two SessionOptions tests):

```rust
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
```

- [ ] **Step 3: Commit**

```bash
git add src/options.rs
git commit -m "$(cat <<'EOF'
feat(options): add VadOptions matching upstream defaults

Per-field humantime-serde and serde defaults match silero's idiom.
Sample-domain conversions (pad_start_frames, min_speech_frames, etc.)
are pub(crate) so the postprocessor can consume them without re-doing
the Duration arithmetic. pad_start is clamped to >= smooth_window_size,
matching upstream StreamVadPostprocessor.__init__.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: src/event.rs — `SpeechSegment`

**Files:**
- Create: `src/event.rs`

- [ ] **Step 1: Write the file with `SpeechSegment` only (the rest of the module follows in Task 9)**

Write `src/event.rs`:

```rust
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
```

- [ ] **Step 2: Commit**

```bash
git add src/event.rs
git commit -m "$(cat <<'EOF'
feat(event): add SpeechSegment with sample-domain accessors

start_sample inclusive, end_sample exclusive. range_usize() slices
PCM directly. duration/start/end return std::time::Duration matching
silero's accessor style.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: src/event.rs — `FrameResult` and `VadEvent`; minimal `src/lib.rs` so `cargo check` works

**Files:**
- Modify: `src/event.rs`
- Create: `src/lib.rs`

- [ ] **Step 1: Append `FrameResult` and `VadEvent` to `src/event.rs`**

Append to `src/event.rs` (before the `#[cfg(test)]` block):

```rust
/// One frame's view of the streaming detector's internal state.
///
/// This is a pure value type — readers can store and compare instances.
/// Frame indices are 0-based; upstream Python uses 1-based indices and
/// we shift on construction. `speech_start_frame` and `speech_end_frame`
/// are the 0-based frame indices of the most-recent segment opening and
/// closing seen so far (`None` until a segment has actually opened).
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

/// Events produced by [`crate::Vad`] in response to streamed PCM.
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
  /// One per 10 ms frame consumed; carries the full per-frame state.
  Frame(FrameResult),
  /// One per closed continuous human-speech window; carries the segment
  /// suitable for slicing the original PCM.
  SegmentClosed(SpeechSegment),
}
```

- [ ] **Step 2: Append targeted unit tests**

Append inside the existing `mod tests` block in `src/event.rs`:

```rust
  #[test]
  fn closed_segment_is_some_only_when_is_speech_end() {
    let result = FrameResult::new(20, 0.9, 0.85, true, false, true, Some(2), Some(20));
    let segment = result.closed_segment().expect("segment closes");
    assert_eq!(segment.start_sample(), 2 * 160);
    assert_eq!(segment.end_sample(), 20 * 160);

    let mid = FrameResult::new(15, 0.8, 0.75, true, false, false, Some(2), None);
    assert!(mid.closed_segment().is_none());
  }

  #[test]
  fn timestamp_uses_frame_shift_samples() {
    let result = FrameResult::new(100, 0.0, 0.0, false, false, false, None, None);
    assert_eq!(result.timestamp(), Duration::from_millis(1_000));
  }

  #[test]
  fn vad_event_carries_payload() {
    let f = FrameResult::new(1, 0.0, 0.0, false, false, false, None, None);
    let s = SpeechSegment::new(0, 160);
    assert_eq!(VadEvent::Frame(f), VadEvent::Frame(f));
    assert_eq!(VadEvent::SegmentClosed(s), VadEvent::SegmentClosed(s));
  }
```

- [ ] **Step 3: Write a minimal `src/lib.rs` so `cargo check` is green**

Write `src/lib.rs`:

```rust
//! Streaming Voice Activity Detection wrapping the FireRedVAD model via ONNX Runtime.
//!
//! See `docs/superpowers/specs/2026-05-08-firered-vad-design.md` for the
//! full design. The remaining modules are landed in subsequent commits;
//! at this point in the implementation the crate exposes only the value
//! types from `error` and `event` plus the configuration types from
//! `options`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod error;
mod event;
mod options;

pub use error::{Error, Result};
pub use event::{FrameResult, SpeechSegment, VadEvent};
pub use options::{GraphOptimizationLevel, SessionOptions, VadOptions};

/// Crate version (matches `CARGO_PKG_VERSION`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

- [ ] **Step 4: Run `cargo check --all-features` to validate the new types**

Run: `cargo check --all-features`
Expected: clean compile, no warnings (since `lints.rust` denies `missing_docs` and warns on others).

- [ ] **Step 5: Run the unit tests for the modules landed so far**

Run: `cargo test --all-features --lib`
Expected: all error/options/event tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/event.rs src/lib.rs
git commit -m "$(cat <<'EOF'
feat(event,lib): add FrameResult, VadEvent, minimal lib.rs

Lands FrameResult (with const-fn accessors and timestamp() / closed_segment()
helpers) and the VadEvent enum that the Sans-I/O surface emits. lib.rs
re-exports error/event/options modules; further modules land later.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: src/features.rs — Cmvn parser

**Files:**
- Create: `src/features.rs`
- Modify: `src/lib.rs` (add `mod features;` declaration; module stays `pub(crate)` — not re-exported)

- [ ] **Step 1: Add the `features` module to `src/lib.rs`**

Edit `src/lib.rs`: add `mod features;` after `mod event;` so module declarations stay alphabetical:

```rust
mod error;
mod event;
mod features;
mod options;
```

- [ ] **Step 2: Write the file with the Cmvn parser only (mel-fbank arrives in Task 11)**

Write `src/features.rs`:

```rust
//! Pure-Rust Kaldi-compatible Mel filterbank + CMVN preprocessing.
//!
//! All public types here are `pub(crate)` — feature extraction is an
//! implementation detail of [`crate::Vad`].

use crate::error::{Error, Result};

/// Number of Mel filterbank bins the model expects.
pub(crate) const NUM_MEL_BINS: usize = 80;

/// Cepstral Mean and Variance Normalization stats parsed from a Kaldi
/// `.ark` file. The 80-dim means and inverse-std-variances are applied
/// to each Mel-fbank feature vector before it is fed to the model.
#[derive(Debug, Clone)]
pub(crate) struct Cmvn {
  means: Vec<f32>,
  inverse_std_variances: Vec<f32>,
}

impl Cmvn {
  /// Parse a Kaldi binary double-matrix `.ark`.
  ///
  /// Format expected:
  ///
  /// ```text
  /// \0B            (2 bytes, magic)
  /// "DM "          (3 bytes, double-matrix marker — note trailing space)
  /// \x04 + i32_le  (5 bytes, rows)
  /// \x04 + i32_le  (5 bytes, cols)
  /// rows*cols*8 bytes f64 LE
  /// ```
  ///
  /// `rows` must be 2 (sums and sum-squares). `cols` must be `NUM_MEL_BINS + 1`
  /// (80 stat columns plus a count column at index 80). `count` lives at
  /// `data[0][NUM_MEL_BINS]`. Each mean is `sums[d] / count`; each
  /// inverse-std-variance is `1 / sqrt(max(1e-20, sum_sq[d]/count - mean[d]^2))`.
  pub(crate) fn from_ark_bytes(bytes: &[u8]) -> Result<Self> {
    let mut i: usize = 0;
    let need = |i: usize, n: usize| -> Result<()> {
      if bytes.len() < i + n {
        Err(Error::InvalidCmvn { reason: "truncated header" })
      } else {
        Ok(())
      }
    };

    need(i, 2)?;
    if &bytes[i..i + 2] != b"\x00B" {
      return Err(Error::InvalidCmvn { reason: "missing \\0B magic" });
    }
    i += 2;

    need(i, 3)?;
    if &bytes[i..i + 3] != b"DM " {
      return Err(Error::InvalidCmvn { reason: "expected double-matrix marker 'DM '" });
    }
    i += 3;

    need(i, 1)?;
    if bytes[i] != 4 {
      return Err(Error::InvalidCmvn { reason: "expected 4-byte int32 size token before rows" });
    }
    i += 1;
    need(i, 4)?;
    let rows = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    i += 4;
    if rows != 2 {
      return Err(Error::InvalidCmvn { reason: "expected exactly 2 rows (sums, sum_sqs)" });
    }

    need(i, 1)?;
    if bytes[i] != 4 {
      return Err(Error::InvalidCmvn { reason: "expected 4-byte int32 size token before cols" });
    }
    i += 1;
    need(i, 4)?;
    let cols = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    i += 4;
    if cols != (NUM_MEL_BINS as i32) + 1 {
      return Err(Error::InvalidCmvn { reason: "expected NUM_MEL_BINS + 1 columns" });
    }

    let total = (rows as usize) * (cols as usize) * 8;
    need(i, total)?;
    let mut data = Vec::with_capacity((rows as usize) * (cols as usize));
    let mut p = i;
    for _ in 0..(rows as usize) * (cols as usize) {
      let chunk = [
        bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3],
        bytes[p + 4], bytes[p + 5], bytes[p + 6], bytes[p + 7],
      ];
      data.push(f64::from_le_bytes(chunk));
      p += 8;
    }

    let count = data[NUM_MEL_BINS]; // first row, last column
    if !(count.is_finite() && count >= 1.0) {
      return Err(Error::InvalidCmvn { reason: "non-positive CMVN count" });
    }

    let mut means = Vec::with_capacity(NUM_MEL_BINS);
    let mut inverse_std_variances = Vec::with_capacity(NUM_MEL_BINS);
    let row_stride = cols as usize;
    for d in 0..NUM_MEL_BINS {
      let sum = data[d];
      let sum_sq = data[row_stride + d];
      let mean = sum / count;
      let mut var = sum_sq / count - mean * mean;
      if var < 1e-20 {
        var = 1e-20;
      }
      let istd = 1.0 / var.sqrt();
      means.push(mean as f32);
      inverse_std_variances.push(istd as f32);
    }

    Ok(Self { means, inverse_std_variances })
  }

  /// Apply CMVN in place to one 80-dim feature vector.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn apply(&self, feature: &mut [f32]) {
    debug_assert_eq!(feature.len(), NUM_MEL_BINS);
    for d in 0..NUM_MEL_BINS {
      feature[d] = (feature[d] - self.means[d]) * self.inverse_std_variances[d];
    }
  }

  pub(crate) fn means(&self) -> &[f32] {
    &self.means
  }

  pub(crate) fn inverse_std_variances(&self) -> &[f32] {
    &self.inverse_std_variances
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The bundled CMVN file is the most reliable parity reference.
  const BUNDLED_CMVN: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/cmvn.ark"));

  #[test]
  fn parses_bundled_cmvn_into_80_means_and_istds() {
    let cmvn = Cmvn::from_ark_bytes(BUNDLED_CMVN).expect("parse cmvn");
    assert_eq!(cmvn.means().len(), NUM_MEL_BINS);
    assert_eq!(cmvn.inverse_std_variances().len(), NUM_MEL_BINS);
    // Means should be roughly in log-mel-energy range; pin the first one so
    // future regressions in parsing immediately surface.
    let first_mean = cmvn.means()[0];
    assert!(first_mean > 5.0 && first_mean < 20.0, "first mean = {first_mean}");
  }

  #[test]
  fn rejects_truncated_input() {
    let bytes = b"\x00BDM ";
    assert!(matches!(
      Cmvn::from_ark_bytes(bytes),
      Err(Error::InvalidCmvn { .. })
    ));
  }

  #[test]
  fn rejects_wrong_magic() {
    let mut bytes = b"\x00BDM \x04\x02\x00\x00\x00\x04\x51\x00\x00\x00".to_vec();
    bytes[0] = 0xFF;
    assert!(matches!(
      Cmvn::from_ark_bytes(&bytes),
      Err(Error::InvalidCmvn { reason: r }) if r.contains("magic")
    ));
  }

  #[test]
  fn apply_subtracts_mean_and_divides_by_std() {
    let cmvn = Cmvn {
      means: vec![1.0; NUM_MEL_BINS],
      inverse_std_variances: vec![2.0; NUM_MEL_BINS],
    };
    let mut feature = vec![3.0f32; NUM_MEL_BINS];
    cmvn.apply(&mut feature);
    for value in &feature {
      assert!((*value - 4.0).abs() < f32::EPSILON);
    }
  }
}
```

- [ ] **Step 3: Run the new tests**

Run: `cargo test --all-features --lib features::tests`
Expected: all 4 tests pass; `parses_bundled_cmvn_into_80_means_and_istds` confirms the parser handles the real vendored file.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/features.rs
git commit -m "$(cat <<'EOF'
feat(features): add Cmvn parser for Kaldi binary double-matrix .ark

Tests pin parsing of the bundled cmvn.ark, header validation for
truncated/wrong-magic inputs, and the apply() in-place transform.
NUM_MEL_BINS = 80; further mel-fbank machinery arrives in Task 11.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: src/features.rs — `MelFilterbank` (window + Mel filters + FFT)

**Files:**
- Modify: `src/features.rs`

This task lands all the math for one feature frame. We do **not** wire the PCM tail buffer or `extract_one` orchestration yet — that's Task 12.

- [ ] **Step 1: Add the constants and struct definitions at the top of `src/features.rs`, just after `NUM_MEL_BINS`**

```rust
/// Sample rate the model expects.
pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;

/// Number of samples in a 25 ms analysis window.
pub(crate) const FRAME_LENGTH_SAMPLES: usize = 400;

/// Number of samples between successive 10 ms frame starts.
pub(crate) const FRAME_SHIFT_SAMPLES: usize = 160;

/// FFT length used for the mel filterbank (next power of 2 ≥ FRAME_LENGTH_SAMPLES).
pub(crate) const FFT_SIZE: usize = 512;

/// Number of unique non-redundant FFT bins (`FFT_SIZE / 2 + 1`).
pub(crate) const FFT_BINS: usize = FFT_SIZE / 2 + 1;

/// Pre-emphasis coefficient (Kaldi default; upstream keeps the default).
const PRE_EMPHASIS: f32 = 0.97;

/// Mel-bin range (low_freq=20, high_freq=Nyquist for 16 kHz).
const MEL_LOW_FREQ_HZ: f32 = 20.0;
const MEL_HIGH_FREQ_HZ: f32 = 8_000.0;

/// Floor for the log of bin energies (Kaldi `log_floor`).
const LOG_FLOOR: f32 = 1e-20;

/// One sparse triangular Mel filter, addressed by `start_bin` and `weights`.
#[derive(Debug, Clone)]
struct MelFilter {
  start_bin: usize,
  weights: Vec<f32>,
}

/// Pure-Rust Kaldi-compatible Mel filterbank.
///
/// Configuration is hard-coded to match upstream FireRedVAD exactly:
/// 16 kHz, 25 ms / 10 ms windows, 80 mel bins, Povey window,
/// pre-emphasis 0.97, DC removal on, snip_edges=true, log floor 1e-20.
#[derive(Debug)]
pub(crate) struct MelFilterbank {
  fft: rustfft::algorithm::Radix2<f32>,
  fft_buf: Vec<rustfft::num_complex::Complex<f32>>,
  povey_window: Vec<f32>,
  filters: Vec<MelFilter>,
}
```

- [ ] **Step 2: Implement the constructor and helper functions**

Append to `src/features.rs` (after the struct definitions):

```rust
#[cfg_attr(not(tarpaulin), inline(always))]
fn hz_to_mel(hz: f32) -> f32 {
  // Kaldi/HTK convention: 1127 * ln(1 + f/700)
  1127.0 * (1.0 + hz / 700.0).ln()
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn mel_to_hz(mel: f32) -> f32 {
  700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Centre frequency of the `i`-th non-redundant FFT bin in Hz.
#[cfg_attr(not(tarpaulin), inline(always))]
fn fft_bin_hz(i: usize) -> f32 {
  (i as f32) * (SAMPLE_RATE_HZ as f32) / (FFT_SIZE as f32)
}

fn build_povey_window() -> Vec<f32> {
  let n = FRAME_LENGTH_SAMPLES;
  let a = std::f32::consts::TAU / ((n - 1) as f32);
  (0..n)
    .map(|i| (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85))
    .collect()
}

fn build_mel_filters() -> Vec<MelFilter> {
  let mel_low = hz_to_mel(MEL_LOW_FREQ_HZ);
  let mel_high = hz_to_mel(MEL_HIGH_FREQ_HZ);
  let mel_step = (mel_high - mel_low) / (NUM_MEL_BINS as f32 + 1.0);

  // The (NUM_MEL_BINS + 2) Mel-frequency anchor points spanning the band:
  // `points[b]` is the left edge of filter b-1, the centre of filter b, and
  // the right edge of filter b+1 (for the matching `b` indices).
  let mut hz_points = Vec::with_capacity(NUM_MEL_BINS + 2);
  for k in 0..(NUM_MEL_BINS + 2) {
    hz_points.push(mel_to_hz(mel_low + (k as f32) * mel_step));
  }

  let mut filters = Vec::with_capacity(NUM_MEL_BINS);
  for b in 0..NUM_MEL_BINS {
    let left = hz_points[b];
    let centre = hz_points[b + 1];
    let right = hz_points[b + 2];

    // Find the FFT bin index range that overlaps [left, right].
    let mut start_bin = FFT_BINS;
    let mut end_bin = 0;
    for i in 0..FFT_BINS {
      let f = fft_bin_hz(i);
      if f > left && f < right {
        if i < start_bin {
          start_bin = i;
        }
        end_bin = i;
      }
    }

    let mut weights = Vec::new();
    if start_bin <= end_bin {
      weights.reserve(end_bin - start_bin + 1);
      for i in start_bin..=end_bin {
        let f = fft_bin_hz(i);
        let w = if f <= centre {
          (f - left) / (centre - left)
        } else {
          (right - f) / (right - centre)
        };
        weights.push(w.max(0.0));
      }
    } else {
      // Filter band falls between FFT bins (very narrow band at low Mel
      // indices); leave it empty — Kaldi behaves the same.
      start_bin = 0;
    }

    filters.push(MelFilter { start_bin, weights });
  }

  filters
}

impl MelFilterbank {
  pub(crate) fn new() -> Self {
    use rustfft::FftDirection;
    Self {
      fft: rustfft::algorithm::Radix2::<f32>::new(FFT_SIZE, FftDirection::Forward),
      fft_buf: vec![rustfft::num_complex::Complex::new(0.0, 0.0); FFT_SIZE],
      povey_window: build_povey_window(),
      filters: build_mel_filters(),
    }
  }

  /// Extract one 80-dim log-Mel feature from a 25 ms window of int16-range
  /// samples. The input is mutated in place (DC removal, pre-emphasis,
  /// windowing happen on a copy inside the FFT buffer; the caller's slice
  /// is **not** mutated).
  pub(crate) fn extract(&mut self, window: &[f32], out: &mut [f32]) {
    debug_assert_eq!(window.len(), FRAME_LENGTH_SAMPLES);
    debug_assert_eq!(out.len(), NUM_MEL_BINS);

    // 1. Copy + remove DC offset.
    let mean: f32 = window.iter().copied().sum::<f32>() / FRAME_LENGTH_SAMPLES as f32;
    let mut samples: [f32; FRAME_LENGTH_SAMPLES] = [0.0; FRAME_LENGTH_SAMPLES];
    for i in 0..FRAME_LENGTH_SAMPLES {
      samples[i] = window[i] - mean;
    }

    // 2. Pre-emphasis: x[i] -= 0.97 * x[i-1] for i = N-1..1; then x[0] -= 0.97 * x[0].
    for i in (1..FRAME_LENGTH_SAMPLES).rev() {
      samples[i] -= PRE_EMPHASIS * samples[i - 1];
    }
    samples[0] -= PRE_EMPHASIS * samples[0];

    // 3. Window with Povey.
    for i in 0..FRAME_LENGTH_SAMPLES {
      samples[i] *= self.povey_window[i];
    }

    // 4. Zero-pad to FFT_SIZE and run the radix-2 FFT.
    for i in 0..FFT_SIZE {
      let re = if i < FRAME_LENGTH_SAMPLES { samples[i] } else { 0.0 };
      self.fft_buf[i].re = re;
      self.fft_buf[i].im = 0.0;
    }
    use rustfft::Fft;
    self.fft.process(&mut self.fft_buf);

    // 5. Power spectrum (|X|^2) for the non-redundant half.
    let mut power: [f32; FFT_BINS] = [0.0; FFT_BINS];
    for i in 0..FFT_BINS {
      let c = self.fft_buf[i];
      power[i] = c.re * c.re + c.im * c.im;
    }

    // 6. Mel filterbank → log.
    for b in 0..NUM_MEL_BINS {
      let f = &self.filters[b];
      let mut energy = 0.0f32;
      for (j, w) in f.weights.iter().enumerate() {
        energy += power[f.start_bin + j] * *w;
      }
      out[b] = energy.max(LOG_FLOOR).ln();
    }
  }
}
```

- [ ] **Step 3: Add unit tests for the building blocks**

Append inside `mod tests` of `src/features.rs`:

```rust
  #[test]
  fn povey_window_endpoints_are_zero_and_centre_is_one() {
    let w = build_povey_window();
    assert_eq!(w.len(), FRAME_LENGTH_SAMPLES);
    assert!(w[0].abs() < 1e-6);
    assert!(w[FRAME_LENGTH_SAMPLES - 1].abs() < 1e-6);
    let centre = (FRAME_LENGTH_SAMPLES - 1) / 2;
    assert!((w[centre] - 1.0).abs() < 1e-3, "centre weight = {}", w[centre]);
  }

  #[test]
  fn mel_filters_cover_the_target_frequency_range() {
    let filters = build_mel_filters();
    assert_eq!(filters.len(), NUM_MEL_BINS);

    // The first filter should start at a low FFT bin.
    let first_centre_hz = fft_bin_hz(filters[0].start_bin + filters[0].weights.len() / 2);
    assert!(first_centre_hz > MEL_LOW_FREQ_HZ);
    assert!(first_centre_hz < 200.0);

    // The last filter should reach close to Nyquist.
    let last = &filters[NUM_MEL_BINS - 1];
    let last_max_bin = last.start_bin + last.weights.len();
    assert!(fft_bin_hz(last_max_bin) > 7_000.0);
  }

  #[test]
  fn mel_filterbank_silence_produces_log_floor_features() {
    let mut bank = MelFilterbank::new();
    let window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    let mut out = vec![0.0f32; NUM_MEL_BINS];
    bank.extract(&window, &mut out);
    let log_floor = LOG_FLOOR.ln();
    for v in &out {
      assert!((*v - log_floor).abs() < 1e-3, "expected log_floor, got {}", v);
    }
  }

  #[test]
  fn mel_filterbank_responds_to_a_pure_tone() {
    let mut bank = MelFilterbank::new();
    let mut window = vec![0.0f32; FRAME_LENGTH_SAMPLES];
    // 1 kHz sinusoid at int16-range amplitude.
    let f = 1_000.0f32;
    let amp = 8_000.0f32;
    for n in 0..FRAME_LENGTH_SAMPLES {
      window[n] = amp * (std::f32::consts::TAU * f * (n as f32) / SAMPLE_RATE_HZ as f32).sin();
    }
    let mut out = vec![0.0f32; NUM_MEL_BINS];
    bank.extract(&window, &mut out);

    // The peak Mel bin should sit somewhere in the lower half of the bank
    // (mel index for 1 kHz is ~28 with these parameters).
    let max_bin = (0..NUM_MEL_BINS).max_by(|a, b| out[*a].partial_cmp(&out[*b]).unwrap()).unwrap();
    assert!((20..40).contains(&max_bin), "peak Mel bin = {max_bin}");
  }
```

- [ ] **Step 4: Run the new tests**

Run: `cargo test --all-features --lib features::tests`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/features.rs
git commit -m "$(cat <<'EOF'
feat(features): add MelFilterbank with Povey window + radix-2 FFT

Pure-Rust Kaldi-compatible 80-bin Mel-fbank: DC removal, pre-emphasis
0.97, Povey window, FFT_SIZE=512 radix-2 (concrete Radix2<f32>, no dyn
dispatch), |X|^2 power spectrum, sparse triangular Mel filters in
[20 Hz, Nyquist], log with 1e-20 floor. Tests pin window endpoints +
centre, filter coverage, the silence -> log_floor invariant, and a
1 kHz tone landing in the expected Mel bin.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: src/features.rs — `FeatureExtractor` (PCM tail buffer + scaling + per-frame orchestration)

**Files:**
- Modify: `src/features.rs`

- [ ] **Step 1: Append the `FeatureExtractor` struct + impl**

Append to `src/features.rs` (after the `MelFilterbank` impl block):

```rust
/// Scale factor applied to incoming PCM before feature extraction.
///
/// Upstream Python reads WAVs as `int16` and feeds raw int16-range
/// values to `kaldi_native_fbank`. We accept f32 in `[-1.0, 1.0]` from
/// callers and multiply by this constant on the way in to keep the
/// downstream filterbank values numerically identical to upstream.
const INT16_SCALE: f32 = 32_768.0;

/// Streaming feature extractor: buffers PCM, emits one 80-dim Mel-fbank
/// feature vector per consumed 10 ms frame.
#[derive(Debug)]
pub(crate) struct FeatureExtractor {
  fbank: MelFilterbank,
  cmvn: Cmvn,
  /// Up to `FRAME_LENGTH_SAMPLES` of pending int16-range samples.
  pcm_tail: Vec<f32>,
  /// Reusable scratch for the 25 ms analysis window.
  window_scratch: Vec<f32>,
  /// Reusable scratch for one 80-dim feature vector.
  feature_scratch: Vec<f32>,
}

impl FeatureExtractor {
  /// Construct from raw CMVN bytes (Kaldi binary `.ark` format).
  pub(crate) fn new(cmvn_bytes: &[u8]) -> Result<Self> {
    Ok(Self {
      fbank: MelFilterbank::new(),
      cmvn: Cmvn::from_ark_bytes(cmvn_bytes)?,
      pcm_tail: Vec::with_capacity(FRAME_LENGTH_SAMPLES),
      window_scratch: vec![0.0; FRAME_LENGTH_SAMPLES],
      feature_scratch: vec![0.0; NUM_MEL_BINS],
    })
  }

  /// Reset all streaming state. Cmvn / fbank / scratch buffers stay allocated.
  pub(crate) fn reset(&mut self) {
    self.pcm_tail.clear();
  }

  /// Append PCM in `[-1.0, 1.0]` range. Internally rescaled to int16-range
  /// to match upstream's input domain.
  pub(crate) fn push_pcm(&mut self, pcm: &[f32]) {
    self.pcm_tail.reserve(pcm.len());
    for &s in pcm {
      self.pcm_tail.push(s * INT16_SCALE);
    }
  }

  /// Number of pending int16-range samples in the tail buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn pending_samples(&self) -> usize {
    self.pcm_tail.len()
  }

  /// True if the tail buffer holds at least one full 25 ms window.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn has_full_window(&self) -> bool {
    self.pcm_tail.len() >= FRAME_LENGTH_SAMPLES
  }

  /// Consume one 25 ms window from the head of the tail and write its
  /// CMVN-normalized 80-dim feature into `out`. Drops the leading
  /// `FRAME_SHIFT_SAMPLES` (10 ms) of the tail so successive calls
  /// produce overlapping 25 ms / 10 ms-hop frames.
  pub(crate) fn extract_one(&mut self, out: &mut [f32]) {
    debug_assert_eq!(out.len(), NUM_MEL_BINS);
    debug_assert!(self.has_full_window());

    // Copy the 25 ms window into reusable scratch (FFT mutates it).
    self.window_scratch.copy_from_slice(&self.pcm_tail[..FRAME_LENGTH_SAMPLES]);

    self.fbank.extract(&self.window_scratch, &mut self.feature_scratch);
    self.cmvn.apply(&mut self.feature_scratch);
    out.copy_from_slice(&self.feature_scratch);

    // Drop the oldest 10 ms (frame shift) so the next call sees the next
    // 25 ms window aligned at +10 ms.
    self.pcm_tail.drain(..FRAME_SHIFT_SAMPLES);
  }
}
```

- [ ] **Step 2: Add tests for the streaming buffer behavior**

Append inside `mod tests`:

```rust
  const BUNDLED_CMVN_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/cmvn.ark"));

  #[test]
  fn feature_extractor_buffers_partial_frames() {
    let mut fx = FeatureExtractor::new(BUNDLED_CMVN_BYTES).expect("init");
    fx.push_pcm(&vec![0.0; 100]);
    assert!(!fx.has_full_window());
    assert_eq!(fx.pending_samples(), 100);

    fx.push_pcm(&vec![0.0; 300]);
    assert!(fx.has_full_window());
    assert_eq!(fx.pending_samples(), 400);

    let mut out = vec![0.0; NUM_MEL_BINS];
    fx.extract_one(&mut out);
    // After consuming one frame, 240 samples (15 ms overlap) remain.
    assert_eq!(fx.pending_samples(), 240);
  }

  #[test]
  fn feature_extractor_emits_consistent_features_for_silence() {
    let mut fx = FeatureExtractor::new(BUNDLED_CMVN_BYTES).expect("init");
    fx.push_pcm(&vec![0.0; FRAME_LENGTH_SAMPLES + 3 * FRAME_SHIFT_SAMPLES]);

    let mut a = vec![0.0; NUM_MEL_BINS];
    let mut b = vec![0.0; NUM_MEL_BINS];
    fx.extract_one(&mut a);
    fx.extract_one(&mut b);
    assert_eq!(a, b, "two consecutive silence frames must produce identical features");
  }

  #[test]
  fn feature_extractor_reset_clears_pending() {
    let mut fx = FeatureExtractor::new(BUNDLED_CMVN_BYTES).expect("init");
    fx.push_pcm(&vec![0.0; 100]);
    fx.reset();
    assert_eq!(fx.pending_samples(), 0);
    assert!(!fx.has_full_window());
  }
```

- [ ] **Step 3: Run all features tests**

Run: `cargo test --all-features --lib features`
Expected: every test passes.

- [ ] **Step 4: Commit**

```bash
git add src/features.rs
git commit -m "$(cat <<'EOF'
feat(features): add FeatureExtractor with streaming PCM buffer

Wraps Cmvn + MelFilterbank with a streaming PCM tail buffer that
emits one CMVN-normalized 80-dim feature per consumed 10 ms hop.
Public API takes f32 in [-1, 1]; we scale by 32768 internally to
match upstream's int16-domain input. Tests pin partial-frame
buffering, silence determinism, and reset() semantics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: src/inference.rs — `OrtRunner`

**Files:**
- Create: `src/inference.rs`
- Modify: `src/lib.rs` (add `mod inference;` declaration)

- [ ] **Step 1: Add the module declaration to `src/lib.rs`**

Edit `src/lib.rs`: add `mod inference;` between `mod features;` and `mod options;`.

- [ ] **Step 2: Write `src/inference.rs`**

Reference: silero's `src/session.rs` for `ort 2.0.0-rc.12` patterns (`OrtSession::builder`, `TensorRef::from_array_view`, `try_extract_tensor`).

Write `src/inference.rs`:

```rust
//! ONNX Runtime wrapper for the FireRedVAD streaming model.

use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::error::{Error, Result};
use crate::features::NUM_MEL_BINS;
use crate::options::SessionOptions;

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
      .map_err(|source| Error::LoadModel { path: path.to_path_buf(), source })?;
    Ok(Self::from_ort_session(session))
  }

  /// Wrap an externally-built `ort::Session`. Caller is responsible for
  /// matching the model contract (`feat` + `caches_in` → `probs` +
  /// `caches_out`); the contract is asserted on first inference.
  pub(crate) fn from_ort_session(inner: OrtSession) -> Self {
    Self {
      inner,
      caches: vec![0.0f32; CACHE_TOTAL],
      feat_scratch: Vec::new(),
      prob_scratch: Vec::new(),
    }
  }

  /// Reset the per-stream cache to zero. Scratch buffers are kept.
  pub(crate) fn reset(&mut self) {
    for v in &mut self.caches {
      *v = 0.0;
    }
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
      &[CACHE_BLOCKS as i64, 1, CACHE_CHANNELS as i64, CACHE_TIME as i64],
    )?;

    self.prob_scratch.extend_from_slice(probs_data);
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
    Err(Error::UnexpectedOutputShape { tensor, shape: actual.to_vec() })
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
    assert_ne!(initial, runner.caches, "caches should change after one inference");
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
```

- [ ] **Step 3: Run the inference tests**

Run: `cargo test --all-features --lib inference::tests`
Expected: all 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/inference.rs
git commit -m "$(cat <<'EOF'
feat(inference): add OrtRunner wrapping ort::Session

ONNX I/O contract pinned: feat[1, T, 80] + caches_in[8, 1, 128, 19]
-> probs[1, T, 1] + caches_out[8, 1, 128, 19]. T-flexible per-call so
the engine can batch every feature pushed since the last infer().
Tests pin empty-input behavior, single-frame inference, cache-
advancement on inference, and reset() zeroing caches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: src/detector.rs — `Postprocessor` (smoothing + 4-state machine + finish)

**Files:**
- Create: `src/detector.rs`
- Modify: `src/lib.rs` (add `mod detector;` declaration)

This task ports upstream `StreamVadPostprocessor` bit-for-bit. Reference the source at `/Users/user/Develop/findit-studio/FireRedVAD/fireredvad/core/stream_vad_postprocessor.py` line-by-line while implementing.

- [ ] **Step 1: Add the module declaration to `src/lib.rs`**

Edit `src/lib.rs`: add `mod detector;` between `mod features;` (or `mod inference;`, whichever sorts before — alphabetical is easier) so the order becomes:

```rust
mod detector;
mod error;
mod event;
mod features;
mod inference;
mod options;
```

- [ ] **Step 2: Write `src/detector.rs` — struct, constructor, smoothing, helpers**

Write `src/detector.rs`:

```rust
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
```

- [ ] **Step 3: Add `push_probability` (state machine) and `finish_active`**

Append to `src/detector.rs` (inside the same `impl Postprocessor` block by re-opening it):

```rust
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
```

- [ ] **Step 4: Add unit tests covering each transition + finish**

Append to the bottom of `src/detector.rs`:

```rust
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
```

- [ ] **Step 5: Run the detector tests**

Run: `cargo test --all-features --lib detector::tests`
Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/detector.rs
git commit -m "$(cat <<'EOF'
feat(detector): add Postprocessor — bit-for-bit port of upstream

4-state machine (SILENCE / POSSIBLE_SPEECH / SPEECH / POSSIBLE_SILENCE),
trailing moving-average smoothing, hit_max_speech re-arm on force-split,
last_speech_end_frame clamping for pad_start. Frame indices are 0-based
on the way out (upstream is 1-based; shifted on construction). Tests
cover each transition: silence-only -> no segments; sub-threshold speech
-> never promoted; min_speech satisfied -> segment opens and closes on
silence; finish() flushes an open segment; max_speech_duration triggers
force-split + immediate restart; smoothing dampens isolated spikes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: src/vad.rs — `Vad` engine (constructors, push/poll/finish/reset)

**Files:**
- Create: `src/vad.rs`
- Modify: `src/lib.rs` (add `mod vad;` and re-export `Vad`)

- [ ] **Step 1: Add the `vad` module to `src/lib.rs`**

Edit `src/lib.rs`. Insert `mod vad;` after `mod options;` (alphabetical). Add `pub use vad::Vad;` to the re-exports. Also add the bundled-feature exports:

```rust
mod detector;
mod error;
mod event;
mod features;
mod inference;
mod options;
mod vad;

pub use error::{Error, Result};
pub use event::{FrameResult, SpeechSegment, VadEvent};
pub use options::{GraphOptimizationLevel, SessionOptions, VadOptions};
pub use vad::Vad;

#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub use vad::{BUNDLED_CMVN, BUNDLED_MODEL};
```

- [ ] **Step 2: Write `src/vad.rs`**

Write `src/vad.rs`:

```rust
//! The Sans-I/O `Vad` engine.

use std::collections::VecDeque;
use std::path::Path;

use crate::detector::Postprocessor;
use crate::error::Result;
use crate::event::{SpeechSegment, VadEvent};
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
```

- [ ] **Step 3: Add the helpers `Postprocessor` needs to expose to `Vad`**

Edit `src/detector.rs`: add three accessors inside the existing `impl Postprocessor` block (after `is_active`):

```rust
  /// Const-fn variant of `options()` used by `Vad::options()`.
  pub(crate) const fn options_const(&self) -> &VadOptions {
    &self.options
  }

  /// Number of frames consumed since the last reset.
  pub(crate) const fn frame_count(&self) -> u64 {
    self.frame_cnt_1based
  }
```

- [ ] **Step 4: Add unit tests for `Vad`**

Append to `src/vad.rs`:

```rust
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
```

- [ ] **Step 5: Run the vad tests (these are the slowest — they touch the bundled ONNX model)**

Run: `cargo test --all-features --lib vad::tests`
Expected: all 5 tests pass.

- [ ] **Step 6: Run the entire library test suite**

Run: `cargo test --all-features --lib`
Expected: every test from every module passes.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs src/detector.rs src/vad.rs
git commit -m "$(cat <<'EOF'
feat(vad): add Sans-I/O Vad engine + Postprocessor accessors

Vad orchestrates FeatureExtractor + OrtRunner + Postprocessor under a
push_samples / poll_event / drain_events / finish / reset surface, with
the constructor matrix (bundled, from_memory, from_file, from_*_with_cmvn,
from_ort_session) gating CMVN-less constructors behind the `bundled`
feature. Tests pin: bundled construction, 1 s of silence emits no
segments, reset clears all state, finish() on an idle stream emits no
segments, and frame events fire once per 10 ms hop after the first 25 ms
window arrives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: tests/integration_test.rs — black-box end-to-end

**Files:**
- Create: `tests/integration_test.rs`

- [ ] **Step 1: Write `tests/integration_test.rs`**

```rust
//! End-to-end black-box tests for the public `firered-vad` API.
//!
//! These tests deliberately avoid the crate-internal modules and instead
//! drive the engine the same way callers do: construct a `Vad`, push
//! PCM, drain events. They run against the bundled ONNX model and
//! CMVN, so they're gated on the `bundled` feature.

#![cfg(feature = "bundled")]

use firered_vad::{Vad, VadEvent};

const SAMPLE_RATE_HZ: u32 = 16_000;

fn synthetic_speech_like(duration_secs: f32) -> Vec<f32> {
  let n = (duration_secs * SAMPLE_RATE_HZ as f32) as usize;
  let mut buf = Vec::with_capacity(n);
  // Sum a few formant-band sinusoids; deterministic and broadband enough
  // that the model usually classifies it as speech.
  let formants = [200.0f32, 700.0, 1_500.0, 2_500.0, 3_500.0];
  for i in 0..n {
    let t = i as f32 / SAMPLE_RATE_HZ as f32;
    let mut sample = 0.0f32;
    for f in formants {
      sample += (core::f32::consts::TAU * f * t).sin();
    }
    buf.push(0.1 * sample);
  }
  buf
}

#[test]
fn bundled_constructs() {
  let _ = Vad::bundled().expect("bundled constructs");
}

#[test]
fn pure_silence_produces_no_segments() {
  let mut vad = Vad::bundled().expect("bundled");
  vad
    .push_samples(&vec![0.0; (SAMPLE_RATE_HZ * 2) as usize])
    .expect("push silence");
  vad.finish().expect("finish");
  let mut segments = 0usize;
  vad.drain_events(|ev| {
    if matches!(ev, VadEvent::SegmentClosed(_)) {
      segments += 1;
    }
  });
  assert_eq!(segments, 0);
}

#[test]
fn synthetic_speech_then_silence_emits_at_least_one_segment() {
  let mut vad = Vad::bundled().expect("bundled");

  let mut pcm = synthetic_speech_like(1.5);          // 1.5 s of "speech"
  pcm.extend(vec![0.0; SAMPLE_RATE_HZ as usize]);    // 1 s of silence
  vad.push_samples(&pcm).expect("push samples");
  vad.finish().expect("finish");

  let mut segments = Vec::new();
  vad.drain_events(|ev| {
    if let VadEvent::SegmentClosed(s) = ev {
      segments.push(s);
    }
  });

  assert!(!segments.is_empty(), "expected at least 1 closed segment");
  for s in &segments {
    assert!(s.start_sample() < s.end_sample());
    assert!(s.end_sample() <= pcm.len() as u64);
  }
}

#[test]
fn pushing_samples_in_arbitrary_chunks_yields_identical_event_stream() {
  let pcm = {
    let mut p = synthetic_speech_like(0.5);
    p.extend(vec![0.0; (SAMPLE_RATE_HZ as f32 * 0.5) as usize]);
    p
  };

  let collect = |chunk_size: usize| -> Vec<VadEvent> {
    let mut vad = Vad::bundled().expect("bundled");
    for chunk in pcm.chunks(chunk_size) {
      vad.push_samples(chunk).expect("push");
    }
    vad.finish().expect("finish");
    let mut out = Vec::new();
    vad.drain_events(|ev| out.push(ev));
    out
  };

  let baseline = collect(SAMPLE_RATE_HZ as usize); // one big push
  for chunk in [1usize, 160, 320, 1024, 4_000, 16_000] {
    assert_eq!(
      collect(chunk),
      baseline,
      "event stream must be deterministic across chunkings (chunk_size={chunk})"
    );
  }
}

#[test]
fn reset_returns_engine_to_freshly_constructed_state() {
  let mut vad = Vad::bundled().expect("bundled");
  vad.push_samples(&vec![0.0; 1_600]).expect("push 100ms");
  vad.reset();

  assert_eq!(vad.frame_count(), 0);
  assert_eq!(vad.pending_events(), 0);
  assert_eq!(vad.pending_samples(), 0);
  assert!(!vad.is_active());
  assert!(!vad.is_finished());
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test --all-features --test integration_test`
Expected: all 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration_test.rs
git commit -m "$(cat <<'EOF'
test: add end-to-end integration tests against the bundled model

Black-box tests drive the public Vad API exactly the way callers do:
construct, push PCM, drain events. Pinned: bundled construction,
silence-only -> no segments, synthetic speech -> at least one segment,
deterministic event stream regardless of chunking (1 sample, 160, 320,
1024, 4000, 16000), reset clears all state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: examples/streaming.rs and examples/detect_file.rs

**Files:**
- Create: `examples/streaming.rs`
- Create: `examples/detect_file.rs`

- [ ] **Step 1: Write `examples/streaming.rs`**

```rust
//! Synthetic streaming demo: alternates 1 s of band-limited noise with
//! 1 s of silence, then prints every closed segment.

use firered_vad::{Vad, VadEvent};

const SAMPLE_RATE_HZ: u32 = 16_000;

fn band_noise(duration_secs: f32) -> Vec<f32> {
  let n = (duration_secs * SAMPLE_RATE_HZ as f32) as usize;
  let mut buf = Vec::with_capacity(n);
  let formants = [200.0f32, 700.0, 1_500.0, 2_500.0, 3_500.0];
  for i in 0..n {
    let t = i as f32 / SAMPLE_RATE_HZ as f32;
    let mut sample = 0.0f32;
    for f in formants {
      sample += (core::f32::consts::TAU * f * t).sin();
    }
    buf.push(0.1 * sample);
  }
  buf
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut vad = Vad::bundled()?;

  let mut pcm = Vec::new();
  for _ in 0..3 {
    pcm.extend(band_noise(1.0));
    pcm.extend(vec![0.0; SAMPLE_RATE_HZ as usize]);
  }

  for chunk in pcm.chunks(1_600) {
    vad.push_samples(chunk)?;
    while let Some(event) = vad.poll_event() {
      if let VadEvent::SegmentClosed(s) = event {
        println!(
          "segment: {:.3}s..{:.3}s ({} samples)",
          s.start().as_secs_f32(),
          s.end().as_secs_f32(),
          s.sample_count()
        );
      }
    }
  }

  vad.finish()?;
  while let Some(event) = vad.poll_event() {
    if let VadEvent::SegmentClosed(s) = event {
      println!(
        "segment (trailing): {:.3}s..{:.3}s ({} samples)",
        s.start().as_secs_f32(),
        s.end().as_secs_f32(),
        s.sample_count()
      );
    }
  }

  Ok(())
}
```

- [ ] **Step 2: Write `examples/detect_file.rs`**

```rust
//! Reads a 16 kHz mono WAV (int16 or float32) and prints every closed
//! speech segment. Designed as the prototype for "feed Whisper" use cases.
//!
//! Usage:
//!     cargo run --example detect_file -- path/to/audio.wav

use firered_vad::{Vad, VadEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = std::env::args()
    .nth(1)
    .ok_or("usage: detect_file <path.wav>")?;
  let mut reader = hound::WavReader::open(&path)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    return Err(format!(
      "expected 16 kHz mono WAV; got {} Hz {}-channel",
      spec.sample_rate, spec.channels
    )
    .into());
  }
  if spec.channels != 1 {
    return Err("expected mono".into());
  }

  let pcm: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => reader
      .samples::<i16>()
      .map(|s| s.map(|s| s as f32 / 32_768.0))
      .collect::<Result<_, _>>()?,
    hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
  };

  let mut vad = Vad::bundled()?;
  for chunk in pcm.chunks(16_000) {
    vad.push_samples(chunk)?;
    while let Some(event) = vad.poll_event() {
      if let VadEvent::SegmentClosed(s) = event {
        println!(
          "{:.3}s..{:.3}s  ({:>6} samples)",
          s.start().as_secs_f32(),
          s.end().as_secs_f32(),
          s.sample_count()
        );
      }
    }
  }
  vad.finish()?;
  while let Some(event) = vad.poll_event() {
    if let VadEvent::SegmentClosed(s) = event {
      println!(
        "{:.3}s..{:.3}s  ({:>6} samples)  (trailing)",
        s.start().as_secs_f32(),
        s.end().as_secs_f32(),
        s.sample_count()
      );
    }
  }
  Ok(())
}
```

- [ ] **Step 3: Verify the examples build**

Run: `cargo build --examples --all-features`
Expected: clean build, no warnings.

- [ ] **Step 4: Commit**

```bash
git add examples/streaming.rs examples/detect_file.rs
git commit -m "$(cat <<'EOF'
docs(examples): add streaming.rs and detect_file.rs demos

streaming.rs alternates synthetic noise/silence and prints emitted
segments — useful for smoke-testing the event ordering. detect_file.rs
reads a 16 kHz mono WAV via hound, drives Vad over the bundled model,
and prints segment timestamps. The detect_file example is the
canonical "feed Whisper" prototype minus the Whisper call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: README.md and CHANGELOG.md

**Files:**
- Create: `README.md`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# firered-vad

Streaming Voice Activity Detection that wraps the [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX model. Bit-for-bit parity with upstream Python's `FireRedStreamVad`, with a Sans-I/O Rust API designed for piping continuous human-speech windows into Whisper or any other downstream consumer.

A sibling crate to [`silero`](https://github.com/uqio/silero) for callers who want a true streaming VAD: 10 ms frame granularity, no externally-managed RNN state, and a built-in postprocessor with smoothing and a 4-state machine.

## Installation

```toml
[dependencies]
firered-vad = "0.1"
```

The default `bundled` feature embeds the ONNX model (~2.3 MB) and CMVN stats. Disable to ship your own:

```toml
[dependencies]
firered-vad = { version = "0.1", default-features = false }
```

## Quick start

```rust
use firered_vad::{Vad, VadEvent};

let mut vad = Vad::bundled()?;
let pcm: Vec<f32> = /* 16 kHz f32 PCM in [-1.0, 1.0] */;

for chunk in pcm.chunks(1_600) {
    vad.push_samples(chunk)?;
    while let Some(event) = vad.poll_event() {
        if let VadEvent::SegmentClosed(segment) = event {
            // Slice the original PCM to recover the speech window.
            let speech = &pcm[segment.range_usize()];
            // ... feed `speech` into Whisper / your transcriber.
        }
    }
}
vad.finish()?;
while let Some(event) = vad.poll_event() {
    if let VadEvent::SegmentClosed(segment) = event {
        // Trailing segment (open at end-of-stream).
    }
}
# Ok::<(), firered_vad::Error>(())
```

## API at a glance

`Vad` is a single Sans-I/O state machine:

| Method | Purpose |
| --- | --- |
| `Vad::bundled()` | Construct from the bundled ONNX + CMVN with default options |
| `Vad::bundled_with(opts)` | Same, with custom `VadOptions` |
| `Vad::from_memory(model)` / `from_file(path)` | Custom model bytes/path with bundled CMVN |
| `Vad::from_memory_with_cmvn` / `Vad::from_file_with_cmvn` | Fully-custom model + CMVN |
| `Vad::from_ort_session(session, cmvn, opts)` | Wrap an externally-built `ort::Session` |
| `push_samples(&[f32])` | Feed PCM, queue events |
| `poll_event() -> Option<VadEvent>` | Pull the next queued event |
| `drain_events(F)` | Closure-based drain over `poll_event` |
| `finish()` | Mark end-of-stream; closes any open segment |
| `reset()` | Wipe all per-stream state |

Events are `VadEvent::Frame(FrameResult)` (per 10 ms frame, with `raw_prob`, `smoothed_prob`, and boundary flags) and `VadEvent::SegmentClosed(SpeechSegment)` (one per closed continuous speech run).

## Tuning

Options reproduce upstream `FireRedStreamVadConfig` defaults exactly. To match upstream's four "mode" presets, configure directly:

```rust
use core::time::Duration;
use firered_vad::VadOptions;

// "Permissive" preset (upstream mode 1):
let opts = VadOptions::new()
    .with_speech_threshold(0.5)
    .with_min_speech_duration(Duration::from_millis(100))
    .with_min_silence_duration(Duration::from_millis(150));

// "Aggressive" — threshold 0.7, min_speech 150 ms, min_silence 100 ms
// "Very aggressive" — threshold 0.9, min_speech 200 ms, min_silence 50 ms
// "Very permissive" — threshold 0.3, min_speech 80 ms, min_silence 200 ms
```

## Features

| Feature | Default | What it does |
| --- | --- | --- |
| `bundled` | yes | Embed the ONNX model + CMVN as `BUNDLED_MODEL` / `BUNDLED_CMVN` constants |
| `serde` | no | `Serialize` / `Deserialize` for `VadOptions` and `SessionOptions`; Duration fields use `humantime-serde` |
| `coreml`, `directml`, `cuda`, `rocm`, `tensorrt`, `openvino` | no | Pass-through to `ort` for the matching execution provider |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. The bundled FireRedVAD model and CMVN stats are Apache-2.0; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
```

- [ ] **Step 2: Write `CHANGELOG.md`**

```markdown
# Changelog

## UNRELEASED

## 0.1.0 — 2026-05-08

Initial release.

- Sans-I/O streaming `Vad` engine: `push_samples` / `poll_event` /
  `drain_events` / `finish` / `reset`.
- Bit-for-bit port of upstream Python's `StreamVadPostprocessor`:
  trailing-mean smoothing, 4-state machine
  (SILENCE / POSSIBLE_SPEECH / SPEECH / POSSIBLE_SILENCE),
  `hit_max_speech` re-arm on force-split,
  `last_speech_end_frame` clamping for `pad_start`.
- Pure-Rust Kaldi-compatible Mel-filterbank + CMVN preprocessing.
  No `dyn` dispatch (concrete `rustfft::algorithm::Radix2<f32>`).
- ONNX Runtime via `ort` 2.0.0-rc.12, contract pinned to
  `feat[1, T, 80] + caches_in[8, 1, 128, 19] -> probs[1, T, 1] + caches_out`.
- `bundled` feature (default) embeds the FireRedVAD streaming ONNX
  model and CMVN stats (Apache-2.0; see `THIRD_PARTY_NOTICES.md`).
- Optional `serde` feature mirrors silero's per-field
  `humantime-serde` idiom.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs: add README.md and CHANGELOG.md for 0.1.0

README documents the Sans-I/O loop, constructor matrix, the four
upstream-mode preset recipes, feature flags, and license. CHANGELOG
records the 0.1.0 surface area.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: .github/workflows/ci.yml

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the CI workflow (mirrors silero's structure exactly)**

```yaml
name: CI

on:
  push:
    branches: [main, "*.x", "0.*"]
    paths-ignore:
      - "**/*.md"
      - "docs/**"
      - "LICENSE*"
      - "THIRD_PARTY_NOTICES.md"
  pull_request:
  schedule:
    - cron: "0 1 1 * *"   # monthly heartbeat
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-Dwarnings"
  RUST_BACKTRACE: 1

jobs:
  rustfmt:
    name: Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  clippy:
    name: Clippy ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - run: cargo clippy --all-features --all-targets -- -D warnings

  build:
    name: Build ${{ matrix.os }} (${{ matrix.features }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        features: ["default", "no-default", "all"]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build (default)
        if: matrix.features == 'default'
        run: cargo build --verbose
      - name: Build (no default features)
        if: matrix.features == 'no-default'
        run: cargo build --no-default-features --verbose
      - name: Build (all features)
        if: matrix.features == 'all'
        run: cargo build --all-features --verbose

  test:
    name: Test ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --all-features --verbose

  coverage:
    name: Coverage
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - uses: Swatinem/rust-cache@v2
      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin --locked
      - name: Run tarpaulin
        env:
          RUSTFLAGS: "--cfg tarpaulin"
        run: cargo tarpaulin --all-features --out Xml --timeout 240
      - uses: codecov/codecov-action@v4
        with:
          files: ./cobertura.xml
          fail_ci_if_error: false
```

- [ ] **Step 2: Verify the YAML parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: no output (parse succeeded).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
ci: add silero-shaped GitHub Actions workflow

rustfmt + clippy (Linux/macOS/Windows) + build matrix
(default, no-default, all-features) + test matrix
(Linux/macOS/Windows) + tarpaulin coverage on Linux nightly with
codecov upload. Drops the template's sanitizer / Miri / Loom jobs —
overkill for a thin ONNX wrapper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Final verification

**Files:** none modified.

- [ ] **Step 1: Format the entire crate**

Run: `cargo fmt --all`
Expected: no output (or formatting fixes; if so, commit them as a separate `style: cargo fmt` commit before continuing).

- [ ] **Step 2: Lint the entire crate**

Run: `cargo clippy --all-features --all-targets -- -D warnings`
Expected: clean, zero warnings.

- [ ] **Step 3: Build with default features**

Run: `cargo build --all-targets`
Expected: clean build, zero warnings.

- [ ] **Step 4: Build without default features**

Run: `cargo build --no-default-features --all-targets`
Expected: clean build. The `bundled`-feature-gated constructors are absent — this is the off-bundled path that downstream consumers use when they ship their own model.

- [ ] **Step 5: Build with every feature**

Run: `cargo build --all-features --all-targets`
Expected: clean build. `serde` derives compile, EP feature flags pass through to `ort`.

- [ ] **Step 6: Run the full test suite**

Run: `cargo test --all-features`
Expected: every unit and integration test passes.

- [ ] **Step 7: Sanity-check `cargo doc`**

Run: `cargo doc --all-features --no-deps`
Expected: clean docs build (the `missing_docs` deny in `lib.rs` will catch any module missing top-level docs).

- [ ] **Step 8: Confirm there are no remaining template artifacts**

Run: `find . -name 'foo.rs' -not -path './target/*' -not -path '.git/*'`
Expected: no output.

- [ ] **Step 9: Commit any cargo-fmt churn (if Step 1 produced any)**

If Step 1 produced changes, commit them now:

```bash
git add -A
git commit -m "$(cat <<'EOF'
style: cargo fmt --all

Apply rustfmt to all newly-introduced sources.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If there were no changes, this step is a no-op.

---

## Out of scope for this plan

The following are mentioned in the spec but deferred:

- **Parity harness (`tests/parity/`)** — manual-only, not part of `cargo test`. Track as a follow-up: it requires a Python virtualenv with `fireredvad` installed plus a fixtures directory, and the IoU scorer is its own standalone deliverable.
- **`tests/parity/run.sh`, `python/run.py`, `rust/main.rs`, `scorer.py`** — same.
- **Resampling helper** — explicit caller responsibility per the spec.
- **AED model** — separate model, separate API.
- **Offline `detect_full` API** — streaming covers the v1 use case (Whisper feeding).
