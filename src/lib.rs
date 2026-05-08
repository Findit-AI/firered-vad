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

/// Crate version (matches `CARGO_PKG_VERSION`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
