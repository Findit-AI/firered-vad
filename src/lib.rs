#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
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
