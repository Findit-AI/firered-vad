//! Architecture-specific SIMD backends for the Mel-fbank inner-loop kernels.
//!
//! Each submodule here is gated on the target architecture it targets.
//! The public dispatcher in [`super::dispatch`] selects among them at
//! call boundaries.
//!
//! Layout mirrors `colconv-be-tier10b/src/row/arch/`. Today only
//! `aarch64` ships with hand-rolled SIMD; x86_64 and wasm32 fall
//! through to scalar via the dispatcher and are easy drop-in additions.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
