//! Architecture-specific SIMD backends for the Mel-fbank inner-loop kernels.
//!
//! Each submodule here is gated on the target architecture it targets.
//! The public dispatcher in [`super::dispatch`] selects among them at
//! call boundaries.
//!
//! Layout mirrors `colconv-be-tier10b/src/row/arch/`. `aarch64` ships
//! NEON; `x86_64` ships SSE4.1, AVX2+FMA, and AVX-512F backends with
//! a runtime-feature-detected cascade in [`super::dispatch`]. wasm32
//! falls through to scalar.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86_avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86_avx512;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86_sse41;
