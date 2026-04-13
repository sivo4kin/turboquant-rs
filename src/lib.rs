//! # TurboQuant
//!
//! KV cache compression via PolarQuant + QJL — Rust port of
//! [turboquant_plus](https://github.com/TheTom/turboquant_plus).
//!
//! ## Overview
//!
//! TurboQuant is an experimental implementation of KV cache compression for LLM
//! inference, based on the TurboQuant research paper (ICLR 2026). It delivers
//! 3.8–6.4× compression ratios using PolarQuant and Walsh-Hadamard rotation.
//!
//! ## Three compression formats
//!
//! - **turbo2**: 2-bit, ~6.4× compression
//! - **turbo3**: 3-bit, ~4.6–5.1× compression
//! - **turbo4**: 4-bit, ~3.8× compression
//!
//! ## Quick start
//!
//! ```rust
//! use turboquant_plus_rs::{TurboQuant, TurboQuantMSE, KVCacheCompressor};
//! use ndarray::Array1;
//!
//! // Single-vector quantization
//! let tq = TurboQuant::new(128, 3, 42, true);
//! let x = Array1::from_shape_fn(128, |i| (i as f64) / 128.0);
//! let compressed = tq.quantize(&x);
//! let x_hat = tq.dequantize(&compressed);
//!
//! // KV cache compression
//! let compressor = KVCacheCompressor::new(128, 3, 3, 42, true);
//! let stats = compressor.memory_stats(1024, 32, 32);
//! println!("Compression ratio: {:.1}×", stats.compression_ratio);
//! ```

pub mod codebook;
pub mod kv_cache;
pub mod outlier;
pub mod polar_quant;
pub mod qjl;
pub mod rotation;
pub mod turboquant;
pub mod utils;

// Re-export primary types at crate root
pub use kv_cache::{CompressedKVCache, KVCacheCompressor, MemoryStats};
pub use outlier::{OutlierCompressedVectors, OutlierTurboQuant};
pub use polar_quant::PolarQuant;
pub use qjl::QJL;
pub use turboquant::{CompressedVectors, TurboQuant, TurboQuantMSE};
pub use utils::{
    memory_footprint_bytes, pack_bits, pack_bits_batch, pack_indices, pack_indices_batch,
    unpack_bits, unpack_indices, unpack_indices_batch, MemoryFootprint,
};
