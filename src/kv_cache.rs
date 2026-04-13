//! KV Cache integration layer for TurboQuant.
//!
//! Compresses transformer KV cache tensors using TurboQuant (for K cache, inner product
//! preservation) and PolarQuant MSE-only (for V cache, MSE preservation).
//!
//! KV cache shape: (num_layers, num_heads, seq_len, head_dim)
//! Quantization is along head_dim — each (head_dim,) vector is quantized independently.

use ndarray::{Array2, Array4};

use crate::turboquant::{CompressedVectors, TurboQuant, TurboQuantMSE};
use ndarray::Array1;

/// Container for a compressed KV cache.
pub struct CompressedKVCache {
    /// Per-layer, per-head compressed K vectors.
    pub k_compressed: Vec<Vec<CompressedVectors>>,
    /// Per-layer, per-head compressed V (indices).
    pub v_indices: Vec<Vec<Array2<usize>>>,
    /// Per-layer, per-head compressed V (norms).
    pub v_norms: Vec<Vec<Array1<f64>>>,

    pub num_layers: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub k_bit_width: usize,
    pub v_bit_width: usize,
}

/// Compress and decompress transformer KV cache tensors.
///
/// Uses:
/// - TurboQuant (Algorithm 2) for K cache — inner product preservation matters
///   for attention score computation (Q @ K^T)
/// - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation matters
///   for value reconstruction (attn_weights @ V)
pub struct KVCacheCompressor {
    pub head_dim: usize,
    pub k_bits: usize,
    pub v_bits: usize,
    k_quantizer: TurboQuant,
    v_quantizer: TurboQuantMSE,
}

impl KVCacheCompressor {
    /// Create a new KV cache compressor.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head vector.
    /// * `k_bits` - Bit-width for K cache (TurboQuant, inner product).
    /// * `v_bits` - Bit-width for V cache (PolarQuant MSE-only).
    /// * `seed` - Random seed.
    /// * `norm_correction` - Whether to apply norm correction in PolarQuant.
    pub fn new(
        head_dim: usize,
        k_bits: usize,
        v_bits: usize,
        seed: u64,
        norm_correction: bool,
    ) -> Self {
        let k_quantizer = TurboQuant::new(head_dim, k_bits, seed, norm_correction);
        let v_quantizer = TurboQuantMSE::new(head_dim, v_bits, seed + 500, norm_correction);

        Self {
            head_dim,
            k_bits,
            v_bits,
            k_quantizer,
            v_quantizer,
        }
    }

    /// Compress full KV cache tensors.
    ///
    /// # Arguments
    /// * `k_cache` - Key cache, shape (num_layers, num_heads, seq_len, head_dim).
    /// * `v_cache` - Value cache, same shape.
    pub fn compress(&self, k_cache: &Array4<f64>, v_cache: &Array4<f64>) -> CompressedKVCache {
        let shape = k_cache.shape();
        let (num_layers, num_heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(head_dim, self.head_dim, "head_dim mismatch");
        assert_eq!(k_cache.shape(), v_cache.shape(), "K and V cache shape mismatch");

        let mut result = CompressedKVCache {
            k_compressed: Vec::with_capacity(num_layers),
            v_indices: Vec::with_capacity(num_layers),
            v_norms: Vec::with_capacity(num_layers),
            num_layers,
            num_heads,
            seq_len,
            head_dim,
            k_bit_width: self.k_bits,
            v_bit_width: self.v_bits,
        };

        for layer in 0..num_layers {
            let mut k_layer = Vec::with_capacity(num_heads);
            let mut v_layer_idx = Vec::with_capacity(num_heads);
            let mut v_layer_norms = Vec::with_capacity(num_heads);

            for head in 0..num_heads {
                // Extract (seq_len, head_dim) slice for this layer/head
                let mut k_vecs = Array2::zeros((seq_len, head_dim));
                let mut v_vecs = Array2::zeros((seq_len, head_dim));
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        k_vecs[[s, d]] = k_cache[[layer, head, s, d]];
                        v_vecs[[s, d]] = v_cache[[layer, head, s, d]];
                    }
                }

                // K: batch quantize all seq positions
                let k_compressed = self.k_quantizer.quantize_batch(&k_vecs);
                k_layer.push(k_compressed);

                // V: MSE quantize
                let (v_indices, v_norms) = self.v_quantizer.quantize_batch(&v_vecs);
                v_layer_idx.push(v_indices);
                v_layer_norms.push(v_norms);
            }

            result.k_compressed.push(k_layer);
            result.v_indices.push(v_layer_idx);
            result.v_norms.push(v_layer_norms);
        }

        result
    }

    /// Decompress back to full KV cache tensors.
    ///
    /// Returns (k_cache, v_cache) both shape (num_layers, num_heads, seq_len, head_dim).
    pub fn decompress(&self, compressed: &CompressedKVCache) -> (Array4<f64>, Array4<f64>) {
        let mut k_cache = Array4::zeros((
            compressed.num_layers,
            compressed.num_heads,
            compressed.seq_len,
            compressed.head_dim,
        ));
        let mut v_cache = Array4::zeros(k_cache.raw_dim());

        for layer in 0..compressed.num_layers {
            for head in 0..compressed.num_heads {
                // K: dequantize
                let k_hat = self.k_quantizer.dequantize(&compressed.k_compressed[layer][head]);
                for s in 0..compressed.seq_len {
                    for d in 0..compressed.head_dim {
                        k_cache[[layer, head, s, d]] = k_hat[[s, d]];
                    }
                }

                // V: dequantize
                let v_hat = self.v_quantizer.dequantize_batch(
                    &compressed.v_indices[layer][head],
                    &compressed.v_norms[layer][head],
                );
                for s in 0..compressed.seq_len {
                    for d in 0..compressed.head_dim {
                        v_cache[[layer, head, s, d]] = v_hat[[s, d]];
                    }
                }
            }
        }

        (k_cache, v_cache)
    }

    /// Compute memory usage statistics.
    pub fn memory_stats(
        &self,
        seq_len: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> MemoryStats {
        let n_vectors = num_layers * num_heads * seq_len;
        let original_bytes = n_vectors * self.head_dim * 2; // fp16

        // K (TurboQuant): b bits per coord + two 32-bit norms (vector + residual)
        let k_bits_total = n_vectors * (self.head_dim * self.k_bits + 64);
        // V (MSE): b bits per coord + one 32-bit norm
        let v_bits_total = n_vectors * (self.head_dim * self.v_bits + 32);

        let compressed_bytes = (k_bits_total + v_bits_total) / 8;

        MemoryStats {
            original_mb: original_bytes as f64 / 1024.0 / 1024.0,
            compressed_mb: compressed_bytes as f64 / 1024.0 / 1024.0,
            compression_ratio: original_bytes as f64 / compressed_bytes as f64,
            k_bits_per_value: self.k_bits,
            v_bits_per_value: self.v_bits,
        }
    }
}

/// Memory usage statistics for KV cache compression.
#[derive(Debug)]
pub struct MemoryStats {
    pub original_mb: f64,
    pub compressed_mb: f64,
    pub compression_ratio: f64,
    pub k_bits_per_value: usize,
    pub v_bits_per_value: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_compress_decompress_roundtrip() {
        let compressor = KVCacheCompressor::new(16, 3, 3, 42, true);

        // Small KV cache: 1 layer, 1 head, 4 tokens, head_dim=16
        let k_cache = Array4::from_shape_fn((1, 1, 4, 16), |(_, _, s, d)| {
            ((s * 16 + d) as f64 + 1.0) / 100.0
        });
        let v_cache = Array4::from_shape_fn((1, 1, 4, 16), |(_, _, s, d)| {
            ((s * 16 + d) as f64 + 50.0) / 100.0
        });

        let compressed = compressor.compress(&k_cache, &v_cache);
        let (k_hat, v_hat) = compressor.decompress(&compressed);

        assert_eq!(k_hat.shape(), k_cache.shape());
        assert_eq!(v_hat.shape(), v_cache.shape());

        // Check that reconstruction is reasonable (not exact, but bounded)
        let k_err: f64 = (&k_cache - &k_hat).mapv(|v| v * v).sum();
        let k_orig: f64 = k_cache.mapv(|v| v * v).sum();
        assert!(
            k_err / k_orig < 1.0,
            "K cache relative MSE: {}",
            k_err / k_orig
        );

        let v_err: f64 = (&v_cache - &v_hat).mapv(|v| v * v).sum();
        let v_orig: f64 = v_cache.mapv(|v| v * v).sum();
        assert!(
            v_err / v_orig < 1.0,
            "V cache relative MSE: {}",
            v_err / v_orig
        );
    }

    #[test]
    fn test_memory_stats() {
        let compressor = KVCacheCompressor::new(128, 3, 3, 42, true);
        let stats = compressor.memory_stats(1024, 32, 32);
        assert!(stats.compression_ratio > 1.0, "Should compress");
        assert!(stats.compressed_mb < stats.original_mb, "Should be smaller");
    }
}
