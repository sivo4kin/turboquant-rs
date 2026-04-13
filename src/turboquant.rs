//! TurboQuant: Full algorithm combining PolarQuant + QJL.
//!
//! Algorithm 2 from the paper — Inner Product TurboQuant.
//!
//! Two-stage process:
//! 1. PolarQuant at (b-1) bits for MSE-optimal compression
//! 2. QJL at 1 bit on the residual for bias elimination
//!
//! Total: b bits per coordinate with near-optimal inner product distortion.

use ndarray::{Array1, Array2, Axis};

use crate::polar_quant::PolarQuant;
use crate::qjl::QJL;

/// Container for TurboQuant-compressed vectors (batch format).
pub struct CompressedVectors {
    /// PolarQuant indices, (b-1)-bit integers. Shape: (batch, d).
    pub mse_indices: Array2<usize>,
    /// Original ‖x‖₂ norms for rescaling. Shape: (batch,).
    pub vector_norms: Array1<f64>,
    /// QJL sign bits {+1, -1}. Shape: (batch, d).
    pub qjl_signs: Array2<i8>,
    /// Residual ‖r‖₂ norms. Shape: (batch,).
    pub residual_norms: Array1<f64>,
    /// Total bits per coordinate.
    pub bit_width: usize,
}

/// Full TurboQuant quantizer: PolarQuant (b-1 bits) + QJL (1 bit).
pub struct TurboQuant {
    pub d: usize,
    pub bit_width: usize,
    polar_quant: PolarQuant,
    qjl: QJL,
}

impl TurboQuant {
    /// Create a new TurboQuant quantizer.
    ///
    /// # Arguments
    /// * `d` - Vector dimension.
    /// * `bit_width` - Total bits per coordinate (b). PolarQuant uses b-1, QJL uses 1.
    /// * `seed` - Random seed for both rotation and projection matrices.
    /// * `norm_correction` - If true, apply norm correction in PolarQuant.
    ///
    /// # Panics
    /// Panics if `bit_width < 2` (need at least 1 bit PolarQuant + 1 bit QJL).
    pub fn new(d: usize, bit_width: usize, seed: u64, norm_correction: bool) -> Self {
        assert!(
            bit_width >= 2,
            "TurboQuant requires bit_width >= 2 (1 bit PolarQuant + 1 bit QJL). \
             For 1-bit, use QJL directly."
        );

        let polar_quant = PolarQuant::new(d, bit_width - 1, seed, norm_correction);
        let qjl = QJL::new(d, seed + 1000);

        Self {
            d,
            bit_width,
            polar_quant,
            qjl,
        }
    }

    /// Quantize a single vector.
    pub fn quantize(&self, x: &Array1<f64>) -> CompressedVectors {
        self.quantize_batch(&x.clone().insert_axis(Axis(0)))
    }

    /// Quantize a batch of vectors.
    ///
    /// # Arguments
    /// * `x` - Input vectors, shape (batch, d).
    ///
    /// # Returns
    /// `CompressedVectors` containing indices, signs, and norms.
    pub fn quantize_batch(&self, x: &Array2<f64>) -> CompressedVectors {
        // Stage 1: PolarQuant (with norm extraction)
        let (mse_indices, vector_norms, residual) =
            self.polar_quant.quantize_and_residual_batch(x);

        // Stage 2: QJL on residual
        let (qjl_signs, residual_norms) = self.qjl.quantize_batch(&residual);

        CompressedVectors {
            mse_indices,
            vector_norms,
            qjl_signs,
            residual_norms,
            bit_width: self.bit_width,
        }
    }

    /// Dequantize back to approximate vectors.
    pub fn dequantize(&self, compressed: &CompressedVectors) -> Array2<f64> {
        // Stage 1: PolarQuant reconstruction
        let x_mse = self
            .polar_quant
            .dequantize_batch(&compressed.mse_indices, &compressed.vector_norms);

        // Stage 2: QJL residual reconstruction
        let x_qjl = self
            .qjl
            .dequantize_batch(&compressed.qjl_signs, &compressed.residual_norms);

        x_mse + x_qjl
    }

    /// Compute total storage in bits for `n_vectors` compressed vectors.
    pub fn compressed_size_bits(&self, n_vectors: usize) -> usize {
        let per_vector = self.d * self.bit_width; // (b-1) + 1 bits per coordinate
        let norms = 64; // two float32 norms per vector: vector_norm + residual_norm
        n_vectors * (per_vector + norms)
    }

    /// Compute compression ratio vs original precision.
    pub fn compression_ratio(&self, original_bits_per_value: usize) -> f64 {
        let original_per_vector = self.d * original_bits_per_value;
        let compressed_per_vector = self.d * self.bit_width + 64; // +64 for two norms
        original_per_vector as f64 / compressed_per_vector as f64
    }
}

/// MSE-only TurboQuant (Algorithm 1) — no QJL stage.
///
/// Use for V cache compression where MSE matters more than inner product.
pub struct TurboQuantMSE {
    pub d: usize,
    pub bit_width: usize,
    polar_quant: PolarQuant,
}

impl TurboQuantMSE {
    pub fn new(d: usize, bit_width: usize, seed: u64, norm_correction: bool) -> Self {
        let polar_quant = PolarQuant::new(d, bit_width, seed, norm_correction);
        Self {
            d,
            bit_width,
            polar_quant,
        }
    }

    /// Quantize a single vector. Returns (indices, norm).
    pub fn quantize(&self, x: &Array1<f64>) -> (Array1<usize>, f64) {
        self.polar_quant.quantize(x)
    }

    /// Quantize a batch of vectors.
    pub fn quantize_batch(&self, x: &Array2<f64>) -> (Array2<usize>, Array1<f64>) {
        self.polar_quant.quantize_batch(x)
    }

    /// Dequantize a single vector.
    pub fn dequantize(&self, indices: &Array1<usize>, norm: f64) -> Array1<f64> {
        self.polar_quant.dequantize(indices, norm)
    }

    /// Dequantize a batch of vectors.
    pub fn dequantize_batch(&self, indices: &Array2<usize>, norms: &Array1<f64>) -> Array2<f64> {
        self.polar_quant.dequantize_batch(indices, norms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turboquant_roundtrip() {
        let tq = TurboQuant::new(32, 3, 42, true);
        let x = Array1::from_shape_fn(32, |i| (i as f64 + 1.0) / 32.0);

        let compressed = tq.quantize(&x);
        let x_hat = tq.dequantize(&compressed);
        let x_hat_vec = x_hat.row(0).to_owned();

        // Check reconstruction error
        let error: f64 = (&x - &x_hat_vec).mapv(|v| v * v).sum();
        let original: f64 = x.mapv(|v| v * v).sum();
        let relative_error = error / original;
        assert!(
            relative_error < 1.0,
            "TurboQuant relative MSE: {relative_error}"
        );
    }

    #[test]
    fn test_turboquant_inner_product_preservation() {
        let tq = TurboQuant::new(64, 3, 42, true);
        let x = Array1::from_shape_fn(64, |i| (i as f64 + 1.0) / 64.0);
        let y = Array1::from_shape_fn(64, |i| (64.0 - i as f64) / 64.0);

        let ip_original = x.dot(&y);

        let cx = tq.quantize(&x);
        let cy = tq.quantize(&y);
        let x_hat = tq.dequantize(&cx).row(0).to_owned();
        let y_hat = tq.dequantize(&cy).row(0).to_owned();
        let ip_approx = x_hat.dot(&y_hat);

        // Should preserve inner product within reasonable bounds
        let relative_error = (ip_original - ip_approx).abs() / ip_original.abs();
        assert!(
            relative_error < 1.0,
            "Inner product relative error: {relative_error}"
        );
    }

    #[test]
    #[should_panic(expected = "bit_width >= 2")]
    fn test_turboquant_1bit_panics() {
        TurboQuant::new(32, 1, 42, true);
    }

    #[test]
    fn test_turboquant_mse_roundtrip() {
        let tq = TurboQuantMSE::new(32, 3, 42, true);
        let x = Array1::from_shape_fn(32, |i| (i as f64 + 1.0) / 32.0);

        let (indices, norm) = tq.quantize(&x);
        let x_hat = tq.dequantize(&indices, norm);

        let error: f64 = (&x - &x_hat).mapv(|v| v * v).sum();
        let original: f64 = x.mapv(|v| v * v).sum();
        let relative_error = error / original;
        assert!(
            relative_error < 1.0,
            "TurboQuantMSE relative MSE: {relative_error}"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let tq = TurboQuant::new(128, 3, 42, true);
        let ratio = tq.compression_ratio(16);
        // 3 bits per coord + 64 bits norm overhead (vector + residual)
        let expected = (128.0 * 16.0) / (128.0 * 3.0 + 64.0);
        assert!(
            (ratio - expected).abs() < 0.01,
            "Compression ratio: {ratio}, expected: {expected}"
        );
    }
}
