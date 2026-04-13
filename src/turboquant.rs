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
use crate::utils::{pack_bits_batch, pack_indices_batch, unpack_bits, unpack_indices_batch};

/// Container for TurboQuant-compressed vectors (batch format).
pub struct CompressedVectors {
    /// PolarQuant indices packed to bytes. Length: batch.
    pub mse_indices_packed: Vec<Vec<u8>>,
    /// Original ‖x‖₂ norms for rescaling (stored as float32). Length: batch.
    pub vector_norms: Vec<f32>,
    /// QJL sign bits packed to bytes. Length: batch.
    pub qjl_signs_packed: Vec<Vec<u8>>,
    /// Residual ‖r‖₂ norms (stored as float32). Length: batch.
    pub residual_norms: Vec<f32>,
    /// Total bits per coordinate.
    pub bit_width: usize,
    /// Vector dimension.
    pub d: usize,
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
            mse_indices_packed: pack_indices_batch(&mse_indices, self.bit_width - 1),
            vector_norms: vector_norms.iter().map(|&v| v as f32).collect(),
            qjl_signs_packed: pack_bits_batch(&qjl_signs),
            residual_norms: residual_norms.iter().map(|&v| v as f32).collect(),
            bit_width: self.bit_width,
            d: self.d,
        }
    }

    /// Dequantize back to approximate vectors.
    pub fn dequantize(&self, compressed: &CompressedVectors) -> Array2<f64> {
        let batch = compressed.vector_norms.len();
        assert_eq!(compressed.d, self.d, "compressed dimension mismatch");
        assert_eq!(
            compressed.mse_indices_packed.len(),
            batch,
            "mse packed batch size mismatch"
        );
        assert_eq!(
            compressed.qjl_signs_packed.len(),
            batch,
            "qjl packed batch size mismatch"
        );
        assert_eq!(
            compressed.residual_norms.len(),
            batch,
            "residual norms batch size mismatch"
        );

        let mse_indices = unpack_indices_batch(
            &compressed.mse_indices_packed,
            self.d,
            compressed.bit_width - 1,
        );
        let vector_norms = Array1::from_vec(
            compressed
                .vector_norms
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
        );
        let mut qjl_signs = Array2::zeros((batch, self.d));
        for (i, packed) in compressed.qjl_signs_packed.iter().enumerate() {
            let signs = unpack_bits(packed, self.d);
            qjl_signs.row_mut(i).assign(&signs);
        }
        let residual_norms = Array1::from_vec(
            compressed
                .residual_norms
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
        );

        // Stage 1: PolarQuant reconstruction
        let x_mse = self.polar_quant.dequantize_batch(&mse_indices, &vector_norms);

        // Stage 2: QJL residual reconstruction
        let x_qjl = self.qjl.dequantize_batch(&qjl_signs, &residual_norms);

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

    fn synthetic_batch(batch: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((batch, d), |(i, j)| {
            let t = (i * 19 + j * 11) as f64;
            let base = (t / d as f64).sin() + 0.25 * (t / 13.0).cos();
            let scale = 0.6 + (i % 7) as f64 * 0.25;
            base * scale
        })
    }

    #[test]
    fn test_turboquant_roundtrip() {
        let d = 128;
        let tq = TurboQuant::new(d, 3, 42, true);
        let x = synthetic_batch(64, d);

        let compressed = tq.quantize_batch(&x);
        let x_hat = tq.dequantize(&compressed);

        let error: f64 = (&x - &x_hat).mapv(|v| v * v).sum();
        let original: f64 = x.mapv(|v| v * v).sum();
        let relative_error = error / original;
        assert!(
            relative_error < 0.6,
            "TurboQuant relative MSE: {relative_error}"
        );
    }

    #[test]
    fn test_turboquant_inner_product_preservation() {
        let d = 128;
        let batch = synthetic_batch(64, d);
        let tq = TurboQuant::new(d, 3, 42, true);

        let c_tq = tq.quantize_batch(&batch);
        let batch_tq = tq.dequantize(&c_tq);

        let mut tq_rel_err = 0.0;
        let mut n_pairs = 0usize;
        for i in 0..32 {
            let j = (i * 9 + 5) % 64;
            let x = batch.row(i);
            let y = batch.row(j);
            let ip_original = x.dot(&y);
            let denom = ip_original.abs().max(1e-8);

            let ip_tq = batch_tq.row(i).dot(&batch_tq.row(j));
            tq_rel_err += (ip_original - ip_tq).abs() / denom;
            n_pairs += 1;
        }

        let tq_mean = tq_rel_err / n_pairs as f64;
        assert!(
            tq_mean < 0.2,
            "TurboQuant mean inner product relative error: {tq_mean}"
        );
    }

    #[test]
    #[should_panic(expected = "bit_width >= 2")]
    fn test_turboquant_1bit_panics() {
        TurboQuant::new(32, 1, 42, true);
    }

    #[test]
    fn test_turboquant_mse_roundtrip() {
        let d = 128;
        let tq = TurboQuantMSE::new(d, 3, 42, true);
        let x = synthetic_batch(64, d);

        let (indices, norms) = tq.quantize_batch(&x);
        let x_hat = tq.dequantize_batch(&indices, &norms);

        let error: f64 = (&x - &x_hat).mapv(|v| v * v).sum();
        let original: f64 = x.mapv(|v| v * v).sum();
        let relative_error = error / original;
        assert!(
            relative_error < 0.6,
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
