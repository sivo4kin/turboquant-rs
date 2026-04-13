//! Outlier channel strategy for non-integer bit precision.
//!
//! Split channels into outlier (higher bits) and non-outlier (lower bits).
//!
//! Examples:
//! - 2.5-bit: 32/128 outlier at 3b + 96/128 normal at 2b = 2.5 avg
//! - 3.5-bit: 64/128 outlier at 4b + 64/128 normal at 3b = 3.5 avg

use ndarray::{Array1, Array2, Axis};

use crate::polar_quant::PolarQuant;
use crate::qjl::QJL;

/// Container for outlier-strategy compressed vector(s).
pub struct OutlierCompressedVectors {
    /// Indices for outlier channels (higher bits). Shape: (batch, n_outlier).
    pub outlier_indices: Array2<usize>,
    /// Norms for outlier channels. Shape: (batch,).
    pub outlier_norms: Array1<f64>,
    /// Indices for normal channels (lower bits). Shape: (batch, n_normal).
    pub normal_indices: Array2<usize>,
    /// Norms for normal channels. Shape: (batch,).
    pub normal_norms: Array1<f64>,
    /// QJL signs for full residual. Shape: (batch, d).
    pub qjl_signs: Array2<i8>,
    /// Residual norms. Shape: (batch,).
    pub residual_norms: Array1<f64>,
    /// Effective bits per channel.
    pub effective_bits: f64,
}

/// Compute how many channels get higher vs lower bit-width.
fn compute_channel_split(d: usize, target_bits: f64) -> (usize, usize, usize, usize) {
    let low_bits = target_bits.floor() as usize;
    let high_bits = low_bits + 1;
    let frac = target_bits - low_bits as f64;

    let n_outlier = (d as f64 * frac).round() as usize;
    let n_normal = d - n_outlier;

    (n_outlier, high_bits, n_normal, low_bits)
}

/// TurboQuant with outlier channel strategy for non-integer bit rates.
///
/// Splits channels into outlier (higher bit-width) and normal (lower bit-width)
/// to achieve fractional average bit rates like 2.5 or 3.5 bits per channel.
pub struct OutlierTurboQuant {
    pub d: usize,
    pub target_bits: f64,
    pub n_outlier: usize,
    pub n_normal: usize,
    pub high_bits: usize,
    pub low_bits: usize,
    pub effective_bits: f64,
    outlier_idx: Vec<usize>,
    normal_idx: Vec<usize>,
    pq_outlier: Option<PolarQuant>,
    pq_normal: Option<PolarQuant>,
    qjl: QJL,
}

impl OutlierTurboQuant {
    /// Create a new OutlierTurboQuant quantizer.
    ///
    /// # Arguments
    /// * `d` - Vector dimension.
    /// * `target_bits` - Target average bits per channel (e.g., 2.5, 3.5).
    /// * `seed` - Random seed.
    pub fn new(d: usize, target_bits: f64, seed: u64) -> Self {
        let (n_outlier, high_bits, n_normal, low_bits) = compute_channel_split(d, target_bits);

        let effective_bits =
            (n_outlier * high_bits + n_normal * low_bits) as f64 / d as f64;

        let outlier_idx: Vec<usize> = (0..n_outlier).collect();
        let normal_idx: Vec<usize> = (n_outlier..d).collect();

        // Separate PolarQuant for outlier and normal channels
        // PolarQuant bit-width is (total - 1) since QJL adds 1 bit
        let pq_outlier = if n_outlier > 0 {
            Some(PolarQuant::new(n_outlier, high_bits - 1, seed, true))
        } else {
            None
        };
        let pq_normal = if n_normal > 0 {
            Some(PolarQuant::new(n_normal, low_bits - 1, seed + 500, true))
        } else {
            None
        };

        let qjl = QJL::new(d, seed + 1000);

        Self {
            d,
            target_bits,
            n_outlier,
            n_normal,
            high_bits,
            low_bits,
            effective_bits,
            outlier_idx,
            normal_idx,
            pq_outlier,
            pq_normal,
            qjl,
        }
    }

    /// Quantize a single vector with outlier channel split.
    pub fn quantize(&self, x: &Array1<f64>) -> OutlierCompressedVectors {
        self.quantize_batch(&x.clone().insert_axis(Axis(0)))
    }

    /// Quantize a batch of vectors with outlier channel split.
    pub fn quantize_batch(&self, x: &Array2<f64>) -> OutlierCompressedVectors {
        let batch = x.nrows();

        // Split channels
        let x_outlier = self.extract_channels(x, &self.outlier_idx);
        let x_normal = self.extract_channels(x, &self.normal_idx);

        // Quantize outlier channels at higher bits
        let (out_idx, out_norms, out_residual) = if let Some(ref pq) = self.pq_outlier {
            pq.quantize_and_residual_batch(&x_outlier)
        } else {
            (
                Array2::zeros((batch, 0)),
                Array1::zeros(batch),
                Array2::zeros((batch, 0)),
            )
        };

        // Quantize normal channels at lower bits
        let (norm_idx, norm_norms, norm_residual) = if let Some(ref pq) = self.pq_normal {
            pq.quantize_and_residual_batch(&x_normal)
        } else {
            (
                Array2::zeros((batch, 0)),
                Array1::zeros(batch),
                Array2::zeros((batch, 0)),
            )
        };

        // Reconstruct full residual
        let mut full_residual = Array2::zeros((batch, self.d));
        for i in 0..batch {
            for (j, &idx) in self.outlier_idx.iter().enumerate() {
                full_residual[[i, idx]] = out_residual[[i, j]];
            }
            for (j, &idx) in self.normal_idx.iter().enumerate() {
                full_residual[[i, idx]] = norm_residual[[i, j]];
            }
        }

        // QJL on full residual
        let (qjl_signs, residual_norms) = self.qjl.quantize_batch(&full_residual);

        OutlierCompressedVectors {
            outlier_indices: out_idx,
            outlier_norms: out_norms,
            normal_indices: norm_idx,
            normal_norms: norm_norms,
            qjl_signs,
            residual_norms,
            effective_bits: self.effective_bits,
        }
    }

    /// Dequantize outlier-strategy compressed vectors.
    pub fn dequantize(&self, compressed: &OutlierCompressedVectors) -> Array2<f64> {
        let batch = compressed.qjl_signs.nrows();

        // Reconstruct outlier channels
        let x_outlier = if let Some(ref pq) = self.pq_outlier {
            pq.dequantize_batch(&compressed.outlier_indices, &compressed.outlier_norms)
        } else {
            Array2::zeros((batch, 0))
        };

        // Reconstruct normal channels
        let x_normal = if let Some(ref pq) = self.pq_normal {
            pq.dequantize_batch(&compressed.normal_indices, &compressed.normal_norms)
        } else {
            Array2::zeros((batch, 0))
        };

        // Reconstruct QJL residual
        let x_qjl = self
            .qjl
            .dequantize_batch(&compressed.qjl_signs, &compressed.residual_norms);

        // Combine
        let mut x_hat = Array2::zeros((batch, self.d));
        for i in 0..batch {
            for (j, &idx) in self.outlier_idx.iter().enumerate() {
                x_hat[[i, idx]] = x_outlier[[i, j]];
            }
            for (j, &idx) in self.normal_idx.iter().enumerate() {
                x_hat[[i, idx]] = x_normal[[i, j]];
            }
        }
        x_hat += &x_qjl;

        x_hat
    }

    /// Compression ratio vs original precision.
    pub fn compression_ratio(&self, original_bits: usize) -> f64 {
        let per_vector_bits = (self.d as f64 * self.effective_bits) as usize + 32 + 64;
        let original = self.d * original_bits;
        original as f64 / per_vector_bits as f64
    }

    /// Extract selected channels from a batch of vectors.
    fn extract_channels(&self, x: &Array2<f64>, channel_indices: &[usize]) -> Array2<f64> {
        let batch = x.nrows();
        let n_channels = channel_indices.len();
        let mut result = Array2::zeros((batch, n_channels));
        for i in 0..batch {
            for (j, &idx) in channel_indices.iter().enumerate() {
                result[[i, j]] = x[[i, idx]];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_split_half() {
        let (n_out, high, n_norm, low) = compute_channel_split(128, 2.5);
        assert_eq!(n_out, 64);
        assert_eq!(high, 3);
        assert_eq!(n_norm, 64);
        assert_eq!(low, 2);
    }

    #[test]
    fn test_channel_split_quarter() {
        let (n_out, high, n_norm, low) = compute_channel_split(128, 3.25);
        assert_eq!(n_out, 32);
        assert_eq!(high, 4);
        assert_eq!(n_norm, 96);
        assert_eq!(low, 3);
    }

    #[test]
    fn test_outlier_roundtrip() {
        let oq = OutlierTurboQuant::new(32, 2.5, 42);
        let x = Array1::from_shape_fn(32, |i| (i as f64 + 1.0) / 32.0);

        let compressed = oq.quantize(&x);
        let x_hat = oq.dequantize(&compressed);
        let x_hat_vec = x_hat.row(0).to_owned();

        let error: f64 = (&x - &x_hat_vec).mapv(|v| v * v).sum();
        let original: f64 = x.mapv(|v| v * v).sum();
        let relative_error = error / original;
        assert!(
            relative_error < 2.0,
            "Outlier TurboQuant relative MSE: {relative_error}"
        );
    }

    #[test]
    fn test_effective_bits() {
        let oq = OutlierTurboQuant::new(128, 2.5, 42);
        assert!(
            (oq.effective_bits - 2.5).abs() < 0.01,
            "Effective bits: {}",
            oq.effective_bits
        );
    }
}
