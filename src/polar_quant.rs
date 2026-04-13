//! PolarQuant: Random rotation + optimal scalar quantization.
//!
//! Algorithm 1 from the TurboQuant paper (AISTATS 2026).
//!
//! After random rotation, coordinates follow a known Beta distribution (Gaussian in
//! high d), enabling optimal scalar quantization per coordinate independently.
//!
//! Important: codebook is calibrated for unit-norm vectors. For non-unit-norm inputs,
//! we extract norms, normalize, quantize, then rescale on dequantization.

use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::codebook::{nearest_centroid_indices_batch, optimal_centroids};
use crate::rotation::random_rotation_dense;

/// MSE-optimized vector quantizer via random rotation + scalar quantization.
///
/// Handles arbitrary-norm vectors by extracting norms before quantization
/// and rescaling after dequantization.
pub struct PolarQuant {
    pub d: usize,
    pub bit_width: usize,
    pub n_centroids: usize,
    pub norm_correction: bool,
    rotation: Array2<f64>,
    centroids: Array1<f64>,
}

impl PolarQuant {
    /// Create a new PolarQuant quantizer.
    ///
    /// # Arguments
    /// * `d` - Vector dimension.
    /// * `bit_width` - Number of bits per coordinate.
    /// * `seed` - Random seed for rotation matrix.
    /// * `norm_correction` - If true, re-normalize reconstructed vectors in rotated domain.
    pub fn new(d: usize, bit_width: usize, seed: u64, norm_correction: bool) -> Self {
        let n_centroids = 1usize << bit_width;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let rotation = random_rotation_dense(d, &mut rng);
        let centroids = optimal_centroids(bit_width, d);

        Self {
            d,
            bit_width,
            n_centroids,
            norm_correction,
            rotation,
            centroids,
        }
    }

    /// Quantize a single vector.
    ///
    /// Returns (indices, norm) where indices are centroid indices and norm is the L2 norm.
    pub fn quantize(&self, x: &Array1<f64>) -> (Array1<usize>, f64) {
        let (indices, norms) = self.quantize_batch(&x.clone().insert_axis(Axis(0)));
        (indices.row(0).to_owned(), norms[0])
    }

    /// Quantize a batch of vectors.
    ///
    /// # Arguments
    /// * `x` - Input vectors, shape (batch, d).
    ///
    /// # Returns
    /// (indices, norms) where indices: (batch, d), norms: (batch,).
    pub fn quantize_batch(&self, x: &Array2<f64>) -> (Array2<usize>, Array1<f64>) {
        let batch = x.nrows();

        // Extract norms and normalize (paper page 5)
        let mut norms = Array1::zeros(batch);
        for i in 0..batch {
            norms[i] = x.row(i).dot(&x.row(i)).sqrt();
        }

        // Normalize: x_normalized = x / ||x||
        let mut x_normalized = x.clone();
        for i in 0..batch {
            let safe_norm = if norms[i] > 0.0 { norms[i] } else { 1.0 };
            for j in 0..self.d {
                x_normalized[[i, j]] /= safe_norm;
            }
        }

        // Rotate normalized vectors: y = (R @ x_normalized.T).T = x_normalized @ R.T
        let y = x_normalized.dot(&self.rotation.t());

        // Nearest centroid per coordinate
        let indices = nearest_centroid_indices_batch(&y, &self.centroids);

        (indices, norms)
    }

    /// Dequantize a single vector.
    pub fn dequantize(&self, indices: &Array1<usize>, norm: f64) -> Array1<f64> {
        let norms = Array1::from_vec(vec![norm]);
        let indices_2d = indices.clone().insert_axis(Axis(0));
        let result = self.dequantize_batch(&indices_2d, &norms);
        result.row(0).to_owned()
    }

    /// Dequantize a batch of vectors.
    ///
    /// # Arguments
    /// * `indices` - Integer indices, shape (batch, d).
    /// * `norms` - Original L2 norms, shape (batch,).
    ///
    /// # Returns
    /// Reconstructed vectors, shape (batch, d).
    pub fn dequantize_batch(&self, indices: &Array2<usize>, norms: &Array1<f64>) -> Array2<f64> {
        let (batch, d) = indices.dim();

        // Look up centroids in the rotated domain
        let mut y_hat = Array2::zeros((batch, d));
        for i in 0..batch {
            for j in 0..d {
                y_hat[[i, j]] = self.centroids[indices[[i, j]]];
            }
        }

        // Norm correction: re-normalize y_hat to unit norm
        if self.norm_correction {
            for i in 0..batch {
                let y_norm = y_hat.row(i).dot(&y_hat.row(i)).sqrt();
                let safe_norm = if y_norm > 1e-10 { y_norm } else { 1.0 };
                for j in 0..d {
                    y_hat[[i, j]] /= safe_norm;
                }
            }
        }

        // Inverse rotation: x_hat_unit = (R.T @ y_hat.T).T = y_hat @ R
        let x_hat_unit = y_hat.dot(&self.rotation);

        // Rescale by original norms
        let mut x_hat = x_hat_unit;
        for i in 0..batch {
            for j in 0..d {
                x_hat[[i, j]] *= norms[i];
            }
        }

        x_hat
    }

    /// Quantize and return indices, norms, and residual error.
    ///
    /// Used by TurboQuant's second stage (QJL on residual).
    pub fn quantize_and_residual(
        &self,
        x: &Array1<f64>,
    ) -> (Array1<usize>, f64, Array1<f64>) {
        let (indices, norm) = self.quantize(x);
        let x_hat = self.dequantize(&indices, norm);
        let residual = x - &x_hat;
        (indices, norm, residual)
    }

    /// Quantize batch and return indices, norms, and residual errors.
    pub fn quantize_and_residual_batch(
        &self,
        x: &Array2<f64>,
    ) -> (Array2<usize>, Array1<f64>, Array2<f64>) {
        let (indices, norms) = self.quantize_batch(x);
        let x_hat = self.dequantize_batch(&indices, &norms);
        let residual = x - &x_hat;
        (indices, norms, residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_batch(batch: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((batch, d), |(i, j)| {
            let phase = (i * 13 + j * 7) as f64;
            let base = (phase / d as f64).sin() + 0.3 * (phase / 11.0).cos();
            let scale = 0.5 + (i % 5) as f64 * 0.35;
            base * scale
        })
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let d = 128;
        let pq = PolarQuant::new(d, 2, 42, true);
        let x = synthetic_batch(64, d);
        let (indices, norms) = pq.quantize_batch(&x);
        let x_hat = pq.dequantize_batch(&indices, &norms);

        assert_eq!(indices.dim(), (64, d));
        assert_eq!(x_hat.dim(), (64, d));

        let error: f64 = (&x - &x_hat).mapv(|v| v * v).sum();
        let original: f64 = x.mapv(|v| v * v).sum();
        let relative_error = error / original;
        assert!(
            relative_error < 0.75,
            "Relative MSE too large: {relative_error}"
        );
    }

    #[test]
    fn test_zero_vector() {
        let pq = PolarQuant::new(16, 2, 42, true);
        let x = Array1::zeros(16);
        let (indices, norm) = pq.quantize(&x);
        let x_hat = pq.dequantize(&indices, norm);
        // Zero vector should reconstruct to approximately zero
        let error: f64 = x_hat.mapv(|v| v * v).sum();
        assert!(error < 1e-20, "Zero vector reconstruction error: {error}");
    }

    #[test]
    fn test_batch_matches_single() {
        let pq = PolarQuant::new(16, 2, 42, true);
        let x1 = Array1::from_shape_fn(16, |i| (i as f64 + 1.0) / 16.0);
        let x2 = Array1::from_shape_fn(16, |i| -(i as f64 + 1.0) / 16.0);

        let (idx1, n1) = pq.quantize(&x1);
        let (idx2, n2) = pq.quantize(&x2);

        let mut batch = Array2::zeros((2, 16));
        batch.row_mut(0).assign(&x1);
        batch.row_mut(1).assign(&x2);
        let (batch_idx, batch_norms) = pq.quantize_batch(&batch);

        assert_eq!(idx1, batch_idx.row(0).to_owned());
        assert_eq!(idx2, batch_idx.row(1).to_owned());
        assert!((n1 - batch_norms[0]).abs() < 1e-15);
        assert!((n2 - batch_norms[1]).abs() < 1e-15);
    }
}
