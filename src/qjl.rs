//! QJL: Quantized Johnson-Lindenstrauss Transform.
//!
//! 1-bit quantization via random projection → sign to compress vectors while
//! preserving inner products. This implementation stores a full d×d projection
//! matrix (O(d²) memory). For large d, a structured/seeded approach would be needed.
//!
//! Key property: unbiased and optimal at 1-bit.
//!     Q_qjl(x) = sign(S · x) where S ~ N(0,1)^(d×d)
//!     Q_qjl_inv(z) = √(π/2) / d · S^T · z

use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

/// √(π/2) constant used in QJL dequantization.
const QJL_CONST: f64 = 1.2533141373155003; // sqrt(pi/2)

/// Quantized Johnson-Lindenstrauss 1-bit quantizer.
pub struct QJL {
    pub d: usize,
    s: Array2<f64>,
}

impl QJL {
    /// Create a new QJL quantizer.
    ///
    /// # Arguments
    /// * `d` - Vector dimension.
    /// * `seed` - Random seed for projection matrix.
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // Random projection matrix S ∈ R^(d×d), entries ~ N(0,1)
        let s = Array2::from_shape_fn((d, d), |_| rng.sample::<f64, _>(StandardNormal));
        Self { d, s }
    }

    /// Quantize a single residual vector to sign bits.
    ///
    /// Returns (signs, norm) where signs are {+1, -1} and norm is ||r||_2.
    pub fn quantize(&self, r: &Array1<f64>) -> (Array1<i8>, f64) {
        let (signs, norms) = self.quantize_batch(&r.clone().insert_axis(Axis(0)));
        (signs.row(0).to_owned(), norms[0])
    }

    /// Quantize a batch of residual vectors to sign bits.
    ///
    /// # Arguments
    /// * `r` - Residual vectors, shape (batch, d).
    ///
    /// # Returns
    /// (signs, norms) where signs: (batch, d) with values {+1, -1}, norms: (batch,).
    pub fn quantize_batch(&self, r: &Array2<f64>) -> (Array2<i8>, Array1<f64>) {
        let batch = r.nrows();

        // Compute norms before projection
        let mut norms = Array1::zeros(batch);
        for i in 0..batch {
            norms[i] = r.row(i).dot(&r.row(i)).sqrt();
        }

        // Project: (S @ r.T).T = r @ S.T
        let projected = r.dot(&self.s.t());

        // Sign quantization: +1 or -1
        let (b, d) = projected.dim();
        let mut signs = Array2::zeros((b, d));
        for i in 0..b {
            for j in 0..d {
                signs[[i, j]] = if projected[[i, j]] >= 0.0 { 1i8 } else { -1i8 };
            }
        }

        (signs, norms)
    }

    /// Dequantize a single set of sign bits back to approximate residual.
    pub fn dequantize(&self, signs: &Array1<i8>, norm: f64) -> Array1<f64> {
        let signs_2d = signs.clone().insert_axis(Axis(0));
        let norms = Array1::from_vec(vec![norm]);
        let result = self.dequantize_batch(&signs_2d, &norms);
        result.row(0).to_owned()
    }

    /// Dequantize a batch of sign bits back to approximate residuals.
    ///
    /// # Arguments
    /// * `signs` - Sign bits, shape (batch, d) with values {+1, -1}.
    /// * `norms` - Residual norms, shape (batch,).
    ///
    /// # Returns
    /// Approximate residuals, shape (batch, d).
    pub fn dequantize_batch(&self, signs: &Array2<i8>, norms: &Array1<f64>) -> Array2<f64> {
        let batch = signs.nrows();

        // Convert signs to f64 for matrix multiply
        let signs_f64 = signs.mapv(|s| s as f64);

        // x̃_qjl = √(π/2) / d · γ · S^T @ signs
        // (S^T @ signs.T).T = signs @ S
        let reconstructed_raw = signs_f64.dot(&self.s);

        // Scale by √(π/2) / d * norm
        let mut reconstructed = reconstructed_raw;
        for i in 0..batch {
            let scale = QJL_CONST / self.d as f64 * norms[i];
            for j in 0..self.d {
                reconstructed[[i, j]] *= scale;
            }
        }

        reconstructed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_batch(batch: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((batch, d), |(i, j)| {
            let t = (i * 17 + j * 5) as f64;
            (t / d as f64).sin() + 0.2 * (t / 9.0).cos()
        })
    }

    #[test]
    fn test_sign_quantization() {
        let qjl = QJL::new(32, 42);
        let r = Array1::from_shape_fn(32, |i| (i as f64 - 16.0) / 10.0);
        let (signs, norm) = qjl.quantize(&r);

        // Signs should be +1 or -1
        for &s in signs.iter() {
            assert!(s == 1 || s == -1, "Sign should be +1 or -1, got {s}");
        }

        // Norm should be positive
        assert!(norm > 0.0);
    }

    #[test]
    fn test_zero_residual() {
        let qjl = QJL::new(16, 42);
        let r = Array1::zeros(16);
        let (signs, norm) = qjl.quantize(&r);
        let r_hat = qjl.dequantize(&signs, norm);

        // Zero residual should reconstruct to zero (norm = 0 → scale = 0)
        let error: f64 = r_hat.mapv(|v| v * v).sum();
        assert!(error < 1e-20, "Zero residual reconstruction error: {error}");
    }

    #[test]
    fn test_inner_product_preservation() {
        let d = 256;
        let qjl = QJL::new(d, 42);
        let batch = synthetic_batch(64, d);
        let (signs, norms) = qjl.quantize_batch(&batch);
        let batch_hat = qjl.dequantize_batch(&signs, &norms);

        let mut total_rel_err = 0.0;
        let mut n_pairs = 0usize;
        for i in 0..32 {
            let j = (i * 7 + 3) % 64;
            let x = batch.row(i);
            let y = batch.row(j);
            let x_hat = batch_hat.row(i);
            let y_hat = batch_hat.row(j);
            let ip_original = x.dot(&y);
            let ip_approx = x_hat.dot(&y_hat);
            let denom = ip_original.abs().max(1e-8);
            total_rel_err += (ip_original - ip_approx).abs() / denom;
            n_pairs += 1;
        }
        let relative_error = total_rel_err / n_pairs as f64;
        assert!(
            relative_error < 2.5,
            "Mean inner product relative error too large: {relative_error}"
        );
    }
}
