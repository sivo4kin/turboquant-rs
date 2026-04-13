//! Codebook construction for PolarQuant.
//!
//! After random rotation, each coordinate follows Beta(d/2, d/2) on [-1/√d, 1/√d],
//! which converges to N(0, 1/d) for large d. We use optimal scalar quantizers for
//! this distribution.
//!
//! Paper provides closed-form centroids for 1-bit and 2-bit. For higher bit-widths,
//! we use Lloyd's algorithm on the Gaussian approximation.

use ndarray::{Array1, Array2};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Compute optimal MSE centroids for the post-rotation coordinate distribution.
///
/// # Arguments
/// * `bit_width` - Number of bits per coordinate (1, 2, 3, 4, ...).
/// * `d` - Vector dimension (affects centroid scale).
///
/// # Returns
/// Sorted array of 2^bit_width centroids.
pub fn optimal_centroids(bit_width: usize, d: usize) -> Array1<f64> {
    let n_centroids = 1usize << bit_width;

    if bit_width == 1 {
        let c = (2.0 / (std::f64::consts::PI * d as f64)).sqrt();
        return Array1::from_vec(vec![-c, c]);
    }

    if bit_width == 2 {
        let inv_sqrt_d = 1.0 / (d as f64).sqrt();
        return Array1::from_vec(vec![
            -1.51 * inv_sqrt_d,
            -0.453 * inv_sqrt_d,
            0.453 * inv_sqrt_d,
            1.51 * inv_sqrt_d,
        ]);
    }

    // For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    lloyds_gaussian(n_centroids, 1.0 / (d as f64).sqrt(), 100)
}

/// Lloyd's algorithm (iterative k-means) for optimal scalar quantization of N(0, σ²).
fn lloyds_gaussian(n_centroids: usize, sigma: f64, n_iter: usize) -> Array1<f64> {
    let normal = Normal::new(0.0, sigma).unwrap();

    // Initialize boundary positions from uniform quantiles
    let mut boundaries = Vec::with_capacity(n_centroids - 1);
    for i in 1..n_centroids {
        let p = i as f64 / n_centroids as f64;
        boundaries.push(normal.inverse_cdf(p));
    }

    let mut centroids = vec![0.0f64; n_centroids];

    // Initial centroids: conditional expectations within each region
    centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0]);
    for i in 1..(n_centroids - 1) {
        centroids[i] = gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i]);
    }
    centroids[n_centroids - 1] =
        gaussian_conditional_expectation(sigma, boundaries[n_centroids - 2], f64::INFINITY);

    for _ in 0..n_iter {
        // Update boundaries (midpoints between consecutive centroids)
        for i in 0..(n_centroids - 1) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        // Update centroids (conditional expectations within each region)
        centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0]);
        for i in 1..(n_centroids - 1) {
            centroids[i] =
                gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i]);
        }
        centroids[n_centroids - 1] =
            gaussian_conditional_expectation(sigma, boundaries[n_centroids - 2], f64::INFINITY);
    }

    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Array1::from_vec(centroids)
}

/// E[X | a < X < b] where X ~ N(0, σ²).
///
/// Uses the formula: E[X | a < X < b] = σ² * (φ(a/σ) − φ(b/σ)) / (Φ(b/σ) − Φ(a/σ))
/// where φ is the PDF and Φ is the CDF of standard normal.
fn gaussian_conditional_expectation(sigma: f64, a: f64, b: f64) -> f64 {
    let std_normal = Normal::new(0.0, 1.0).unwrap();

    let a_std = if a.is_finite() { a / sigma } else { a };
    let b_std = if b.is_finite() { b / sigma } else { b };

    // Compute P(a < X/σ < b) using numerically stable formulation
    let prob = if !a_std.is_finite() && a_std < 0.0 {
        // a = -inf
        std_normal.cdf(b_std)
    } else if !b_std.is_finite() && b_std > 0.0 {
        // b = +inf: use sf(a) = cdf(-a) for better stability
        std_normal.cdf(-a_std)
    } else {
        std_normal.cdf(b_std) - std_normal.cdf(a_std)
    };

    if prob < 1e-15 {
        // Asymptotic approximation for extreme regions
        if a.is_finite() && !b.is_finite() {
            return a + sigma; // E[X | X > a] ≈ a + σ
        } else if !a.is_finite() && b.is_finite() {
            return b - sigma;
        } else if a.is_finite() && b.is_finite() {
            return (a + b) / 2.0;
        } else {
            return 0.0;
        }
    }

    let pdf_diff = std_normal.pdf(a_std) - std_normal.pdf(b_std);
    sigma * pdf_diff / prob
}

/// Find nearest centroid index for each value (single vector). Vectorized via binary search.
///
/// Uses `searchsorted` on midpoint boundaries — O(n log k) instead of O(n * k).
pub fn nearest_centroid_indices(values: &Array1<f64>, centroids: &Array1<f64>) -> Array1<usize> {
    let boundaries = compute_boundaries(centroids);
    Array1::from_shape_fn(values.len(), |i| {
        boundaries.partition_point(|&b| b < values[i])
    })
}

/// Find nearest centroid index for each value (batch of vectors).
pub fn nearest_centroid_indices_batch(
    values: &Array2<f64>,
    centroids: &Array1<f64>,
) -> Array2<usize> {
    let boundaries = compute_boundaries(centroids);
    let (batch, d) = values.dim();
    Array2::from_shape_fn((batch, d), |(i, j)| {
        boundaries.partition_point(|&b| b < values[[i, j]])
    })
}

/// Compute midpoint boundaries between sorted centroids.
fn compute_boundaries(centroids: &Array1<f64>) -> Vec<f64> {
    let n = centroids.len();
    (0..n - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1bit_centroids_symmetric() {
        let c = optimal_centroids(1, 128);
        assert_eq!(c.len(), 2);
        assert!((c[0] + c[1]).abs() < 1e-15, "1-bit centroids should be symmetric");
    }

    #[test]
    fn test_2bit_centroids_count() {
        let c = optimal_centroids(2, 128);
        assert_eq!(c.len(), 4);
        // Should be sorted
        for i in 1..4 {
            assert!(c[i] > c[i - 1], "Centroids should be sorted");
        }
    }

    #[test]
    fn test_3bit_centroids_via_lloyds() {
        let c = optimal_centroids(3, 128);
        assert_eq!(c.len(), 8);
        // Should be sorted and symmetric around 0
        for i in 1..8 {
            assert!(c[i] > c[i - 1], "Centroids should be sorted");
        }
        // Check approximate symmetry
        for i in 0..4 {
            assert!(
                (c[i] + c[7 - i]).abs() < 1e-6,
                "Centroids should be approximately symmetric: {} vs {}",
                c[i],
                c[7 - i]
            );
        }
    }

    #[test]
    fn test_nearest_centroid_basic() {
        let centroids = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let values = Array1::from_vec(vec![-0.8, 0.1, 0.7, -0.1]);
        let indices = nearest_centroid_indices(&values, &centroids);
        assert_eq!(indices[0], 0); // -0.8 closest to -1.0
        assert_eq!(indices[1], 1); // 0.1 closest to 0.0
        assert_eq!(indices[2], 2); // 0.7 closest to 1.0
        assert_eq!(indices[3], 1); // -0.1 closest to 0.0
    }
}
