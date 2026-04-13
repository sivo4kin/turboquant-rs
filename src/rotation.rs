//! Random rotation matrix generation for PolarQuant.
//!
//! Two implementations:
//! 1. Dense Haar-distributed rotation via QR decomposition — O(d²) multiply, exact
//! 2. Fast structured rotation via Hadamard + random sign flips — O(d log d), approximate

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::StandardNormal;

/// Generate a Haar-distributed random rotation matrix via QR decomposition.
///
/// Uses Modified Gram-Schmidt for the QR factorization, then adjusts
/// signs to ensure a proper rotation (det = +1).
pub fn random_rotation_dense(d: usize, rng: &mut impl Rng) -> Array2<f64> {
    assert!(d >= 1, "d must be >= 1, got {d}");

    // Random Gaussian matrix
    let g = Array2::from_shape_fn((d, d), |_| rng.sample::<f64, _>(StandardNormal));

    // QR decomposition via Modified Gram-Schmidt
    let (mut q, r) = qr_mgs(&g);

    // Make Q Haar-distributed by fixing signs via diagonal of R
    for j in 0..d {
        let sign = if r[[j, j]] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..d {
            q[[i, j]] *= sign;
        }
    }

    // Ensure proper rotation (det = +1) — flip first column if det = -1
    let sign = det_sign(&q);
    if sign < 0.0 {
        for i in 0..d {
            q[[i, 0]] = -q[[i, 0]];
        }
    }

    q
}

/// Modified Gram-Schmidt QR decomposition.
///
/// Returns (Q, R) where A = Q * R, Q is orthogonal, R is upper triangular.
fn qr_mgs(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = a.clone();
    let mut r = Array2::zeros((n, n));

    for j in 0..n {
        // Orthogonalize column j against previous columns
        for i in 0..j {
            let rij: f64 = q.column(i).dot(&q.column(j));
            r[[i, j]] = rij;
            let qi = q.column(i).to_owned();
            for k in 0..m {
                q[[k, j]] -= rij * qi[k];
            }
        }
        let rjj = q.column(j).dot(&q.column(j)).sqrt();
        r[[j, j]] = rjj;
        if rjj > 1e-15 {
            for k in 0..m {
                q[[k, j]] /= rjj;
            }
        }
    }

    (q, r)
}

/// Compute the sign of the determinant of a square matrix via LU decomposition
/// with partial pivoting.
///
/// Returns +1.0 or -1.0 (or 0.0 for singular matrices).
fn det_sign(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut sign = 1.0f64;

    for col in 0..n {
        // Partial pivoting: find row with largest absolute value in this column
        let mut max_row = col;
        let mut max_val = lu[[col, col]].abs();
        for row in (col + 1)..n {
            let val = lu[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_row != col {
            // Swap rows
            for j in 0..n {
                let tmp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
            sign = -sign;
        }

        if lu[[col, col]].abs() < 1e-15 {
            return 0.0; // Singular
        }

        // Eliminate below pivot
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];
            for j in col..n {
                lu[[row, j]] -= factor * lu[[col, j]];
            }
        }
    }

    // Sign of product of diagonal
    for i in 0..n {
        if lu[[i, i]] < 0.0 {
            sign = -sign;
        }
    }

    sign
}

/// Return the smallest power of 2 >= n.
fn next_power_of_2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Generate a normalized Hadamard matrix of size n (must be power of 2).
///
/// Uses the recursive Sylvester construction.
pub fn hadamard_matrix(n: usize) -> Array2<f64> {
    assert!(
        n >= 1 && n.is_power_of_two(),
        "n must be a positive power of 2, got {n}"
    );
    if n == 1 {
        return Array2::from_elem((1, 1), 1.0);
    }
    let half = hadamard_matrix(n / 2);
    let mut h = Array2::zeros((n, n));
    let hn = n / 2;
    for i in 0..hn {
        for j in 0..hn {
            let v = half[[i, j]];
            h[[i, j]] = v;
            h[[i, j + hn]] = v;
            h[[i + hn, j]] = v;
            h[[i + hn, j + hn]] = -v;
        }
    }
    h
}

/// Fast Walsh-Hadamard Transform, O(n log n).
///
/// Input length must be a positive power of 2. Returns a new normalized array.
pub fn fast_walsh_hadamard_transform(x: &Array1<f64>) -> Array1<f64> {
    let n = x.len();
    assert!(
        n >= 1 && n.is_power_of_two(),
        "Input length must be a positive power of 2, got {n}"
    );
    let mut out = x.clone();
    let mut h = 1usize;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let a = out[j];
                let b = out[j + h];
                out[j] = a + b;
                out[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let norm = (n as f64).sqrt();
    out.mapv_inplace(|v| v / norm);
    out
}

/// Components for a fast structured random rotation: D2 @ H @ D1 (random signs + Hadamard).
pub struct FastRotation {
    pub signs1: Array1<f64>,
    pub signs2: Array1<f64>,
    pub padded_d: usize,
    pub original_d: usize,
}

/// Generate a fast structured random rotation: D @ H @ D' (random signs + Hadamard).
///
/// For large d, this is O(d log d) to apply instead of O(d²).
pub fn random_rotation_fast(d: usize, rng: &mut impl Rng) -> FastRotation {
    let padded_d = next_power_of_2(d);
    let signs1 = Array1::from_shape_fn(padded_d, |_| if rng.gen::<bool>() { 1.0 } else { -1.0 });
    let signs2 = Array1::from_shape_fn(padded_d, |_| if rng.gen::<bool>() { 1.0 } else { -1.0 });
    FastRotation {
        signs1,
        signs2,
        padded_d,
        original_d: d,
    }
}

/// Apply the structured random rotation to a vector.
pub fn apply_fast_rotation(x: &Array1<f64>, rot: &FastRotation) -> Array1<f64> {
    let d = x.len();
    let mut padded = Array1::zeros(rot.padded_d);
    padded.slice_mut(ndarray::s![..d]).assign(x);
    // D1 @ x
    padded *= &rot.signs1;
    // H @ D1 @ x (normalized)
    padded = fast_walsh_hadamard_transform(&padded);
    // D2 @ H @ D1 @ x
    padded *= &rot.signs2;
    padded.slice(ndarray::s![..d]).to_owned()
}

/// Apply the transpose of the structured random rotation.
///
/// Since D and H are their own transposes (symmetric), the transpose is D1 @ H @ D2.
pub fn apply_fast_rotation_transpose(y: &Array1<f64>, rot: &FastRotation) -> Array1<f64> {
    let d = y.len();
    let mut padded = Array1::zeros(rot.padded_d);
    padded.slice_mut(ndarray::s![..d]).assign(y);
    // Reverse order: D2^T = D2, H^T = H, D1^T = D1
    padded *= &rot.signs2;
    padded = fast_walsh_hadamard_transform(&padded);
    padded *= &rot.signs1;
    padded.slice(ndarray::s![..d]).to_owned()
}

/// Apply structured rotation to a batch of vectors. Shape: (batch, d).
pub fn apply_fast_rotation_batch(x: &Array2<f64>, rot: &FastRotation) -> Array2<f64> {
    let (batch, d) = x.dim();
    let mut padded = Array2::zeros((batch, rot.padded_d));
    padded
        .slice_mut(ndarray::s![.., ..d])
        .assign(x);
    // D1: multiply each row by signs1
    for mut row in padded.rows_mut() {
        row *= &rot.signs1;
    }

    // Vectorized Walsh-Hadamard on each row
    let n = rot.padded_d;
    let mut h = 1usize;
    while h < n {
        // Process butterfly operations
        let mut new_padded = padded.clone();
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                for b in 0..batch {
                    let a_val = padded[[b, j]];
                    let b_val = padded[[b, j + h]];
                    new_padded[[b, j]] = a_val + b_val;
                    new_padded[[b, j + h]] = a_val - b_val;
                }
            }
        }
        padded = new_padded;
        h *= 2;
    }
    let norm = (n as f64).sqrt();
    padded.mapv_inplace(|v| v / norm);

    // D2: multiply each row by signs2
    for mut row in padded.rows_mut() {
        row *= &rot.signs2;
    }

    padded
        .slice(ndarray::s![.., ..d])
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn test_random_rotation_is_orthogonal() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let q = random_rotation_dense(16, &mut rng);

        // Q^T @ Q should be identity
        let qt = q.t();
        let eye = qt.dot(&q);
        for i in 0..16 {
            for j in 0..16 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye[[i, j]] - expected).abs() < 1e-10,
                    "Q^T Q[{i},{j}] = {}, expected {expected}",
                    eye[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_random_rotation_det_positive() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let q = random_rotation_dense(16, &mut rng);
        let sign = det_sign(&q);
        assert!(sign > 0.0, "det(Q) should be +1, got sign {sign}");
    }

    #[test]
    fn test_hadamard_orthogonal() {
        let h = hadamard_matrix(8);
        let hth = h.t().dot(&h);
        // H^T H = 8 * I (unnormalized Hadamard)
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 8.0 } else { 0.0 };
                assert!(
                    (hth[[i, j]] - expected).abs() < 1e-10,
                    "H^T H[{i},{j}] = {}, expected {expected}",
                    hth[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_fwht_matches_hadamard_matrix() {
        let h = hadamard_matrix(8);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let via_fwht = fast_walsh_hadamard_transform(&x);
        // Direct matrix multiply (unnormalized H @ x / sqrt(n))
        let via_matrix = h.dot(&x) / (8.0_f64).sqrt();
        for i in 0..8 {
            assert!(
                (via_fwht[i] - via_matrix[i]).abs() < 1e-10,
                "FWHT[{i}] = {}, matrix[{i}] = {}",
                via_fwht[i],
                via_matrix[i]
            );
        }
    }

    #[test]
    fn test_fast_rotation_roundtrip() {
        // d must be power of 2 for lossless roundtrip (no padding truncation)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let d = 16;
        let rot = random_rotation_fast(d, &mut rng);
        let x = Array1::from_shape_fn(d, |i| (i + 1) as f64);
        let y = apply_fast_rotation(&x, &rot);
        let x_back = apply_fast_rotation_transpose(&y, &rot);
        for i in 0..d {
            assert!(
                (x[i] - x_back[i]).abs() < 1e-10,
                "Roundtrip failed at index {i}: {} vs {}",
                x[i],
                x_back[i]
            );
        }
    }
}
