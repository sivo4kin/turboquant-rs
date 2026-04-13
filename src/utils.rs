//! Utility functions for bit packing and memory measurement.

use ndarray::{Array1, Array2};

/// Pack {+1, -1} sign array into a compact bit vector.
///
/// 8 signs per byte. +1 → 1, -1 → 0.
pub fn pack_bits(signs: &Array1<i8>) -> Vec<u8> {
    let d = signs.len();
    let n_bytes = (d + 7) / 8;
    let mut packed = vec![0u8; n_bytes];

    for i in 0..d {
        if signs[i] > 0 {
            packed[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    packed
}

/// Pack batch of {+1, -1} sign arrays into compact bit vectors.
pub fn pack_bits_batch(signs: &Array2<i8>) -> Vec<Vec<u8>> {
    let batch = signs.nrows();
    (0..batch)
        .map(|i| pack_bits(&signs.row(i).to_owned()))
        .collect()
}

/// Unpack bit vector back to {+1, -1} signs.
///
/// # Arguments
/// * `packed` - Packed bytes from `pack_bits`.
/// * `d` - Original dimension (to truncate padding).
pub fn unpack_bits(packed: &[u8], d: usize) -> Array1<i8> {
    let mut signs = Array1::zeros(d);
    for i in 0..d {
        let bit = (packed[i / 8] >> (7 - (i % 8))) & 1;
        signs[i] = if bit == 1 { 1 } else { -1 };
    }
    signs
}

/// Pack b-bit indices into compact byte array.
///
/// For bit_width <= 4, packs multiple indices per byte using bit manipulation.
/// For bit_width 5-8, uses uint8 directly.
pub fn pack_indices(indices: &Array1<usize>, bit_width: usize) -> Vec<u8> {
    assert!(
        (1..=8).contains(&bit_width),
        "bit_width must be 1-8, got {bit_width}"
    );

    if bit_width <= 4 {
        // Convert each index to bit_width binary digits, then pack 8 bits per byte
        let n = indices.len();
        let total_bits = n * bit_width;
        let n_bytes = (total_bits + 7) / 8;
        let mut packed = vec![0u8; n_bytes];

        let mut bit_pos = 0usize;
        for i in 0..n {
            let val = indices[i] as u8;
            for b in 0..bit_width {
                let bit = (val >> (bit_width - 1 - b)) & 1;
                if bit == 1 {
                    packed[bit_pos / 8] |= 1 << (7 - (bit_pos % 8));
                }
                bit_pos += 1;
            }
        }

        packed
    } else {
        // 5-8 bit: just use uint8
        indices.iter().map(|&v| v as u8).collect()
    }
}

/// Unpack b-bit indices from compact byte array.
///
/// # Arguments
/// * `packed` - Packed bytes from `pack_indices`.
/// * `n` - Number of original indices.
/// * `bit_width` - Bits per index.
pub fn unpack_indices(packed: &[u8], n: usize, bit_width: usize) -> Array1<usize> {
    assert!(
        (1..=8).contains(&bit_width),
        "bit_width must be 1-8, got {bit_width}"
    );

    if bit_width <= 4 {
        let mut indices = Array1::zeros(n);
        let mut bit_pos = 0usize;

        for i in 0..n {
            let mut val = 0u8;
            for b in 0..bit_width {
                let bit = (packed[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1;
                val |= bit << (bit_width - 1 - b);
                bit_pos += 1;
            }
            indices[i] = val as usize;
        }

        indices
    } else {
        Array1::from_vec(packed[..n].iter().map(|&v| v as usize).collect())
    }
}

/// Calculate memory footprint of compressed KV cache.
pub fn memory_footprint_bytes(n_vectors: usize, d: usize, bit_width: usize) -> MemoryFootprint {
    let mse_bits = bit_width - 1; // PolarQuant uses b-1 bits
    let qjl_bits = 1;

    let mse_bytes = (n_vectors * d * mse_bits + 7) / 8;
    let qjl_bytes = (n_vectors * d * qjl_bits + 7) / 8;
    let norm_bytes = n_vectors * 4; // float32 per vector
    let total = mse_bytes + qjl_bytes + norm_bytes;
    let original = n_vectors * d * 2; // fp16

    MemoryFootprint {
        mse_indices_bytes: mse_bytes,
        qjl_signs_bytes: qjl_bytes,
        norms_bytes: norm_bytes,
        total_bytes: total,
        original_fp16_bytes: original,
        compression_ratio: if total > 0 {
            original as f64 / total as f64
        } else {
            f64::INFINITY
        },
    }
}

/// Memory footprint breakdown for compressed vectors.
#[derive(Debug)]
pub struct MemoryFootprint {
    pub mse_indices_bytes: usize,
    pub qjl_signs_bytes: usize,
    pub norms_bytes: usize,
    pub total_bytes: usize,
    pub original_fp16_bytes: usize,
    pub compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_bits_roundtrip() {
        let signs = Array1::from_vec(vec![1i8, -1, 1, 1, -1, -1, 1, -1, 1, 1]);
        let packed = pack_bits(&signs);
        let unpacked = unpack_bits(&packed, signs.len());
        assert_eq!(signs, unpacked);
    }

    #[test]
    fn test_pack_unpack_bits_exact_byte() {
        let signs = Array1::from_vec(vec![1i8, -1, 1, 1, -1, -1, 1, -1]);
        let packed = pack_bits(&signs);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_bits(&packed, 8);
        assert_eq!(signs, unpacked);
    }

    #[test]
    fn test_pack_unpack_indices_2bit() {
        let indices = Array1::from_vec(vec![0usize, 1, 2, 3, 0, 2, 1, 3]);
        let packed = pack_indices(&indices, 2);
        let unpacked = unpack_indices(&packed, indices.len(), 2);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_indices_3bit() {
        let indices = Array1::from_vec(vec![0usize, 1, 2, 3, 4, 5, 6, 7]);
        let packed = pack_indices(&indices, 3);
        let unpacked = unpack_indices(&packed, indices.len(), 3);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_indices_8bit() {
        let indices = Array1::from_vec(vec![0usize, 127, 255, 42]);
        let packed = pack_indices(&indices, 8);
        let unpacked = unpack_indices(&packed, indices.len(), 8);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_memory_footprint() {
        let fp = memory_footprint_bytes(1000, 128, 3);
        assert!(fp.compression_ratio > 1.0);
        assert!(fp.total_bytes < fp.original_fp16_bytes);
    }
}
