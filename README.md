# turboquant-rs

Rust port of [turboquant_plus](https://github.com/TheTom/turboquant_plus) — KV cache compression for LLM inference via PolarQuant + QJL.

## Background

This project implements the algorithms described in the **TurboQuant** paper (ICLR 2026, [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)) and the related **PolarQuant** paper (AISTATS 2026, [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)).

The original Python reference implementation by [TheTom](https://github.com/TheTom/turboquant_plus) demonstrated that KV cache tensors in transformer models can be aggressively compressed with minimal quality loss, achieving 3.8–6.4× compression ratios. This crate is a faithful Rust port of that work.

## Key ideas

1. **V compression is nearly free** — value cache compression down to 2 bits shows negligible quality impact when key precision is maintained.
2. **K precision dominates** — all observed quality degradation stems from key cache compression.
3. **Boundary layers matter** — protecting the first and last two layers at higher precision recovers 37–91% of the quality gap.

## Compression formats

| Format | Bits/value | Compression | PPL vs q8_0 |
|--------|-----------|-------------|-------------|
| turbo4 | 4.25 | 3.8× | +0.23% |
| turbo3 | 3.50 | 4.6× | +1.06% |
| turbo2 | 2.50 | 6.4× | +6.48% |

## Architecture

The two-stage TurboQuant algorithm (Algorithm 2 from the paper):

1. **PolarQuant** (b−1 bits) — random rotation (Haar-distributed or fast Walsh-Hadamard) followed by optimal scalar quantization using Lloyd's algorithm on the post-rotation Gaussian distribution.
2. **QJL** (1 bit) — Quantized Johnson-Lindenstrauss sign projection on the residual to eliminate inner-product bias.

For V cache, only the MSE-optimal PolarQuant stage is used (Algorithm 1), since value reconstruction prioritises MSE over inner-product preservation.

## Modules

| Module | Description |
|--------|-------------|
| `rotation` | Haar-distributed rotation via QR, fast Walsh-Hadamard transform |
| `codebook` | Optimal MSE centroids — closed-form (1–2 bit) and Lloyd's algorithm (3+ bit) |
| `polar_quant` | PolarQuant quantizer with norm extraction and correction |
| `qjl` | QJL 1-bit sign quantizer preserving inner products |
| `turboquant` | Combined TurboQuant and MSE-only variant |
| `kv_cache` | KV cache compressor — TurboQuant for K, PolarQuant-MSE for V |
| `outlier` | Non-integer bit rates via outlier/normal channel splitting |
| `utils` | Bit packing, index packing, memory footprint calculation |

## Usage

```rust
use turboquant::{TurboQuant, KVCacheCompressor};
use ndarray::Array1;

// Quantize a single vector at 3 bits
let tq = TurboQuant::new(128, 3, 42, true);
let x = Array1::from_shape_fn(128, |i| (i as f64) / 128.0);
let compressed = tq.quantize(&x);
let x_hat = tq.dequantize(&compressed);

// KV cache memory savings
let compressor = KVCacheCompressor::new(128, 3, 3, 42, true);
let stats = compressor.memory_stats(1024, 32, 32);
println!("Compression ratio: {:.1}×", stats.compression_ratio);
```

## Building and testing

```bash
cargo build
cargo test
```

## References

- **TurboQuant**: Ankur Moitra, Mark Sellke, *TurboQuant: Online Vector Quantization for Efficient KV Cache Compression*, ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: Ankur Moitra, Mark Sellke, *PolarQuant: Provably Good Quantization via Polar Coordinates*, AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **Reference implementation**: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) (Python)

## License

MIT
