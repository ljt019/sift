# Sift EmbeddingGemma bit-exact references

These fixtures freeze Sift's established numerical behavior for fast
optimization work. They complement rather than replace
`embedding_gemma_goldens.json`:

- the Sentence Transformers fixture checks Sift against an independent model
  implementation with tight numerical error bounds;
- these raw fixtures require every Sift output bit to remain unchanged.

Do not bless a changed exact fixture merely to make a candidate pass. A
non-exact candidate must first be justified and requalified against the
independent Python goldens.

## Captured output

Each file is a headerless concatenation of row-major, little-endian `f32`
values produced with model revision
`57c266a740f537b4dc058e1b0cda161fd15afa75`, in this order:

1. one document at 768 dimensions;
2. one mixed-length batch of eight documents at 768 dimensions;
3. the same batch at 128 dimensions.
4. two retrieval queries at 768 dimensions.

That is 9,472 values and 37,888 bytes per backend.

| fixture | SHA-256 |
|---|---|
| `embedding_gemma_cpu_zen5_f32.bin` | `7af0cbdff9549a87810150b19f70b1868f7636d7213115250b6e776d35125de8` |
| `embedding_gemma_cuda_sm86_cuda12.9_f32.bin` | `59261aceecd6f42a2d28fb8d28b3a1c35b93a1130fffe192212309ed3b1cba37` |

Both fixtures reproduced identically in two independent runs immediately
after capture.

## Capture scope

- Rust 1.95.0 (`59807616e1fa2540724bfbac14d7976d7e4a3860`), LLVM 21.1.8
- CPU: AMD Ryzen 9 9950X3D, 16 cores / 32 threads, x86-64
- GPU: NVIDIA GeForce RTX 3060, compute capability 8.6
- CUDA toolkit 12.9.86
- NVIDIA driver 595.84

Exact floating-point behavior can depend on hardware, compiler, CUDA, and
cuBLAS. A mismatch after changing that environment is evidence requiring
review, not automatically a model regression.

## Quick iteration

Run exactness before benchmarking:

```console
cargo test --release --test test_embedding_gemma_exact \
  embedding_gemma_cpu_is_bit_exact -- --ignored --nocapture
cargo test --release --features cuda \
  --test test_embedding_gemma_exact embedding_gemma_cuda_is_bit_exact \
  -- --ignored --nocapture
```

On failure, the harness reports the scenario, document, dimension, expected
and actual bit patterns, and both floating-point values. Only benchmark an
optimization after the relevant exactness command reports `EXACT=PASS`.

The matching performance harness is:

```console
cargo bench --bench embedding_gemma -- cpu
cargo bench --features cuda --bench embedding_gemma -- cuda
```

## Intentional regeneration

First pass the independent Python golden test and review why the established
numerical behavior should change. Regeneration then requires both an explicit
flag and opt-in environment variable:

```console
SIFT_BLESS_EXACT=1 cargo test --release --test test_embedding_gemma_exact \
  embedding_gemma_cpu_is_bit_exact -- --ignored --nocapture
SIFT_BLESS_EXACT=1 cargo test --release --features cuda \
  --test test_embedding_gemma_exact embedding_gemma_cuda_is_bit_exact \
  -- --ignored --nocapture
```

Update this provenance and the recorded SHA-256 whenever either fixture is
intentionally replaced.
