---
name: maintain-embedding-gemma
description: Maintain or optimize Sift's specialized EmbeddingGemma implementation while protecting independent correctness, backend-specific bit exactness, and measured performance. Use when changing src/embeddings, crates/sift-embedding-runtime, crates/sift-embedding-kernels, embedding tokenization or batching, CPU/CUDA execution, numerical output, embedding fixtures, or embedding benchmarks.
---

# Maintain EmbeddingGemma

Preserve the distinction between portable model correctness, machine-specific
numerical stability, and performance. Let the implementation and fixture
documentation remain the source of truth; do not duplicate them into this
skill.

## Orient the change

Inspect the relevant source and current diff before editing:

- `src/embeddings/embedding_gemma.rs` owns model loading and execution.
- `src/embeddings/worker.rs` owns the async boundary and warm model worker.
- `crates/sift-embedding-runtime` is a private tensor runtime specialized for
  Sift's model, not a general public framework.
- `crates/sift-embedding-kernels` owns the CUDA kernels and their build.

Read `tests/fixtures/README.md` before changing model semantics, fixture
generation, tolerances, or the pinned model revision. Read
`tests/fixtures/exact/README.md` before optimizing, benchmarking, investigating
an exactness failure, or replacing an exact fixture.

## Select the evidence level

Always run formatting and focused unit tests for touched code:

```console
cargo fmt --all -- --check
cargo clippy --workspace --exclude sift-embedding-kernels --all-targets -- -D warnings
cargo test --workspace --exclude sift-embedding-kernels --lib
```

The CUDA kernel crate requires a toolkit even for direct workspace checks; the
production Docker build is its compile gate.

Run `test_embedding_gemma` in release mode when a change can affect
tokenization, prompts, truncation, batching, tensor operations, model output,
or either backend. This is the authoritative comparison against independently
generated Python goldens:

```console
cargo test --release --test test_embedding_gemma -- --nocapture
cargo test --release --features cuda --test test_embedding_gemma -- --nocapture
```

The CUDA command exercises both CPU and CUDA. Run it only when CUDA is relevant
and a working NVIDIA device and toolkit are available.

For an optimization intended to preserve behavior:

1. Pass the independent golden test.
2. Run the relevant ignored bit-exact test using the command in
   `tests/fixtures/exact/README.md`.
3. Benchmark only after exactness reports `EXACT=PASS`.
4. Compare before and after using the same build mode, backend, hardware, input,
   and batch sizes.

Treat exact fixtures as backend- and environment-specific. A mismatch after a
compiler, CPU, GPU, CUDA, or cuBLAS change requires investigation, but portable
correctness is decided by the independent golden test.

## Protect the baselines

- Never set `SIFT_BLESS_EXACT=1` merely to make a candidate pass.
- Requalify an intentional numerical change against the independent goldens
  before replacing an exact fixture, then update its recorded provenance and
  SHA-256.
- Regenerate the independent fixture only for an intentional reference-model
  update, using `scripts/generate_embeddinggemma_fixtures.py` and the procedure
  in `tests/fixtures/README.md`.
- Keep query and document paths distinct; they use different task prefixes.
- Preserve `LICENSE-CANDLE-MIT` in both derived internal crates. Preserve the
  applicable MIT notice if more upstream-derived code is introduced.

## Keep the implementation narrow

Add only the tensor operations and kernels EmbeddingGemma needs. Prefer
explicit model shapes and private operations over expanding the runtime into a
general tensor API. Preserve batched execution unless evidence justifies a
change, and measure rather than infer performance.

Report which validation layers passed, which were skipped because the required
model or hardware was unavailable, and any environment mismatch that limits an
exactness conclusion.
