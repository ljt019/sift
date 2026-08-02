# EmbeddingGemma correctness fixtures

`embedding_gemma_goldens.json` is generated independently for Sift from the
canonical `google/embeddinggemma-300m` model through Sentence Transformers. It
contains model outputs only.

The fixture records the exact model commit, Python package versions, device,
dtype, normalization setting, and maximum sequence length used to generate it.
Inputs cover query and document prompts, batch inputs, 768- and 128-dimensional
outputs, empty and minimal documents, unrelated content, and an input longer
than the model context window.

## Current golden

- Model revision: `57c266a740f537b4dc058e1b0cda161fd15afa75`
- Fixture SHA-256: `b90539603e85c95a2713c493067b275bbd440b7405c7bbaf42d7f307c93103bd`

Two consecutive generations with the pinned environment produced this same
fixture hash.

The Rust test requires cosine similarity above `0.9999995`. CPU and CUDA use a
maximum absolute element error below `3e-7` and root-mean-square error below
`1e-7`; both remain inside those bounds with observed maximum absolute error
below `2.74e-7` and RMSE below `7.87e-8` on the capture host. The intentionally
truncated 2,000-token stress case has separate portable bounds of `2e-6` and
`1e-6` to account for accumulated floating-point drift across hardware and its
second normalization at 128 dimensions. ROCm uses bounds of `1e-6` and `2e-7`,
respectively, for ordinary inputs because of its different GEMM and
transcendental implementations. On gfx1151 with
ROCm 7.2, the hipBLAS and HIPRTC-matmul paths both passed, with combined worst
cases of `5.97e-7` maximum absolute error and `1.71e-7` RMSE.

Backend-specific raw output references for optimization work are documented
under `exact/`. They preserve Sift's established bits but are not substitutes
for this independent correctness fixture.

## Regeneration

Access to the gated model and an accepted Gemma license are required. Set a
Hugging Face token that has access, then run the generator from the repository
root:

```console
HF_TOKEN=... uv run scripts/generate_embeddinggemma_fixtures.py
```

The generator resolves `main` to an immutable model commit before loading the
model and writes that commit into the fixture. Its PEP 723 metadata pins the
Python dependencies used for generation.

Review the recorded revision and package versions whenever regenerating the
file. A changed fixture should be treated as an intentional golden update, not
as routine formatting.
