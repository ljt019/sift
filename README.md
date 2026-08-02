# Sift

Sift is a self-hosted web search API for agents. It searches through
SearXNG, fetches and extracts result pages as Markdown, and uses
EmbeddingGemma to return the most relevant passages within a requested context
budget.

Individual page failures fall back to the SearXNG snippet instead of failing
the entire search.

## Run with Docker

Copy the example env and set `HF_TOKEN`. The token must have access to
the gated `google/embeddinggemma-300m` model.

```console
cp .env.example .env
```

To use the bundled SearXNG, keep `COMPOSE_PROFILES=bundled`.
If SearXNG is already deployed elsewhere, remove `COMPOSE_PROFILES` and set
`SEARXNG_URL` to its address.

```console
docker compose up --build
```

Sift listens on port `8099`. The bundled SearXNG and Valkey remain private to
the Docker network.

The default image uses CUDA. For an AMD GPU, install ROCm on the Linux host and
start the ROCm image with the HIP devices passed through:

```console
docker compose -f compose.yml -f compose.rocm.yml up --build
```

The ROCm image defaults to `EMBEDDING_DEVICE=rocm`. Set `rocm:N` to select a
different visible GPU.

## Native GPU builds

CUDA and ROCm are opt-in Cargo features:

```console
EMBEDDING_DEVICE=cuda cargo run --release --features cuda
EMBEDDING_DEVICE=rocm cargo run --release --features rocm
```

ROCm builds require Linux plus the HIP runtime and HIPRTC. Sift uses hipBLAS
when it is installed and otherwise falls back to its bundled HIPRTC matrix
multiplication kernel. It looks in `SIFT_ROCM_LIB_DIR`, then `ROCM_PATH/lib`
and `HIP_PATH/lib`, followed by the system library search path. Kernels are
compiled for the selected GPU at runtime, so the binary does not bake in a
particular `gfx` architecture. Set `SIFT_ROCM_FORCE_HIPRTC_MATMUL=1` to test or
force the dependency-free matrix multiplication path.

## API

```console
curl http://localhost:8099/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "how to handle serde_json errors",
    "numResults": 8,
    "contextMaxCharacters": 24000
  }'
```

`numResults` defaults to 8 and may be between 1 and 10.
`contextMaxCharacters` defaults to 24,000 and may be between 1 and 100,000.

Health is available at `GET /health`.

## Performance

Warm EmbeddingGemma document embedding at 768 dimensions, using mixed-length
inputs and production batching. Rows are batch sizes; values are mean latency over five runs after warmup. Model loading is excluded.

| Batch size | CPU | RTX 3060 | Radeon 8060S hipBLAS | Radeon 8060S HIPRTC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 90.5 ms | 11.8 ms | 32.3 ms | 29.8 ms |
| 2 | 107.9 ms | 12.9 ms | 34.5 ms | 33.1 ms |
| 4 | 126.8 ms | 15.8 ms | 35.6 ms | 39.5 ms |
| 8 | 234.3 ms | 35.7 ms | 60.9 ms | 77.0 ms |
| 16 | 373.8 ms | 62.1 ms | 111.9 ms | 145.7 ms |
| 32 | 675.1 ms | 114.1 ms | 350.6 ms | 393.3 ms |

Measured with Rust 1.95.0, CUDA 12.9.1, and ROCm 7.2.3. Search latency
also includes SearXNG, page fetching, extraction, and tokenization.

## License

[MIT](LICENSE)
