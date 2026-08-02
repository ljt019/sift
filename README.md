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

| Batch size | CPU | RTX 3060 |
| ---: | ---: | ---: |
| 1 | 90.5 ms | 11.8 ms |
| 2 | 107.9 ms | 12.9 ms |
| 4 | 126.8 ms | 15.8 ms |
| 8 | 234.3 ms | 35.7 ms |
| 16 | 373.8 ms | 62.1 ms |
| 32 | 675.1 ms | 114.1 ms |

Measured with Rust 1.95.0 and CUDA 12.9.1. Search latency also includes SearXNG, page fetching, extraction, and tokenization.

## License

[MIT](LICENSE)
