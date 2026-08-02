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

<--Performance Matrix Here-->

## License

[MIT](LICENSE)
