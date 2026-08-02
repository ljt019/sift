# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==1.6.0",
#   "numpy==2.5.1",
#   "safetensors==0.8.0",
#   "sentence-transformers==5.4.1",
#   "tokenizers==0.22.2",
#   "torch==2.10.0",
#   "transformers==5.14.1",
# ]
# ///

"""Generate Sift's independent EmbeddingGemma correctness fixtures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-300m"
DEFAULT_OUTPUT = Path("tests/fixtures/embedding_gemma_goldens.json")

QUERIES = [
    ("rust_json", "how do I parse json in rust"),
    ("capital_of_france", "capital of france"),
]

DOCUMENTS = [
    ("serde_json", "serde_json is the standard crate for JSON in Rust."),
    ("paris", "Paris is the capital and largest city of France."),
    ("unrelated", "The quick brown fox jumps over the lazy dog."),
    ("empty", ""),
    ("minimal", "a"),
    ("over_context_window", "word " * 3000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"fixture destination (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def resolve_revision(token: str) -> str:
    info = HfApi().model_info(MODEL_ID, revision="main", token=token)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {MODEL_ID}")
    return info.sha


def token_count(model: SentenceTransformer, text: str, prompt_name: str) -> int:
    prompt = model.prompts[prompt_name]
    encoded = model.tokenizer(
        prompt + text,
        add_special_tokens=True,
        truncation=False,
    )
    return len(encoded["input_ids"])


def encode(
    model: SentenceTransformer,
    cases: list[tuple[str, str]],
    prompt_name: str,
    dimensions: int,
) -> list[dict[str, Any]]:
    texts = [text for _, text in cases]
    embeddings = model.encode(
        texts,
        prompt_name=prompt_name,
        truncate_dim=dimensions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    fixtures = []
    for (name, text), embedding in zip(cases, embeddings, strict=True):
        count = token_count(model, text, prompt_name)
        fixtures.append(
            {
                "name": name,
                "text": text,
                "token_count_before_truncation": count,
                "was_truncated": count > model.max_seq_length,
                "embedding": embedding.astype("float32").tolist(),
            }
        )
    return fixtures


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required because google/embeddinggemma-300m is gated"
        )

    revision = resolve_revision(token)
    model = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        token=token,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )

    fixture = {
        "metadata": {
            "schema_version": 1,
            "model": MODEL_ID,
            "revision": revision,
            "device": "cpu",
            "dtype": "float32",
            "normalize_embeddings": True,
            "max_sequence_length": model.max_seq_length,
            "python": platform.python_version(),
            "packages": {
                "huggingface-hub": package_version("huggingface-hub"),
                "numpy": package_version("numpy"),
                "safetensors": package_version("safetensors"),
                "sentence-transformers": package_version("sentence-transformers"),
                "tokenizers": package_version("tokenizers"),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
            },
        },
        "queries_768": encode(model, QUERIES, "query", 768),
        "documents_768": encode(model, DOCUMENTS, "document", 768),
        "documents_128": encode(model, DOCUMENTS, "document", 128),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} from {MODEL_ID}@{revision}")


if __name__ == "__main__":
    main()
