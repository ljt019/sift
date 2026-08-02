# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.3.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

ARG RUST_VERSION=1.95.0

ENV CUDA_ROOT=/usr/local/cuda \
    PATH=/root/.cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
        https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"

WORKDIR /build
COPY . .

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/build/target \
    cargo build --locked --release --features cuda --bin sift \
    && cp target/release/sift /usr/local/bin/sift

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 sift \
    && useradd --uid 10001 --gid sift --create-home --shell /usr/sbin/nologin sift \
    && install --directory --owner sift --group sift /var/cache/sift/huggingface

COPY --from=builder /usr/local/bin/sift /usr/local/bin/sift
COPY --from=builder /build/LICENSE /usr/share/doc/sift/LICENSE
COPY --from=builder /build/crates/sift-embedding-runtime/LICENSE-CANDLE-MIT /usr/share/doc/sift/LICENSE-CANDLE-MIT

ENV HF_HOME=/var/cache/sift/huggingface

WORKDIR /app
USER sift

ENTRYPOINT ["/usr/local/bin/sift"]
