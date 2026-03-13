# syntax=docker/dockerfile:1
# Multi-stage: build Rust server with CUDA, then minimal runtime
# Модели и GPU прокидываются с хоста через volumes/deploy
#
# Кэш sherpa-onnx: BuildKit cache mounts сохраняют target/ и cargo registry
# между сборками. При изменении только нашего кода sherpa-onnx не пересобирается.
# Сборка: DOCKER_BUILDKIT=1 docker build ...

# ── Stage 1: Builder ─────────────────────────────────────────────────────
# CUDA 11.8: ort/ONNX Runtime требует libcublasLt.so.11 (CUDA 11)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder-base

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_INIT_SKIP_PATH_CHECK=yes

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

COPY Cargo.toml.docker ./Cargo.toml
COPY Cargo.lock ./
COPY server-reliable ./server-reliable

# builder-shell: интерактивная отладка (cargo build — вручную)
FROM builder-base AS builder-shell
CMD ["/bin/bash"]

# builder: полная сборка с кэшем target/ и cargo registry
# Артефакты копируем в /build/out — cache mount не попадает в образ
FROM builder-base AS builder
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/build/target \
    cargo build -p live-transcribe-server-reliable --release && \
    mkdir -p /build/out /build/sherpa-libs && \
    cp /build/target/release/live-transcribe-server-reliable /build/out/ && \
    (cp -n /build/target/release/*.so /build/sherpa-libs/ 2>/dev/null || true) && \
    (cp -n /build/target/release/deps/*.so /build/sherpa-libs/ 2>/dev/null || true)

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary and sherpa-onnx shared libs from builder
COPY --from=builder /build/out/live-transcribe-server-reliable /app/
COPY --from=builder /build/sherpa-libs/ /app/

# Default settings (paths for mounted volumes)
COPY server-reliable/settings.docker.json /app/settings.json

# Audio buffer (writable)
RUN mkdir -p /app/audio

ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

EXPOSE 9745

CMD ["./live-transcribe-server-reliable", "--host", "0.0.0.0", "--port", "9745"]
