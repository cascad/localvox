# syntax=docker/dockerfile:1
# Multi-stage: build Rust server with CUDA, then minimal runtime
# Модели и GPU прокидываются с хоста через volumes/deploy
#
# Стратегия кэширования (Docker layer cache, без BuildKit cache mounts):
#   1. deps   — системные пакеты + Rust + cargo build с dummy main.rs
#              → все зависимости (sherpa-rs, whisper-rs, ort, rustls …) компилируются здесь
#              → слой кэшируется пока Cargo.toml / Cargo.lock не меняются
#   2. builder — COPY реального кода + cargo build → пересобирается только наш крейт
#   3. runtime — минимальный образ с бинарником
#
# При добавлении/обновлении зависимостей deps пересоберётся (10–20 мин),
# при изменении только кода — быстрая пересборка (~30 сек).

# ── Stage 1: deps — toolchain + все зависимости ──────────────────────────
# CUDA 11.8: ort/ONNX Runtime требует libcublasLt.so.11 (CUDA 11)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive

# apt не через HTTP(S)_PROXY: на Windows часто пробрасывается 127.0.0.1:1080x — в билде это не ваш прокси.
RUN printf '%s\n' \
    'Acquire::http::Proxy "false";' \
    'Acquire::https::Proxy "false";' \
    > /etc/apt/apt.conf.d/99docker-direct && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Сброс прокси на время скачивания rustup/crates (та же проблема, что и у apt).
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Копируем ТОЛЬКО манифесты — изменение исходников не инвалидирует этот слой
COPY Cargo.toml.docker ./Cargo.toml
COPY Cargo.lock ./
COPY server-reliable/Cargo.toml ./server-reliable/Cargo.toml

# Заглушка: cargo build скомпилирует все зависимости, но не наш код
RUN mkdir -p server-reliable/src && echo 'fn main() {}' > server-reliable/src/main.rs
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
    cargo build -p live-transcribe-server-reliable --release

# Убираем fingerprint нашего крейта — при COPY реального кода cargo пересоберёт только его
RUN rm -rf target/release/.fingerprint/live-transcribe-server-reliable-* \
    target/release/deps/live_transcribe_server_reliable-* \
    target/release/live-transcribe-server-reliable

# ── Stage 2: builder — компиляция нашего кода (быстро) ───────────────────
FROM deps AS builder

# Копируем реальные исходники (инвалидирует только этот слой)
COPY server-reliable ./server-reliable

RUN env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
    cargo build -p live-transcribe-server-reliable --release && \
    mkdir -p /build/out /build/sherpa-libs && \
    cp target/release/live-transcribe-server-reliable /build/out/ && \
    (cp -n target/release/*.so /build/sherpa-libs/ 2>/dev/null || true) && \
    (cp -n target/release/deps/*.so /build/sherpa-libs/ 2>/dev/null || true)

# builder-shell: интерактивная отладка (cargo build — вручную)
FROM deps AS builder-shell
COPY server-reliable ./server-reliable
CMD ["/bin/bash"]

# ── Stage 3: Runtime ────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN printf '%s\n' \
    'Acquire::http::Proxy "false";' \
    'Acquire::https::Proxy "false";' \
    > /etc/apt/apt.conf.d/99docker-direct && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/out/live-transcribe-server-reliable /app/
COPY --from=builder /build/sherpa-libs/ /app/

COPY server-reliable/settings.docker.example.json /app/settings.json

RUN mkdir -p /app/audio

ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

EXPOSE 9745

CMD ["./live-transcribe-server-reliable", "--host", "0.0.0.0", "--port", "9745"]
