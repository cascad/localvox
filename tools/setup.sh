#!/bin/bash
# LocalVox: развёртывание на машине — nvidia-container-toolkit, модели, Ollama, docker compose
# Запуск: ./tools/setup.sh  (из корня репо) или bash tools/setup.sh
#
# Требования: Docker, NVIDIA GPU + драйверы. Ollama — опционально.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="$REPO_ROOT/models"
WHISPER_DIR="$MODELS_DIR/whisper"
GIGAAM_DIR="$MODELS_DIR/gigaam-v3-ctc-ru"
PARAKEET_DIR="$MODELS_DIR/parakeet-tdt-0.6b-v3-int8"

echo "=== LocalVox setup ==="
echo "Repo root: $REPO_ROOT"
echo ""

# ── 0. nvidia-container-toolkit (Ubuntu/Debian) ─────────────────────────────
install_nvidia_container_toolkit() {
  if command -v nvidia-ctk &>/dev/null || dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
    echo "[OK] nvidia-container-toolkit уже установлен"
    return 0
  fi
  if [ -n "${SKIP_NVIDIA_TOOLKIT:-}" ]; then
    echo "[skip] nvidia-container-toolkit пропущен (SKIP_NVIDIA_TOOLKIT=1)"
    return 0
  fi
  if ! command -v apt-get &>/dev/null; then
    echo "[!] apt-get не найден. Установите nvidia-container-toolkit вручную: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    return 1
  fi
  echo "Установка nvidia-container-toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
  sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null || true
  echo "[OK] nvidia-container-toolkit установлен"
}

if nvidia-smi &>/dev/null; then
  install_nvidia_container_toolkit || true
else
  echo "[!] nvidia-smi не найден. GPU не обнаружен или драйверы не установлены."
  echo "    Для CPU-режима можно продолжить (сервер будет медленным)."
  if [ -t 0 ]; then
    read -p "Продолжить? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
  fi
fi

echo ""

# ── 1. Модели ASR ────────────────────────────────────────────────────────
mkdir -p "$MODELS_DIR"

# Whisper (обязательна)
WHISPER_FILE="$WHISPER_DIR/ggml-large-v3-turbo.bin"
if [ -f "$WHISPER_FILE" ]; then
  echo "[OK] Whisper уже есть: $WHISPER_FILE"
else
  echo "[1/3] Скачивание Whisper (~1.5 GB)..."
  mkdir -p "$WHISPER_DIR"
  curl -L -o "$WHISPER_FILE" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
  echo "[OK] Whisper: $WHISPER_FILE"
fi

# GigaAM (опционально)
if [ -f "$GIGAAM_DIR/model.int8.onnx" ]; then
  echo "[OK] GigaAM уже есть: $GIGAAM_DIR"
else
  echo "[2/3] Скачивание GigaAM..."
  mkdir -p "$GIGAAM_DIR"
  BASE="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main"
  for f in model.int8.onnx tokens.txt; do
    [ -f "$GIGAAM_DIR/$f" ] || curl -L -o "$GIGAAM_DIR/$f" "$BASE/$f"
  done
  echo "[OK] GigaAM: $GIGAAM_DIR"
fi

# Parakeet (опционально, для мультиязычности; не используется в settings.docker.json по умолчанию)
if [ -n "${SKIP_PARAKEET:-}" ]; then
  echo "[skip] Parakeet пропущен (SKIP_PARAKEET=1)"
elif [ -f "$PARAKEET_DIR/encoder.int8.onnx" ]; then
  echo "[OK] Parakeet уже есть: $PARAKEET_DIR"
else
  echo "[3/3] Скачивание Parakeet (~640 MB)..."
  mkdir -p "$PARAKEET_DIR"
  ARCHIVE="/tmp/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
  curl -L -o "$ARCHIVE" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
  tar -xjf "$ARCHIVE" -C "$PARAKEET_DIR" --strip-components=1
  rm -f "$ARCHIVE"
  echo "[OK] Parakeet: $PARAKEET_DIR"
fi

echo ""

# ── 2. Ollama (LLM-корректор и суммаризация) ──────────────────────────────
if command -v ollama &>/dev/null; then
  echo "Ollama найден. Загрузка моделей..."
  ollama pull qwen3.5:4b  2>/dev/null || true  # LLM-корректор (server)
  ollama pull qwen3.5:9b  2>/dev/null || true  # Суммаризация (client)
  echo "[OK] Ollama модели готовы"
else
  echo "[!] Ollama не установлен. Для LLM-корректора и суммаризации: https://ollama.com"
  echo "    Без Ollama сервер будет работать, но llm_correction_enabled лучше выключить."
fi

echo ""

# ── 3. Docker Compose ─────────────────────────────────────────────────────
echo "Запуск docker compose..."
docker compose pull
if [ -z "${SKIP_DOCKER_UP:-}" ]; then
  docker compose up -d
else
  echo "[skip] docker compose up пропущен (SKIP_DOCKER_UP=1)"
fi

echo ""
echo "=== Готово ==="
echo "Сервер: ws://localhost:9745"
echo "Клиент: cargo run -p live-transcribe-client-reliable -- --loopback default-output"
echo ""
echo "Остановить: docker compose down"
