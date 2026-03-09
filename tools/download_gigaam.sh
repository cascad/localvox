#!/bin/bash
# Download GigaAM v3 CTC ONNX model (sherpa-onnx format) for Russian ASR
# Source: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16

set -e
BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/models/gigaam-v3-ctc-ru"

mkdir -p "$OUT_DIR"

for name in model.int8.onnx tokens.txt; do
  path="$OUT_DIR/$name"
  if [ -f "$path" ]; then
    echo "Already exists: $path"
  else
    echo "Downloading $name..."
    curl -L -o "$path" "$BASE_URL/$name"
    echo "Saved: $path"
  fi
done

echo "Done. Model dir: $OUT_DIR"
