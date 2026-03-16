# LocalVox

**Клиент-серверная система потоковой транскрипции речи** — не совсем realtime, но с задержкой в несколько секунд. Сервер: ансамбль Whisper + GigaAM и LLM-корректор (Ollama). TUI-клиент: микрофон + loopback системного звука, запись в файл, экспорт и суммаризация.

## Quickstart

### Сервер

```bash
# Из корня репо (Linux/macOS)
chmod +x tools/setup.sh
./tools/setup.sh
```

Скрипт скачает модели, установит nvidia-container-toolkit и запустит сервер через Docker Compose. После — сервер работает на `ws://localhost:9745`.

На Windows: WSL или Git Bash (`bash tools/setup.sh`).

### Клиент

Скачать бинарник из [Releases](https://github.com/cascad/localvox/releases) и запустить. При первом запуске автоматически создаётся `client-config.json` с дефолтами — если сервер на той же машине, работает из коробки. Если на другой — поменять `server` в конфиге.

Или из исходников:

```bash
cargo run -p live-transcribe-client-reliable
```

---

## Подробнее

### Архитектура

- **Сервер** — WebSocket, принимает аудио, распознаёт через Whisper/GigaAM, опционально корректирует через LLM
- **Клиент** — TUI (ratatui), захватывает микрофон и/или системный звук, отправляет на сервер, отображает транскрипцию

### Развёртывание сервера (Docker)

`tools/setup.sh` делает всё за вас:
1. Устанавливает nvidia-container-toolkit (Ubuntu/Debian) — пропуск: `SKIP_NVIDIA_TOOLKIT=1`
2. Скачивает Whisper, GigaAM, Parakeet в `models/` — Parakeet: `SKIP_PARAKEET=1`
3. Ollama: `ollama pull qwen3.5:4b` + `qwen3.5:9b`
4. `docker compose pull && docker compose up -d`

**Требования:** Docker, NVIDIA GPU + драйверы. На Ubuntu/Debian скрипт сам поставит toolkit.

Управление: `docker compose down` / `docker compose up -d`.

### Развёртывание сервера (Cargo)

**1. Скачать модели**

В `settings.json` задаётся массив `models` — можно использовать одну, две или все три.

| Модель | Описание | Скачать |
|--------|----------|---------|
| **Whisper** | GGML через whisper.cpp. Обязательна. CUDA/CPU. | Папка `models/whisper/`, файл [ggml-large-v3-turbo.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin) |
| **GigaAM** | NeMo CTC (sherpa-onnx), русский. Опционально. | Папка `models/gigaam-v3-ctc-ru/`: [model.int8.onnx](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main/model.int8.onnx), [tokens.txt](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main/tokens.txt). Или `.\tools\download_gigaam.ps1` |
| **Parakeet** | NeMo TDT 0.6B (sherpa-onnx), 25 языков. Опционально. | `.\tools\download_parakeet.ps1` или [архив](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2) → `models/parakeet-tdt-0.6b-v3-int8/` |

**Ollama** (опционально): установить [Ollama](https://ollama.com), `ollama pull qwen3.5:4b`. В `settings.json`: `llm_correction_enabled: true`.

**2. Настроить и запустить**

```bash
cp server-reliable/settings.example.json server-reliable/settings.json
# Отредактировать models[].model_path под свои пути

cargo run -p live-transcribe-server-reliable --release -- --host 0.0.0.0 --port 9745
```

Пример `models` в settings.json:
```json
"models": [
  { "type": "whisper", "model_path": "models/whisper/ggml-large-v3-turbo.bin", "use_gpu": true },
  { "type": "gigaam", "model_path": "models/gigaam-v3-ctc-ru", "use_gpu": true },
  { "type": "parakeet", "model_path": "models/parakeet-tdt-0.6b-v3-int8", "use_gpu": true }
]
```

### Клиент (TUI)

- Микрофон + loopback системного звука (Windows: WASAPI, macOS: BlackHole и т.п.)
- Автогенерация `client-config.json` при первом запуске (`device: "default"`, `loopback: "default-output"`, `summarize_enabled: true`)
- **Клавиши:** `r` — запись, `x` — завершить сессию, `e` — экспорт + суммаризация, `F2` — настройки, `q`/Esc — выход
- Прокрутка: ↑↓ / jk / Tab, Home/End

`--loopback default-output` — дефолтный вывод системы. `--loopback Razer` — по имени устройства. Без `--loopback` — только микрофон.

### Конвейер распознавания

Каждый этап настраивается в `settings.json`:

1. **ASR-модели** — Whisper (обязательна), GigaAM и/или Parakeet (опционально). При нескольких моделях запускаются параллельно.
2. **Ансамбль** — word-level alignment (Levenshtein), fallback при пустом результате. Per-model фильтр галлюцинаций.
3. **LLM-корректор** — Ollama, получает все варианты + merged, выбирает лучший. `llm_correction_enabled: true`.

**Режимы:** только Whisper | Whisper + GigaAM/Parakeet | с LLM-корректором.

### Прочее

- **Reliable-буфер** — дисковая очередь аудио, VAD-сегментация, overlap-мерж
- **CPU-only сборка** (CI, без GPU): `cargo build -p live-transcribe-server-reliable --no-default-features`

## Релизы

```bash
git tag v0.1.0
git push origin v0.1.0
```

После push тега `v*` GitHub Actions собирает:
- **Клиент** — Windows (x64), macOS (x64, arm64). [Releases](https://github.com/cascad/localvox/releases)
- **Сервер (Docker)** — `ghcr.io/cascad/localvox:v0.1.0` (GPU), `ghcr.io/cascad/localvox-cpu:v0.1.0` (CPU)

## Бенчмарк SOVA

```bash
python -m venv venv && source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install websockets jiwer datasets

python tools/bench_sova.py --dataset hf:bond005/sova_rudevices --split test --limit 100 -v -o tools/sova_results.json
```

Другие источники: `--dataset local:./folder` (manifest.jsonl) или `--dataset dir:./wavs` (.wav + .txt).
