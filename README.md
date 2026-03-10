# LocalVox

Сервер потоковой транскрипции речи с ансамблем Whisper + GigaAM и LLM-корректором. TUI-клиент с микрофоном и loopback системного звука. Для локальной распознавания речи в реальном времени.

## Quickstart

**1. Скачать модели**

Пути относительно корня репозитория (откуда запускаете `cargo`):

- **Whisper:** создать папку `models/whisper/`, скачать [ggml-large-v3-turbo.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin) и положить туда. Итог: `models/whisper/ggml-large-v3-turbo.bin`
- **GigaAM** (опционально): создать папку `models/gigaam-v3-ctc-ru/`, скачать [model.int8.onnx](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main/model.int8.onnx) и [tokens.txt](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main/tokens.txt) туда
- **Ollama** (опционально, для LLM-корректора): установить [Ollama](https://ollama.com), затем `ollama pull qwen2.5:7b-instruct`. Включить в `settings.json`: `llm_correction_enabled: true`

**2. Настроить сервер**

```bash
cp server-reliable/settings.example.json server-reliable/settings.json
# Отредактировать model_path и при необходимости cuda_path
```

**3. Запуск**

```bash
# Терминал 1
cargo run -p live-transcribe-server-reliable --release -- --host 0.0.0.0 --port 9745

# Терминал 2
cargo run -p live-transcribe-client-reliable -- --device 0 --loopback Razer --output transcript.txt
```

`--loopback Razer` — захват системного звука. Без него — только микрофон. F2 — настройки в TUI.

**Loopback:** `--loopback default-output` — дефолтный вывод системы (работает на Windows и macOS 14.2+). Или имя/индекс устройства.

## Возможности

### Сервер: конвейер распознавания

Сервер обрабатывает аудио по цепочке. Каждый этап можно включить или выключить в `settings.json`:

1. **Whisper (GGML)** — базовая модель, всегда работает. whisper.cpp, CUDA/CPU.

2. **GigaAM (NeMo CTC)** — вторая модель через sherpa-onnx. Опциональна: задайте `gigaam_model_dir` и `ensemble_enabled: true`. Запускается параллельно с Whisper.

3. **Ансамбль** — при `ensemble_enabled` Whisper и GigaAM запускаются параллельно. Алгоритмический мерж (word-level alignment) даёт промежуточный результат, но основная арбитрация — в LLM. Если GigaAM выключен или пустой — идёт только Whisper. Переключатель: `ensemble_enabled`.

4. **LLM-корректор** — постобработка через Ollama. Получает все три варианта (Whisper, GigaAM, algorithmic merged) и выбирает/объединяет лучший, исправляет ошибки. Именно здесь происходит реальный выбор между моделями. Переключатель: `llm_correction_enabled`. Требует запущенный Ollama.

**Режимы работы:**
- Только Whisper: `ensemble_enabled: false`
- Whisper + GigaAM: `ensemble_enabled: true`, `gigaam_model_dir` задан
- С корректором: `llm_correction_enabled: true`

### Остальное

- **Reliable-буфер** — дисковая очередь аудио, VAD-сегментация, overlap-мерж
- **TUI-клиент** — ratatui, микрофон + loopback системного звука, запись в файл

## Бенчмарк SOVA

Сравнение с эталоном на датасете [bond005/sova_rudevices](https://huggingface.co/datasets/bond005/sova_rudevices):

```bash
# Создать venv и установить зависимости
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

pip install websockets jiwer datasets

# Запустить сервер (в отдельном терминале)
cargo run -p live-transcribe-server-reliable --release -- --host 0.0.0.0 --port 9745

# Запустить бенчмарк
python tools/bench_sova.py --dataset hf:bond005/sova_rudevices --split test --limit 100 -v -o tools/sova_results.json
```

По умолчанию подключается к `ws://localhost:9745`. Другой сервер: `--server ws://host:port`.

Другие источники датасета:
```bash
# Локальная папка с manifest.json (path, transcription)
python tools/bench_sova.py --dataset local:./sova_test --limit 50

# Папка с .wav и .txt с одинаковыми именами
python tools/bench_sova.py --dataset dir:./wavs --limit 50
```
