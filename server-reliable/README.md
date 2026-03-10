# live-transcribe-server-reliable (Rust)

Сервер потоковой транскрипции речи. WebSocket, дисковая буферизация, ансамбль Whisper + GigaAM, LLM-корректор.

## Архитектура

```
                          ┌─────────────────────────────────────────────────┐
                          │                    Server                       │
  Client ──WebSocket──►   │  AudioWriter ──► WAV segments on disk           │
  (PCM 16kHz i16)         │       │                                         │
                          │       ▼                                         │
                          │  ┌──────────── ASR Dispatcher ───────────────┐  │
                          │  │  Global priority queue (live > youtube)   │  │
                          │  │                                           │  │
                          │  │  Worker 1 ─┐                              │  │
                          │  │  Worker 2 ─┤── Whisper + GigaAM (parallel)│  │
                          │  │  Worker N ─┘                              │  │
                          │  │       │                                   │  │
                          │  │       ▼                                   │  │
                          │  │  Finalize Queue (per-session, ordered)    │  │
                          │  │  seq 0 → seq 1 → seq 2 → ...             │  │
                          │  │       │                                   │  │
                          │  │       ▼                                   │  │
                          │  │  merge_overlap + prev_tail (sequential)   │  │
                          │  └───────┬───────────────────────────────────┘  │
                          │          ▼                                      │
                          │  ┌─ LLM Dispatcher (optional) ──────────────┐  │
                          │  │  Worker 1 ─┐                              │  │
                          │  │  Worker M ─┘── Ollama correction          │  │
                          │  └───────┬───────────────────────────────────┘  │
                          │          ▼                                      │
                          │  OutputSink ──► transcript.jsonl + WebSocket    │
                          └─────────────────────────────────────────────────┘
```

### Пайплайн обработки

1. **AudioWriter** — принимает PCM, нарезает на WAV-сегменты (VAD, `max_chunk_duration_sec`, `overlap_sec`), пишет на диск с `.meta.json` (start/end time).

2. **ASR Dispatcher** — глобальная очередь с приоритетами (live > youtube). N воркеров (`asr_workers`) берут задачи параллельно:
   - Whisper (whisper.cpp, CUDA/CPU) + GigaAM (sherpa-onnx) запускаются одновременно в отдельных потоках
   - Ensemble merge (word-level alignment) объединяет результаты

3. **Finalize Queue** — per-session per-source очередь с монотонным `seq_id`. ASR-инференс параллелен, но `merge_overlap` + `prev_tail` update + отправка клиенту выполняются строго по порядку. Это устраняет race condition при параллельной обработке без потери скорости.

4. **LLM Dispatcher** (опционально) — M воркеров (`llm_pool_size`) отправляют результат в Ollama для финальной коррекции. Получает все три варианта (Whisper, GigaAM, merged) + контекст предыдущих фраз.

5. **OutputSink** — dual-write: `transcript.jsonl` на диск (для replay при reconnect) + WebSocket клиенту.

### Ключевые решения

- **Параллельный ASR, последовательный merge**: ASR-инференс (3-5 сек) работает на всех воркерах параллельно. merge_overlap (~0 мс) выполняется строго по порядку через finalize queue. Результат: скорость N воркеров, 0 дублей.
- **Дисковый буфер**: аудио пишется на диск → сервер переживает перезапуск, клиент может переподключиться.
- **Per-session state**: `prev_tail`, `context_lines`, `transcribed_end` — на каждую сессию и source отдельно.

## Сборка и запуск

```bash
cargo build -p live-transcribe-server-reliable --release
```

Настройки в `settings.json` рядом с бинарником или в текущей директории. Обязательно: `model_path`.

```bash
cargo run -p live-transcribe-server-reliable --release -- --host 0.0.0.0 --port 9745
```

## Протокол

- **Клиент → сервер**: бинарные `[source_id: u8][PCM 16 kHz mono i16]`; JSON `{"type":"recording","enabled":bool}`, `{"type":"config","source_count":1|2}`, `{"type":"end_of_stream"}`
- **Сервер → клиент**: JSON `{"type":"transcript","text":"...","source":0|1,"start_sec":N,"end_sec":N}`, `{"type":"status",...}`, `{"type":"done"}`

Аудио: `audio/session_YYYYMMDD_HHMMSS/` (WAV + `.meta.json`), после обработки удаляется.
