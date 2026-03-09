# live-transcribe-server-reliable (Rust)

Сервер надёжной дисковой транскрипции на Rust. Тот же протокол и поведение, что у Python `reliable-transcribe`: аудио пишется на диск по сегментам (WAV + `.meta.json`), обрабатывается очередью через GGML Whisper, текст сливается с учётом перекрытий, клиенту отправляются `transcript` и `status` (включая `lag_sec`, `audio_dir_size_mb`).

## Модель

Используется **whisper.cpp** через крейт `whisper-rs`. Модель — один GGML `.bin` файл (например `ggml-large-v3-turbo.bin`).

## Сборка и запуск

```bash
cargo build -p live-transcribe-server-reliable --release
```

Настройки в `settings.json` рядом с бинарником или в текущей директории (см. `config::load_settings`). Обязательно задать `model_path` — путь к `.bin` модели.

```bash
# из корня репозитория, с указанием модели
cargo run -p live-transcribe-server-reliable --release -- --model "F:\project_models\ggml_whisper\ggml-large-v3-turbo.bin"

# или положить model_path в server-reliable/settings.json и запустить
cargo run -p live-transcribe-server-reliable --release -- --host 0.0.0.0 --port 9745
```

## Протокол

Совместим с клиентом `live-transcribe-client-reliable` и Python `reliable-transcribe`:

- **Клиент → сервер**: бинарные сообщения `[source_id: u8][PCM 16 kHz mono i16]`; JSON `{"type":"recording","enabled":bool}`, `{"type":"config","source_count":1|2}`.
- **Сервер → клиент**: JSON `{"type":"transcript","text":"...","source":0|1}`, `{"type":"status", ...}` (в т.ч. `lag_sec`, `lag_sec_0`, `lag_sec_1`, `audio_dir_size_mb`).

Аудио пишется в `audio/session_YYYYMMDD_HHMMSS/` (WAV + `.meta.json`), после обработки файлы удаляются.
