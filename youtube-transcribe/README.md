# youtube-transcribe

TUI-клиент для очереди транскрипции YouTube через localvox. Durability: состояние сохраняется в JSON, перезапуск продолжает работу.

## Требования

- **yt-dlp** — по умолчанию ищет `yt-dlp.exe` в текущей папке, затем рядом с exe. Либо укажите путь в `settings.json`
- **ffmpeg** — в PATH или в settings
- **localvox сервер** — должен быть запущен

## Использование

```bash
# Запустить TUI
cargo run -p youtube-transcribe

# Добавить URL при старте
cargo run -p youtube-transcribe -- "https://youtube.com/watch?v=..."

# Несколько URL
cargo run -p youtube-transcribe -- "https://..." "https://..."

# Папка для транскриптов
cargo run -p youtube-transcribe -- --output-dir ./my_transcripts
```

## TUI

- **a** — добавить URL (ввод в нижней панели)
- **d** — удалить выбранный (только pending)
- **j/k** или стрелки — навигация
- **q** — выход

## Настройки

Файл `settings.json` (в текущей папке или рядом с exe):

```json
{
  "yt_dlp_path": "yt-dlp.exe",
  "ffmpeg_path": "ffmpeg",
  "server": "ws://127.0.0.1:9745",
  "output_dir": "transcripts",
  "state_file": "queue.json"
}
```

По умолчанию `yt_dlp_path` ищет в текущей папке, затем рядом с exe.

## Durability

Состояние сохраняется в `queue.json` (рядом с exe или в текущей папке). При перезапуске:
- `in_progress` → `pending` (retry)
- Очередь обрабатывается с начала

Транскрипты сохраняются в `transcripts/{uuid}.txt` (по умолчанию).
