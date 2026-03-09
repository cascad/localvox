# live-transcribe-client-reliable

Копия клиента для использования с **reliable-transcribe** сервером. Отображает отставание распознавания и размер папки с аудио на сервере.

## Запуск

```bash
cargo run -p live-transcribe-client-reliable -- --device 0 --loopback Razer --output transcript.txt
```

Параметры те же, что у `live-transcribe-client`. В блоке статуса добавлена строка:

- **Отставание** — на сколько секунд распознавание отстаёт от записи (общее и по mic/sys).
- **Папка** — размер папки с аудио на сервере (MB).

Сервер должен быть reliable-transcribe (`reliable-transcribe/server.py`). Если сервер не присылает поля `lag_sec`, `audio_dir_size_mb`, в интерфейсе будет «—».
