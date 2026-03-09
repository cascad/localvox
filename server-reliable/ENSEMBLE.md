# Ансамбль Whisper + GigaAM

## Включение

1. Скачайте модель GigaAM:
   ```powershell
   powershell -ExecutionPolicy Bypass -File ..\tools\download_gigaam.ps1
   ```

2. В `settings.json`:
   - `gigaam_model_dir`: путь к папке с model.int8.onnx и tokens.txt (напр. `models/gigaam-v3-ctc-ru`)
   - `ensemble_enabled`: `true`

## Сравнение качества

1. **Только Whisper**: `ensemble_enabled: false`
2. **Ансамбль**: `ensemble_enabled: true`, `gigaam_model_dir` задан

Запустите сервер, подключите клиент, говорите в микрофон. Сравните transcript.txt при разных настройках.

## Merge-стратегия

Алгоритмический мерж: fallback (если один пустой — другой) + word-level alignment (Levenshtein) + при конфликте — выбор более длинного варианта. LLM-корректор получает все три варианта (Whisper, GigaAM, merged) и выбирает/объединяет лучший.

## Чёрный список галлюцинаций

`hallucination.rs` — отдельные списки для каждой модели (чтобы не выкидывать валидный контент другой):
- **filter_whisper**: DROP (редакторы субтитров, музыка, подписки) + STRIP («Субтитры создал X»)
- **filter_gigaam**: пока пусто — добавлять по мере обнаружения
