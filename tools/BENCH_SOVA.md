# Бенчмарк SOVA

Утилита `bench_sova.py` прогоняет датасет SOVA через live-transcribe-server-reliable и сравнивает результат с эталоном (WER).

## Подготовка

1. Запусти сервер:
   ```bash
   cargo run -p live-transcribe-server-reliable --release -- --host 0.0.0.0 --port 9745
   ```

2. Установи зависимости:
   ```bash
   pip install websockets jiwer
   ```

3. Для HuggingFace датасета:
   ```bash
   pip install datasets
   ```

## Использование

### HuggingFace bond005/sova_rudevices

```bash
python tools/bench_sova.py --dataset hf:bond005/sova_rudevices --split test --limit 100 -v -o results.json
```

### Локальная папка с manifest

Создай `manifest.jsonl`:
```json
{"path": "audio/001.wav", "transcription": "привет мир"}
{"path": "audio/002.wav", "transcription": "как дела"}
```

```bash
python tools/bench_sova.py --dataset local:./my_sova --limit 50 -o results.json
```

### Папка с .wav и .txt

Структура:
```
wavs/
  001.wav
  001.txt   # эталонный текст
  002.wav
  002.txt
```

```bash
python tools/bench_sova.py --dataset dir:./wavs --limit 50
```

## Формат результатов

С `-o results.json` сохраняется:
- `avg_wer` — средний WER (взвешенный по числу слов в эталоне)
- `results` — массив с path, reference, hypothesis, wer по каждому файлу
