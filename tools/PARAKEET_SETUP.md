# Parakeet TDT 0.6B v3 — ASR для русского и 24 европейских языков

Модель **NVIDIA Parakeet TDT 0.6B v3** (sherpa-onnx) поддерживает **25 языков**, включая **русский**.

## Скачать модель

### Вариант 1: PowerShell (Windows)

```powershell
.\tools\download_parakeet.ps1
```

### Вариант 2: Вручную

1. Скачать архив:
   - **https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2**

2. Распаковать в `models/parakeet-tdt-0.6b-v3-int8/`:
   - `encoder.int8.onnx` (~622 MB)
   - `decoder.int8.onnx` (~12 MB)
   - `joiner.int8.onnx` (~6 MB)
   - `tokens.txt`

## Конфигурация

В `settings.json` добавить в массив `models`:

```json
{
  "models": [
    { "type": "whisper", "model_path": "models/whisper/ggml-large-v3-turbo.bin", "use_gpu": true },
    { "type": "gigaam", "model_path": "models/gigaam-v3-ctc-ru" },
    { "type": "parakeet", "model_path": "models/parakeet-tdt-0.6b-v3-int8" }
  ]
}
```

Или только Parakeet для русского:

```json
{
  "models": [
    { "type": "whisper", "model_path": "models/whisper/ggml-large-v3-turbo.bin" },
    { "type": "parakeet", "model_path": "models/parakeet-tdt-0.6b-v3-int8" }
  ]
}
```

## Поддерживаемые языки

bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, **ru**, uk

## Размер

~640 MB (INT8 квантизация)

## GPU (CUDA)

Parakeet и GigaAM поддерживают `use_gpu: true` — тогда используется провайдер `cuda`.

**Важно:** По умолчанию sherpa-rs собирается с CPU-бинарниками. Для GPU нужно пересобрать с feature `cuda`:

```toml
# Cargo.toml — заменить строку sherpa-rs:
sherpa-rs = { version = "0.6", features = ["sys", "cuda"] }
```

При этом feature `download-binaries` несовместим с `cuda` — придётся собирать sherpa-onnx из исходников (с CUDA toolkit). Без этого `use_gpu: true` приведёт к ошибке при загрузке.
