# Silero STT: загрузка и настройка для localvox

## 1. Скачать модель и токены

### Русский (через Whisper/GigaAM — Silero STT пока нет для ru)
Официальные Silero STT модели: **en**, **de**, **es**, **ua** (украинский).

### Английский (рекомендуется для теста)

| Файл | URL |
|------|-----|
| **Модель ONNX** | https://models.silero.ai/models/en/en_v5.onnx |
| **Словарь (labels)** | https://models.silero.ai/models/en/en_v1_labels.json |

### Украинский

| Файл | URL |
|------|-----|
| **Модель ONNX** | https://models.silero.ai/models/ua/ua_v3.onnx |
| **Словарь (labels)** | https://models.silero.ai/models/ua/ua_v1_labels.json |

### Немецкий

| Файл | URL |
|------|-----|
| **Модель ONNX** | https://models.silero.ai/models/de/de_v1.onnx |
| **Словарь (labels)** | https://models.silero.ai/models/de/de_v1_labels.json |

### Испанский

| Файл | URL |
|------|-----|
| **Модель ONNX** | https://models.silero.ai/models/es/es_v1.onnx |
| **Словарь (labels)** | https://models.silero.ai/models/es/es_v1_labels.json |

---

## 2. Конвертация labels.json в tokens.txt

Адаптер ожидает `tokens.txt` — один токен на строку. Labels приходят в формате JSON-массива.

```powershell
# PowerShell: конвертация en_v1_labels.json -> tokens.txt
$json = Get-Content "en_v1_labels.json" | ConvertFrom-Json
$json | ForEach-Object { $_ } | Set-Content tokens.txt
```

Или вручную: скачать JSON, открыть, скопировать массив и построчно записать в `tokens.txt` (каждый элемент — отдельная строка).

---

## 3. Структура папки

```
models/
  silero-en/
    en_v5.onnx      # модель
    tokens.txt      # токены (из en_v1_labels.json)
```

---

## 4. Настройка settings.json

```json
{
  "models": [
    {"type": "whisper", "model_path": "models/whisper/ggml-large-v3-turbo.bin", "use_gpu": true},
    {"type": "silero", "model_path": "models/silero-en/en_v5.onnx", "tokens_path": "models/silero-en/tokens.txt"}
  ]
}
```

### Только Silero (без ансамбля)

```json
{
  "models": [
    {"type": "silero", "model_path": "models/silero-en/en_v5.onnx", "tokens_path": "models/silero-en/tokens.txt"}
  ]
}
```

---

## 5. Быстрый тест (PowerShell)

```powershell
# Скачать модель и labels
Invoke-WebRequest -Uri "https://models.silero.ai/models/en/en_v5.onnx" -OutFile "models/silero-en/en_v5.onnx"
Invoke-WebRequest -Uri "https://models.silero.ai/models/en/en_v1_labels.json" -OutFile "models/silero-en/en_v1_labels.json"

# Конвертация labels -> tokens.txt
$json = Get-Content "models/silero-en/en_v1_labels.json" -Raw
$labels = $json | ConvertFrom-Json
$labels | ForEach-Object { $_ } | Set-Content "models/silero-en/tokens.txt"
```

---

## Ссылки

- **Репозиторий Silero**: https://github.com/snakers4/silero-models  
- **Модели**: https://models.silero.ai/  
- **models.yml** (список моделей): https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml  
