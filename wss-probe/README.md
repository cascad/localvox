# wss-probe

Минимальная проверка **WSS** (или WS) до сервера LocalVox: TCP → TLS → WebSocket handshake, затем отправка такого же `config`, как у клиента, и **печать входящих сообщений** в stdout.

## Сборка

```bash
cargo build -p wss-probe --release
```

## Примеры

Внешний IP, свой PEM, API key из окружения:

```bash
set LOCALVOX_API_KEY=your-key
wss-probe wss://203.0.113.10:9745 --tls-ca-path certs/server.pem
```

Локально, небезопасно (любой сертификат):

```bash
wss-probe wss://127.0.0.1:9745 --tls-insecure --api-key YOUR_KEY
```

Остановка через 60 с или после 5 сообщений:

```bash
wss-probe wss://... --tls-ca-path certs/server.pem --for-sec 60 --max-msgs 5
```

Без лимита времени (пока не прервёшь Ctrl+C): `--for-sec 0`.

## Флаги

| Флаг | Описание |
|------|----------|
| `URL` | `wss://host:port` или `ws://...` |
| `--api-key` | Или `LOCALVOX_API_KEY` |
| `--tls-ca-path` | PEM доверенного CA / self-signed сервера |
| `--tls-insecure` | Не проверять сертификат сервера |
| `--client-id` | Опционально, в JSON `config` |
| `--for-sec` | Секунды чтения (0 = бесконечно), по умолчанию 30 |
| `--max-msgs` | Остановка после N сообщений (0 = без лимита) |
