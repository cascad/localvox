//! Client configuration file handling.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_SUMMARIZE_PROMPT: &str = r#"Ты — суммаризатор встреч. Транскрипт ниже — запись разговора (mic = микрофон, sys = системный звук/удалённые участники).

Сделай структурированную суммаризацию:

1. Заголовок: «Суммаризация: [тема встречи]»

2. УЧАСТНИКИ — «— Имя (mic/sys) — роль». Если имён нет — «— участник (mic)», «— участник (sys)».

3. СТАТУСЫ ПО ЗАДАЧАМ — по участникам: что делает, вопросы. Пропусти «что сделал вчера», «блокеры», если нет информации. Не пиши «Не указано».

4. РАЗБОР (если есть дискуссии) — Контекст, Суть, Решение.

5. ПРОЧЕЕ — организационные вопросы, следующие шаги.

Только русский. Только факты из транскрипта.

---

{text}"#;

#[derive(Deserialize, Serialize, Debug, Default, Clone)]
pub struct ClientConfig {
    #[serde(default)]
    pub server: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    /// Системный звук (loopback): устройство вывода по имени или индексу
    #[serde(default)]
    pub loopback: Option<String>,
    /// Второй микрофон (по имени или индексу). Игнорируется если задан loopback.
    #[serde(default)]
    pub device2: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
    /// Папка для экспорта result_{datetime}.txt (по умолчанию — рядом с output)
    #[serde(default)]
    pub export_dir: Option<String>,
    /// Включить LLM-суммаризацию экспортированного куска (асинхронно после сохранения)
    #[serde(default)]
    pub summarize_enabled: Option<bool>,
    /// Бэкенд суммаризации: "ollama" (пока единственный)
    #[serde(default)]
    pub summarize_backend: Option<String>,
    /// URL Ollama API (напр. http://localhost:11434)
    #[serde(default)]
    pub summarize_url: Option<String>,
    /// Модель Ollama (напр. qwen3.5:9b)
    #[serde(default)]
    pub summarize_model: Option<String>,
    /// Промпт суммаризации. Плейсхолдер {text} — текст экспорта
    #[serde(default)]
    pub summarize_prompt: Option<String>,
    /// Идентификатор сессии для переподключения к серверу. Пусто/нет поля + задан `api_key` → см. [`effective_client_id`].
    #[serde(default)]
    pub client_id: Option<String>,
    /// API key for server auth (or env LOCALVOX_API_KEY).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Accept self-signed TLS certs (dev only).
    #[serde(default)]
    pub tls_insecure: Option<bool>,
    /// Path to custom CA cert for TLS (PEM).
    #[serde(default)]
    pub tls_ca_path: Option<String>,
}

/// Эффективный `client_id` для WebSocket `config`: явное значение из конфига, иначе при непустом `api_key` — стабильный `k-` + 16 hex символов от SHA-256 ключа.
///
/// Если оба устройства используют **один** API key, задайте им **разные** `client_id` вручную, иначе сессии на сервере конфликтуют.
pub fn effective_client_id(configured: Option<String>, api_key: Option<&str>) -> Option<String> {
    let configured = configured.and_then(|s| {
        let t = s.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_string())
        }
    });
    if let Some(id) = configured {
        return Some(id);
    }
    let key = api_key.map(str::trim).filter(|k| !k.is_empty())?;
    let hash = Sha256::digest(key.as_bytes());
    let hex = hex::encode(hash);
    Some(format!("k-{}", &hex[..16]))
}

pub fn find_config_path() -> Option<PathBuf> {
    let name = "client-config.json";
    let mut candidates: Vec<Option<PathBuf>> = vec![
        std::env::current_dir().ok().map(|p| p.join(name)),
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|pp| pp.join(name))),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|pp| pp.join(name))),
    ];
    candidates.push(
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|base| base.join("live-transcribe").join(name)),
    );
    candidates.into_iter().flatten().find(|p| p.is_file())
}

/// Конфиг по умолчанию: device и loopback = default, summarize_enabled = true.
pub fn default_config() -> ClientConfig {
    ClientConfig {
        server: Some("ws://localhost:9745".into()),
        device: Some("default".into()),
        loopback: Some("default-output".into()),
        device2: None,
        output: Some("transcript.txt".into()),
        export_dir: None,
        summarize_enabled: Some(true),
        summarize_backend: Some("ollama".into()),
        summarize_url: Some("http://localhost:11434".into()),
        summarize_model: Some("qwen3.5:9b".into()),
        summarize_prompt: Some(DEFAULT_SUMMARIZE_PROMPT.into()),
        client_id: Some("live".into()),
        api_key: None,
        tls_insecure: None,
        // Self-signed из tools/gen-dev-certs.* — путь от cwd или абсолютный.
        tls_ca_path: Some("certs/server.pem".into()),
    }
}

pub fn load_config() -> ClientConfig {
    match find_config_path() {
        Some(path) => {
            eprintln!("Конфиг: {}", path.display());
            match std::fs::read_to_string(&path) {
                Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                    eprintln!("Ошибка парсинга конфига: {e}");
                    ClientConfig::default()
                }),
                Err(e) => {
                    eprintln!("Не удалось прочитать конфиг: {e}");
                    ClientConfig::default()
                }
            }
        }
        None => {
            let cfg = default_config();
            let path = get_config_save_path();
            if let Ok(text) = serde_json::to_string_pretty(&cfg) {
                if let Err(e) = std::fs::write(&path, text) {
                    eprintln!(
                        "Не удалось сохранить конфиг по умолчанию в {}: {e}",
                        path.display()
                    );
                } else {
                    eprintln!("Создан конфиг по умолчанию: {}", path.display());
                }
            }
            cfg
        }
    }
}

/// Путь для сохранения конфига. Предпочитает тот же файл, из которого загружаем.
pub fn get_config_save_path() -> PathBuf {
    if let Some(path) = find_config_path() {
        return path;
    }
    let name = "client-config.json";
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.is_dir() {
            return cwd.join(name);
        }
    }
    if let Some(base) = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    {
        let dir = base.join("live-transcribe");
        let _ = std::fs::create_dir_all(&dir);
        return dir.join(name);
    }
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::effective_client_id;

    #[test]
    fn effective_client_id_prefers_configured() {
        assert_eq!(
            effective_client_id(Some("  my-id  ".into()), Some("secret")),
            Some("my-id".into())
        );
    }

    #[test]
    fn effective_client_id_derives_from_api_key() {
        assert_eq!(
            effective_client_id(None, Some("secret")),
            Some("k-2bb80d537b1da3e3".into())
        );
        assert_eq!(
            effective_client_id(Some("".into()), Some("secret")),
            Some("k-2bb80d537b1da3e3".into())
        );
    }

    #[test]
    fn effective_client_id_none_without_key() {
        assert_eq!(effective_client_id(None, None), None);
        assert_eq!(effective_client_id(None, Some("")), None);
    }
}
