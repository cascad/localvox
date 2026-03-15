//! Async summarization via Ollama.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use crate::types::UiEvent;

/// Асинхронная суммаризация через Ollama. Сохраняет в {base}_summary.txt.
pub fn spawn_summarize_ollama(
    file_path: PathBuf,
    url: &str,
    model: &str,
    prompt_template: &str,
    ui_tx: mpsc::Sender<UiEvent>,
) {
    let url = url.trim_end_matches('/').to_string();
    let model = model.to_string();
    let prompt_template = prompt_template.to_string();
    let _ = thread::spawn(move || {
        let content = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Debug {
                    text: format!("[summarize] ошибка чтения: {}", e),
                });
                let _ = ui_tx.send(UiEvent::SummarizeDone {
                    msg: format!("Ошибка чтения: {}", e),
                });
                return;
            }
        };
        let prompt = prompt_template.replace("{text}", &content);
        let is_thinking = model.contains("3.5") || model.to_lowercase().contains("deepseek");
        let mut body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": { "temperature": 0.2, "num_predict": 1024 }
        });
        if is_thinking {
            body["think"] = serde_json::json!(false);
        }
        let api_url = format!("{}/api/generate", url);
        let _ = ui_tx.send(UiEvent::Debug {
            text: format!("[summarize] запрос → {} ({})", api_url, model),
        });
        let resp = ureq::post(&api_url)
            .timeout(std::time::Duration::from_secs(120))
            .send_json(&body);
        let summary = match resp {
            Ok(r) => match r.into_json::<serde_json::Value>() {
                Ok(j) => j
                    .get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                Err(e) => {
                    let _ = ui_tx.send(UiEvent::Debug {
                        text: format!("[summarize] ошибка JSON: {}", e),
                    });
                    let _ = ui_tx.send(UiEvent::SummarizeDone {
                        msg: format!("Ошибка JSON: {}", e),
                    });
                    return;
                }
            },
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Debug {
                    text: format!("[summarize] ошибка Ollama: {}", e),
                });
                let _ = ui_tx.send(UiEvent::SummarizeDone {
                    msg: format!("Ошибка Ollama: {}", e),
                });
                return;
            }
        };
        if summary.is_empty() {
            let _ = ui_tx.send(UiEvent::Debug {
                text: "[summarize] LLM вернул пустой ответ".into(),
            });
            let _ = ui_tx.send(UiEvent::SummarizeDone {
                msg: "LLM вернул пустой ответ".into(),
            });
            return;
        }
        let stem = file_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let summary_path = file_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(format!("{}_summary.txt", stem));
        match std::fs::write(&summary_path, &summary) {
            Ok(_) => {
                let _ = ui_tx.send(UiEvent::Debug {
                    text: format!("[summarize] ответ → {}", summary_path.display()),
                });
                let _ = ui_tx.send(UiEvent::SummarizeDone {
                    msg: format!("Суммаризация → {}", summary_path.display()),
                });
            }
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Debug {
                    text: format!("[summarize] ошибка записи: {}", e),
                });
                let _ = ui_tx.send(UiEvent::SummarizeDone {
                    msg: format!("Ошибка записи: {}", e),
                });
            }
        }
    });
}
