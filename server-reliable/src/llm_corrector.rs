//! Async LLM correction via Ollama API.
//! Combines arbiter (picking best variant) and corrector (fixing ASR errors) in one pass.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

const CONTEXT_LINES: usize = 5;
const TIMEOUT_SEC: u64 = 30;

pub struct LlmCorrector {
    ollama_url: String,
    model: String,
    context: Mutex<VecDeque<String>>,
}

impl LlmCorrector {
    pub fn new(ollama_url: &str, model: &str) -> Self {
        Self {
            ollama_url: ollama_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            context: Mutex::new(VecDeque::with_capacity(CONTEXT_LINES + 1)),
        }
    }

    pub fn push_context(&self, text: &str) {
        let mut ctx = self.context.lock().unwrap();
        ctx.push_back(text.to_string());
        while ctx.len() > CONTEXT_LINES {
            ctx.pop_front();
        }
    }

    fn build_context_string(&self) -> String {
        let ctx = self.context.lock().unwrap();
        if ctx.is_empty() {
            return "(нет)".to_string();
        }
        ctx.iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Calls Ollama to correct/arbitrate between two ASR outputs.
    /// Returns corrected text, or None on error/timeout.
    pub fn correct(
        &self,
        whisper_text: &str,
        gigaam_text: &str,
        merged_text: &str,
    ) -> Option<String> {
        let context_str = self.build_context_string();

        let prompt = if gigaam_text.is_empty() {
            format!(
                "Исправь ошибки распознавания речи в тексте. \
                 Контекст предыдущих фраз: {context_str}\n\
                 Текст: \"{merged_text}\"\n\
                 Верни только исправленный текст, без кавычек и пояснений."
            )
        } else {
            format!(
                "Даны два варианта распознавания одного аудиофрагмента.\n\
                 Вариант A (Whisper): \"{whisper_text}\"\n\
                 Вариант B (GigaAM): \"{gigaam_text}\"\n\
                 Алгоритмический мерж: \"{merged_text}\"\n\
                 Контекст предыдущих фраз: {context_str}\n\n\
                 Выбери лучший вариант или объедини их. Исправь явные ошибки распознавания. \
                 Верни только исправленный текст, без кавычек и пояснений."
            )
        };

        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.1,
                "num_predict": 256,
            }
        });

        let url = format!("{}/api/generate", self.ollama_url);
        let t0 = Instant::now();

        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(TIMEOUT_SEC))
            .send_json(&body);

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM request failed: {}", e);
                return None;
            }
        };

        let json: serde_json::Value = match resp.into_json() {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("LLM response parse error: {}", e);
                return None;
            }
        };

        let response_text = json
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim()
            .to_string();

        let elapsed = t0.elapsed().as_secs_f64();
        tracing::debug!("LLM correction took {:.2}s: {:?}", elapsed, &response_text);

        if response_text.is_empty() || response_text.len() > merged_text.len() * 3 {
            return None;
        }

        Some(response_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_push_context() {
        let c = LlmCorrector::new("http://localhost:11434", "model");
        for i in 0..10 {
            c.push_context(&format!("line {}", i));
        }
        // Context caps at CONTEXT_LINES, no panic
    }
}
