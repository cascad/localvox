//! Async LLM correction via Ollama API.
//! Combines arbiter (picking best variant) and corrector (fixing ASR errors) in one pass.
//! Stateless: context is passed in by the caller (enables parallel LLM workers).

use crate::processor::ModelOutput;
use std::time::Instant;

pub struct LlmCorrector {
    ollama_url: String,
    model: String,
    prompt_single: String,
    prompt_ensemble: String,
    timeout_sec: u64,
}

impl LlmCorrector {
    pub fn new(
        ollama_url: &str,
        model: &str,
        prompt_single: &str,
        prompt_ensemble: &str,
        timeout_sec: u64,
    ) -> Self {
        Self {
            ollama_url: ollama_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            prompt_single: prompt_single.to_string(),
            prompt_ensemble: prompt_ensemble.to_string(),
            timeout_sec: timeout_sec.max(1),
        }
    }

    /// Calls Ollama to correct/arbitrate between ASR outputs.
    /// Returns corrected text, or None on error/timeout.
    /// Overlap between segments is handled algorithmically after LLM; LLM only corrects and picks variants.
    pub fn correct(
        &self,
        model_outputs: &[ModelOutput],
        merged_text: &str,
        context_str: &str,
    ) -> Option<String> {
        let context_str = if context_str.is_empty() { "(нет)" } else { context_str };

        let prompt = if model_outputs.len() <= 1 {
            self.prompt_single
                .replace("{context_str}", context_str)
                .replace("{merged_text}", merged_text)
        } else {
            let model_variants: String = model_outputs
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    let label = (b'A' + i as u8) as char;
                    format!("Вариант {} ({}): \"{}\"", label, o.model_name, o.text)
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mut p = self.prompt_ensemble
                .replace("{context_str}", context_str)
                .replace("{merged_text}", merged_text)
                .replace("{model_variants}", &model_variants);
            if model_outputs.len() >= 2 {
                let w = model_outputs.iter().find(|o| o.model_name == "whisper").map(|o| o.text.as_str()).unwrap_or("");
                let g = model_outputs.iter().find(|o| o.model_name == "gigaam").map(|o| o.text.as_str()).unwrap_or("");
                p = p.replace("{whisper_text}", w).replace("{gigaam_text}", g);
            }
            p
        };

        self.do_request(&prompt, merged_text)
    }

    fn do_request(&self, prompt: &str, merged_text: &str) -> Option<String> {
        let is_thinking_model = self.model.contains("3.5") || self.model.to_lowercase().contains("deepseek");
        let num_predict = merged_text.len().max(96).min(384);
        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.1,
                "num_predict": num_predict,
            }
        });
        if is_thinking_model {
            body["think"] = serde_json::json!(false);
        }

        let url = format!("{}/api/generate", self.ollama_url);
        let t0 = Instant::now();

        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_sec))
            .send_json(&body);

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                let elapsed = t0.elapsed().as_secs_f64();
                tracing::warn!("LLM request failed after {:.1}s: {}", elapsed, e);
                return None;
            }
        };

        let json: serde_json::Value = match resp.into_json() {
            Ok(j) => j,
            Err(e) => {
                let elapsed = t0.elapsed().as_secs_f64();
                tracing::warn!("LLM response parse error after {:.1}s: {}", elapsed, e);
                return None;
            }
        };

        let raw = json
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim()
            .to_string();

        let response_text = strip_llm_reasoning(&raw);
        let elapsed = t0.elapsed().as_secs_f64();

        if response_text.is_empty() {
            tracing::warn!("LLM returned empty response after {:.1}s", elapsed);
            return None;
        }
        if response_text.len() > merged_text.len() * 3 {
            let trunc_target = merged_text.len().max(50) * 2;
            if let Some(truncated) = truncate_at_sentence_boundary(&response_text, trunc_target) {
                if truncated.len() >= merged_text.len() / 2 {
                    tracing::warn!(
                        "LLM response truncated ({} → {} chars, input {} chars) after {:.1}s",
                        response_text.len(),
                        truncated.len(),
                        merged_text.len(),
                        elapsed
                    );
                    return Some(truncated);
                }
            }
            tracing::warn!(
                "LLM response too long ({} vs {} chars) after {:.1}s, discarding",
                response_text.len(),
                merged_text.len(),
                elapsed
            );
            tracing::warn!("  [raw ASR] {}", merged_text);
            tracing::warn!("  [LLM proposed] {}", response_text);
            return None;
        }

        tracing::info!("LLM correction {:.1}s: {:?}", elapsed, response_text.chars().take(60).collect::<String>());
        Some(response_text)
    }
}

fn truncate_at_sentence_boundary(text: &str, target_bytes: usize) -> Option<String> {
    if target_bytes >= text.len() {
        return Some(text.to_string());
    }
    let mut end = target_bytes;
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    let slice = &text[..end];
    let cut = slice
        .rfind(". ")
        .map(|p| p + 1)
        .or_else(|| slice.rfind("? ").map(|p| p + 1))
        .or_else(|| slice.rfind("! ").map(|p| p + 1))
        .or_else(|| slice.rfind(", ").map(|p| p + 1))
        .or_else(|| slice.rfind(' '))
        .unwrap_or(end);
    let result = text[..cut].trim();
    if result.is_empty() {
        None
    } else {
        Some(result.to_string())
    }
}

fn strip_llm_reasoning(text: &str) -> String {
    let t = text.trim();

    // Только паттерны, которые невозможны в реальной речи — это мета-рассуждения LLM
    let colon_prefixes = [
        "используем вариант",
        "выбираем вариант",
        "вариант а (whisper)",
        "вариант б (gigaam)",
        "вариант в (silero)",
        "вариант в (parakeet)",
        "вариант a (whisper)",
        "вариант b (gigaam)",
        "вариант c (silero)",
        "вариант c (parakeet)",
        "variant a",
        "variant b",
        "variant c",
        "corrected:",
        "corrected text:",
    ];
    let lower = t.to_lowercase();
    for prefix in colon_prefixes {
        if lower.starts_with(prefix) {
            if let Some(pos) = t.find(':') {
                let after = t[pos + 1..].trim();
                let after = after.trim_start_matches(|c: char| c == '"' || c == '«' || c == '"');
                let after = after.trim_end_matches(|c: char| c == '"' || c == '»' || c == '"');
                let after = after.trim();
                if !after.is_empty() {
                    return after.to_string();
                }
            }
        }
    }

    // Strip wrapping quotes: «текст» or "текст"
    let stripped = t
        .trim_start_matches(|c: char| c == '«' || c == '"' || c == '"')
        .trim_end_matches(|c: char| c == '»' || c == '"' || c == '"')
        .trim();
    if !stripped.is_empty() && stripped != t {
        return stripped.to_string();
    }

    t.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stateless() {
        let _ = LlmCorrector::new(
            "http://localhost:11434",
            "model",
            "{context_str} {merged_text}",
            "{context_str} {merged_text} {whisper_text} {gigaam_text}",
            10,
        );
    }

    #[test]
    fn test_strip_variant_prefix_ru() {
        let r = strip_llm_reasoning("Используем вариант А (Whisper): \"Во-первых, у меня самые лучшие\"");
        assert_eq!(r, "Во-первых, у меня самые лучшие");
    }

    #[test]
    fn test_strip_variant_b_gigaam() {
        let r = strip_llm_reasoning("Вариант Б (GigaAM): текст после двоеточия");
        assert_eq!(r, "текст после двоеточия");
    }

    #[test]
    fn test_strip_quotes() {
        let r = strip_llm_reasoning("«Привет, как дела?»");
        assert_eq!(r, "Привет, как дела?");
    }

    #[test]
    fn test_strip_corrected_prefix() {
        let r = strip_llm_reasoning("Corrected: Вот правильный вариант");
        assert_eq!(r, "Вот правильный вариант");
    }

    #[test]
    fn test_real_speech_preserved() {
        assert_eq!(strip_llm_reasoning("Результат: мы получили отличный"), "Результат: мы получили отличный");
        assert_eq!(strip_llm_reasoning("Ответ: да, конечно"), "Ответ: да, конечно");
        assert_eq!(strip_llm_reasoning("Исправленный текст: вот он"), "Исправленный текст: вот он");
    }

    #[test]
    fn test_passthrough_clean() {
        let r = strip_llm_reasoning("Просто нормальный текст без мусора");
        assert_eq!(r, "Просто нормальный текст без мусора");
    }
}
