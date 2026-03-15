//! Чёрный список типичных галлюцинаций Whisper и GigaAM.
//! Отдельные списки для каждой модели — чтобы не выкидывать валидный контент другой.

/// Whisper: отбрасываем весь сегмент.
const WHISPER_DROP: &[&str] = &[
    "редактор субтитров",
    "субтитры субтитров",
    "субтитры и субтитры",
    "субтитры в мастерской",
    "субтитры в киеве",
    "субтитры в ",
    "субтитры сделал",
    "субтитры подготовил",
    "спасибо за субтитры",
    "корректор а.егорова",
    "смотрите на видео",
    "смотрите продолжение",
    "до скорого",
    "увидимся",
    "subtitles by",
    "translated by",
    "подписывайтесь на канал",
    "подпишись на канал",
    "подпишись!",
    "спасибо за подписку",
    "спасибо за просмотр",
    "thanks for watching",
    "subscribe",
    "продолжение следует",
    "www.",
    "http",
    "amara.org",
    // Музыка, звуки (Whisper на тишине/шуме)
    "веселая музыка",
    "спокойная музыка",
    "грустная мелодия",
    "лирическая музыка",
    "динамичная музыка",
    "таинственная музыка",
    "торжественная музыка",
    "интригующая музыка",
    "напряженная музыка",
    "печальная музыка",
    "тревожная музыка",
    "музыкальная заставка",
    "перестрелка",
    "гудок поезда",
    "рёв мотора",
    "шум двигателя",
    "сигнал автомобиля",
    "лай собак",
    "пес лает",
    "кашель",
    "выстрелы",
    "шум дождя",
    "песня",
    "взрыв",
    "шум мотора",
    "плеск воды",
    "гудок автомобиля",
    "лай собаки",
    "по тв.",
    "аплодисменты",
    "городской шум",
    "полиция",
    "городской гудок",
    "сигнал машины",
    "смех",
    "стук в дверь",
    "полицейская сирена",
    "звонок в дверь",
];

/// Whisper: вырезаем встроенные фразы (субтитры создал X и т.п.).
const WHISPER_STRIP: &[(&str, bool)] = &[
    ("субтитры создал", true),
    ("субтитры добавил", true),
    ("субтитры подогнал", true),
    // Редактор субтитров А.Семкин, корректор А.Егорова и т.п.
    ("редактор субтитров ", false),
    ("а.семкин, корректор а.егорова. ", false),
    ("а.семкин, корректор а.егорова, ", false),
    (", корректор а.егорова. ", false),
    (", корректор а.егорова, ", false),
];

/// GigaAM: отбрасываем весь сегмент (редактор субтитров, корректор и т.п.).
const GIGAAM_DROP: &[&str] = &[
    "редактор субтитров",
    "корректор а.егорова",
];

/// GigaAM: вырезаем встроенные фразы (редактор субтитров, корректор и т.п.).
const GIGAAM_STRIP: &[(&str, bool)] = &[
    ("редактор субтитров ", false),
    ("а.семкин, корректор а.егорова. ", false),
    ("а.семкин, корректор а.егорова, ", false),
    (", корректор а.егорова. ", false),
    (", корректор а.егорова, ", false),
];

/// Parakeet: отбрасываем весь сегмент при типичных галлюцинациях (короткие сегменты).
/// Parakeet TDT 0.6B часто выдаёт случайные английские слова на тишине/шуме.
/// yeah — самая частая галлюцинация (sherpa-onnx #3267), pinch — из практики.
const PARAKEET_DROP: &[&str] = &[
    "yeah",
    "yeah.",
    "pinch",
    "pinch.",
];

/// Parakeet: вырезаем встроенные фразы.
const PARAKEET_STRIP: &[(&str, bool)] = &[];

/// Фильтрует галлюцинации Parakeet.
pub fn filter_parakeet(text: &str) -> String {
    apply_filter(text, PARAKEET_DROP, PARAKEET_STRIP)
}

/// Фильтрует галлюцинации Whisper.
pub fn filter_whisper(text: &str) -> String {
    apply_filter(text, WHISPER_DROP, WHISPER_STRIP)
}

/// Фильтрует галлюцинации GigaAM.
pub fn filter_gigaam(text: &str) -> String {
    apply_filter(text, GIGAAM_DROP, GIGAAM_STRIP)
}

fn apply_filter(text: &str, drop_patterns: &[&str], strip_phrases: &[(&str, bool)]) -> String {
    let t = text.trim();
    if t.is_empty() {
        return String::new();
    }

    let dominated_by_brackets = {
        let total = t.chars().count();
        let bracket_chars: usize = t
            .split(|c| c == '[' || c == '*' || c == '(')
            .skip(1)
            .map(|s| s.chars().count() + 1)
            .sum();
        bracket_chars * 2 > total
    };
    if dominated_by_brackets {
        return String::new();
    }

    let lower = t.to_lowercase();
    let word_count = t.split_whitespace().count();
    for pat in drop_patterns {
        if lower.contains(pat) {
            let pat_words = pat.split_whitespace().count();
            if word_count <= pat_words + 2 {
                return String::new();
            }
        }
    }

    strip_embedded_phrases(t, strip_phrases)
}

fn strip_embedded_phrases(text: &str, strip_phrases: &[(&str, bool)]) -> String {
    let mut result = text.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        for (phrase, remove_next_word) in strip_phrases {
            let lower = result.to_lowercase();
            if let Some(pos) = lower.find(phrase) {
                let phrase_end = pos + phrase.len();
                let remove_end = if *remove_next_word {
                    let rest = result[phrase_end..].trim_start();
                    let leading = result[phrase_end..].len() - rest.len();
                    let word_end = rest
                        .find(|c: char| c.is_whitespace() || ".!?,;".contains(c))
                        .unwrap_or(rest.len());
                    phrase_end + leading + word_end
                } else {
                    phrase_end
                };

                let before = result[..pos].trim_end();
                let after = result[remove_end..].trim_start();
                result = format!("{} {}", before, after).trim().to_string();
                result = result.replace("  ", " ");
                changed = true;
                break;
            }
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_drop_subtitles() {
        assert_eq!(filter_whisper("Субтитры создал DimaTorzhok"), "");
    }

    #[test]
    fn test_whisper_strip_embedded() {
        let t = filter_whisper("Но они называют это. Субтитры создал DimaTorzhok с высоким уровнем. Нет.");
        assert!(!t.to_lowercase().contains("субтитр"));
        assert!(t.contains("Но они"));
        assert!(t.contains("Нет"));
    }

    #[test]
    fn test_whisper_keep_normal() {
        assert_eq!(filter_whisper("Привет, как дела?"), "Привет, как дела?");
    }

    #[test]
    fn test_whisper_drop_music() {
        assert_eq!(filter_whisper("Веселая музыка играет"), "");
    }

    #[test]
    fn test_whisper_drop_subscribe() {
        assert_eq!(filter_whisper("Подпишись на канал!"), "");
    }

    #[test]
    fn test_whisper_drop_editor() {
        assert_eq!(filter_whisper("Редактор субтитров М.Иванов"), "");
    }

    #[test]
    fn test_whisper_strip_editor_corrector() {
        let t = filter_whisper("Редактор субтитров А.Семкин, корректор А.Егорова. Ему пора.");
        assert_eq!(t.trim(), "Ему пора.");
    }

    #[test]
    fn test_whisper_brackets_dominated() {
        assert_eq!(filter_whisper("[музыка] [аплодисменты]"), "");
    }

    #[test]
    fn test_whisper_empty_input() {
        assert_eq!(filter_whisper(""), "");
        assert_eq!(filter_whisper("   "), "");
    }

    #[test]
    fn test_gigaam_passthrough() {
        assert_eq!(filter_gigaam("Привет, как дела?"), "Привет, как дела?");
    }

    #[test]
    fn test_gigaam_whisper_pattern_not_dropped() {
        // GigaAM не фильтрует по Whisper-паттернам
        assert_eq!(filter_gigaam("Веселая музыка"), "Веселая музыка");
    }
}
