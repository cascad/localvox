//! Постобработка транскрипта: объединение сегментов, ансамбль, исправления.

use crate::overlap::merge_overlap;

/// Обрабатывает новый сегмент транскрипта с учётом хвоста предыдущего.
/// Возвращает (текст для отправки клиенту, новый хвост для следующей итерации).
pub fn process_segment(prev_tail: &str, new_text: &str) -> (String, String) {
    let (merged, new_tail) = merge_overlap(prev_tail, new_text);
    (merged, new_tail)
}

/// Word-level edit distance (Levenshtein) alignment between two word sequences.
/// Returns aligned pairs: (Option<word_a>, Option<word_b>) — None means gap.
fn align_words<'a>(words_a: &[&'a str], words_b: &[&'a str]) -> Vec<(Option<&'a str>, Option<&'a str>)> {
    let n = words_a.len();
    let m = words_b.len();

    // dp[i][j] = edit distance for words_a[..i] vs words_b[..j]
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if words_a[i - 1].to_lowercase() == words_b[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    // Backtrack
    let mut result = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = if words_a[i - 1].to_lowercase() == words_b[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                result.push((Some(words_a[i - 1]), Some(words_b[j - 1])));
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            result.push((Some(words_a[i - 1]), None));
            i -= 1;
        } else {
            result.push((None, Some(words_b[j - 1])));
            j -= 1;
        }
    }
    result.reverse();
    result
}

/// Picks the better word variant from two aligned candidates.
/// Heuristic: matching words kept as-is; on conflict — prefer longer word
/// (ASR models rarely hallucinate extra characters, but often truncate).
fn pick_best_word<'a>(a: Option<&'a str>, b: Option<&'a str>) -> &'a str {
    match (a, b) {
        (Some(wa), Some(wb)) => {
            if wa.to_lowercase() == wb.to_lowercase() {
                wa
            } else if wb.chars().count() > wa.chars().count() {
                wb
            } else {
                wa
            }
        }
        (Some(w), None) | (None, Some(w)) => w,
        (None, None) => "",
    }
}

/// Объединяет результаты Whisper и GigaAM через word-level alignment.
pub fn ensemble_merge(whisper_text: &str, gigaam_text: &str) -> String {
    let w = whisper_text.trim();
    let g = gigaam_text.trim();
    if w.is_empty() {
        return g.to_string();
    }
    if g.is_empty() {
        return w.to_string();
    }

    let w_words: Vec<&str> = w.split_whitespace().collect();
    let g_words: Vec<&str> = g.split_whitespace().collect();

    let aligned = align_words(&w_words, &g_words);
    let merged: Vec<&str> = aligned
        .iter()
        .map(|(a, b)| pick_best_word(*a, *b))
        .filter(|s| !s.is_empty())
        .collect();

    merged.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        assert_eq!(ensemble_merge("hello world", "hello world"), "hello world");
    }

    #[test]
    fn test_one_empty() {
        assert_eq!(ensemble_merge("hello", ""), "hello");
        assert_eq!(ensemble_merge("", "world"), "world");
    }

    #[test]
    fn test_both_empty() {
        assert_eq!(ensemble_merge("", ""), "");
    }

    #[test]
    fn test_word_alignment_insertion() {
        let result = ensemble_merge("привет мир", "привет прекрасный мир");
        assert_eq!(result, "привет прекрасный мир");
    }

    #[test]
    fn test_word_alignment_conflict() {
        // "сто" vs "стой" — longer wins
        let result = ensemble_merge("не сто у меня", "не стой у меня");
        assert_eq!(result, "не стой у меня");
    }

    #[test]
    fn test_word_alignment_whisper_longer() {
        // "слово" vs "короткое" — longer wins per word
        let result = ensemble_merge("длинное слово", "длинное короткое");
        assert_eq!(result, "длинное короткое");
    }

    #[test]
    fn test_process_segment_passthrough() {
        let (merged, _) = process_segment("", "first segment");
        assert_eq!(merged, "first segment");
    }

    #[test]
    fn test_process_segment_with_overlap() {
        let (merged, tail) = process_segment("one two three", "two three four");
        assert_eq!(merged, "four");
        assert!(!tail.is_empty());
    }

    #[test]
    fn test_process_segment_empty_new() {
        let (merged, tail) = process_segment("prev tail", "");
        assert_eq!(merged, "");
        assert_eq!(tail, "prev tail");
    }
}
