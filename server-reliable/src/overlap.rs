//! Text overlap merging at chunk boundaries.

const TAIL_WORDS: usize = 10;

/// Remove duplication at chunk boundaries.
/// Returns (merged_text_to_emit, new_tail for next iteration).
pub fn merge_overlap(prev_tail: &str, new_text: &str) -> (String, String) {
    let new_text = new_text.trim();
    if new_text.is_empty() {
        return (String::new(), prev_tail.to_string());
    }
    if prev_tail.trim().is_empty() {
        let words: Vec<&str> = new_text.split_whitespace().collect();
        let tail = if words.len() >= TAIL_WORDS {
            words[words.len().saturating_sub(TAIL_WORDS)..].join(" ")
        } else {
            new_text.to_string()
        };
        return (new_text.to_string(), tail);
    }

    let prev_words: Vec<&str> = prev_tail.split_whitespace().collect();
    let new_words: Vec<&str> = new_text.split_whitespace().collect();
    if new_words.is_empty() {
        return (String::new(), prev_tail.to_string());
    }

    // 1) Partial word at boundary: prev="...оф", new="офис ..." -> emit "ис ..."
    let last_prev = prev_words.last().unwrap_or(&"");
    let first_new = new_words.first().unwrap_or(&"");
    if !last_prev.is_empty()
        && !first_new.is_empty()
        && first_new.starts_with(last_prev)
        && first_new.len() > last_prev.len()
    {
        let completion = &first_new[last_prev.len()..];
        let rest: Vec<&str> = new_words[1..].to_vec();
        let merged = if rest.is_empty() {
            completion.to_string()
        } else {
            format!("{} {}", completion, rest.join(" "))
        };
        let merged = merged.trim().to_string();
        let words: Vec<&str> = merged.split_whitespace().collect();
        let tail = if words.len() >= TAIL_WORDS {
            words[words.len().saturating_sub(TAIL_WORDS)..].join(" ")
        } else if merged.is_empty() {
            new_text.to_string()
        } else {
            merged.clone()
        };
        return (merged, tail);
    }

    // 2) Exact word overlap (longest match first)
    let n_max = prev_words.len().min(new_words.len());
    let mut best_len = 0usize;
    for n in (1..=n_max).rev() {
        let prev_suffix: String = prev_words[prev_words.len() - n..].join(" ");
        let new_prefix: String = new_words[..n].join(" ");
        if prev_suffix == new_prefix {
            best_len = n;
            break;
        }
    }

    let merged = if best_len > 0 {
        new_words[best_len..].join(" ")
    } else {
        new_text.to_string()
    };
    let merged = merged.trim().to_string();
    let words: Vec<&str> = merged.split_whitespace().collect();
    let tail = if words.len() >= TAIL_WORDS {
        words[words.len().saturating_sub(TAIL_WORDS)..].join(" ")
    } else if merged.is_empty() {
        new_text.to_string()
    } else {
        merged.clone()
    };
    (merged, tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_prev() {
        let (m, t) = merge_overlap("", "hello world");
        assert_eq!(m, "hello world");
        assert_eq!(t, "hello world");
    }

    #[test]
    fn test_empty_new() {
        let (m, t) = merge_overlap("prev tail", "");
        assert_eq!(m, "");
        assert_eq!(t, "prev tail");
    }

    #[test]
    fn test_exact_overlap() {
        let (m, _) = merge_overlap("one two three", "two three four");
        assert_eq!(m, "four");
    }

    #[test]
    fn test_partial_word_overlap() {
        let (m, _) = merge_overlap("привет оф", "офис открыт");
        assert_eq!(m, "ис открыт");
    }

    #[test]
    fn test_single_word_prev() {
        let (m, t) = merge_overlap("hello", "hello world");
        assert_eq!(m, "world");
        assert_eq!(t, "world"); // tail = last words of merged
    }

    #[test]
    fn test_no_overlap() {
        let (m, _) = merge_overlap("one two", "three four");
        assert_eq!(m, "three four");
    }

    #[test]
    fn test_prev_empty_trimmed() {
        let (m, t) = merge_overlap("   ", "new text");
        assert_eq!(m, "new text");
        assert_eq!(t, "new text");
    }
}
