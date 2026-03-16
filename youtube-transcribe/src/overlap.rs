//! Text overlap merging at chunk boundaries (client-side deduplication).

const TAIL_WORDS: usize = 10;

fn words_match(prev_suffix: &[&str], new_prefix: &[&str]) -> bool {
    prev_suffix.len() == new_prefix.len()
        && prev_suffix.iter().zip(new_prefix.iter()).all(|(a, b)| {
            let a = a.trim_matches(|c: char| ",.!?;:".contains(c));
            let b = b.trim_matches(|c: char| ",.!?;:".contains(c));
            !a.is_empty() && !b.is_empty() && a.to_lowercase() == b.to_lowercase()
        })
}

/// Remove duplication at chunk boundaries.
/// Returns (merged_text_to_emit, new_tail, no_space_before) where no_space_before
/// means the emit continues the previous word (e.g. "умны" + "е" -> "умные").
pub fn merge_overlap(prev_tail: &str, new_text: &str) -> (String, String, bool) {
    let new_text = new_text.trim();
    if new_text.is_empty() {
        return (String::new(), prev_tail.to_string(), false);
    }
    if prev_tail.trim().is_empty() {
        let words: Vec<&str> = new_text.split_whitespace().collect();
        let tail = if words.len() >= TAIL_WORDS {
            words[words.len().saturating_sub(TAIL_WORDS)..].join(" ")
        } else {
            new_text.to_string()
        };
        return (new_text.to_string(), tail, false);
    }

    let prev_words: Vec<&str> = prev_tail.split_whitespace().collect();
    let new_words: Vec<&str> = new_text.split_whitespace().collect();
    if new_words.is_empty() {
        return (String::new(), prev_tail.to_string(), false);
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
        return (merged, tail, true); // no space: "умны" + "е" -> "умные"
    }

    // 2) Word overlap (longest match first, case-insensitive, punctuation-ignoring)
    let n_max = prev_words.len().min(new_words.len());
    let mut best_len = 0usize;
    for n in (1..=n_max).rev() {
        let prev_suffix = &prev_words[prev_words.len() - n..];
        let new_prefix = &new_words[..n];
        if words_match(prev_suffix, new_prefix) {
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

    // Server may send pre-merged "е даже" (completion of "умны"->"умные"). Glue without space.
    let no_space = best_len == 0 && !merged.is_empty() && !prev_words.is_empty() && {
        let first_emit = words
            .first()
            .map(|s| s.trim_matches(|c: char| ",.!?;:".contains(c)));
        let last_prev = prev_words
            .last()
            .unwrap_or(&"")
            .trim_matches(|c: char| ",.!?;:".contains(c));
        first_emit.map_or(false, |f| {
            f.chars().count() == 1
                && f.chars()
                    .next()
                    .map_or(false, |c| c.is_alphabetic() && !c.is_ascii())
                && last_prev.chars().count() >= 3
                && last_prev
                    .chars()
                    .last()
                    .map_or(false, |c| c.is_alphabetic())
        })
    };

    (merged, tail, no_space)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merge_chunks(chunks: &[&str]) -> String {
        let mut merged = String::new();
        let mut prev_tail = String::new();
        for text in chunks {
            let (to_emit, new_tail, no_space_before) = merge_overlap(&prev_tail, text);
            if !merged.is_empty() && !to_emit.is_empty() && !no_space_before {
                merged.push(' ');
            }
            merged.push_str(&to_emit);
            prev_tail = new_tail;
        }
        merged
    }

    #[test]
    fn test_umny_umnye() {
        // "умны" / "умные" - partial word overlap, "е" glues without space
        let r = merge_chunks(&["Мы все недостаточно умны", "умные даже ученые ракетчик"]);
        assert!(!r.contains("умны умные"), "should not duplicate: {}", r);
        assert!(r.contains("умные"), "should have умные: {}", r);
        assert!(!r.contains("умны е "), "е should glue to умны: {}", r);
    }

    #[test]
    fn test_umny_umnye_server_premerged() {
        // Server sends "е даже" (pre-merged), client must glue "умны" + "е"
        let r = merge_chunks(&["Мы все недостаточно умны", "е даже ученые ракетчик"]);
        assert!(!r.contains("умны е "), "е should glue: {}", r);
        assert!(r.contains("умные"), "should glue to умные: {}", r);
    }

    #[test]
    fn test_malchikom_laboratorom() {
        let r = merge_chunks(&[
            "прошу говорите со мной как с мальчишкой или лаборатором",
            "С мальчишкой или лаборатором большой успех не является",
        ]);
        assert!(
            !r.contains("лаборатором С мальчишкой"),
            "should not duplicate phrase: {}",
            r
        );
        assert!(
            r.contains("большой успех"),
            "should have continuation: {}",
            r
        );
    }

    #[test]
    fn test_exact_word_overlap() {
        let r = merge_chunks(&["привет как дела", "как дела отлично"]);
        assert_eq!(r, "привет как дела отлично");
    }

    #[test]
    fn test_full_duplicate_chunk() {
        let r = merge_chunks(&["Привет как дела", "Привет как дела отлично"]);
        assert_eq!(r, "Привет как дела отлично");
    }
}
