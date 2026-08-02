use anyhow::{Result, ensure};

use crate::embeddings::DocumentTokenSpan;

#[derive(Debug)]
pub struct Chunk {
    pub document: usize,
    pub ordinal: usize,
    pub start: usize,
    pub end: usize,
    pub tokens: usize,
    pub text: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Assembled {
    pub content: String,
    pub truncated: bool,
}

pub fn split(document: usize, text: &str, spans: &[DocumentTokenSpan]) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::with_capacity(spans.len());
    let mut previous_start = None;

    for span in spans {
        ensure!(span.tokens > 0, "tokenized chunk has no tokens");
        ensure!(
            span.start < span.end
                && span.end <= text.len()
                && text.is_char_boundary(span.start)
                && text.is_char_boundary(span.end),
            "tokenized chunk has invalid text offsets {}..{}",
            span.start,
            span.end
        );
        if let Some(previous_start) = previous_start {
            ensure!(
                span.start > previous_start,
                "tokenized chunk offsets do not advance"
            );
        }
        previous_start = Some(span.start);

        let (start, end) = clean_bounds(text, span.start, span.end);
        if start == end {
            continue;
        }
        chunks.push(Chunk {
            document,
            ordinal: chunks.len(),
            start,
            end,
            tokens: span.tokens,
            text: text[start..end].to_owned(),
        });
    }

    Ok(chunks)
}

pub fn assemble(text: &str, chunks: &[&Chunk]) -> Assembled {
    if chunks.is_empty() {
        return Assembled {
            content: String::new(),
            truncated: contains_text(text),
        };
    }

    let mut ranges = chunks
        .iter()
        .map(|chunk| (chunk.start, chunk.end))
        .collect::<Vec<_>>();
    ranges.sort_unstable();

    let mut merged = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        match merged.last_mut() {
            Some((_, previous_end)) if start <= *previous_end => {
                *previous_end = (*previous_end).max(end);
            }
            _ => merged.push((start, end)),
        }
    }

    let ranges = merged
        .into_iter()
        .map(|(start, end)| {
            let (start, end) = clean_bounds(text, start, end);
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .fold(Vec::<(usize, usize)>::new(), |mut ranges, (start, end)| {
            match ranges.last_mut() {
                Some((_, previous_end)) if !contains_text(&text[*previous_end..start]) => {
                    *previous_end = end;
                }
                _ => ranges.push((start, end)),
            }
            ranges
        });

    let Some(&(first_start, _)) = ranges.first() else {
        return Assembled {
            content: String::new(),
            truncated: contains_text(text),
        };
    };
    let last_end = ranges.last().expect("ranges is not empty").1;
    let omitted_prefix = contains_text(&text[..first_start]);
    let omitted_suffix = contains_text(&text[last_end..]);
    let mut sections = Vec::with_capacity(ranges.len() * 2 + 2);
    if omitted_prefix {
        sections.push("…".to_owned());
    }
    for (index, &(start, end)) in ranges.iter().enumerate() {
        if index > 0 {
            sections.push("…".to_owned());
        }
        sections.push(text[start..end].to_owned());
    }
    if omitted_suffix {
        sections.push("…".to_owned());
    }

    Assembled {
        content: sections.join("\n\n"),
        truncated: omitted_prefix || ranges.len() > 1 || omitted_suffix,
    }
}

fn contains_text(text: &str) -> bool {
    text.chars().any(|character| !character.is_whitespace())
}

fn clean_bounds(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    let original = (start, end);
    trim_whitespace(text, &mut start, &mut end);

    if start > 0
        && start < end
        && !text[..start]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        && !text[start..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        while start < end {
            let character = text[start..].chars().next().expect("start is in bounds");
            start += character.len_utf8();
            if character.is_whitespace() {
                break;
            }
        }
    }
    if end < text.len()
        && start < end
        && !text[..end]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        && !text[end..].chars().next().is_some_and(char::is_whitespace)
    {
        while start < end {
            let character = text[..end].chars().next_back().expect("end is in bounds");
            end -= character.len_utf8();
            if character.is_whitespace() {
                break;
            }
        }
    }
    trim_whitespace(text, &mut start, &mut end);

    if start == end {
        let (mut start, mut end) = original;
        trim_whitespace(text, &mut start, &mut end);
        (start, end)
    } else {
        (start, end)
    }
}

fn trim_whitespace(text: &str, start: &mut usize, end: &mut usize) {
    while *start < *end {
        let character = text[*start..].chars().next().expect("start is in bounds");
        if !character.is_whitespace() {
            break;
        }
        *start += character.len_utf8();
    }
    while *start < *end {
        let character = text[..*end].chars().next_back().expect("end is in bounds");
        if !character.is_whitespace() {
            break;
        }
        *end -= character.len_utf8();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_at_character(text: &str, character: usize) -> usize {
        text.char_indices()
            .nth(character)
            .map_or(text.len(), |(offset, _)| offset)
    }

    fn span(text: &str, start: usize, end: usize, tokens: usize) -> DocumentTokenSpan {
        DocumentTokenSpan {
            start: byte_at_character(text, start),
            end: byte_at_character(text, end),
            tokens,
        }
    }

    #[test]
    fn builds_unicode_chunks_from_tokenizer_offsets() {
        let text = "aé中🙂bc";
        let spans = [span(text, 0, 4, 4), span(text, 3, 6, 3)];
        let chunks = split(2, text, &spans).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "aé中🙂");
        assert_eq!(chunks[1].text, "🙂bc");
        assert_eq!((chunks[1].document, chunks[1].ordinal), (2, 1));
        assert_eq!(chunks[0].tokens, 4);
    }

    #[test]
    fn assembles_overlapping_chunks_without_duplicate_text() {
        let text = "abcdefghij";
        let spans = [span(text, 0, 6, 6), span(text, 4, 10, 6)];
        let chunks = split(0, text, &spans).unwrap();

        assert_eq!(
            assemble(text, &[&chunks[0], &chunks[1]]),
            Assembled {
                content: text.into(),
                truncated: false,
            }
        );
    }

    #[test]
    fn marks_gaps_between_selected_chunks() {
        let text = "aaaa bbbb cccc";
        let spans = [span(text, 0, 4, 4), span(text, 10, 14, 4)];
        let chunks = split(0, text, &spans).unwrap();

        let assembled = assemble(text, &[&chunks[0], &chunks[1]]);

        assert_eq!(assembled.content, "aaaa\n\n…\n\ncccc");
        assert!(assembled.truncated);
    }

    #[test]
    fn marks_omitted_prefix_and_suffix() {
        let text = "prefix middle suffix";
        let spans = [span(text, 7, 13, 6)];
        let chunks = split(0, text, &spans).unwrap();
        let assembled = assemble(text, &[&chunks[0]]);

        assert_eq!(assembled.content, "…\n\nmiddle\n\n…");
        assert!(assembled.truncated);
    }

    #[test]
    fn whitespace_between_selected_chunks_is_not_an_omission() {
        let text = "aaaa   bbbb";
        let spans = [span(text, 0, 4, 4), span(text, 7, 11, 4)];
        let chunks = split(0, text, &spans).unwrap();
        let assembled = assemble(text, &[&chunks[0], &chunks[1]]);

        assert_eq!(assembled.content, text);
        assert!(!assembled.truncated);
    }

    #[test]
    fn removes_partial_words_at_tokenized_edges() {
        let text = "serialization next section continues";
        let spans = [DocumentTokenSpan {
            start: 10,
            end: 25,
            tokens: 5,
        }];
        let chunks = split(0, text, &spans).unwrap();

        assert_eq!(chunks[0].text, "next");
    }
}
