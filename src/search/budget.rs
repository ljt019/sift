use super::Document;
use super::Raw;
use super::rank::RankedDocument;

pub fn allocate(ranked: Vec<RankedDocument>, max_characters: usize) -> Vec<Document> {
    allocate_content(
        ranked
            .into_iter()
            .map(|ranked| (ranked.raw, ranked.content, ranked.truncated)),
        max_characters,
    )
}

pub fn allocate_unranked(raw: Vec<Raw>, max_characters: usize) -> Vec<Document> {
    allocate_content(
        raw.into_iter().map(|mut raw| {
            let content = std::mem::take(&mut raw.content);
            (raw, content, false)
        }),
        max_characters,
    )
}

fn allocate_content(
    sources: impl IntoIterator<Item = (Raw, String, bool)>,
    max_characters: usize,
) -> Vec<Document> {
    let sources = sources.into_iter().collect::<Vec<_>>();
    let lengths = sources
        .iter()
        .map(|(_, content, _)| content.trim().chars().count())
        .collect::<Vec<_>>();
    let allocations = fair_allocations(&lengths, max_characters);

    sources
        .into_iter()
        .zip(allocations)
        .filter_map(
            |((raw, selected_content, upstream_truncated), allocation)| {
                let (content, budget_truncated) = fit_content(&selected_content, allocation);
                if content.is_empty() {
                    return None;
                }
                Some(Document {
                    hit: raw.hit,
                    content,
                    full_length: raw.full_length,
                    from_snippet: raw.from_snippet,
                    truncated: upstream_truncated || budget_truncated,
                })
            },
        )
        .collect()
}

fn fair_allocations(lengths: &[usize], max_characters: usize) -> Vec<usize> {
    let mut allocations = vec![0; lengths.len()];
    let mut remaining = max_characters;
    let mut pending = (0..lengths.len()).collect::<Vec<_>>();

    while !pending.is_empty() {
        let pending_count = pending.len();
        let share = remaining / pending_count;
        let saturated = pending
            .iter()
            .copied()
            .filter(|&index| lengths[index] <= share)
            .collect::<Vec<_>>();
        if saturated.is_empty() {
            for (position, index) in pending.into_iter().enumerate() {
                allocations[index] = share + usize::from(position < remaining % pending_count);
            }
            break;
        }

        for index in &saturated {
            allocations[*index] = lengths[*index];
            remaining -= lengths[*index];
        }
        pending.retain(|index| !saturated.contains(index));
    }

    allocations
}

fn fit_content(content: &str, max_characters: usize) -> (String, bool) {
    let content = content.trim();
    if max_characters == 0 {
        return (String::new(), !content.is_empty());
    }
    if content.chars().count() <= max_characters {
        return (content.to_owned(), false);
    }
    if max_characters == 1 {
        return ("…".to_owned(), true);
    }

    let mut clipped = content.chars().take(max_characters - 1).collect::<String>();
    clipped.truncate(clipped.trim_end().len());
    clipped.push('…');
    (clipped, true)
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::search::Raw;
    use crate::search::searxng::Hit;

    fn ranked(title: &str, content: &str) -> RankedDocument {
        RankedDocument {
            raw: Raw {
                hit: Hit {
                    title: title.into(),
                    url: Url::parse("https://example.com").unwrap(),
                    date: None,
                    snippet: String::new(),
                },
                content: content.into(),
                full_length: content.chars().count(),
                from_snippet: false,
            },
            content: content.into(),
            truncated: false,
        }
    }

    #[test]
    fn shares_tiny_budgets_on_character_boundaries() {
        let documents = allocate(
            vec![ranked("first", "aé中🙂"), ranked("second", "later")],
            3,
        );

        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].hit.title, "first");
        assert_eq!(documents[0].content, "a…");
        assert_eq!(documents[1].content, "…");
        assert!(documents[0].truncated);
        assert!(documents[1].truncated);
        assert_eq!(
            documents
                .iter()
                .map(|document| document.content.chars().count())
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn one_character_budget_is_an_ellipsis() {
        let documents = allocate(vec![ranked("first", "content")], 1);

        assert_eq!(documents[0].content, "…");
        assert!(documents[0].truncated);
    }

    #[test]
    fn preserves_upstream_truncation_without_budget_clipping() {
        let mut ranked = ranked("first", "selected");
        ranked.truncated = true;
        let documents = allocate(vec![ranked], 100);

        assert_eq!(documents[0].content, "selected");
        assert!(documents[0].truncated);
    }

    #[test]
    fn unranked_results_keep_searxng_order() {
        let raw = vec![
            ranked("first", "first result").raw,
            ranked("second", "second result").raw,
        ];
        let documents = allocate_unranked(raw, 100);

        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].hit.title, "first");
        assert_eq!(documents[1].hit.title, "second");
        assert!(!documents[0].truncated);
        assert!(!documents[1].truncated);
    }

    #[test]
    fn long_documents_share_the_budget() {
        let documents = allocate(
            vec![
                ranked("first", "aaaaaaaaaa"),
                ranked("second", "bbbbbbbbbb"),
            ],
            10,
        );

        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].content, "aaaa…");
        assert_eq!(documents[1].content, "bbbb…");
    }

    #[test]
    fn unused_share_is_redistributed() {
        let documents = allocate(
            vec![ranked("short", "abc"), ranked("long", "bbbbbbbbbb")],
            10,
        );

        assert_eq!(documents[0].content, "abc");
        assert_eq!(documents[1].content, "bbbbbb…");
        assert_eq!(
            documents
                .iter()
                .map(|document| document.content.chars().count())
                .sum::<usize>(),
            10
        );
    }
}
