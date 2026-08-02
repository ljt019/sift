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
    let mut remaining = max_characters;
    let mut documents = Vec::new();

    for (raw, selected_content, upstream_truncated) in sources {
        if remaining == 0 {
            break;
        }
        let (content, budget_truncated) = fit_content(&selected_content, remaining);
        if content.is_empty() {
            continue;
        }
        remaining -= content.chars().count();
        documents.push(Document {
            hit: raw.hit,
            content,
            full_length: raw.full_length,
            from_snippet: raw.from_snippet,
            truncated: upstream_truncated || budget_truncated,
        });
    }

    documents
}

fn fit_content(content: &str, max_characters: usize) -> (String, bool) {
    let content = content.trim();
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

    fn ranked(title: &str, content: &str, score: f32) -> RankedDocument {
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
            score,
        }
    }

    #[test]
    fn spends_budget_in_rank_order_on_character_boundaries() {
        let documents = allocate(
            vec![
                ranked("first", "aé中🙂", 1.0),
                ranked("second", "later", 0.5),
            ],
            3,
        );

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].hit.title, "first");
        assert_eq!(documents[0].content, "aé…");
        assert!(documents[0].truncated);
        assert_eq!(documents[0].content.chars().count(), 3);
    }

    #[test]
    fn one_character_budget_is_an_ellipsis() {
        let documents = allocate(vec![ranked("first", "content", 1.0)], 1);

        assert_eq!(documents[0].content, "…");
        assert!(documents[0].truncated);
    }

    #[test]
    fn preserves_upstream_truncation_without_budget_clipping() {
        let mut ranked = ranked("first", "selected", 1.0);
        ranked.truncated = true;
        let documents = allocate(vec![ranked], 100);

        assert_eq!(documents[0].content, "selected");
        assert!(documents[0].truncated);
    }

    #[test]
    fn unranked_results_keep_searxng_order() {
        let raw = vec![
            ranked("first", "first result", 0.0).raw,
            ranked("second", "second result", 0.0).raw,
        ];
        let documents = allocate_unranked(raw, 100);

        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].hit.title, "first");
        assert_eq!(documents[1].hit.title, "second");
        assert!(!documents[0].truncated);
        assert!(!documents[1].truncated);
    }
}
