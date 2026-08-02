mod budget;
mod cache;
mod candidate;
mod chunk;
mod debug;
mod dedup;
mod evidence;
mod extract;
mod fetch;
mod github;
mod rank;
mod searxng;
mod source_follow;
mod stackexchange;

use anyhow::Context;
use tokio::task::JoinSet;

use crate::state::AppState;

pub(crate) use cache::{PageCache, SearchCache};
pub(crate) use fetch::PublicDnsResolver;
pub use searxng::Hit;
pub(crate) use stackexchange::StackExchangeClient;

pub struct Params {
    num_results: usize,
    context_max_characters: usize,
}

impl Params {
    pub fn new(num_results: usize, context_max_characters: usize) -> Self {
        Self {
            num_results,
            context_max_characters,
        }
    }
}

pub struct Document {
    pub hit: Hit,
    pub content: String,
    pub full_length: usize,
    pub from_snippet: bool,
    pub truncated: bool,
}

struct Raw {
    hit: Hit,
    source_priority: bool,
    upstream_consensus: bool,
    content: String,
    full_length: usize,
    from_snippet: bool,
}

fn exact_identifier_query(query: &str) -> Option<String> {
    if let Some((start, end, diagnostic)) = quoted_diagnostic(query) {
        return Some(compact_diagnostic_query(
            diagnostic,
            query[..start]
                .split_whitespace()
                .chain(query[end..].split_whitespace()),
        ));
    }
    if let Some(code) = compiler_error_code(query) {
        return Some(compact_diagnostic_query(
            code,
            query.split_whitespace().filter(|word| {
                word.trim_matches(|character: char| !character.is_alphanumeric()) != code
            }),
        ));
    }

    let words = query.split_whitespace().collect::<Vec<_>>();
    let (index, identifier) = words.iter().enumerate().find_map(|(index, word)| {
        let token = word.trim_matches(|character: char| {
            !character.is_alphanumeric() && character != '_' && character != ':'
        });
        let alphanumeric_count = token
            .chars()
            .filter(|character| character.is_alphanumeric())
            .count();
        ((token.contains('_') || token.contains("::")) && alphanumeric_count >= 2)
            .then_some((index, token))
    })?;
    let companion = (!identifier.contains("::"))
        .then(|| {
            words[index + 1..]
                .iter()
                .chain(words[..index].iter().rev())
                .map(|word| word.trim_matches(|character: char| !character.is_alphanumeric()))
                .find(|word| {
                    word.len() >= 3
                        && !matches!(
                            word.to_ascii_lowercase().as_str(),
                            "and"
                                | "error"
                                | "for"
                                | "from"
                                | "import"
                                | "into"
                                | "rust"
                                | "the"
                                | "using"
                                | "with"
                        )
                })
        })
        .flatten();

    Some(match companion {
        Some(companion) => format!("\"{identifier}\" {companion}"),
        None => format!("\"{identifier}\""),
    })
}

fn quoted_diagnostic(query: &str) -> Option<(usize, usize, &str)> {
    let mut quotes = query.match_indices('"').map(|(index, _)| index);
    while let (Some(start), Some(end)) = (quotes.next(), quotes.next()) {
        let phrase = &query[start + 1..end];
        let lower = phrase.to_ascii_lowercase();
        let tokens = lower
            .split(|character: char| !character.is_alphanumeric())
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        let strong_marker = tokens.iter().any(|token| {
            matches!(
                *token,
                "cannot"
                    | "denied"
                    | "error"
                    | "exception"
                    | "excessively"
                    | "failed"
                    | "failure"
                    | "invalid"
                    | "mismatch"
                    | "panic"
                    | "unable"
                    | "unauthorized"
                    | "undefined"
                    | "unexpected"
                    | "unresolved"
                    | "unsupported"
            )
        });
        let not_diagnostic = [
            "must not",
            "not authorized",
            "not found",
            "not implemented",
            "not satisfied",
            "not supported",
        ]
        .iter()
        .any(|marker| lower.contains(marker));
        if tokens.len() >= 3 && (strong_marker || not_diagnostic) {
            return Some((start, end + 1, phrase));
        }
    }
    None
}

fn compiler_error_code(query: &str) -> Option<&str> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .find(|token| {
            let letters = token
                .chars()
                .take_while(|character| character.is_ascii_uppercase())
                .count();
            let prefix = &token[..letters];
            let digits = token.len().saturating_sub(letters);
            matches!(prefix, "E" | "C" | "CS" | "FS" | "TS" | "GHC" | "LNK")
                && (3..=5).contains(&digits)
                && token[letters..]
                    .chars()
                    .all(|character| character.is_ascii_digit())
        })
}

fn compact_diagnostic_query<'a>(
    diagnostic: &str,
    context: impl Iterator<Item = &'a str>,
) -> String {
    let context = context
        .map(|word| {
            word.trim_matches(|character: char| {
                !character.is_alphanumeric()
                    && !matches!(character, '_' | ':' | '-' | '.' | '+' | '#')
            })
        })
        .filter(|word| {
            word.chars().any(|character| character.is_alphanumeric())
                && !matches!(
                    word.to_ascii_lowercase().as_str(),
                    "a" | "after"
                        | "an"
                        | "and"
                        | "before"
                        | "because"
                        | "error"
                        | "example"
                        | "fix"
                        | "for"
                        | "from"
                        | "in"
                        | "into"
                        | "is"
                        | "of"
                        | "on"
                        | "the"
                        | "to"
                        | "using"
                        | "when"
                        | "while"
                        | "with"
                        | "workaround"
                )
        })
        .take(8)
        .collect::<Vec<_>>();
    if context.is_empty() {
        format!("\"{diagnostic}\"")
    } else {
        format!("\"{diagnostic}\" {}", context.join(" "))
    }
}

pub async fn run(state: &AppState, query: &str, params: &Params) -> crate::Result<Vec<Document>> {
    let candidate_limit = (params.num_results * 4).clamp(params.num_results, 32);
    let hits = searxng::search(state, query, candidate_limit).await?;
    let hits = match &state.embeddings {
        Some(embeddings) => {
            // SearXNG can only rank titles and snippets; the answer-bearing
            // passage may live in a result whose summary is mediocre. Fetch a
            // wider shortlist before full-text ranking so half of the search
            // pool is not discarded on metadata alone.
            let shortlist_limit = (params.num_results * 3).min(candidate_limit);
            candidate::select(embeddings, query, hits, shortlist_limit)
                .await
                .context("failed to shortlist search candidates")?
        }
        None => hits.into_iter().take(params.num_results).collect(),
    };
    let stackexchange_urls = hits
        .iter()
        .enumerate()
        .filter(|(_, hit)| stackexchange::recognizes(&hit.url))
        .map(|(index, hit)| (index, hit.url.clone()))
        .collect::<Vec<_>>();
    let mut tasks = JoinSet::new();
    let mut stackexchange_hits = Vec::new();
    for (index, hit) in hits.into_iter().enumerate() {
        if stackexchange::recognizes(&hit.url) {
            stackexchange_hits.push((index, hit));
            continue;
        }
        let state = state.clone();
        tasks.spawn(async move { (index, resolve(&state, hit, None).await) });
    }

    // Generic page fetches above run while this batched API request is in
    // flight, so one slow source does not delay unrelated results.
    let mut stackexchange = state.stackexchange.fetch(&stackexchange_urls).await;
    for (index, hit) in stackexchange_hits {
        let state = state.clone();
        let specialized_content = stackexchange.remove(&index);
        tasks.spawn(async move {
            (
                index,
                resolve(&state, hit, specialized_content.as_deref()).await,
            )
        });
    }

    let mut raw = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(result) => raw.push(result),
            Err(error) => tracing::error!(error = ?error, "page resolution task failed"),
        }
    }
    raw.sort_by_key(|(index, _)| *index);
    let mut raw = raw.into_iter().map(|(_, raw)| raw).collect::<Vec<_>>();
    if let Some(candidate) = source_follow::cited_candidate(
        query,
        raw.iter()
            .map(|result| (&result.hit.url, result.content.as_str())),
    ) {
        let mut recovered = resolve(state, candidate, None).await;
        if !recovered.from_snippet {
            recovered.hit.title =
                source_follow::recovered_title(&recovered.content, &recovered.hit.title);
            tracing::debug!(url = %recovered.hit.url, "recovered cited first-party source");
            raw.push(recovered);
        }
    }
    let raw = dedup::refine(query, raw);

    match &state.embeddings {
        Some(embeddings) => {
            let ranked = rank::select(embeddings, query, raw, params.num_results)
                .await
                .context("failed to rank search results")?;
            Ok(budget::allocate(ranked, params.context_max_characters))
        }
        None => Ok(budget::allocate_unranked(
            raw,
            params.context_max_characters,
        )),
    }
}

/// Page-level failures are intentionally contained here. Search itself is
/// required, but an individual page always degrades to its SearXNG snippet.
async fn resolve(
    state: &AppState,
    candidate: searxng::SearchCandidate,
    specialized_content: Option<&str>,
) -> Raw {
    let searxng::SearchCandidate {
        mut hit,
        source_priority,
        upstream_consensus,
        fetch_urls,
    } = candidate;
    if let Some(content) = specialized_content {
        return choose_content(
            hit,
            source_priority,
            upstream_consensus,
            Some(content.to_owned()),
        );
    }

    if stackexchange::recognizes(&hit.url)
        && let Some(printer_url) = stackexchange::printer_url(&hit.url)
    {
        let extracted = state
            .page_cache
            .get_or_extract(&printer_url, || async {
                fetch_and_extract(state, &printer_url)
                    .await
                    .filter(|content| !stackexchange::printer_unavailable(content))
            })
            .await
            .map(|content| content.to_string());
        if extracted.is_some() {
            return choose_content(hit, source_priority, upstream_consensus, extracted);
        }
    }

    let snippet_length = hit.snippet.trim().chars().count();
    for url in fetch_urls {
        let extracted = state
            .page_cache
            .get_or_extract(&url, || fetch_and_extract(state, &url))
            .await
            .map(|content| content.to_string());
        if extracted
            .as_deref()
            .is_some_and(|content| content.trim().chars().count() > snippet_length)
        {
            hit.url = url;
            return choose_content(hit, source_priority, upstream_consensus, extracted);
        }
    }

    choose_content(hit, source_priority, upstream_consensus, None)
}

async fn fetch_and_extract(state: &AppState, url: &url::Url) -> Option<String> {
    match fetch::get(state, url).await {
        Ok(fetch::Page::Html(html)) => {
            let debug_html = state.config.sift_debug_dir.is_some().then(|| html.clone());
            let extracted = extract_bounded(state, html, url.clone()).await;
            debug::capture(
                state.config.sift_debug_dir.as_deref(),
                url,
                debug_html.as_deref().unwrap_or_default(),
                extracted.as_deref().unwrap_or_default(),
            )
            .await;
            extracted
        }
        Ok(fetch::Page::Text(text)) => {
            let extracted = text.trim();
            let extracted = (!extracted.is_empty()).then(|| extracted.to_owned());
            debug::capture(
                state.config.sift_debug_dir.as_deref(),
                url,
                &text,
                extracted.as_deref().unwrap_or_default(),
            )
            .await;
            extracted
        }
        Ok(fetch::Page::Pdf(pdf)) => extract_pdf_bounded(state, pdf, url.clone()).await,
        Err(error) => {
            let failure = error.diagnostic();
            tracing::debug!(
                url = %url,
                final_url = failure.final_url.as_ref().map(url::Url::as_str),
                status = failure.status,
                content_type = failure.content_type.as_deref(),
                headers = ?failure.headers,
                body_preview = failure.body_preview.as_deref(),
                error = ?error,
                "page fetch failed; using snippet"
            );
            debug::capture_fetch_failure(state.config.sift_debug_dir.as_deref(), url, failure)
                .await;
            None
        }
    }
}

fn choose_content(
    hit: Hit,
    source_priority: bool,
    upstream_consensus: bool,
    extracted: Option<String>,
) -> Raw {
    let snippet_length = hit.snippet.trim().chars().count();
    match extracted {
        Some(content) if content.trim().chars().count() > snippet_length => {
            let content = content.trim().to_owned();
            Raw {
                hit,
                source_priority,
                upstream_consensus,
                full_length: content.chars().count(),
                content,
                from_snippet: false,
            }
        }
        _ => {
            let content = hit.snippet.trim().to_owned();
            Raw {
                hit,
                source_priority,
                upstream_consensus,
                full_length: content.chars().count(),
                content,
                from_snippet: true,
            }
        }
    }
}

async fn extract_bounded(state: &AppState, html: String, url: url::Url) -> Option<String> {
    let permit = match state.extract_permits.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(error) => {
            tracing::error!(error = %error, "extraction semaphore closed; using snippet");
            return None;
        }
    };
    let extraction_url = url.clone();
    match tokio::task::spawn_blocking(move || {
        // Keep the permit in the blocking task so request cancellation cannot
        // release capacity while extraction is still consuming a thread.
        let _permit = permit;
        extract::run(&html, &extraction_url)
    })
    .await
    {
        Ok(Ok(content)) => Some(content),
        Ok(Err(error)) => {
            tracing::debug!(url = %url, error = ?error, "page extraction failed; using snippet");
            None
        }
        Err(error) => {
            tracing::error!(url = %url, error = ?error, "page extraction task failed; using snippet");
            None
        }
    }
}

async fn extract_pdf_bounded(state: &AppState, pdf: Vec<u8>, url: url::Url) -> Option<String> {
    let permit = match state.extract_permits.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(error) => {
            tracing::error!(error = %error, "extraction semaphore closed; using snippet");
            return None;
        }
    };
    let extraction = tokio::task::spawn_blocking(move || {
        // PDF parsing is CPU-bound and shares the same bounded pool as HTML
        // extraction. A parser panic remains contained to this task.
        let _permit = permit;
        pdf_extract::extract_text_from_mem(&pdf)
    })
    .await;

    match extraction {
        Ok(Ok(content)) if !content.trim().is_empty() => Some(content.trim().to_owned()),
        Ok(Ok(_)) => {
            tracing::debug!(url = %url, "PDF extractor returned no content; using snippet");
            None
        }
        Ok(Err(error)) => {
            tracing::debug!(url = %url, error = ?error, "PDF extraction failed; using snippet");
            None
        }
        Err(error) => {
            tracing::error!(url = %url, error = ?error, "PDF extraction task failed; using snippet");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;

    fn hit(snippet: &str) -> Hit {
        Hit {
            title: "Result".into(),
            url: Url::parse("https://example.com/result").unwrap(),
            date: None,
            snippet: snippet.into(),
        }
    }

    #[test]
    fn exact_identifier_query_ignores_generic_placeholders() {
        assert_eq!(exact_identifier_query("_"), None);
        assert_eq!(exact_identifier_query("Result<Infallible, _>"), None);
        assert_eq!(
            exact_identifier_query(
                "Rust E0277 the trait FromResidual<Result<Infallible, _>> is not implemented"
            )
            .as_deref(),
            Some("\"E0277\" Rust trait FromResidual<Result<Infallible not implemented")
        );
    }

    #[test]
    fn exact_identifier_query_preserves_real_identifiers() {
        assert_eq!(
            exact_identifier_query("invalid_reference_casting tokenizers Rust 1.73").as_deref(),
            Some("\"invalid_reference_casting\" tokenizers")
        );
        assert_eq!(
            exact_identifier_query("rust unresolved import tower_http::timeout").as_deref(),
            Some("\"tower_http::timeout\"")
        );
    }

    #[test]
    fn exact_identifier_query_preserves_quoted_diagnostics_with_context() {
        assert_eq!(
            exact_identifier_query(
                "Rust E0277 \"FromResidual<Result<Infallible, _>> is not implemented\" using ? with a custom error enum"
            )
            .as_deref(),
            Some(
                "\"FromResidual<Result<Infallible, _>> is not implemented\" Rust E0277 custom enum"
            )
        );
        assert_eq!(
            exact_identifier_query(
                "PostgreSQL \"cached plan must not change result type\" after ALTER TABLE prepared statement fix"
            )
            .as_deref(),
            Some(
                "\"cached plan must not change result type\" PostgreSQL ALTER TABLE prepared statement"
            )
        );
        assert_eq!(
            exact_identifier_query(
                "TypeScript TS2589 \"Type instantiation is excessively deep and possibly infinite\" Zod recursive schema workaround"
            )
            .as_deref(),
            Some(
                "\"Type instantiation is excessively deep and possibly infinite\" TypeScript TS2589 Zod recursive schema"
            )
        );
    }

    #[test]
    fn exact_identifier_query_focuses_compiler_codes_but_not_ordinary_quotes() {
        assert_eq!(
            exact_identifier_query("Rust E0502 cannot borrow mutable because immutable example")
                .as_deref(),
            Some("\"E0502\" Rust cannot borrow mutable immutable")
        );
        assert_eq!(
            exact_identifier_query("books containing \"not all who wander are lost\""),
            None
        );
        assert_eq!(
            exact_identifier_query("PostgreSQL \"prepared statements\" performance"),
            None
        );
        assert_eq!(exact_identifier_query("HTTP 404 response"), None);
    }

    #[test]
    fn failed_or_weaker_extraction_degrades_to_snippet() {
        let failed = choose_content(hit("use serde_json"), false, false, None);
        assert!(failed.from_snippet);
        assert_eq!(failed.content, "use serde_json");

        let weaker = choose_content(
            hit("a useful search snippet"),
            false,
            false,
            Some("short".into()),
        );
        assert!(weaker.from_snippet);
        assert_eq!(weaker.content, "a useful search snippet");
    }

    #[test]
    fn stronger_extraction_replaces_snippet() {
        let raw = choose_content(
            hit("brief snippet"),
            false,
            false,
            Some("a substantially longer extracted page body".into()),
        );

        assert!(!raw.from_snippet);
        assert_eq!(raw.full_length, 42);
        assert_eq!(raw.content, "a substantially longer extracted page body");
    }
}
