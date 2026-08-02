mod budget;
mod cache;
mod chunk;
mod debug;
mod extract;
mod fetch;
mod rank;
mod searxng;
mod stackexchange;

use anyhow::Context;
use tokio::task::JoinSet;

use crate::state::AppState;

pub(crate) use cache::PageCache;
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
    content: String,
    full_length: usize,
    from_snippet: bool,
}

pub async fn run(state: &AppState, query: &str, params: &Params) -> crate::Result<Vec<Document>> {
    let hits = searxng::search(state, query, params.num_results).await?;
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
    let raw = raw.into_iter().map(|(_, raw)| raw).collect();

    match &state.embeddings {
        Some(embeddings) => {
            let ranked = rank::select(embeddings, query, raw)
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
async fn resolve(state: &AppState, hit: Hit, specialized_content: Option<&str>) -> Raw {
    if let Some(content) = specialized_content {
        return choose_content(hit, Some(content.to_owned()));
    }

    let url = hit.url.clone();
    let extracted = state
        .page_cache
        .get_or_extract(&url, || fetch_and_extract(state, &url))
        .await
        .map(|content| content.to_string());

    choose_content(hit, extracted)
}

async fn fetch_and_extract(state: &AppState, url: &url::Url) -> Option<String> {
    match fetch::get(state, url).await {
        Ok(html) => {
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

fn choose_content(hit: Hit, extracted: Option<String>) -> Raw {
    let snippet_length = hit.snippet.trim().chars().count();
    match extracted {
        Some(content) if content.trim().chars().count() > snippet_length => {
            let content = content.trim().to_owned();
            Raw {
                hit,
                full_length: content.chars().count(),
                content,
                from_snippet: false,
            }
        }
        _ => {
            let content = hit.snippet.trim().to_owned();
            Raw {
                hit,
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
    fn failed_or_weaker_extraction_degrades_to_snippet() {
        let failed = choose_content(hit("use serde_json"), None);
        assert!(failed.from_snippet);
        assert_eq!(failed.content, "use serde_json");

        let weaker = choose_content(hit("a useful search snippet"), Some("short".into()));
        assert!(weaker.from_snippet);
        assert_eq!(weaker.content, "a useful search snippet");
    }

    #[test]
    fn stronger_extraction_replaces_snippet() {
        let raw = choose_content(
            hit("brief snippet"),
            Some("a substantially longer extracted page body".into()),
        );

        assert!(!raw.from_snippet);
        assert_eq!(raw.full_length, 42);
        assert_eq!(raw.content, "a substantially longer extracted page body");
    }
}
