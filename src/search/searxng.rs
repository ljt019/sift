use std::collections::HashSet;

use serde::Deserialize;
use tokio::time::{Instant, sleep_until};
use url::Url;

use crate::error::{AppError, Result};
use crate::state::AppState;

#[derive(Debug)]
pub struct Hit {
    pub title: String,
    pub url: Url,
    pub date: Option<String>,
    pub snippet: String,
}

pub async fn search(state: &AppState, query: &str, limit: usize) -> Result<Vec<Hit>> {
    let _permit = state
        .searxng_permits
        .acquire()
        .await
        .expect("SearXNG semaphore is never closed");
    wait_for_rate_limit(state).await;

    let url = search_url(
        &state.config.searxng_url,
        query,
        &state.config.searxng_categories,
    );

    let response = state
        .http
        .get(url)
        .timeout(state.config.searxng_timeout)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(AppError::SearchBackend)?;

    let body: SearxResponse = response.json().await.map_err(|error| {
        AppError::SearchBackendResponse(format!("could not decode JSON: {error}"))
    })?;

    Ok(into_hits(body, limit))
}

async fn wait_for_rate_limit(state: &AppState) {
    let mut next_request = state.searxng_next_request.lock().await;
    let now = Instant::now();
    if *next_request > now {
        sleep_until(*next_request).await;
    }
    *next_request = Instant::now() + state.config.searxng_min_interval;
}

fn search_url(base: &Url, query: &str, categories: &str) -> Url {
    let mut url = base.clone();
    url.set_path("/search");
    let mut parameters = url.query_pairs_mut();
    parameters
        .append_pair("q", query)
        .append_pair("format", "json")
        .append_pair("language", "en")
        .append_pair("safesearch", "0");
    if !categories.is_empty() {
        parameters.append_pair("categories", categories);
    }
    drop(parameters);
    url
}

fn into_hits(body: SearxResponse, limit: usize) -> Vec<Hit> {
    if limit == 0 {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut hits = Vec::with_capacity(limit.min(body.results.len()));

    for result in body.results {
        let Ok(mut url) = Url::parse(&result.url) else {
            tracing::debug!(url = result.url, "discarding invalid search result URL");
            continue;
        };
        if !matches!(url.scheme(), "http" | "https") {
            tracing::debug!(url = %url, "discarding unsupported search result URL");
            continue;
        }
        url.set_fragment(None);
        if !seen.insert(url.clone()) {
            continue;
        }

        hits.push(Hit {
            title: result.title.unwrap_or_else(|| "Untitled".into()),
            url,
            date: result
                .published_date
                .map(|date| date.chars().take(10).collect()),
            snippet: result.content.unwrap_or_default(),
        });
        if hits.len() == limit {
            break;
        }
    }

    hits
}

/// SearXNG fields other than `url` are optional in real responses.
#[derive(Deserialize)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<SearxResult>,
}

#[derive(Deserialize)]
struct SearxResult {
    url: String,
    title: Option<String>,
    content: Option<String>,
    published_date: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_deduplicates_and_limits_results() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://example.com/a#one","title":"A","content":"first"},
                    {"url":"https://example.com/a#two","title":"duplicate"},
                    {"url":"ftp://example.com/file"},
                    {"url":"not a URL"},
                    {"url":"https://example.com/b","published_date":"2026-08-01T12:00:00"},
                    {"url":"https://example.com/c"}
                ]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, 2);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url.as_str(), "https://example.com/a");
        assert_eq!(hits[0].snippet, "first");
        assert_eq!(hits[1].date.as_deref(), Some("2026-08-01"));
    }

    #[test]
    fn requests_selected_categories_without_overriding_engines() {
        let url = search_url(
            &Url::parse("http://localhost:8080").unwrap(),
            "axum timeout",
            "general,it,science,news",
        );
        let parameters = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(
            parameters.get("q").map(|value| value.as_ref()),
            Some("axum timeout")
        );
        assert_eq!(
            parameters.get("categories").map(|value| value.as_ref()),
            Some("general,it,science,news")
        );
        assert!(!parameters.contains_key("engines"));
    }
}
