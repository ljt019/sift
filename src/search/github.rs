use std::time::Duration;

use serde::Deserialize;
use tokio::time::{Instant, sleep_until};
use url::Url;

use super::{Hit, exact_identifier_query};
use crate::state::AppState;

const SEARCH_INTERVAL: Duration = Duration::from_secs(6);
const SEARCH_TIMEOUT: Duration = Duration::from_secs(8);

pub async fn search(state: &AppState, query: &str, limit: usize) -> Vec<Hit> {
    let Some(query) = exact_identifier_query(query) else {
        return Vec::new();
    };
    if limit == 0 {
        return Vec::new();
    }

    wait_for_rate_limit(state).await;
    match search_issues(state, &query, limit).await {
        Ok(hits) => {
            tracing::debug!(
                query,
                results = hits.len(),
                "received GitHub issue candidates"
            );
            hits
        }
        Err(error) => {
            tracing::debug!(error = ?error, "GitHub issue search failed; continuing without exact-identifier candidates");
            Vec::new()
        }
    }
}

async fn wait_for_rate_limit(state: &AppState) {
    let mut next_request = state.github_next_request.lock().await;
    let now = Instant::now();
    if *next_request > now {
        sleep_until(*next_request).await;
    }
    *next_request = Instant::now() + SEARCH_INTERVAL;
}

async fn search_issues(state: &AppState, query: &str, limit: usize) -> anyhow::Result<Vec<Hit>> {
    let mut url = Url::parse("https://api.github.com/search/issues").expect("valid literal");
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("per_page", &limit.clamp(1, 10).to_string());
    let response: SearchResponse = state
        .http
        .get(url)
        .header("accept", "application/vnd.github+json")
        .header("x-github-api-version", "2022-11-28")
        .timeout(SEARCH_TIMEOUT)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok(response
        .items
        .into_iter()
        .filter_map(Issue::into_hit)
        .take(limit)
        .collect())
}

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    items: Vec<Issue>,
}

#[derive(Deserialize)]
struct Issue {
    title: String,
    html_url: String,
    created_at: String,
    body: Option<String>,
}

impl Issue {
    fn into_hit(self) -> Option<Hit> {
        let url = Url::parse(&self.html_url).ok()?;
        let snippet = self
            .body
            .as_deref()
            .map(|body| body.chars().take(800).collect::<String>())
            .unwrap_or_default();
        Some(Hit {
            title: self.title,
            url,
            date: Some(self.created_at.chars().take(10).collect()),
            snippet,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focuses_exact_identifiers_without_triggering_ordinary_queries() {
        assert_eq!(
            exact_identifier_query("invalid_reference_casting tokenizers Rust 1.73").as_deref(),
            Some("\"invalid_reference_casting\" tokenizers")
        );
        assert_eq!(
            exact_identifier_query("rust unresolved import tower_http::timeout").as_deref(),
            Some("\"tower_http::timeout\"")
        );
        assert_eq!(exact_identifier_query("serde internally tagged enum"), None);
    }

    #[test]
    fn parses_issue_candidates() {
        let response: SearchResponse = serde_json::from_str(
            r#"{
                "items": [{
                    "title": "Fix exact_identifier",
                    "html_url": "https://github.com/example/project/issues/1",
                    "created_at": "2026-08-01T12:00:00Z",
                    "body": "The answer-bearing issue body"
                }]
            }"#,
        )
        .unwrap();

        let hit = response
            .items
            .into_iter()
            .next()
            .unwrap()
            .into_hit()
            .unwrap();

        assert_eq!(hit.title, "Fix exact_identifier");
        assert_eq!(hit.date.as_deref(), Some("2026-08-01"));
        assert_eq!(hit.snippet, "The answer-bearing issue body");
    }
}
