use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    pub status: Status,
    pub version: &'static str,
    pub searxng: Dependency,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Degraded,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Dependency {
    pub url: String,
    pub reachable: bool,
    pub latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_num_results")]
    pub num_results: usize,
    pub context_max_characters: Option<usize>,
}

fn default_num_results() -> usize {
    8
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    pub content: String,
    pub full_length: usize,
    pub from_snippet: bool,
    pub truncated: bool,
}

impl From<crate::search::Document> for SearchResult {
    fn from(document: crate::search::Document) -> Self {
        Self {
            title: document.hit.title,
            url: document.hit.url.into(),
            date: document.hit.date,
            content: document.content,
            full_length: document.full_length,
            from_snippet: document.from_snippet,
            truncated: document.truncated,
        }
    }
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::search::{Document, Hit};

    #[test]
    fn search_result_exposes_content_provenance_and_truncation() {
        let result = SearchResult::from(Document {
            hit: Hit {
                title: "Example".into(),
                url: Url::parse("https://example.com/page").unwrap(),
                date: Some("2026-08-01".into()),
                snippet: "snippet".into(),
            },
            content: "selected content\n\n…".into(),
            full_length: 4_096,
            from_snippet: false,
            truncated: true,
        });

        let value = serde_json::to_value(result).unwrap();
        assert_eq!(value["fullLength"], 4_096);
        assert_eq!(value["fromSnippet"], false);
        assert_eq!(value["truncated"], true);
        assert_eq!(value["content"], "selected content\n\n…");
    }
}
