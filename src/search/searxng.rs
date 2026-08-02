use std::collections::HashSet;

use serde::Deserialize;
use tokio::time::{Instant, sleep_until};
use url::Url;

use super::exact_identifier_query;
use crate::error::{AppError, Result};
use crate::state::AppState;

const SOURCE_PROMOTION_LIMIT: usize = 2;

#[derive(Clone, Debug)]
pub struct Hit {
    pub title: String,
    pub url: Url,
    pub date: Option<String>,
    pub snippet: String,
}

pub async fn search(state: &AppState, query: &str, limit: usize) -> Result<Vec<Hit>> {
    state
        .search_cache
        .get_or_search(query, limit, || search_uncached(state, query, limit))
        .await
}

async fn search_uncached(state: &AppState, query: &str, limit: usize) -> Result<Vec<Hit>> {
    let (searxng, github) = tokio::join!(
        search_searxng(state, query, limit),
        super::github::search(state, query, 4.min(limit)),
    );
    let mut hits = searxng?;
    merge_auxiliary_hits(&mut hits, github, limit);
    Ok(hits)
}

async fn search_searxng(state: &AppState, query: &str, limit: usize) -> Result<Vec<Hit>> {
    let _permit = state
        .searxng_permits
        .acquire()
        .await
        .expect("SearXNG semaphore is never closed");
    wait_for_rate_limit(state).await;

    let route = route(query, &state.config.searxng_categories);
    let mut body = request(state, &route.query, &route.categories, None).await?;
    let discovery_query = route.supplemental_query.clone().map(|supplemental_query| {
        let categories = has_news_intent(query).then_some("general");
        let time_range = has_discovery_recency_intent(&supplemental_query).then_some("year");
        SupplementalQuery::Discovery {
            query: supplemental_query,
            categories,
            time_range,
        }
    });
    let supplemental_query = source_aware_query(query, &route.query, &body.results)
        .map(SupplementalQuery::Source)
        .or(discovery_query)
        .or_else(|| {
            has_category(&state.config.searxng_categories, "map")
                .then(|| local_map_query(query, &route.query))
                .flatten()
                .map(SupplementalQuery::Local)
        });
    tracing::debug!(
        original_query = query,
        routed_query = route.query,
        categories = route.categories,
        supplemental = ?supplemental_query,
        "planned SearXNG search"
    );
    if let Some(supplemental_query) = supplemental_query {
        wait_for_rate_limit(state).await;
        match request(
            state,
            supplemental_query.query(),
            supplemental_query.categories(&route.categories),
            supplemental_query.time_range(),
        )
        .await
        {
            Ok(supplemental) => {
                tracing::debug!(
                    query = supplemental_query.query(),
                    results = supplemental.results.len(),
                    "received supplemental SearXNG results"
                );
                match supplemental_query {
                    SupplementalQuery::Discovery { .. } => {
                        body.results.extend(supplemental.results);
                    }
                    SupplementalQuery::Local(_) => {
                        merge_local_results(&mut body, supplemental);
                    }
                    SupplementalQuery::Source(source) => {
                        if source.host.is_none()
                            && let Some(host) =
                                infer_official_host(&route.query, &supplemental.results)
                        {
                            let search_host = registrable_host(&host);
                            let focused_query = format!("{} site:{search_host}", route.query);
                            wait_for_rate_limit(state).await;
                            match request(state, &focused_query, &route.categories, None).await {
                                Ok(focused) if !focused.results.is_empty() => {
                                    tracing::debug!(
                                        query = focused_query,
                                        host,
                                        results = focused.results.len(),
                                        "received focused first-party SearXNG results"
                                    );
                                    merge_source_results(&mut body, focused, Some(&host));
                                    body.results.extend(supplemental.results);
                                }
                                Ok(_) => {
                                    merge_source_results(&mut body, supplemental, Some(&host));
                                }
                                Err(error) => {
                                    tracing::debug!(
                                        query = focused_query,
                                        error = ?error,
                                        "focused first-party SearXNG query failed"
                                    );
                                    merge_source_results(&mut body, supplemental, Some(&host));
                                }
                            }
                        } else {
                            merge_source_results(&mut body, supplemental, source.host.as_deref());
                        }
                    }
                }
            }
            Err(error) => tracing::debug!(
                query = supplemental_query.query(),
                error = ?error,
                "supplemental SearXNG query failed"
            ),
        }
    }

    Ok(into_hits(body, limit))
}

async fn request(
    state: &AppState,
    query: &str,
    categories: &str,
    time_range: Option<&str>,
) -> Result<SearxResponse> {
    let url = search_url(&state.config.searxng_url, query, categories, time_range);
    let response = state
        .http
        .get(url)
        .timeout(state.config.searxng_timeout)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(AppError::SearchBackend)?;

    response
        .json()
        .await
        .map_err(|error| AppError::SearchBackendResponse(format!("could not decode JSON: {error}")))
}

async fn wait_for_rate_limit(state: &AppState) {
    let mut next_request = state.searxng_next_request.lock().await;
    let now = Instant::now();
    if *next_request > now {
        sleep_until(*next_request).await;
    }
    *next_request = Instant::now() + state.config.searxng_min_interval;
}

#[derive(Debug, PartialEq, Eq)]
struct Route {
    query: String,
    categories: String,
    supplemental_query: Option<String>,
}

#[derive(Debug)]
enum SupplementalQuery {
    Discovery {
        query: String,
        categories: Option<&'static str>,
        time_range: Option<&'static str>,
    },
    Local(String),
    Source(SourceQuery),
}

impl SupplementalQuery {
    fn query(&self) -> &str {
        match self {
            Self::Discovery { query, .. } | Self::Local(query) => query,
            Self::Source(source) => &source.query,
        }
    }

    fn categories<'a>(&'a self, default: &'a str) -> &'a str {
        match self {
            Self::Local(_) => "map",
            Self::Discovery { categories, .. } => categories.unwrap_or(default),
            Self::Source(_) => default,
        }
    }

    fn time_range(&self) -> Option<&str> {
        match self {
            Self::Discovery { time_range, .. } => *time_range,
            Self::Local(_) | Self::Source(_) => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SourceQuery {
    query: String,
    host: Option<String>,
}

fn route(query: &str, configured_categories: &str) -> Route {
    let categories = configured_categories
        .split(',')
        .map(str::trim)
        .filter(|category| !category.is_empty())
        .collect::<Vec<_>>();
    let has = |category: &str| categories.contains(&category);

    if has("news") && has_news_intent(query) {
        return Route {
            query: normalize_news_query(query),
            categories: "news".to_owned(),
            supplemental_query: Some(news_discovery_query(query)),
        };
    }

    let normalized_query = normalize_intent_modifiers(query);
    let mut routed_categories = Vec::new();
    if has("general") {
        routed_categories.push("general");
    }
    if has("it") && has_it_intent(query) {
        routed_categories.push("it");
    }
    if has("science") && has_science_intent(query) {
        routed_categories.push("science");
    }
    Route {
        supplemental_query: exact_identifier_query(&normalized_query).or_else(|| {
            if has_discovery_recency_intent(query) {
                Some(query.to_owned())
            } else {
                normalize_temporal_query(&normalized_query)
            }
        }),
        query: normalized_query,
        categories: if routed_categories.is_empty() {
            categories.first().copied().unwrap_or_default().to_owned()
        } else {
            routed_categories.join(",")
        },
    }
}

fn has_it_intent(query: &str) -> bool {
    exact_identifier_query(query).is_some()
        || query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| {
                matches!(
                    token.to_ascii_lowercase().as_str(),
                    "axum"
                        | "compiler"
                        | "cuda"
                        | "docker"
                        | "javascript"
                        | "kubernetes"
                        | "linux"
                        | "nixos"
                        | "postgres"
                        | "postgresql"
                        | "python"
                        | "react"
                        | "rust"
                        | "serde"
                        | "tokio"
                        | "typescript"
                )
            })
}

fn has_science_intent(query: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "arxiv"
                    | "clinical"
                    | "diabetes"
                    | "doi"
                    | "evidence"
                    | "guideline"
                    | "guidelines"
                    | "journal"
                    | "paper"
                    | "papers"
                    | "research"
                    | "study"
                    | "studies"
                    | "superconductor"
                    | "trial"
                    | "trials"
            )
        })
}

fn has_discovery_recency_intent(query: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "latest" | "newest" | "recent" | "recently"
            )
        })
}

fn has_category(configured_categories: &str, expected: &str) -> bool {
    configured_categories
        .split(',')
        .map(str::trim)
        .any(|category| category == expected)
}

fn local_map_query(original_query: &str, search_query: &str) -> Option<String> {
    let lower = original_query.to_ascii_lowercase();
    let local_intent =
        lower.contains(" near ") || lower.contains(" nearby") || lower.contains(" closest ");
    let unsuitable_for_map = lower
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| matches!(token, "hike" | "hikes" | "hiking" | "trail" | "trails"));
    if !local_intent || unsuitable_for_map {
        return None;
    }

    let query = search_query
        .split_whitespace()
        .filter(|word| {
            !matches!(
                word.trim_matches(|character: char| !character.is_alphanumeric())
                    .to_ascii_lowercase()
                    .as_str(),
                "closest" | "late" | "near" | "nearby" | "now" | "open" | "coworking"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    (!query.is_empty()).then_some(query)
}

fn has_news_intent(query: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "news" | "headline" | "headlines" | "breaking"
            )
        })
}

fn normalize_intent_modifiers(query: &str) -> String {
    let normalized = query
        .split_whitespace()
        .filter(|word| {
            let token = word.trim_matches(|character: char| !character.is_alphanumeric());
            !matches!(
                token.to_ascii_lowercase().as_str(),
                "best"
                    | "a"
                    | "an"
                    | "are"
                    | "can"
                    | "could"
                    | "current"
                    | "currently"
                    | "did"
                    | "do"
                    | "documentation"
                    | "does"
                    | "how"
                    | "i"
                    | "is"
                    | "latest"
                    | "me"
                    | "my"
                    | "newest"
                    | "official"
                    | "recent"
                    | "recently"
                    | "should"
                    | "the"
                    | "today"
                    | "traditional"
                    | "was"
                    | "were"
                    | "what"
                    | "when"
                    | "where"
                    | "which"
                    | "why"
                    | "would"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    if normalized.is_empty() {
        query.to_owned()
    } else {
        normalized
    }
}

fn normalize_news_query(query: &str) -> String {
    let words = query.split_whitespace().collect::<Vec<_>>();
    let mut normalized = Vec::with_capacity(words.len());
    let mut index = 0;

    while index < words.len() {
        let token = words[index].trim_matches(|character: char| !character.is_alphanumeric());
        let lower = token.to_ascii_lowercase();
        if is_month(&lower) {
            index += 1;
            if index < words.len() && is_day(words[index]) {
                index += 1;
            }
            if index < words.len() && is_year(words[index]) {
                index += 1;
            }
            continue;
        }
        if matches!(
            lower.as_str(),
            "news"
                | "headline"
                | "headlines"
                | "today"
                | "yesterday"
                | "breaking"
                | "latest"
                | "recent"
                | "recently"
                | "from"
                | "about"
        ) {
            index += 1;
            continue;
        }
        normalized.push(words[index]);
        index += 1;
    }

    let normalized = normalized.join(" ");
    if normalized.trim().is_empty() {
        query.to_owned()
    } else {
        normalized
    }
}

fn news_discovery_query(query: &str) -> String {
    let normalized = query
        .split_whitespace()
        .filter(|word| {
            !matches!(
                word.trim_matches(|character: char| !character.is_alphanumeric())
                    .to_ascii_lowercase()
                    .as_str(),
                "breaking" | "headline" | "headlines" | "news"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        query.to_owned()
    } else {
        normalized
    }
}

fn normalize_temporal_query(query: &str) -> Option<String> {
    let normalized = query
        .split_whitespace()
        .filter(|word| {
            let token = word.trim_matches(|character: char| !character.is_alphanumeric());
            !is_year(token)
                && !matches!(
                    token.to_ascii_lowercase().as_str(),
                    "latest" | "newest" | "current" | "currently" | "recent" | "recently" | "today"
                )
        })
        .collect::<Vec<_>>()
        .join(" ");

    (normalized != query && !normalized.is_empty()).then_some(normalized)
}

fn source_aware_query(
    original_query: &str,
    search_query: &str,
    results: &[SearxResult],
) -> Option<SourceQuery> {
    let lower = original_query.to_ascii_lowercase();
    if let Some(host) = named_authority_host(original_query) {
        let query = if host == "arxiv.org" {
            if let Some(title) = arxiv_title(original_query) {
                format!("\"{title}\" site:{host}")
            } else {
                format!("{search_query} site:{host}")
            }
        } else {
            format!("{search_query} site:{host}")
        };
        return Some(SourceQuery {
            query,
            host: Some(host.into()),
        });
    }
    if lower.contains("primary source") {
        let subject = original_query
            .split_whitespace()
            .filter(|word| {
                !matches!(
                    word.trim_matches(|character: char| !character.is_alphanumeric())
                        .to_ascii_lowercase()
                        .as_str(),
                    "primary" | "source" | "sources"
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        return (!subject.is_empty()).then(|| SourceQuery {
            query: format!("{subject} report site:.gov"),
            host: Some(".gov".into()),
        });
    }
    if let Some(code) = compiler_error_code(original_query) {
        if query_contains_token(original_query, "rust") && code.starts_with('E') {
            return Some(SourceQuery {
                query: format!("{code} site:doc.rust-lang.org"),
                host: Some("doc.rust-lang.org".into()),
            });
        }
        if let Some(host) = infer_official_host(search_query, results) {
            let search_host = registrable_host(&host);
            return Some(SourceQuery {
                query: format!("{code} site:{search_host}"),
                host: Some(host),
            });
        }
    }
    if !has_source_intent(&lower) {
        return None;
    }

    if let Some(host) = infer_official_host(search_query, results) {
        let search_host = registrable_host(&host);
        return Some(SourceQuery {
            query: format!("{search_query} site:{search_host}"),
            host: Some(host),
        });
    }

    has_documentation_intent(&lower).then(|| SourceQuery {
        query: original_query
            .split_whitespace()
            .map(|word| {
                if word.eq_ignore_ascii_case("documentation") {
                    "docs"
                } else {
                    word
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        host: None,
    })
}

pub(super) fn named_authority_host(query: &str) -> Option<&'static str> {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();

    if tokens
        .windows(3)
        .any(|tokens| tokens == ["federal", "funds", "rate"])
    {
        return Some("federalreserve.gov");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["geneva", "convention"] || tokens == ["geneva", "conventions"])
    {
        return Some("icrc.org");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["mortgage", "rate"] || tokens == ["mortgage", "rates"])
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "data" | "official"))
    {
        return Some("freddiemac.com");
    }
    if tokens.iter().any(|token| token == "cpi")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "data" | "official"))
    {
        return Some("bls.gov");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["united", "nations"])
    {
        return Some("un.org");
    }
    if tokens.iter().any(|token| token == "acc")
        && tokens.iter().any(|token| token == "aha")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "guideline" | "guidelines"))
    {
        return Some("jacc.org");
    }
    if tokens.iter().any(|token| token == "ada") && tokens.iter().any(|token| token == "diabetes") {
        return Some("diabetesjournals.org");
    }
    for token in &tokens {
        if token == "fda" {
            return Some("fda.gov");
        }
        if token == "nasa" {
            return Some("nasa.gov");
        }
        if token == "nixos" {
            return Some("wiki.nixos.org");
        }
        if token == "tokio" {
            return Some("docs.rs");
        }
        if token == "arxiv" {
            return Some("arxiv.org");
        }
        if token == "who" {
            return Some("who.int");
        }
    }
    None
}

fn query_contains_token(query: &str, expected: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case(expected))
}

pub(super) fn arxiv_title(query: &str) -> Option<String> {
    if !query_contains_token(query, "arxiv") {
        return None;
    }
    let title = query
        .split_whitespace()
        .filter(|word| {
            !matches!(
                word.trim_matches(|character: char| !character.is_alphanumeric())
                    .to_ascii_lowercase()
                    .as_str(),
                "arxiv" | "original" | "paper" | "papers"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    (title.split_whitespace().count() >= 3).then_some(title)
}

fn compiler_error_code(query: &str) -> Option<&str> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .find(|token| {
            let letters = token
                .chars()
                .take_while(|character| character.is_ascii_alphabetic())
                .count();
            let digits = token.chars().count().saturating_sub(letters);
            (1..=4).contains(&letters)
                && (3..=5).contains(&digits)
                && token
                    .chars()
                    .skip(letters)
                    .all(|character| character.is_ascii_digit())
        })
}

fn merge_source_results(
    primary: &mut SearxResponse,
    mut supplemental: SearxResponse,
    host: Option<&str>,
) {
    let mut results = Vec::with_capacity(primary.results.len() + supplemental.results.len());
    match host {
        Some(host) => {
            results.extend(
                supplemental
                    .results
                    .iter()
                    .chain(&primary.results)
                    .filter(|result| result_matches_host(result, host))
                    .take(SOURCE_PROMOTION_LIMIT)
                    .cloned(),
            );
            results.append(&mut primary.results);
            results.append(&mut supplemental.results);
        }
        None => {
            results.append(&mut supplemental.results);
            results.append(&mut primary.results);
        }
    }
    primary.results = results;
}

fn merge_local_results(primary: &mut SearxResponse, supplemental: SearxResponse) {
    let mut local = supplemental.results.into_iter();
    let promoted = local.by_ref().take(4).collect::<Vec<_>>();
    let mut results = Vec::with_capacity(promoted.len() + primary.results.len() + local.len());
    results.extend(promoted);
    results.append(&mut primary.results);
    results.extend(local);
    primary.results = results;
}

fn result_matches_host(result: &SearxResult, expected: &str) -> bool {
    Url::parse(&result.url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            if expected.starts_with('.') {
                host.ends_with(expected)
            } else {
                host == expected || host.ends_with(&format!(".{expected}"))
            }
        })
}

pub(super) fn has_source_intent(lower_query: &str) -> bool {
    query_contains_token(lower_query, "arxiv")
        || lower_query.contains("full text")
        || lower_query.contains("primary source")
        || lower_query.contains("release notes")
        || lower_query.split_whitespace().any(|word| {
            matches!(
                word.trim_matches(|character: char| !character.is_alphanumeric()),
                "documentation" | "docs" | "guideline" | "guidelines" | "official"
            )
        })
}

fn has_documentation_intent(lower_query: &str) -> bool {
    lower_query.split_whitespace().any(|word| {
        matches!(
            word.trim_matches(|character: char| !character.is_alphanumeric()),
            "documentation" | "docs" | "official"
        )
    })
}

fn infer_official_host(search_query: &str, results: &[SearxResult]) -> Option<String> {
    let query_tokens = distinctive_tokens(search_query);
    let query_words = search_query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let public_authority_intent = query_words
        .iter()
        .any(|word| matches!(word.as_str(), "election" | "voter" | "voting"))
        && query_words
            .iter()
            .any(|word| matches!(word.as_str(), "deadline" | "registration"));
    let entity = search_query
        .split(|character: char| !character.is_alphanumeric())
        .find(|token| token.len() >= 2)
        .map(str::to_ascii_lowercase);
    let entity_brand = entity.as_deref().map(brand_token);
    results
        .iter()
        .take(12)
        .enumerate()
        .filter_map(|(index, result)| {
            let url = Url::parse(&result.url).ok()?;
            let host = url.host_str()?;
            let host = host.strip_prefix("www.").unwrap_or(host);
            if is_generic_host(host) {
                return None;
            }
            let normalized_host = host
                .chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();
            let title = result
                .title
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            let host_match = entity_brand.is_some_and(|entity| normalized_host.contains(entity));
            let stable_brand_match = entity_brand.is_some_and(|entity| {
                host.split('.').any(|label| {
                    label
                        .chars()
                        .filter(|character| character.is_alphanumeric())
                        .flat_map(char::to_lowercase)
                        .eq(entity.chars())
                })
            });
            let title_matches = query_tokens
                .iter()
                .filter(|token| title.contains(token.as_str()))
                .take(2)
                .count();
            let institutional = host.ends_with(".gov")
                || host.ends_with(".edu")
                || host.ends_with(".int")
                || host.ends_with(".ac.uk");
            if public_authority_intent && !host.ends_with(".gov") {
                return None;
            }
            let documentation = host.starts_with("doc.")
                || host.starts_with("docs.")
                || url.path().split('/').any(|segment| {
                    matches!(
                        segment,
                        "doc" | "docs" | "documentation" | "guidelines" | "reference"
                    )
                });
            let entity_documentation = documentation
                && entity.as_ref().is_some_and(|entity| {
                    title
                        .split(|character: char| !character.is_alphanumeric())
                        .any(|word| word == entity)
                });
            let score = usize::from(host_match) * 8
                + usize::from(stable_brand_match) * 3
                + usize::from(institutional) * 2
                + title_matches
                + usize::from(documentation)
                + usize::from(entity_documentation) * 3;
            (score >= 4).then(|| (score, index, official_search_host(host)))
        })
        .max_by(|left, right| left.0.cmp(&right.0).then_with(|| right.1.cmp(&left.1)))
        .map(|(_, _, host)| host)
}

fn registrable_host(host: &str) -> String {
    let labels = host.split('.').collect::<Vec<_>>();
    if labels.len() <= 2 {
        return host.to_owned();
    }
    let suffix = labels[labels.len() - 2..].join(".");
    let labels_to_keep = if matches!(
        suffix.as_str(),
        "ac.uk" | "co.jp" | "co.uk" | "com.au" | "com.br" | "com.cn" | "org.uk"
    ) {
        3
    } else {
        2
    };
    labels[labels.len() - labels_to_keep..].join(".")
}

fn official_search_host(host: &str) -> String {
    if host.starts_with("doc.") || host.starts_with("docs.") {
        host.to_owned()
    } else {
        registrable_host(host)
    }
}

fn distinctive_tokens(query: &str) -> Vec<String> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|token| token.len() >= 3)
        .filter(|token| !token.chars().all(|character| character.is_ascii_digit()))
        .filter(|token| {
            !matches!(
                token.as_str(),
                "and"
                    | "documentation"
                    | "example"
                    | "for"
                    | "full"
                    | "guideline"
                    | "guidelines"
                    | "how"
                    | "notes"
                    | "official"
                    | "release"
                    | "text"
                    | "the"
                    | "using"
                    | "what"
                    | "with"
            )
        })
        .collect()
}

fn brand_token(token: &str) -> &str {
    let stem = token.trim_end_matches(|character: char| character.is_ascii_digit());
    if stem.len() >= 3 { stem } else { token }
}

fn is_generic_host(host: &str) -> bool {
    [
        "github.com",
        "medium.com",
        "reddit.com",
        "stackoverflow.com",
        "wikipedia.org",
        "youtube.com",
    ]
    .iter()
    .any(|generic| host == *generic || host.ends_with(&format!(".{generic}")))
}

fn is_month(token: &str) -> bool {
    matches!(
        token,
        "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
            | "jan"
            | "feb"
            | "mar"
            | "apr"
            | "jun"
            | "jul"
            | "aug"
            | "sep"
            | "sept"
            | "oct"
            | "nov"
            | "dec"
    )
}

fn is_day(word: &str) -> bool {
    word.trim_matches(|character: char| !character.is_ascii_digit())
        .parse::<u8>()
        .is_ok_and(|day| (1..=31).contains(&day))
}

fn is_year(word: &str) -> bool {
    let word = word.trim_matches(|character: char| !character.is_ascii_digit());
    word.len() == 4
        && word
            .parse::<u16>()
            .is_ok_and(|year| (1900..=2200).contains(&year))
}

fn search_url(base: &Url, query: &str, categories: &str, time_range: Option<&str>) -> Url {
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
    if let Some(time_range) = time_range {
        parameters.append_pair("time_range", time_range);
    }
    drop(parameters);
    url
}

fn into_hits(body: SearxResponse, limit: usize) -> Vec<Hit> {
    if limit == 0 {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut seen_titles = HashSet::new();
    let mut hits = Vec::with_capacity(limit.min(body.results.len()));

    for result in body.results {
        let preferred_url = result
            .links
            .iter()
            .find(|link| link.label.eq_ignore_ascii_case("official website"))
            .map(|link| link.url.as_str())
            .unwrap_or(&result.url);
        let Ok(mut url) = Url::parse(preferred_url) else {
            tracing::debug!(url = preferred_url, "discarding invalid search result URL");
            continue;
        };
        if !matches!(url.scheme(), "http" | "https") {
            tracing::debug!(url = %url, "discarding unsupported search result URL");
            continue;
        }
        url.set_fragment(None);
        if !seen.insert(canonical_key(&url)) {
            continue;
        }

        let snippet = result
            .content
            .filter(|content| !content.trim().is_empty())
            .or_else(|| format_map_details(result.address, result.data))
            .unwrap_or_default();
        let title = result.title.unwrap_or_else(|| "Untitled".into());
        let title_key = normalized_title_key(&title);
        if title != "Untitled" && !seen_titles.insert(title_key) {
            continue;
        }

        hits.push(Hit {
            title,
            url,
            date: result
                .published_date
                .map(|date| date.chars().take(10).collect()),
            snippet,
        });
        if hits.len() == limit {
            break;
        }
    }

    hits
}

fn merge_auxiliary_hits(primary: &mut Vec<Hit>, auxiliary: Vec<Hit>, limit: usize) {
    let mut urls = primary
        .iter()
        .map(|hit| canonical_key(&hit.url))
        .collect::<HashSet<_>>();
    let mut titles = primary
        .iter()
        .map(|hit| normalized_title_key(&hit.title))
        .collect::<HashSet<_>>();
    let auxiliary = auxiliary.into_iter().filter(|hit| {
        urls.insert(canonical_key(&hit.url)) && titles.insert(normalized_title_key(&hit.title))
    });
    let insert_at = primary.len().min(4);
    primary.splice(insert_at..insert_at, auxiliary);
    primary.truncate(limit);
}

fn normalized_title_key(title: &str) -> String {
    let mut seen = HashSet::new();
    title
        .replace("%20", " ")
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|word| {
            !(word.len() >= 4 && word.chars().all(|character| character.is_ascii_digit()))
        })
        .filter(|word| !word.is_empty())
        .filter(|word| seen.insert(word.clone()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn canonical_key(url: &Url) -> String {
    let mut key = url.clone();
    if key.scheme() == "http" {
        let _ = key.set_scheme("https");
    }
    if let Some(host) = key.host_str().map(str::to_owned)
        && let Some(host) = host.strip_prefix("www.")
    {
        let _ = key.set_host(Some(host));
    }
    if key.path().len() > 1 {
        let trimmed = key.path().trim_end_matches('/').to_owned();
        key.set_path(&trimmed);
    }

    let retained = key
        .query_pairs()
        .filter(|(name, _)| {
            let name = name.to_ascii_lowercase();
            !name.starts_with("utm_")
                && !matches!(
                    name.as_str(),
                    "fbclid" | "gclid" | "ref" | "referrer" | "source"
                )
        })
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    key.set_query(None);
    if !retained.is_empty() {
        key.query_pairs_mut().extend_pairs(retained);
    }
    key.to_string()
}

/// SearXNG fields other than `url` are optional in real responses.
#[derive(Deserialize)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<SearxResult>,
}

#[derive(Clone, Deserialize)]
struct SearxResult {
    url: String,
    title: Option<String>,
    content: Option<String>,
    #[serde(rename = "publishedDate", alias = "published_date")]
    published_date: Option<String>,
    address: Option<SearxAddress>,
    #[serde(default)]
    data: Vec<SearxData>,
    #[serde(default)]
    links: Vec<SearxLink>,
}

#[derive(Clone, Deserialize)]
struct SearxAddress {
    house_number: Option<String>,
    road: Option<String>,
    locality: Option<String>,
    postcode: Option<String>,
    country: Option<String>,
}

#[derive(Clone, Deserialize)]
struct SearxData {
    key: String,
    value: String,
}

#[derive(Clone, Deserialize)]
struct SearxLink {
    label: String,
    url: String,
}

fn format_map_details(address: Option<SearxAddress>, data: Vec<SearxData>) -> Option<String> {
    let mut details = address
        .map(format_address)
        .filter(|address| !address.is_empty())
        .into_iter()
        .collect::<Vec<_>>();
    details.extend(
        data.into_iter()
            .filter(|item| item.key == "opening_hours")
            .map(|item| format!("Hours: {}", item.value)),
    );
    (!details.is_empty()).then(|| details.join(". "))
}

fn format_address(address: SearxAddress) -> String {
    let street = [address.house_number, address.road]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    let address = [
        (!street.is_empty()).then_some(street),
        address.locality,
        address.postcode,
        address.country,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(", ");
    if address.is_empty() {
        String::new()
    } else {
        format!("Address: {address}")
    }
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
                    {"url":"http://www.example.com/a/?utm_source=test","title":"canonical duplicate"},
                    {"url":"https://mirror.example/a","title":"  A  "},
                    {"url":"ftp://example.com/file"},
                    {"url":"not a URL"},
                    {"url":"https://example.com/b","publishedDate":"2026-08-01T12:00:00"},
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
    fn deduplicates_noisy_numeric_title_variants() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://lake.example/a","title":"Lake Mendota water temperature 2742 2742 | LakeMonster"},
                    {"url":"https://lake.example/b","title":"Lake%20Mendota Water Temperature | LakeMonster"},
                    {"url":"https://lake.example/c","title":"Mendota research buoy"}
                ]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, 3);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].title, "Mendota research buoy");
        assert_ne!(
            normalized_title_key("PostgreSQL 11 Released"),
            normalized_title_key("PostgreSQL 18 Released")
        );
    }

    #[test]
    fn routes_explicit_news_broadly_and_preserves_dates_for_discovery() {
        assert_eq!(
            route("OpenAI news August 1 2026", "general,it,science,news"),
            Route {
                query: "OpenAI".into(),
                categories: "news".into(),
                supplemental_query: Some("OpenAI August 1 2026".into()),
            }
        );
        assert_eq!(
            route("what causes the aurora borealis", "general,it,science,news"),
            Route {
                query: "causes aurora borealis".into(),
                categories: "general".into(),
                supplemental_query: None,
            }
        );
    }

    #[test]
    fn normalizes_search_intent_without_misrouting_current_conditions() {
        assert_eq!(
            route("Lake Mendota water temperature today", "general,news"),
            Route {
                query: "Lake Mendota water temperature".into(),
                categories: "general".into(),
                supplemental_query: None,
            }
        );
        assert_eq!(
            route(
                "What is the US federal funds rate in August 2026?",
                "general,news"
            ),
            Route {
                query: "US federal funds rate in August 2026?".into(),
                categories: "general".into(),
                supplemental_query: Some("US federal funds rate in August".into()),
            }
        );
        assert_eq!(
            route(
                "current FDA approved Alzheimer's drugs 2026",
                "general,news"
            ),
            Route {
                query: "FDA approved Alzheimer's drugs 2026".into(),
                categories: "general".into(),
                supplemental_query: Some("FDA approved Alzheimer's drugs".into()),
            }
        );
    }

    #[test]
    fn focuses_clear_local_intent_for_map_search() {
        assert_eq!(
            local_map_query(
                "best coffee shops near Madison Wisconsin open late",
                "coffee shops near Madison Wisconsin open late"
            ),
            Some("coffee shops Madison Wisconsin".into())
        );
        assert_eq!(
            local_map_query(
                "best coworking cafes near Austin Texas open late",
                "coworking cafes Austin Texas"
            ),
            Some("cafes Austin Texas".into())
        );
        assert_eq!(
            local_map_query(
                "coffee shops near Madison Wisconsin",
                "coffee shops near Madison Wisconsin"
            ),
            Some("coffee shops Madison Wisconsin".into())
        );
        assert_eq!(
            local_map_query(
                "Lake Mendota water temperature today",
                "Lake Mendota water temperature"
            ),
            None
        );
        assert_eq!(
            local_map_query(
                "best hiking trails near Madison Wisconsin",
                "hiking trails near Madison Wisconsin"
            ),
            None
        );
        assert!(has_category("general,news,map", "map"));
    }

    #[test]
    fn respects_restricted_category_configuration() {
        assert_eq!(
            route("axum timeout", "it,science"),
            Route {
                query: "axum timeout".into(),
                categories: "it".into(),
                supplemental_query: None,
            }
        );
    }

    #[test]
    fn supplements_temporal_queries_without_the_date_constraint() {
        assert_eq!(
            route(
                "2026 hypertension first line treatment guideline adults",
                "general,science"
            ),
            Route {
                query: "2026 hypertension first line treatment guideline adults".into(),
                categories: "general,science".into(),
                supplemental_query: Some(
                    "hypertension first line treatment guideline adults".into()
                ),
            }
        );
        assert_eq!(
            route("latest room temperature superconductor", "general,science").supplemental_query,
            Some("latest room temperature superconductor".into())
        );
        assert_eq!(
            route("latest room temperature superconductor", "general,science").categories,
            "general,science"
        );
    }

    #[test]
    fn adds_specialist_categories_only_for_matching_intent() {
        assert_eq!(
            route(
                "Attention Is All You Need arXiv paper",
                "general,it,science"
            )
            .categories,
            "general,science"
        );
        assert_eq!(
            route("Rust E0502 borrow error", "general,it,science").categories,
            "general,it"
        );
        assert_eq!(
            route("traditional shakshuka recipe", "general,it,science").categories,
            "general"
        );
    }

    #[test]
    fn supplements_code_identifiers_with_a_focused_exact_query() {
        assert_eq!(
            route(
                "invalid_reference_casting tokenizers Rust 1.73",
                "general,it"
            )
            .supplemental_query,
            Some("\"invalid_reference_casting\" tokenizers".into())
        );
        assert_eq!(
            route("rust unresolved import tower_http timeout", "general,it").supplemental_query,
            Some("\"tower_http\" timeout".into())
        );
    }

    #[test]
    fn expands_primary_source_intent_to_institutional_reports() {
        assert_eq!(
            source_aware_query(
                "primary sources causes of the 2008 financial crisis",
                "primary sources causes of the 2008 financial crisis",
                &[]
            ),
            Some(SourceQuery {
                query: "causes of the 2008 financial crisis report site:.gov".into(),
                host: Some(".gov".into()),
            })
        );
        assert!(has_source_intent(
            "primary sources causes of the 2008 financial crisis"
        ));
    }

    #[test]
    fn named_authorities_receive_a_first_party_supplement() {
        assert_eq!(
            source_aware_query(
                "current FDA approved Alzheimer's drugs 2026",
                "FDA approved Alzheimer's drugs 2026",
                &[]
            ),
            Some(SourceQuery {
                query: "FDA approved Alzheimer's drugs 2026 site:fda.gov".into(),
                host: Some("fda.gov".into()),
            })
        );
        assert_eq!(
            named_authority_host("FDA-approved medication"),
            Some("fda.gov")
        );
        assert_eq!(named_authority_host("cve database"), None);
        assert_eq!(
            named_authority_host("NASA mission status"),
            Some("nasa.gov")
        );
        assert_eq!(
            named_authority_host("NixOS Docker setup"),
            Some("wiki.nixos.org")
        );
        assert_eq!(
            named_authority_host("WHO adult physical activity guidelines"),
            Some("who.int")
        );
        assert_eq!(
            named_authority_host("US federal funds rate August 2026"),
            Some("federalreserve.gov")
        );
        assert_eq!(
            named_authority_host("Geneva Convention Common Article 3 full text"),
            Some("icrc.org")
        );
        assert_eq!(
            named_authority_host("current US mortgage rate official data"),
            Some("freddiemac.com")
        );
        assert_eq!(named_authority_host("best mortgage rates"), None);
        assert_eq!(
            named_authority_host("current US CPI inflation official data"),
            Some("bls.gov")
        );
        assert_eq!(
            named_authority_host("Attention Is All You Need arXiv paper"),
            Some("arxiv.org")
        );
        assert_eq!(
            named_authority_host("United Nations Charter Article 51 full text"),
            Some("un.org")
        );
        assert_eq!(
            named_authority_host("2026 ACC AHA cholesterol guideline"),
            Some("jacc.org")
        );
        assert_eq!(
            named_authority_host("2026 ADA diabetes standards"),
            Some("diabetesjournals.org")
        );
        assert_eq!(
            named_authority_host("Rust tokio select documentation"),
            Some("docs.rs")
        );
        assert_eq!(named_authority_host("ACC conference schedule"), None);
        assert!(has_source_intent("attention is all you need arxiv paper"));
        assert_eq!(
            source_aware_query(
                "Attention Is All You Need arXiv paper",
                "Attention Is All You Need arXiv paper",
                &[]
            ),
            Some(SourceQuery {
                query: "\"Attention Is All You Need\" site:arxiv.org".into(),
                host: Some("arxiv.org".into()),
            })
        );
        assert_eq!(
            source_aware_query(
                "What is the US federal funds rate in August 2026?",
                "US federal funds rate in August 2026?",
                &[]
            ),
            Some(SourceQuery {
                query: "US federal funds rate in August 2026? site:federalreserve.gov".into(),
                host: Some("federalreserve.gov".into()),
            })
        );
        assert_eq!(
            source_aware_query(
                "NixOS Docker NVIDIA container toolkit setup",
                "NixOS Docker NVIDIA container toolkit setup",
                &[]
            ),
            Some(SourceQuery {
                query: "NixOS Docker NVIDIA container toolkit setup site:wiki.nixos.org".into(),
                host: Some("wiki.nixos.org".into()),
            })
        );
        assert_eq!(named_authority_host("approved medication"), None);
    }

    #[test]
    fn compiler_errors_use_a_compact_first_party_query() {
        let results: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://rustfaq.org/e0502","title":"Rust E0502 guide"},
                    {"url":"https://doc.rust-lang.org/error_codes/E0502.html","title":"E0502 - Error codes index - Learn Rust"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            source_aware_query(
                "Rust E0502 cannot borrow mutable because immutable example",
                "Rust E0502 cannot borrow mutable because immutable example",
                &results.results
            ),
            Some(SourceQuery {
                query: "E0502 site:doc.rust-lang.org".into(),
                host: Some("doc.rust-lang.org".into()),
            })
        );
        assert_eq!(
            source_aware_query(
                "Rust E0502 cannot borrow mutable because immutable example",
                "Rust E0502 cannot borrow mutable because immutable example",
                &[]
            ),
            Some(SourceQuery {
                query: "E0502 site:doc.rust-lang.org".into(),
                host: Some("doc.rust-lang.org".into()),
            })
        );
        assert_eq!(compiler_error_code("HTTP 404 response"), None);
    }

    #[test]
    fn infers_first_party_documentation_hosts() {
        let results: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://stackoverflow.com/questions/1","title":"React useEffect loop"},
                    {"url":"https://react.dev/reference/react/useEffect","title":"useEffect - React"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            source_aware_query(
                "React useEffect infinite loop official documentation",
                "React useEffect infinite loop",
                &results.results
            ),
            Some(SourceQuery {
                query: "React useEffect infinite loop site:react.dev".into(),
                host: Some("react.dev".into()),
            })
        );
        assert_eq!(registrable_host("publications.iarc.who.int"), "who.int");
        assert_eq!(registrable_host("docs.nvidia.com"), "nvidia.com");
        assert_eq!(registrable_host("law.ox.ac.uk"), "ox.ac.uk");
        assert_eq!(official_search_host("docs.nvidia.com"), "docs.nvidia.com");
        assert_eq!(official_search_host("download.nvidia.com"), "nvidia.com");
    }

    #[test]
    fn first_party_inference_rejects_branded_third_party_docs() {
        let sqlite: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://coddy.tech/docs/sqlite/wal-mode","title":"SQLite WAL mode and concurrency"},
                    {"url":"https://sqlite.org/index.html","title":"SQLite Home Page"},
                    {"url":"https://sqlite.org/wal.html","title":"Write-Ahead Logging"}
                ]
            }"#,
        )
        .unwrap();
        let qwen: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://qwen3.app/","title":"Qwen3 model guide"},
                    {"url":"https://qwen.ai/blog?id=qwen3-coder","title":"Qwen3-Coder: Agentic Coding in the World"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            infer_official_host(
                "SQLite WAL mode concurrent readers writers",
                &sqlite.results
            ),
            Some("sqlite.org".into())
        );
        assert_eq!(
            infer_official_host("Qwen3 Coder model context length", &qwen.results),
            Some("qwen.ai".into())
        );

        let election: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://industry.visitcalifornia.com/meeting","title":"California 2026 registration"},
                    {"url":"https://www.sos.ca.gov/elections/upcoming-elections/general-election-november-3-2026","title":"California 2026 General Election"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(
            infer_official_host(
                "California 2026 election voter registration deadline",
                &election.results
            ),
            Some("ca.gov".into())
        );
    }

    #[test]
    fn falls_back_to_a_compact_official_docs_query() {
        assert_eq!(
            source_aware_query(
                "React useEffect infinite loop official documentation",
                "React useEffect infinite loop",
                &[]
            ),
            Some(SourceQuery {
                query: "React useEffect infinite loop official docs".into(),
                host: None,
            })
        );
    }

    #[test]
    fn source_results_promote_the_inferred_host() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://example.com/react","title":"Generic guide"},
                    {"url":"https://react.dev/reference/react/useEffect","title":"useEffect - React"}
                ]
            }"#,
        )
        .unwrap();
        let supplemental: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://github.com/facebook/react","title":"React source"},
                    {"url":"https://react.dev/learn/synchronizing-with-effects","title":"Synchronizing with Effects"},
                    {"url":"https://react.dev/learn/removing-effect-dependencies","title":"Removing Effect Dependencies"},
                    {"url":"https://react.dev/learn/lifecycle-of-reactive-effects","title":"Lifecycle of Reactive Effects"}
                ]
            }"#,
        )
        .unwrap();

        merge_source_results(&mut primary, supplemental, Some("react.dev"));

        assert_eq!(
            primary.results[0].url,
            "https://react.dev/learn/synchronizing-with-effects"
        );
        assert_eq!(
            primary.results[1].url,
            "https://react.dev/learn/removing-effect-dependencies"
        );
        assert_eq!(primary.results[2].url, "https://example.com/react");
        assert_eq!(
            primary.results[3].url,
            "https://react.dev/reference/react/useEffect"
        );
    }

    #[test]
    fn local_results_promote_places_without_discarding_web_results() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/late-hours","title":"Late-night guide"},
                {"url":"https://example.com/reviews","title":"Local reviews"}
            ]}"#,
        )
        .unwrap();
        let local: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://www.openstreetmap.org/node/1","title":"Cafe 1"},
                {"url":"https://www.openstreetmap.org/node/2","title":"Cafe 2"},
                {"url":"https://www.openstreetmap.org/node/3","title":"Cafe 3"},
                {"url":"https://www.openstreetmap.org/node/4","title":"Cafe 4"},
                {"url":"https://www.openstreetmap.org/node/5","title":"Cafe 5"}
            ]}"#,
        )
        .unwrap();

        merge_local_results(&mut primary, local);

        assert_eq!(primary.results[0].title.as_deref(), Some("Cafe 1"));
        assert_eq!(primary.results[3].title.as_deref(), Some("Cafe 4"));
        assert_eq!(
            primary.results[4].title.as_deref(),
            Some("Late-night guide")
        );
        assert_eq!(primary.results[6].title.as_deref(), Some("Cafe 5"));
    }

    #[test]
    fn exact_identifier_candidates_join_after_the_consensus_head() {
        let make_hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let mut primary = (0..8)
            .map(|index| {
                make_hit(
                    &format!("Web result {index}"),
                    &format!("https://example.com/{index}"),
                )
            })
            .collect::<Vec<_>>();
        let auxiliary = vec![
            make_hit(
                "Fix invalid_reference_casting",
                "https://github.com/example/project/issues/1",
            ),
            make_hit(
                "Web result 2",
                "https://github.com/example/project/issues/2",
            ),
        ];

        merge_auxiliary_hits(&mut primary, auxiliary, 8);

        assert_eq!(primary.len(), 8);
        assert_eq!(primary[3].title, "Web result 3");
        assert_eq!(primary[4].title, "Fix invalid_reference_casting");
        assert_eq!(primary[5].title, "Web result 4");
    }

    #[test]
    fn map_addresses_become_fallback_snippets() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [{
                    "url":"https://openstreetmap.org/node/1",
                    "title":"Colectivo Coffee",
                    "content":"",
                    "address": {
                        "house_number":"25",
                        "road":"South Pinckney Street",
                        "locality":"Madison",
                        "postcode":"53703",
                        "country":"United States"
                    }
                }]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, 1);

        assert_eq!(
            hits[0].snippet,
            "Address: 25 South Pinckney Street, Madison, 53703, United States"
        );
    }

    #[test]
    fn map_results_keep_hours_and_prefer_the_official_website() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [{
                    "url":"https://openstreetmap.org/node/1",
                    "title":"Cuppa Austin Coffee",
                    "address": {
                        "house_number":"9225",
                        "road":"West Parmer Lane",
                        "locality":"Austin",
                        "postcode":"78717",
                        "country":"United States"
                    },
                    "data": [{
                        "key":"opening_hours",
                        "label":"open days",
                        "value":"Mo-Fr 06:30-21:00; Sa,Su 07:00-21:00"
                    }],
                    "links": [{
                        "label":"official website",
                        "url":"https://www.cuppaaustin.com/"
                    }]
                }]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, 1);

        assert_eq!(hits[0].url.as_str(), "https://www.cuppaaustin.com/");
        assert_eq!(
            hits[0].snippet,
            "Address: 9225 West Parmer Lane, Austin, 78717, United States. Hours: Mo-Fr 06:30-21:00; Sa,Su 07:00-21:00"
        );
    }

    #[test]
    fn requests_selected_categories_without_overriding_engines() {
        let url = search_url(
            &Url::parse("http://localhost:8080").unwrap(),
            "axum timeout",
            "general,it,science,news",
            None,
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

    #[test]
    fn requests_freshness_window_without_changing_the_category() {
        let url = search_url(
            &Url::parse("http://localhost:8080").unwrap(),
            "latest OpenSSH vulnerability",
            "general",
            Some("year"),
        );
        let parameters = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(
            parameters.get("categories").map(|value| value.as_ref()),
            Some("general")
        );
        assert_eq!(
            parameters.get("time_range").map(|value| value.as_ref()),
            Some("year")
        );
    }
}
