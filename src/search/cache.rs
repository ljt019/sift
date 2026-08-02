use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, OnceCell};
use tokio::time::Instant;
use url::Url;

use super::searxng::SearchCandidate;
use crate::{AppError, Result};

const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
// A pool is degraded only when the primary metasearch produces no usable web
// result and auxiliary providers have to carry it. Individual engine and
// optional-supplement failures do not shorten a useful pool's normal TTL.
const DEGRADED_SEARCH_TTL: Duration = Duration::from_secs(3 * 60);
const STALE_SEARCH_TTL: Duration = Duration::from_secs(60 * 60);
const MAX_CACHE_ENTRIES: usize = 256;
const MAX_CACHE_ENTRY_BYTES: usize = 512 * 1024;

type SharedContent = Arc<str>;
type SharedLoad = Arc<OnceCell<Option<SharedContent>>>;
type SharedSearch = Arc<OnceCell<CachedSearch>>;

#[derive(Clone)]
pub(crate) struct SearchOutcome {
    candidates: Arc<[SearchCandidate]>,
    healthy: bool,
}

impl SearchOutcome {
    pub(crate) fn new(candidates: Vec<SearchCandidate>, healthy: bool) -> Self {
        Self {
            candidates: candidates.into(),
            healthy,
        }
    }
}

pub(crate) struct PageCache {
    entries: Mutex<HashMap<Url, CacheEntry>>,
}

struct CacheEntry {
    inserted: Instant,
    load: SharedLoad,
}

pub(crate) struct SearchCache {
    entries: Mutex<HashMap<String, SearchCacheEntry>>,
}

struct SearchCacheEntry {
    inserted: Instant,
    load: SharedSearch,
    stale: Option<StaleSearch>,
}

#[derive(Clone)]
struct StaleSearch {
    inserted: Instant,
    candidates: Arc<[SearchCandidate]>,
}

struct SearchLoad {
    load: SharedSearch,
    stale: Option<Arc<[SearchCandidate]>>,
}

enum CachedSearch {
    Outcome {
        outcome: SearchOutcome,
        completed: Instant,
    },
    Failure {
        error: CachedSearchError,
        completed: Instant,
    },
}

#[derive(Debug)]
enum CachedSearchError {
    SearchBackend(String),
    SearchBackendResponse(String),
    BadRequest(String),
    Internal(String),
}

impl CachedSearchError {
    fn capture(error: &AppError) -> Self {
        match error {
            AppError::SearchBackend(error) => Self::SearchBackend(error.to_string()),
            AppError::SearchBackendResponse(message) => {
                Self::SearchBackendResponse(message.clone())
            }
            AppError::BadRequest(message) => Self::BadRequest(message.clone()),
            AppError::Internal(error) => Self::Internal(format!("{error:?}")),
        }
    }

    fn to_app_error(&self) -> AppError {
        match self {
            // reqwest::Error is not cloneable or constructible, so retries during the
            // backoff retain the backend-error boundary but not timeout-specific status.
            Self::SearchBackend(message) | Self::SearchBackendResponse(message) => {
                AppError::SearchBackendResponse(message.clone())
            }
            Self::BadRequest(message) => AppError::BadRequest(message.clone()),
            Self::Internal(message) => AppError::Internal(anyhow::anyhow!(message.clone())),
        }
    }
}

impl CachedSearch {
    fn completed(&self) -> Instant {
        match self {
            Self::Outcome { completed, .. } | Self::Failure { completed, .. } => *completed,
        }
    }

    fn ttl(&self) -> Duration {
        match self {
            Self::Outcome { outcome, .. } if outcome.healthy => CACHE_TTL,
            Self::Outcome { .. } | Self::Failure { .. } => DEGRADED_SEARCH_TTL,
        }
    }
}

impl SearchCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) async fn get_or_search<F, Fut>(
        &self,
        query: &str,
        limit: usize,
        search: F,
    ) -> Result<Vec<SearchCandidate>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<SearchOutcome>>,
    {
        // A SearXNG first-page response does not depend on how many results a
        // Sift caller ultimately wants. The search layer always loads one
        // maximal candidate pool, so requests for 4 and 8 results can share
        // the same upstream work and slice it at the boundary.
        let key = search_cache_key(query);
        let search_load = self.load_for(key).await;
        let original_error = std::sync::Mutex::new(None);
        let cached = search_load
            .load
            .get_or_init(|| async {
                match search().await {
                    Ok(outcome) => CachedSearch::Outcome {
                        outcome,
                        completed: Instant::now(),
                    },
                    Err(error) => {
                        let cached = CachedSearchError::capture(&error);
                        *original_error
                            .lock()
                            .expect("original search error mutex is not poisoned") = Some(error);
                        CachedSearch::Failure {
                            error: cached,
                            completed: Instant::now(),
                        }
                    }
                }
            })
            .await;

        match cached {
            CachedSearch::Outcome { outcome, .. } if outcome.healthy => {
                Ok(outcome.candidates.to_vec())
            }
            CachedSearch::Outcome { outcome, .. } => {
                if let Some(stale) = search_load.stale {
                    tracing::debug!(
                        degraded_results = outcome.candidates.len(),
                        stale_results = stale.len(),
                        "backfilling degraded SearXNG results from the last healthy search pool"
                    );
                    Ok(backfill_degraded(&outcome.candidates, &stale, limit))
                } else {
                    Ok(outcome.candidates.to_vec())
                }
            }
            CachedSearch::Failure { error, .. } => {
                if let Some(stale) = search_load.stale {
                    tracing::debug!(
                        error = ?error,
                        stale_results = stale.len(),
                        "serving the last healthy search pool after a SearXNG failure"
                    );
                    Ok(stale.to_vec())
                } else {
                    let original = original_error
                        .lock()
                        .expect("original search error mutex is not poisoned")
                        .take();
                    Err(original.unwrap_or_else(|| error.to_app_error()))
                }
            }
        }
    }

    async fn load_for(&self, key: String) -> SearchLoad {
        let now = Instant::now();
        let mut entries = self.entries.lock().await;
        if let Some(entry) = entries.get(&key)
            && search_entry_is_fresh(entry, now)
        {
            return SearchLoad {
                load: entry.load.clone(),
                stale: fresh_stale(entry, now),
            };
        }

        let stale = entries
            .remove(&key)
            .as_ref()
            .and_then(|entry| stale_from_entry(entry, now));
        if entries.len() >= MAX_CACHE_ENTRIES
            && let Some(oldest) = entries
                .iter()
                .filter(|(_, entry)| entry.load.initialized())
                .min_by_key(|(_, entry)| entry.inserted)
                .map(|(key, _)| key.clone())
        {
            entries.remove(&oldest);
        }

        let load = Arc::new(OnceCell::new());
        if entries.len() < MAX_CACHE_ENTRIES {
            entries.insert(
                key,
                SearchCacheEntry {
                    inserted: now,
                    load: load.clone(),
                    stale: stale.clone(),
                },
            );
        }
        SearchLoad {
            load,
            stale: stale.map(|stale| stale.candidates),
        }
    }
}

fn backfill_degraded(
    current: &[SearchCandidate],
    stale: &[SearchCandidate],
    limit: usize,
) -> Vec<SearchCandidate> {
    if current.is_empty() {
        return stale.iter().take(limit).cloned().collect();
    }

    let mut results = Vec::with_capacity(limit.min(current.len() + stale.len()));
    let mut urls = HashSet::with_capacity(limit);
    for candidate in current.iter().chain(stale) {
        if urls.insert(candidate.hit.url.clone()) {
            results.push(candidate.clone());
            if results.len() == limit {
                break;
            }
        }
    }
    results
}

fn search_cache_key(query: &str) -> String {
    let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
    let preserves_case = normalized
        .split(|character: char| {
            !character.is_alphanumeric() && character != '_' && character != ':'
        })
        .filter(|token| !token.is_empty())
        .any(|token| {
            let uppercase = token
                .chars()
                .filter(|character| character.is_ascii_uppercase())
                .count();
            (token
                .chars()
                .all(|character| character.is_ascii_uppercase())
                && super::evidence::recommendation_authority(token).is_some())
                || token.contains('_')
                || token.contains("::")
                || (uppercase >= 2
                    && token
                        .chars()
                        .any(|character| character.is_ascii_lowercase()))
        });
    if preserves_case {
        normalized
    } else {
        normalized.to_ascii_lowercase()
    }
}

fn search_entry_is_fresh(entry: &SearchCacheEntry, now: Instant) -> bool {
    entry
        .load
        .get()
        .is_none_or(|cached| now.saturating_duration_since(cached.completed()) < cached.ttl())
}

fn fresh_stale(entry: &SearchCacheEntry, now: Instant) -> Option<Arc<[SearchCandidate]>> {
    entry
        .stale
        .as_ref()
        .filter(|stale| now.saturating_duration_since(stale.inserted) < STALE_SEARCH_TTL)
        .map(|stale| stale.candidates.clone())
}

fn stale_from_entry(entry: &SearchCacheEntry, now: Instant) -> Option<StaleSearch> {
    if let Some(CachedSearch::Outcome { outcome, completed }) = entry.load.get()
        && outcome.healthy
        && now.saturating_duration_since(*completed) < STALE_SEARCH_TTL
    {
        return Some(StaleSearch {
            inserted: *completed,
            candidates: outcome.candidates.clone(),
        });
    }
    entry
        .stale
        .as_ref()
        .filter(|stale| now.saturating_duration_since(stale.inserted) < STALE_SEARCH_TTL)
        .cloned()
}

impl PageCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) async fn get_or_extract<F, Fut>(
        &self,
        url: &Url,
        extract: F,
    ) -> Option<SharedContent>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Option<String>>,
    {
        let load = self.load_for(url).await;
        let content = load
            .get_or_init(|| async { extract().await.map(SharedContent::from) })
            .await
            .clone();

        if content
            .as_ref()
            .is_none_or(|content| content.len() > MAX_CACHE_ENTRY_BYTES)
        {
            self.remove_if_same(url, &load).await;
        }

        content
    }

    async fn load_for(&self, url: &Url) -> SharedLoad {
        let now = Instant::now();
        let mut entries = self.entries.lock().await;
        entries.retain(|_, entry| now.duration_since(entry.inserted) < CACHE_TTL);

        if let Some(entry) = entries.get(url) {
            return entry.load.clone();
        }

        if entries.len() >= MAX_CACHE_ENTRIES
            && let Some(oldest) = entries
                .iter()
                .filter(|(_, entry)| entry.load.initialized())
                .min_by_key(|(_, entry)| entry.inserted)
                .map(|(url, _)| url.clone())
        {
            entries.remove(&oldest);
        }

        let load = Arc::new(OnceCell::new());
        if entries.len() < MAX_CACHE_ENTRIES {
            entries.insert(
                url.clone(),
                CacheEntry {
                    inserted: now,
                    load: load.clone(),
                },
            );
        }
        load
    }

    async fn remove_if_same(&self, url: &Url, load: &SharedLoad) {
        let mut entries = self.entries.lock().await;
        if entries
            .get(url)
            .is_some_and(|entry| Arc::ptr_eq(&entry.load, load))
        {
            entries.remove(url);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::search::searxng::Hit;

    fn hit(title: &str) -> SearchCandidate {
        hit_at(title, "https://example.com/result")
    }

    fn hit_at(title: &str, url: &str) -> SearchCandidate {
        let url = Url::parse(url).unwrap();
        SearchCandidate {
            hit: Hit {
                title: title.into(),
                url: url.clone(),
                date: None,
                snippet: "result".into(),
            },
            source_priority: false,
            upstream_consensus: false,
            fetch_urls: vec![url],
        }
    }

    fn healthy(title: &str) -> SearchOutcome {
        SearchOutcome::new(vec![hit(title)], true)
    }

    #[tokio::test]
    async fn successful_extractions_are_cached() {
        let cache = PageCache::new();
        let url = Url::parse("https://example.com/article").unwrap();
        let calls = AtomicUsize::new(0);

        for _ in 0..2 {
            let content = cache
                .get_or_extract(&url, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Some("markdown".to_owned())
                })
                .await;
            assert_eq!(content.as_deref(), Some("markdown"));
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn concurrent_extractions_are_coalesced() {
        let cache = PageCache::new();
        let url = Url::parse("https://example.com/article").unwrap();
        let calls = AtomicUsize::new(0);

        let (left, right) = tokio::join!(
            cache.get_or_extract(&url, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                Some("markdown".to_owned())
            }),
            cache.get_or_extract(&url, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Some("duplicate".to_owned())
            }),
        );

        assert_eq!(left.as_deref(), Some("markdown"));
        assert_eq!(right.as_deref(), Some("markdown"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn failed_extractions_are_retried() {
        let cache = PageCache::new();
        let url = Url::parse("https://example.com/article").unwrap();
        let calls = AtomicUsize::new(0);

        for _ in 0..2 {
            assert!(
                cache
                    .get_or_extract(&url, || async {
                        calls.fetch_add(1, Ordering::Relaxed);
                        None
                    })
                    .await
                    .is_none()
            );
        }

        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn successful_searches_are_whitespace_normalized_and_cached() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        for query in ["Rust  timeout", "Rust timeout"] {
            let hits = cache
                .get_or_search(query, 8, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(healthy("Timeout"))
                })
                .await
                .unwrap();
            assert_eq!(hits[0].title, "Timeout");
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn requested_result_counts_share_one_search_pool() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        for limit in [4, 32] {
            let hits = cache
                .get_or_search("same query", limit, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(healthy("shared"))
                })
                .await
                .unwrap();
            assert_eq!(hits[0].title, "shared");
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn ordinary_capitalization_does_not_multiply_upstream_searches() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        for query in ["Rust timeout", "rust timeout", "RUST TIMEOUT"] {
            cache
                .get_or_search(query, 8, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(healthy("Timeout"))
                })
                .await
                .unwrap();
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn case_sensitive_search_plans_do_not_share_cache_entries() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        for query in ["WHO guidelines", "who guidelines"] {
            cache
                .get_or_search(query, 8, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(healthy(query))
                })
                .await
                .unwrap();
        }

        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn degraded_ttl_starts_when_a_slow_search_completes() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);
        let query = "query";

        let degraded = cache
            .get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(CACHE_TTL).await;
                Ok(SearchOutcome::new(vec![hit("partial")], false))
            })
            .await
            .unwrap();
        assert_eq!(degraded[0].title, "partial");

        tokio::time::advance(DEGRADED_SEARCH_TTL - Duration::from_secs(1)).await;
        let cached = cache
            .get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(healthy("too early"))
            })
            .await
            .unwrap();
        assert_eq!(cached[0].title, "partial");
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        tokio::time::advance(Duration::from_secs(2)).await;
        let recovered = cache
            .get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(healthy("recovered"))
            })
            .await
            .unwrap();
        assert_eq!(recovered[0].title, "recovered");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn degraded_results_precede_deduplicated_stale_backfill() {
        let cache = SearchCache::new();
        let query = "latest status";

        cache
            .get_or_search(query, 3, || async {
                Ok(SearchOutcome::new(
                    vec![
                        hit_at("old one", "https://example.com/one"),
                        hit_at("old two", "https://example.com/two"),
                        hit_at("old three", "https://example.com/three"),
                    ],
                    true,
                ))
            })
            .await
            .unwrap();

        tokio::time::advance(CACHE_TTL).await;
        let results = cache
            .get_or_search(query, 3, || async {
                Ok(SearchOutcome::new(
                    vec![
                        hit_at("current", "https://example.com/current"),
                        hit_at("updated two", "https://example.com/two"),
                    ],
                    false,
                ))
            })
            .await
            .unwrap();

        assert_eq!(
            results
                .iter()
                .map(|candidate| candidate.title.as_str())
                .collect::<Vec<_>>(),
            ["current", "updated two", "old one"]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_failed_refreshes_share_stale_and_back_off() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);
        let query = "Europa Clipper status";

        cache
            .get_or_search(query, 8, || async { Ok(healthy("healthy")) })
            .await
            .unwrap();
        tokio::time::advance(CACHE_TTL).await;

        let (first, second, third) = tokio::join!(
            cache.get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_secs(5)).await;
                Err(anyhow::anyhow!("first refresh failed").into())
            }),
            cache.get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_secs(5)).await;
                Err(anyhow::anyhow!("second refresh failed").into())
            }),
            cache.get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_secs(5)).await;
                Err(anyhow::anyhow!("third refresh failed").into())
            }),
        );

        for result in [first, second, third] {
            assert_eq!(result.unwrap()[0].title, "healthy");
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let cached_stale = cache
            .get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(healthy("too early"))
            })
            .await
            .unwrap();
        assert_eq!(cached_stale[0].title, "healthy");
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        tokio::time::advance(DEGRADED_SEARCH_TTL).await;
        let recovered = cache
            .get_or_search(query, 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(healthy("recovered"))
            })
            .await
            .unwrap();
        assert_eq!(recovered[0].title, "recovered");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_initial_failures_are_coalesced_until_retry_backoff_expires() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        let (first, second) = tokio::join!(
            cache.get_or_search("query", 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_secs(5)).await;
                Err(anyhow::anyhow!("first search failed").into())
            }),
            cache.get_or_search("query", 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_secs(5)).await;
                Err(anyhow::anyhow!("second search failed").into())
            }),
        );
        assert!(first.is_err());
        assert!(second.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let cached_failure = cache
            .get_or_search("query", 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(healthy("too early"))
            })
            .await;
        assert!(cached_failure.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        tokio::time::advance(DEGRADED_SEARCH_TTL).await;
        let recovered = cache
            .get_or_search("query", 8, || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(healthy("recovered"))
            })
            .await
            .unwrap();

        assert_eq!(recovered[0].title, "recovered");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn initializing_caller_receives_the_original_search_error() {
        let cache = SearchCache::new();

        let error = cache
            .get_or_search("query", 8, || async {
                Err(AppError::SearchBackendResponse("original detail".into()))
            })
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            AppError::SearchBackendResponse(detail) if detail == "original detail"
        ));
    }
}
