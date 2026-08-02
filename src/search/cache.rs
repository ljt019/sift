use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, OnceCell};
use tokio::time::Instant;
use url::Url;

use super::searxng::Hit;
use crate::Result;

const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_CACHE_ENTRIES: usize = 256;
const MAX_CACHE_ENTRY_BYTES: usize = 512 * 1024;

type SharedContent = Arc<str>;
type SharedLoad = Arc<OnceCell<Option<SharedContent>>>;
type SharedSearch = Arc<OnceCell<Arc<[Hit]>>>;

pub(crate) struct PageCache {
    entries: Mutex<HashMap<Url, CacheEntry>>,
}

struct CacheEntry {
    inserted: Instant,
    load: SharedLoad,
}

pub(crate) struct SearchCache {
    entries: Mutex<HashMap<(String, usize), SearchCacheEntry>>,
}

struct SearchCacheEntry {
    inserted: Instant,
    load: SharedSearch,
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
    ) -> Result<Vec<Hit>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Vec<Hit>>>,
    {
        let key = (
            query
                .split_whitespace()
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
                .join(" "),
            limit,
        );
        let load = self.load_for(key.clone()).await;
        match load
            .get_or_try_init(|| async { search().await.map(Arc::from) })
            .await
        {
            Ok(hits) => Ok(hits.to_vec()),
            Err(error) => {
                self.remove_if_same(&key, &load).await;
                Err(error)
            }
        }
    }

    async fn load_for(&self, key: (String, usize)) -> SharedSearch {
        let now = Instant::now();
        let mut entries = self.entries.lock().await;
        entries.retain(|_, entry| now.duration_since(entry.inserted) < CACHE_TTL);

        if let Some(entry) = entries.get(&key) {
            return entry.load.clone();
        }
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
                },
            );
        }
        load
    }

    async fn remove_if_same(&self, key: &(String, usize), load: &SharedSearch) {
        let mut entries = self.entries.lock().await;
        if entries
            .get(key)
            .is_some_and(|entry| Arc::ptr_eq(&entry.load, load))
        {
            entries.remove(key);
        }
    }
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

    fn hit(title: &str) -> Hit {
        Hit {
            title: title.into(),
            url: Url::parse("https://example.com/result").unwrap(),
            date: None,
            snippet: "result".into(),
        }
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
    async fn successful_searches_are_normalized_and_cached() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        for query in ["Rust  timeout", "rust timeout"] {
            let hits = cache
                .get_or_search(query, 8, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(vec![hit("Timeout")])
                })
                .await
                .unwrap();
            assert_eq!(hits[0].title, "Timeout");
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn failed_searches_are_retried() {
        let cache = SearchCache::new();
        let calls = AtomicUsize::new(0);

        for _ in 0..2 {
            let result = cache
                .get_or_search("query", 8, || async {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Err(anyhow::anyhow!("search failed").into())
                })
                .await;
            assert!(result.is_err());
        }

        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
