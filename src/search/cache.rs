use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, OnceCell};
use tokio::time::Instant;
use url::Url;

const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_CACHE_ENTRIES: usize = 256;
const MAX_CACHE_ENTRY_BYTES: usize = 512 * 1024;

type SharedContent = Arc<str>;
type SharedLoad = Arc<OnceCell<Option<SharedContent>>>;

pub(crate) struct PageCache {
    entries: Mutex<HashMap<Url, CacheEntry>>,
}

struct CacheEntry {
    inserted: Instant,
    load: SharedLoad,
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
}
