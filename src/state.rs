use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, Semaphore};
use tokio::time::Instant;

use crate::config::Config;
use crate::embeddings::EmbeddingWorker;
use crate::search::{PageCache, PublicDnsResolver, SearchCache, StackExchangeClient};

const CLIENT_UA: &str = concat!("sift/", env!("CARGO_PKG_VERSION"));
const MAX_CONCURRENT_FETCHES: usize = 32;
const MAX_CONCURRENT_EXTRACTIONS: usize = 8;

#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub config: Config,
    pub http: reqwest::Client,
    pub page_http: reqwest::Client,
    pub searxng_permits: Semaphore,
    pub searxng_next_request: Mutex<Instant>,
    pub github_next_request: Mutex<Instant>,
    pub fetch_permits: Semaphore,
    pub extract_permits: Arc<Semaphore>,
    pub(crate) page_cache: PageCache,
    pub(crate) search_cache: SearchCache,
    pub embeddings: Option<EmbeddingWorker>,
    pub(crate) stackexchange: StackExchangeClient,
}

impl AppState {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            // Per-request timeouts are applied explicitly; these are ceilings.
            .connect_timeout(Duration::from_secs(5))
            .pool_max_idle_per_host(8)
            .redirect(reqwest::redirect::Policy::limited(5))
            .user_agent(CLIENT_UA)
            .build()?;

        let page_http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .pool_max_idle_per_host(8)
            // A system proxy could resolve a blocked destination on the
            // server's behalf, bypassing the resolver policy below.
            .no_proxy()
            .dns_resolver(PublicDnsResolver)
            // Fetch redirects are followed manually so every target is checked.
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(CLIENT_UA)
            .build()?;

        let searxng_permits = Semaphore::new(config.searxng_concurrency.get());
        let searxng_next_request = Mutex::new(Instant::now());
        let github_next_request = Mutex::new(Instant::now());
        let fetch_permits = Semaphore::new(MAX_CONCURRENT_FETCHES);
        let extract_permits = Arc::new(Semaphore::new(extraction_concurrency()));
        let page_cache = PageCache::new();
        let search_cache = SearchCache::new();
        let stackexchange = StackExchangeClient::new(
            http.clone(),
            config
                .stackexchange_key
                .as_ref()
                .map(|key| key.expose().to_owned()),
        );
        let embeddings = config
            .embedding_enabled
            .then(|| EmbeddingWorker::spawn(config.embedding_backend))
            .transpose()?;

        Ok(Self(Arc::new(Inner {
            config,
            http,
            page_http,
            searxng_permits,
            searxng_next_request,
            github_next_request,
            fetch_permits,
            extract_permits,
            page_cache,
            search_cache,
            embeddings,
            stackexchange,
        })))
    }
}

fn extraction_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|threads| threads.get().div_ceil(2).min(MAX_CONCURRENT_EXTRACTIONS))
        .unwrap_or(1)
}

impl std::ops::Deref for AppState {
    type Target = Inner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
