use std::fmt;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;
use url::Url;

use crate::embeddings::EmbeddingBackend;

#[derive(Clone)]
pub struct StackExchangeKey(String);

impl StackExchangeKey {
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for StackExchangeKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[redacted]")
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bind: SocketAddr,
    pub searxng_url: Url,
    pub searxng_timeout: Duration,
    pub searxng_categories: String,
    pub searxng_concurrency: NonZeroUsize,
    pub searxng_min_interval: Duration,
    pub stackexchange_key: Option<StackExchangeKey>,
    pub sift_debug_dir: Option<PathBuf>,
    pub embedding_enabled: bool,
    pub embedding_backend: EmbeddingBackend,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{key} is not valid: {source}")]
    Invalid {
        key: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

fn var<T>(key: &'static str, default: T) -> Result<T, ConfigError>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match std::env::var(key) {
        Ok(raw) if !raw.trim().is_empty() => {
            raw.trim()
                .parse()
                .map_err(|e: T::Err| ConfigError::Invalid {
                    key,
                    source: Box::new(e),
                })
        }
        _ => Ok(default),
    }
}

fn secs(key: &'static str, default: u64) -> Result<Duration, ConfigError> {
    Ok(Duration::from_secs(var(key, default)?))
}

fn optional_var(key: &'static str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut searxng_url: Url = var(
            "SEARXNG_URL",
            Url::parse("http://localhost:8080").expect("valid literal"),
        )?;
        searxng_url.set_path("");
        searxng_url.set_query(None);
        searxng_url.set_fragment(None);

        let embedding_backend = var("EMBEDDING_DEVICE", EmbeddingBackend::Cpu)?;

        Ok(Self {
            bind: var("BIND", "0.0.0.0:8099".parse().expect("valid literal"))?,
            searxng_url,
            searxng_timeout: secs("SEARXNG_TIMEOUT_SECS", 25)?,
            searxng_categories: var(
                "SEARXNG_CATEGORIES",
                "general,it,science,news,map".to_owned(),
            )?,
            searxng_concurrency: var(
                "SEARXNG_CONCURRENCY",
                NonZeroUsize::new(2).expect("non-zero literal"),
            )?,
            searxng_min_interval: Duration::from_millis(var("SEARXNG_MIN_INTERVAL_MS", 3_000)?),
            stackexchange_key: optional_var("STACKEXCHANGE_KEY").map(StackExchangeKey),
            sift_debug_dir: std::env::var_os("SIFT_DEBUG_DIR")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from),
            embedding_enabled: var("EMBEDDING_ENABLED", true)?,
            embedding_backend,
        })
    }
}
