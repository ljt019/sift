use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use anyhow::{Context, Result, bail};
use encoding_rs::{Encoding, UTF_8, WINDOWS_1252};
use reqwest::StatusCode;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::header::{CONTENT_TYPE, LOCATION, USER_AGENT};
use url::{Host, Url};

use crate::state::AppState;

const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(12);
const CHROME_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36";
const FAILURE_PREVIEW_BYTES: usize = 2 * 1024;
const DIAGNOSTIC_HEADERS: &[&str] = &[
    "server",
    "cf-mitigated",
    "cf-ray",
    "retry-after",
    "x-cache",
    "x-cache-status",
    "x-sucuri-id",
    "x-vercel-id",
];

#[derive(Debug)]
pub struct FetchError {
    source: anyhow::Error,
    diagnostic: FetchFailure,
}

#[derive(Clone, Debug)]
pub struct FetchFailure {
    pub final_url: Option<Url>,
    pub status: Option<u16>,
    pub content_type: Option<String>,
    pub headers: Vec<(String, String)>,
    pub body_preview: Option<String>,
    pub error: String,
}

impl FetchError {
    pub fn diagnostic(&self) -> &FetchFailure {
        &self.diagnostic
    }

    fn from_error(error: anyhow::Error) -> Self {
        let diagnostic = match error.downcast_ref::<ResponseFailure>() {
            Some(failure) => failure.diagnostic.clone(),
            None => FetchFailure {
                final_url: None,
                status: None,
                content_type: None,
                headers: Vec::new(),
                body_preview: None,
                error: format!("{error:#}"),
            },
        };
        Self {
            source: error,
            diagnostic,
        }
    }
}

impl fmt::Display for FetchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}

impl std::error::Error for FetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct ResponseFailure {
    message: String,
    diagnostic: FetchFailure,
}

#[derive(Clone, Copy)]
pub(crate) struct PublicDnsResolver;

impl Resolve for PublicDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await?
                .collect::<Vec<_>>();
            if addresses.is_empty() {
                return Err(
                    std::io::Error::other(format!("{host} resolved to no addresses")).into(),
                );
            }
            if let Some(address) = addresses
                .iter()
                .map(std::net::SocketAddr::ip)
                .find(|address| !is_public_ip(*address))
            {
                return Err(std::io::Error::other(format!(
                    "{host} resolved to non-public address {address}"
                ))
                .into());
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

pub async fn get(state: &AppState, url: &Url) -> std::result::Result<String, FetchError> {
    let _permit = state
        .fetch_permits
        .acquire()
        .await
        .context("fetch semaphore closed")
        .map_err(FetchError::from_error)?;

    match tokio::time::timeout(FETCH_TIMEOUT, get_with_redirects(state, url)).await {
        Ok(result) => result.map_err(FetchError::from_error),
        Err(error) => Err(FetchError::from_error(
            anyhow::Error::new(error)
                .context(format!("page fetch timed out after {FETCH_TIMEOUT:?}")),
        )),
    }
}

async fn get_with_redirects(state: &AppState, url: &Url) -> Result<String> {
    let mut current = url.clone();
    let mut redirects = 0;
    let mut spoof_browser = true;
    let mut retried_cloudflare = false;

    loop {
        validate_destination(&current)?;

        let mut request = state.page_http.get(current.clone());
        if spoof_browser {
            request = request.header(USER_AGENT, CHROME_USER_AGENT);
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("failed to fetch {current}"))?;

        if response.status() == StatusCode::FORBIDDEN
            && response.headers().contains_key("cf-mitigated")
            && spoof_browser
            && !retried_cloudflare
        {
            retried_cloudflare = true;
            spoof_browser = false;
            continue;
        }

        if response.status().is_redirection() {
            if redirects == MAX_REDIRECTS {
                bail!("too many redirects while fetching {url}");
            }
            let location = response
                .headers()
                .get(LOCATION)
                .context("redirect response omitted Location")?
                .to_str()
                .context("redirect Location is not valid ASCII")?;
            current = current
                .join(location)
                .with_context(|| format!("invalid redirect target {location:?}"))?;
            redirects += 1;
            continue;
        }

        if !response.status().is_success() {
            return Err(http_status_failure(response).await);
        }

        let mut response = response;

        if let Some(length) = response.content_length()
            && length > MAX_BODY_BYTES as u64
        {
            bail!("page body exceeds {MAX_BODY_BYTES} bytes");
        }

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let diagnostic_headers = diagnostic_headers(response.headers());
        ensure_supported_content_type(content_type.as_deref())?;

        let mut body = Vec::with_capacity(
            response
                .content_length()
                .unwrap_or(16 * 1024)
                .min(MAX_BODY_BYTES as u64) as usize,
        );
        while let Some(chunk) = response.chunk().await.context("failed to read page body")? {
            append_limited(&mut body, &chunk, MAX_BODY_BYTES)?;
        }

        let body = decode(&body, content_type.as_deref());
        if is_known_access_wall(&current, &body) {
            let message = format!("page returned a verification wall for {current}");
            let diagnostic = FetchFailure {
                final_url: Some(current.clone()),
                status: Some(StatusCode::OK.as_u16()),
                content_type,
                headers: diagnostic_headers,
                body_preview: Some(preview(&body)),
                error: message.clone(),
            };
            return Err(anyhow::Error::new(ResponseFailure {
                message,
                diagnostic,
            }));
        }
        return Ok(body);
    }
}

async fn http_status_failure(mut response: reqwest::Response) -> anyhow::Error {
    let status = response.status().as_u16();
    let final_url = response.url().clone();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let headers = diagnostic_headers(response.headers());
    let body_preview = match response.chunk().await {
        Ok(Some(chunk)) => {
            let length = chunk.len().min(FAILURE_PREVIEW_BYTES);
            Some(preview(&decode(&chunk[..length], content_type.as_deref())))
        }
        Ok(None) => None,
        Err(error) => Some(format!("<failed to read response preview: {error}>")),
    };
    let diagnostic = FetchFailure {
        final_url: Some(final_url.clone()),
        status: Some(status),
        content_type,
        headers,
        body_preview,
        error: format!("page returned HTTP {status} for {final_url}"),
    };
    anyhow::Error::new(ResponseFailure {
        message: diagnostic.error.clone(),
        diagnostic,
    })
}

fn diagnostic_headers(headers: &reqwest::header::HeaderMap) -> Vec<(String, String)> {
    DIAGNOSTIC_HEADERS
        .iter()
        .filter_map(|&name| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(|value| (name.to_owned(), value.to_owned()))
        })
        .collect()
}

fn preview(body: &str) -> String {
    sanitize_preview(body)
        .chars()
        .take(FAILURE_PREVIEW_BYTES)
        .collect()
}

fn sanitize_preview(body: &str) -> String {
    body.chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_known_access_wall(url: &Url, body: &str) -> bool {
    let is_reddit = url.domain().is_some_and(|domain| {
        domain.eq_ignore_ascii_case("reddit.com")
            || domain.to_ascii_lowercase().ends_with(".reddit.com")
    });
    if !is_reddit {
        return false;
    }

    let redirected_to_login = url.path().starts_with("/login/")
        && url
            .query_pairs()
            .any(|(name, value)| name == "reason" && value.starts_with("lor"));
    redirected_to_login
        || body.contains("Reddit - Please wait for verification")
        || body.contains("You've been blocked by network security")
        || body.contains("whoa there, pardner")
}

fn append_limited(body: &mut Vec<u8>, chunk: &[u8], limit: usize) -> Result<()> {
    if body.len().saturating_add(chunk.len()) > limit {
        bail!("page body exceeds {limit} bytes");
    }
    body.extend_from_slice(chunk);
    Ok(())
}

fn ensure_supported_content_type(content_type: Option<&str>) -> Result<()> {
    let Some(media_type) = content_type.and_then(|value| value.split(';').next()) else {
        return Ok(());
    };
    if matches!(
        media_type.trim().to_ascii_lowercase().as_str(),
        "text/html" | "application/xhtml+xml" | "text/plain"
    ) {
        Ok(())
    } else {
        bail!("unsupported page content type {media_type:?}")
    }
}

fn decode(bytes: &[u8], content_type: Option<&str>) -> String {
    let encoding = Encoding::for_bom(bytes)
        .map(|(encoding, _)| encoding)
        .or_else(|| content_type.and_then(charset).and_then(Encoding::for_label))
        .unwrap_or(UTF_8);
    let (decoded, _, had_errors) = encoding.decode(bytes);
    if had_errors && encoding == UTF_8 {
        WINDOWS_1252.decode(bytes).0.into_owned()
    } else {
        decoded.into_owned()
    }
}

fn charset(content_type: &str) -> Option<&[u8]> {
    content_type.split(';').skip(1).find_map(|parameter| {
        let (name, value) = parameter.split_once('=')?;
        name.trim()
            .eq_ignore_ascii_case("charset")
            .then(|| value.trim().trim_matches(['\'', '"']).as_bytes())
    })
}

fn validate_destination(url: &Url) -> Result<()> {
    if !matches!(url.scheme(), "http" | "https") {
        bail!("unsupported URL scheme {:?}", url.scheme());
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("page URLs may not contain credentials");
    }

    let host = url.host().context("page URL has no host")?;
    match host {
        Host::Ipv4(address) => ensure_public_ip(IpAddr::V4(address)),
        Host::Ipv6(address) => ensure_public_ip(IpAddr::V6(address)),
        Host::Domain(domain) => {
            if domain.eq_ignore_ascii_case("localhost")
                || domain.to_ascii_lowercase().ends_with(".localhost")
            {
                bail!("local page destinations are not allowed");
            }
            Ok(())
        }
    }
}

fn ensure_public_ip(address: IpAddr) -> Result<()> {
    if is_public_ip(address) {
        Ok(())
    } else {
        bail!("non-public page destination {address} is not allowed")
    }
}

fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224)
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4() {
        return is_public_ipv4(mapped);
    }
    let segments = address.segments();
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_multicast()
        || segments[0] & 0xfe00 == 0xfc00
        || segments[0] & 0xffc0 == 0xfe80
        || segments[0] & 0xffc0 == 0xfec0
        // IANA's local-use NAT64 prefix can translate to private IPv4 even
        // though the URL itself contains a globally shaped IPv6 literal.
        || (segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2] == 0x0001)
        || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        || (segments[0] == 0x2001 && segments[1] == 0x0002))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_public_addresses() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "169.254.169.254",
            "172.20.0.1",
            "192.168.1.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "64:ff9b:1::c0a8:1",
            "::ffff:127.0.0.1",
            "::192.168.1.1",
        ] {
            assert!(
                ensure_public_ip(address.parse().unwrap()).is_err(),
                "{address}"
            );
        }
        assert!(ensure_public_ip("1.1.1.1".parse().unwrap()).is_ok());
        assert!(ensure_public_ip("2606:4700:4700::1111".parse().unwrap()).is_ok());
    }

    #[test]
    fn decodes_declared_charset() {
        assert_eq!(
            decode(
                &[0x63, 0x61, 0x66, 0xe9],
                Some("text/html; charset=iso-8859-1")
            ),
            "café"
        );
    }

    #[test]
    fn validates_content_types() {
        assert!(ensure_supported_content_type(None).is_ok());
        assert!(ensure_supported_content_type(Some("text/html; charset=utf-8")).is_ok());
        assert!(ensure_supported_content_type(Some("application/pdf")).is_err());
    }

    #[test]
    fn enforces_streaming_body_limit() {
        let mut body = b"abc".to_vec();
        append_limited(&mut body, b"de", 5).unwrap();
        assert_eq!(body, b"abcde");
        assert!(append_limited(&mut body, b"f", 5).is_err());
        assert_eq!(body, b"abcde");
    }

    #[test]
    fn failure_previews_are_single_line_and_readable() {
        assert_eq!(
            sanitize_preview("  <html>\n\tAccess\r denied </html>  "),
            "<html> Access denied </html>"
        );
    }

    #[test]
    fn recognizes_reddit_verification_wall() {
        let url = Url::parse("https://www.reddit.com/r/rust/comments/example").unwrap();

        assert!(is_known_access_wall(
            &url,
            "<title>Reddit - Please wait for verification</title>"
        ));
        assert!(!is_known_access_wall(&url, "<title>A real post</title>"));

        let login = Url::parse(
            "https://old.reddit.com/login/?reason=lor2&dest=https%3A%2F%2Fold.reddit.com%2Fr%2Frust",
        )
        .unwrap();
        assert!(is_known_access_wall(&login, ""));
        assert!(is_known_access_wall(
            &url,
            "You've been blocked by network security."
        ));
    }

    #[test]
    fn rejects_local_and_credentialed_urls_before_fetching() {
        for url in [
            "http://127.0.0.1/admin",
            "http://[::1]/admin",
            "http://localhost/admin",
            "https://user:password@example.com/",
            "file:///etc/passwd",
        ] {
            assert!(
                validate_destination(&Url::parse(url).unwrap()).is_err(),
                "{url}"
            );
        }
        assert!(validate_destination(&Url::parse("https://example.com/").unwrap()).is_ok());
    }
}
