use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};
use url::Url;

use super::fetch::FetchFailure;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExtractionRecord<'a> {
    url: &'a str,
    domain: &'a str,
    raw_byte_length: usize,
    extracted_char_length: usize,
    extracted_raw_ratio: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FetchFailureRecord<'a> {
    url: &'a str,
    domain: &'a str,
    final_url: Option<&'a str>,
    status: Option<u16>,
    content_type: Option<&'a str>,
    headers: &'a [(String, String)],
    body_preview: Option<&'a str>,
    error: &'a str,
}

pub async fn capture(directory: Option<&Path>, url: &Url, html: &str, markdown: &str) {
    let Some(directory) = directory else {
        return;
    };
    if let Err(error) = write_capture(directory, url, html, markdown).await {
        tracing::debug!(url = %url, path = %directory.display(), error = ?error, "failed to write extraction diagnostics");
    }
}

pub async fn capture_fetch_failure(directory: Option<&Path>, url: &Url, failure: &FetchFailure) {
    let Some(directory) = directory else {
        return;
    };
    if let Err(error) = write_fetch_failure(directory, url, failure).await {
        tracing::debug!(url = %url, path = %directory.display(), error = ?error, "failed to write fetch diagnostics");
    }
}

async fn write_fetch_failure(
    directory: &Path,
    url: &Url,
    failure: &FetchFailure,
) -> anyhow::Result<()> {
    tokio::fs::create_dir_all(directory).await?;
    let record = FetchFailureRecord {
        url: url.as_str(),
        domain: url.domain().unwrap_or_default(),
        final_url: failure.final_url.as_ref().map(Url::as_str),
        status: failure.status,
        content_type: failure.content_type.as_deref(),
        headers: &failure.headers,
        body_preview: failure.body_preview.as_deref(),
        error: &failure.error,
    };
    let path = directory.join(format!("{}.fetch.json", url_hash(url)));
    tokio::fs::write(path, serde_json::to_vec_pretty(&record)?).await?;
    Ok(())
}

async fn write_capture(
    directory: &Path,
    url: &Url,
    html: &str,
    markdown: &str,
) -> anyhow::Result<()> {
    tokio::fs::create_dir_all(directory).await?;
    let stem = url_hash(url);
    let raw_byte_length = html.len();
    let extracted_char_length = markdown.chars().count();
    let extracted_raw_ratio = if raw_byte_length == 0 {
        0.0
    } else {
        extracted_char_length as f64 / raw_byte_length as f64
    };
    let record = ExtractionRecord {
        url: url.as_str(),
        domain: url.domain().unwrap_or_default(),
        raw_byte_length,
        extracted_char_length,
        extracted_raw_ratio,
    };
    let metadata = serde_json::to_vec_pretty(&record)?;

    tokio::try_join!(
        tokio::fs::write(directory.join(format!("{stem}.html")), html),
        tokio::fs::write(directory.join(format!("{stem}.md")), markdown),
        tokio::fs::write(directory.join(format!("{stem}.json")), metadata),
    )?;
    Ok(())
}

fn url_hash(url: &Url) -> String {
    let digest = Sha256::digest(url.as_str().as_bytes());
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(hash, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_the_three_url_keyed_artifacts() {
        let directory =
            std::env::temp_dir().join(format!("sift-extraction-debug-test-{}", std::process::id()));
        let url = Url::parse("https://example.com/article").unwrap();
        write_capture(&directory, &url, "<p>héllo</p>", "héllo")
            .await
            .unwrap();

        let stem = url_hash(&url);
        assert_eq!(
            tokio::fs::read_to_string(directory.join(format!("{stem}.html")))
                .await
                .unwrap(),
            "<p>héllo</p>"
        );
        assert_eq!(
            tokio::fs::read_to_string(directory.join(format!("{stem}.md")))
                .await
                .unwrap(),
            "héllo"
        );
        let metadata = tokio::fs::read_to_string(directory.join(format!("{stem}.json")))
            .await
            .unwrap();
        assert!(metadata.contains("\"rawByteLength\": 13"));
        assert!(metadata.contains("\"extractedCharLength\": 5"));

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn unset_directory_is_a_no_op() {
        capture(
            None,
            &Url::parse("https://example.com").unwrap(),
            "html",
            "markdown",
        )
        .await;
    }

    #[tokio::test]
    async fn writes_fetch_failure_separately_from_extraction_artifacts() {
        let directory =
            std::env::temp_dir().join(format!("sift-fetch-debug-test-{}", std::process::id()));
        let url = Url::parse("https://example.com/blocked").unwrap();
        let failure = FetchFailure {
            final_url: Some(Url::parse("https://www.example.com/wall").unwrap()),
            status: Some(403),
            content_type: Some("text/html".into()),
            headers: vec![("server".into(), "example".into())],
            body_preview: Some("Access denied".into()),
            error: "page returned HTTP 403".into(),
        };

        write_fetch_failure(&directory, &url, &failure)
            .await
            .unwrap();

        let stem = url_hash(&url);
        let metadata = tokio::fs::read_to_string(directory.join(format!("{stem}.fetch.json")))
            .await
            .unwrap();
        assert!(metadata.contains("\"status\": 403"));
        assert!(metadata.contains("\"bodyPreview\": \"Access denied\""));
        assert!(!directory.join(format!("{stem}.html")).exists());

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn write_failures_are_swallowed() {
        let path = std::env::temp_dir().join(format!(
            "sift-extraction-debug-file-test-{}",
            std::process::id()
        ));
        tokio::fs::write(&path, "not a directory").await.unwrap();

        capture(
            Some(&path),
            &Url::parse("https://example.com").unwrap(),
            "html",
            "markdown",
        )
        .await;

        tokio::fs::remove_file(path).await.unwrap();
    }
}
