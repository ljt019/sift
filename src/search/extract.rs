use anyhow::{Result, bail};
use url::Url;

const MAX_TREE_ELEMENTS: usize = 100_000;

pub fn run(html: &str, url: &Url) -> Result<String> {
    let options = trafilatura::Options::default()
        .with_fallback(true)
        .with_links(true)
        .with_exclude_comments(true)
        .with_max_tree_size(MAX_TREE_ELEMENTS)
        .with_url(url.clone());
    let result = trafilatura::extract(html, &options)?;
    let markdown = result.content_markdown();
    let markdown = markdown.trim();
    if markdown.is_empty() {
        bail!("extractor returned no content");
    }
    Ok(markdown.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_article_as_markdown() {
        let html = r#"
            <html><body><main><article>
              <h1>Rust error handling</h1>
              <p>Use structured error types at library boundaries and attach context at application boundaries.</p>
              <p>This second paragraph makes the article substantial enough for extraction.</p>
            </article></main></body></html>
        "#;

        let markdown = run(html, &Url::parse("https://example.com/errors").unwrap()).unwrap();

        assert!(markdown.contains("Rust error handling"));
        assert!(markdown.contains("structured error types"));
    }
}
