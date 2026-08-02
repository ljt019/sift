use std::borrow::Cow;

use anyhow::{Result, bail};
use scraper::{Html, Selector};
use url::Url;

const MAX_TREE_ELEMENTS: usize = 100_000;

pub fn run(html: &str, url: &Url) -> Result<String> {
    // Large client-side state blobs can overwhelm the visible document and
    // occasionally win extraction as escaped JSON. They cannot contribute
    // readable page content, so discard them before building the extraction
    // tree. This is especially important for Discourse's `data-preloaded`
    // payload, which duplicates every visible post inside one script tag.
    let html = discourse_content(html)
        .map(Cow::Owned)
        .unwrap_or_else(|| without_scripts(html));
    let options = trafilatura::Options::default()
        .with_fallback(true)
        .with_links(true)
        .with_exclude_comments(true)
        .with_max_tree_size(MAX_TREE_ELEMENTS)
        .with_url(url.clone());
    let result = trafilatura::extract(&html, &options)?;
    let markdown = result.content_markdown();
    let markdown = markdown.trim();
    if markdown.is_empty() {
        bail!("extractor returned no content");
    }
    if client_only_shell(markdown) {
        bail!("page only exposed a client-rendered shell");
    }
    Ok(markdown.to_owned())
}

fn discourse_content(html: &str) -> Option<String> {
    if !html.contains("crawler-post") || !html.contains("Discourse") {
        return None;
    }

    // Discourse serves the crawler view directly to simple clients, but wraps
    // that same complete HTML in `<noscript>` for browsers. HTML parsers treat
    // a noscript body as text when scripting is enabled, which is why generic
    // extraction escaped every tag instead of reading the posts.
    let crawler = discourse_noscript_fragment(html).unwrap_or(html);
    let document = Html::parse_fragment(crawler);
    let title_selector = Selector::parse("#topic-title h1").ok()?;
    let post_selector = Selector::parse(".crawler-post .post").ok()?;
    let posts = document.select(&post_selector).collect::<Vec<_>>();
    if posts.is_empty() {
        return None;
    }

    let mut focused = String::from("<main><article>");
    if let Some(title) = document.select(&title_selector).next() {
        focused.push_str(&title.html());
    }
    for post in posts {
        focused.push_str(&post.html());
    }
    focused.push_str("</article></main>");
    Some(focused)
}

fn discourse_noscript_fragment(html: &str) -> Option<&str> {
    let lower = html.to_ascii_lowercase();
    let mut search_from = 0;
    while let Some(relative_start) = lower[search_from..].find("<noscript") {
        let start = search_from + relative_start;
        let open_end = lower[start..].find('>')? + start + 1;
        let close = lower[open_end..].find("</noscript>")? + open_end;
        if lower[open_end..close].contains("crawler-post") {
            return Some(&html[open_end..close]);
        }
        search_from = close + "</noscript>".len();
    }
    None
}

fn without_scripts(html: &str) -> Cow<'_, str> {
    let lower = html.to_ascii_lowercase();
    let mut search_from = 0;
    let mut retained_from = 0;
    let mut cleaned = None::<String>;

    while let Some(relative_start) = lower[search_from..].find("<script") {
        let start = search_from + relative_start;
        let boundary = lower.as_bytes().get(start + "<script".len()).copied();
        if !boundary.is_some_and(|byte| byte == b'>' || byte == b'/' || byte.is_ascii_whitespace())
        {
            search_from = start + "<script".len();
            continue;
        }

        let output = cleaned.get_or_insert_with(|| String::with_capacity(html.len()));
        output.push_str(&html[retained_from..start]);
        let Some(relative_close) = lower[start + "<script".len()..].find("</script") else {
            retained_from = html.len();
            break;
        };
        let close = start + "<script".len() + relative_close;
        let Some(relative_end) = lower[close..].find('>') else {
            retained_from = html.len();
            break;
        };
        retained_from = close + relative_end + 1;
        search_from = retained_from;
    }

    match cleaned {
        Some(mut cleaned) => {
            cleaned.push_str(&html[retained_from..]);
            Cow::Owned(cleaned)
        }
        None => Cow::Borrowed(html),
    }
}

fn client_only_shell(markdown: &str) -> bool {
    if markdown.chars().count() > 1_000 {
        return false;
    }
    let lower = markdown.to_ascii_lowercase();
    let messages = [
        "enable javascript to run this app",
        "javascript is required to view this",
        "please enable javascript",
        "this site requires javascript",
    ];
    let Some(message) = messages.iter().find(|message| lower.contains(*message)) else {
        return false;
    };
    lower
        .replace(message, "")
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count()
        < 200
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

    #[test]
    fn ignores_large_preloaded_script_payloads() {
        let escaped_post =
            r#"{\"cooked\":\"\\u003cp\\u003eWrong escaped copy\\u003c/p\\u003e\"}"#.repeat(2_000);
        let html = format!(
            r#"
                <html><body>
                  <script type="application/json" id="data-preloaded">{escaped_post}</script>
                  <main><article>
                    <h1>Forum answer</h1>
                    <p>The visible answer explains how to return Result from the async block.</p>
                    <pre><code>async {{ operation()?; Ok(()) }}</code></pre>
                  </article></main>
                </body></html>
            "#
        );

        let markdown = run(
            &html,
            &Url::parse("https://forum.example/t/answer/1").unwrap(),
        )
        .unwrap();

        assert!(markdown.contains("visible answer"));
        assert!(!markdown.contains("Wrong escaped copy"));
        assert!(!markdown.contains(r#"\u003c"#));
    }

    #[test]
    fn script_stripping_does_not_match_longer_tag_names() {
        let html = "<scripting>keep this</scripting><p>and this</p>";

        assert_eq!(without_scripts(html), html);
    }

    #[test]
    fn extracts_discourse_crawler_posts_from_noscript() {
        let html = r#"
            <html><head>
              <meta name="generator" content="Discourse 2026.8.0">
              <script type="application/json" id="data-preloaded">
                {"cooked":"\\u003cp\\u003eEscaped duplicate\\u003c/p\\u003e"}
              </script>
            </head><body>
              <noscript>
                <div id="topic-title"><h1>Async error help</h1></div>
                <div class="topic-body crawler-post">
                  <div class="post">
                    <p>So the issue here is that the future returns the unit type.</p>
                    <pre><code>async { operation()?; Ok(()) }</code></pre>
                  </div>
                </div>
                <div id="related-topics">Unrelated topic list</div>
              </noscript>
            </body></html>
        "#;

        let markdown = run(
            html,
            &Url::parse("https://forum.example/t/answer/1").unwrap(),
        )
        .unwrap();

        assert!(markdown.contains("future returns the unit type"));
        assert!(!markdown.contains(r#"\<pre"#));
        assert!(!markdown.contains(r#"\u003c"#));
        assert!(!markdown.contains("Unrelated topic list"));
    }

    #[test]
    fn rejects_client_only_application_shells() {
        let html = r#"
            <html><body><main>
              <h1>Treaty database</h1>
              <p>You need to enable JavaScript to run this app.</p>
            </main></body></html>
        "#;

        let error = run(html, &Url::parse("https://example.com/article-3").unwrap()).unwrap_err();

        assert!(error.to_string().contains("client-rendered shell"));
    }

    #[test]
    fn keeps_concise_content_that_happens_to_include_a_noscript_warning() {
        let html = r#"
            <html><body><main><article>
              <h1>Service status</h1>
              <p>The spacecraft remains healthy in its inner cruise phase. Mission controllers completed the latest navigation campaign and verified all science instruments after the Mars gravity assist. The next planned milestone is an Earth flyby, followed by arrival at Jupiter.</p>
              <p>You need to enable JavaScript to run this app.</p>
            </article></main></body></html>
        "#;

        let markdown = run(html, &Url::parse("https://example.com/status").unwrap()).unwrap();

        assert!(markdown.contains("spacecraft remains healthy"));
    }
}
