use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::Mutex;
use tokio::time::Instant;
use url::Url;

use super::{Hit, exact_identifier_query};

const API_BASE: &str = "https://api.stackexchange.com/2.3";
// Immutable filter created through /filters/create. It includes only the
// question, answer, attribution, quota, and backoff fields used below.
const API_FILTER: &str = "2hElTY3zn8AeeG)anHfU.obxyjY10FWBuiLATZQeMSje.t";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_CACHE_ENTRIES: usize = 256;
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ANSWERS: usize = 10;
const MAX_SEARCH_RESULTS: usize = 10;
const SEARCH_BODY_CHARACTERS: usize = 400;
const SEARCH_ANSWER_CHARACTERS: usize = 800;

pub(crate) struct StackExchangeClient {
    http: Client,
    api_key: Option<String>,
    request_gate: Mutex<()>,
    state: Mutex<ClientState>,
}

struct ClientState {
    cache: HashMap<QuestionRef, CacheEntry>,
    next_request: Instant,
}

struct CacheEntry {
    markdown: String,
    inserted: Instant,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QuestionRef {
    site: String,
    id: u64,
}

struct Target {
    result_index: usize,
    question: QuestionRef,
}

impl StackExchangeClient {
    pub fn new(http: Client, api_key: Option<String>) -> Self {
        Self {
            http,
            api_key,
            request_gate: Mutex::new(()),
            state: Mutex::new(ClientState {
                cache: HashMap::new(),
                next_request: Instant::now(),
            }),
        }
    }

    /// Searches Stack Overflow directly so discovery and answer retrieval use
    /// one quota, one optional API key, and one backoff gate. SearXNG's three
    /// separate Stack Exchange engines cannot share any of those controls.
    pub async fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        if limit == 0 {
            return Vec::new();
        }
        let query = exact_identifier_query(query).unwrap_or_else(|| query.to_owned());
        match self
            .search_questions(&query, limit.min(MAX_SEARCH_RESULTS))
            .await
        {
            Ok(hits) => {
                tracing::debug!(
                    query,
                    results = hits.len(),
                    "received Stack Overflow candidates"
                );
                hits
            }
            Err(error) => {
                tracing::debug!(error = ?error, "Stack Overflow search failed; continuing without direct candidates");
                Vec::new()
            }
        }
    }

    async fn search_questions(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let _request_guard = self.request_gate.lock().await;
        self.ensure_not_backed_off().await?;

        let mut request = self
            .http
            .get(format!("{API_BASE}/search/advanced"))
            .query(&[
                ("q", query),
                ("site", "stackoverflow"),
                ("sort", "relevance"),
                ("order", "desc"),
                ("filter", API_FILTER),
            ])
            .query(&[("pagesize", limit)])
            .timeout(REQUEST_TIMEOUT);
        if let Some(api_key) = &self.api_key {
            request = request.query(&[("key", api_key)]);
        }

        let response = request
            .send()
            .await
            .context("failed to search the Stack Exchange API")?;
        let status = response.status();
        let body = read_limited(response).await?;
        if !status.is_success() {
            self.record_api_error(status, &body).await?;
        }
        let response: ApiResponse =
            serde_json::from_slice(&body).context("invalid Stack Exchange search response")?;
        self.apply_response_limits(&response).await;

        Ok(response
            .items
            .into_iter()
            .filter_map(ApiQuestion::into_search_hit)
            .take(limit)
            .collect())
    }

    /// Fetches all recognized Stack Exchange questions in as few API calls as
    /// possible. Failures are deliberately omitted so the caller can fall back
    /// to the ordinary page fetch and ultimately the SearXNG snippet.
    pub async fn fetch(&self, urls: &[(usize, Url)]) -> HashMap<usize, String> {
        let targets = urls
            .iter()
            .filter_map(|(result_index, url)| {
                question_ref(url).map(|question| Target {
                    result_index: *result_index,
                    question,
                })
            })
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return HashMap::new();
        }

        let missing = self.missing_questions(&targets).await;
        let mut groups = BTreeMap::<String, Vec<u64>>::new();
        for question in missing {
            groups.entry(question.site).or_default().push(question.id);
        }

        for (site, mut ids) in groups {
            ids.sort_unstable();
            if let Err(error) = self.fetch_group(&site, &ids).await {
                tracing::debug!(site, ids = ?ids, error = ?error, "Stack Exchange API fetch failed; using page fallback");
            }
        }

        let state = self.state.lock().await;
        targets
            .into_iter()
            .filter_map(|target| {
                state
                    .cache
                    .get(&target.question)
                    .map(|entry| (target.result_index, entry.markdown.clone()))
            })
            .collect()
    }

    async fn missing_questions(&self, targets: &[Target]) -> Vec<QuestionRef> {
        let now = Instant::now();
        let mut state = self.state.lock().await;
        state
            .cache
            .retain(|_, entry| now.duration_since(entry.inserted) < CACHE_TTL);

        targets
            .iter()
            .map(|target| target.question.clone())
            .filter(|question| !state.cache.contains_key(question))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    async fn fetch_group(&self, site: &str, ids: &[u64]) -> Result<()> {
        let _request_guard = self.request_gate.lock().await;
        // Another search may have populated these entries while this request
        // waited at the gate. Recheck so identical concurrent searches do not
        // spend quota twice.
        let ids = {
            let state = self.state.lock().await;
            ids.iter()
                .copied()
                .filter(|id| {
                    !state.cache.contains_key(&QuestionRef {
                        site: site.to_owned(),
                        id: *id,
                    })
                })
                .collect::<Vec<_>>()
        };
        if ids.is_empty() {
            return Ok(());
        }

        self.ensure_not_backed_off().await?;

        let ids = ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";");
        let mut request = self
            .http
            .get(format!("{API_BASE}/questions/{ids}"))
            .query(&[("site", site), ("filter", API_FILTER)])
            .timeout(REQUEST_TIMEOUT);
        if let Some(api_key) = &self.api_key {
            request = request.query(&[("key", api_key)]);
        }

        let response = request
            .send()
            .await
            .context("failed to call Stack Exchange API")?;
        let status = response.status();
        let body = read_limited(response).await?;
        if !status.is_success() {
            self.record_api_error(status, &body).await?;
        }
        let mut response: ApiResponse =
            serde_json::from_slice(&body).context("invalid Stack Exchange API response")?;

        let now = Instant::now();
        self.apply_response_limits(&response).await;
        let mut state = self.state.lock().await;

        for mut question in response.items.drain(..) {
            let reference = QuestionRef {
                site: site.to_owned(),
                id: question.question_id,
            };
            let markdown = render_question(&mut question);
            if markdown.trim().is_empty() {
                continue;
            }
            if state.cache.len() >= MAX_CACHE_ENTRIES
                && let Some(oldest) = state
                    .cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.inserted)
                    .map(|(question, _)| question.clone())
            {
                state.cache.remove(&oldest);
            }
            state.cache.insert(
                reference,
                CacheEntry {
                    markdown,
                    inserted: now,
                },
            );
        }
        Ok(())
    }

    async fn ensure_not_backed_off(&self) -> Result<()> {
        let now = Instant::now();
        let next_request = self.state.lock().await.next_request;
        if next_request > now {
            bail!(
                "Stack Exchange API is backed off for {} more seconds",
                next_request.duration_since(now).as_secs()
            );
        }
        Ok(())
    }

    async fn record_api_error(&self, status: reqwest::StatusCode, body: &[u8]) -> Result<()> {
        let error = serde_json::from_slice::<ApiError>(body).ok();
        if error
            .as_ref()
            .is_some_and(|error| error.error_name == "throttle_violation")
        {
            let retry_after = error
                .as_ref()
                .and_then(|error| retry_after_seconds(&error.error_message))
                .unwrap_or(15 * 60);
            self.state.lock().await.next_request =
                Instant::now() + Duration::from_secs(retry_after);
        }
        let message = error
            .map(|error| error.error_message)
            .unwrap_or_else(|| "unknown API error".to_owned());
        bail!("Stack Exchange API returned {status}: {message}")
    }

    async fn apply_response_limits(&self, response: &ApiResponse) {
        if let Some(backoff) = response.backoff {
            self.state.lock().await.next_request = Instant::now() + Duration::from_secs(backoff);
            tracing::warn!(
                backoff_seconds = backoff,
                "Stack Exchange API requested backoff"
            );
        }
        if let Some(quota_remaining) = response.quota_remaining
            && quota_remaining < 50
        {
            tracing::warn!(
                quota_remaining,
                quota_max = ?response.quota_max,
                "Stack Exchange API quota is low"
            );
        }
    }
}

pub(super) fn printer_url(url: &Url) -> Option<Url> {
    let question = question_ref(url)?;
    let service = printer_service(url.host_str()?)?;
    let mut printer = Url::parse("https://www.stackprinter.com/export").expect("valid literal");
    printer
        .query_pairs_mut()
        .append_pair("question", &question.id.to_string())
        .append_pair("service", &service)
        .append_pair("language", "en")
        .append_pair("hideAnswers", "false")
        .append_pair("showAll", "true")
        .append_pair("width", "640");
    Some(printer)
}

pub(super) fn printer_unavailable(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    lower.contains("the stackexchange server is too busy")
        || lower.contains("please try again later")
        || lower.contains("too many requests")
        || lower.contains("for the love it bears to fair maidens")
        || lower.contains("stackprinter - the stack exchange printer suite")
}

fn retry_after_seconds(message: &str) -> Option<u64> {
    message
        .split_once("more requests available in ")?
        .1
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

pub(super) fn recognizes(url: &Url) -> bool {
    question_ref(url).is_some()
}

async fn read_limited(mut response: reqwest::Response) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        bail!("Stack Exchange API response exceeds {MAX_RESPONSE_BYTES} bytes");
    }

    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed to read Stack Exchange API response")?
    {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            bail!("Stack Exchange API response exceeds {MAX_RESPONSE_BYTES} bytes");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn question_ref(url: &Url) -> Option<QuestionRef> {
    let site = api_site(url.host_str()?)?;
    let mut segments = url.path_segments()?;
    let kind = segments.next()?;
    if !matches!(kind, "questions" | "q") {
        return None;
    }
    let id = segments.next()?.parse().ok()?;
    Some(QuestionRef { site, id })
}

fn api_site(host: &str) -> Option<String> {
    let normalized = host.to_ascii_lowercase();
    let host = normalized.strip_prefix("www.").unwrap_or(&normalized);

    let site = match host {
        "stackoverflow.com" => "stackoverflow".to_owned(),
        "serverfault.com" => "serverfault".to_owned(),
        "superuser.com" => "superuser".to_owned(),
        "askubuntu.com" => "askubuntu".to_owned(),
        "mathoverflow.net" => "mathoverflow".to_owned(),
        "stackapps.com" => "stackapps".to_owned(),
        _ if host.ends_with(".stackexchange.com") => host
            .strip_suffix(".stackexchange.com")
            .filter(|site| !site.is_empty())?
            .to_owned(),
        _ if host.ends_with(".stackoverflow.com") => host
            .strip_suffix(".com")
            .filter(|site| !site.is_empty())?
            .to_owned(),
        _ => return None,
    };
    Some(site)
}

fn printer_service(host: &str) -> Option<String> {
    let normalized = host.to_ascii_lowercase();
    let host = normalized.strip_prefix("www.").unwrap_or(&normalized);

    match host {
        "stackoverflow.com" => Some("stackoverflow".into()),
        "serverfault.com" => Some("serverfault".into()),
        "superuser.com" => Some("superuser".into()),
        "askubuntu.com" => Some("askubuntu".into()),
        "mathoverflow.net" => Some("mathoverflow".into()),
        "stackapps.com" => Some("stackapps".into()),
        _ if host.ends_with(".stackexchange.com") => host
            .strip_suffix(".com")
            .filter(|service| !service.is_empty())
            .map(str::to_owned),
        _ if host.ends_with(".stackoverflow.com") => host
            .strip_suffix(".com")
            .filter(|service| !service.is_empty())
            .map(str::to_owned),
        _ => None,
    }
}

fn render_question(question: &mut ApiQuestion) -> String {
    question.answers.sort_by(|left, right| {
        right
            .is_accepted
            .cmp(&left.is_accepted)
            .then_with(|| right.score.cmp(&left.score))
    });

    let mut output = String::new();
    let _ = writeln!(output, "# {}\n", question.title.trim());
    write_attribution(
        &mut output,
        "Question",
        question.owner.as_ref(),
        question.score,
        &question.link,
        question.content_license.as_deref(),
    );
    let _ = writeln!(output, "\n{}", question.body_markdown.trim());

    if !question.answers.is_empty() {
        output.push_str("\n\n# Answers\n");
    }
    for answer in question.answers.iter().take(MAX_ANSWERS) {
        let heading = if answer.is_accepted {
            "Accepted answer"
        } else {
            "Answer"
        };
        let _ = writeln!(output, "\n## {heading}\n");
        write_attribution(
            &mut output,
            heading,
            answer.owner.as_ref(),
            answer.score,
            &answer.share_link,
            answer.content_license.as_deref(),
        );
        let _ = writeln!(output, "\n{}", answer.body_markdown.trim());
    }

    output.trim().to_owned()
}

fn write_attribution(
    output: &mut String,
    kind: &str,
    owner: Option<&ApiUser>,
    score: i64,
    source: &str,
    license: Option<&str>,
) {
    let _ = write!(output, "> {kind}");
    if let Some(owner) = owner {
        let name = escape_link_text(&owner.display_name);
        if let Some(link) = &owner.link {
            let _ = write!(output, " by [{name}]({link})");
        } else {
            let _ = write!(output, " by {name}");
        }
    }
    let _ = write!(output, " · score {score} · [source]({source})");
    if let Some(license) = license {
        let _ = write!(output, " · {license}");
    }
    output.push('\n');
}

fn escape_link_text(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

#[derive(Deserialize)]
struct ApiResponse {
    #[serde(default)]
    items: Vec<ApiQuestion>,
    backoff: Option<u64>,
    quota_remaining: Option<u64>,
    quota_max: Option<u64>,
}

#[derive(Deserialize)]
struct ApiError {
    error_name: String,
    error_message: String,
}

#[derive(Deserialize)]
struct ApiQuestion {
    question_id: u64,
    title: String,
    #[serde(default)]
    body_markdown: String,
    link: String,
    #[serde(default)]
    score: i64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    is_answered: bool,
    owner: Option<ApiUser>,
    content_license: Option<String>,
    #[serde(default)]
    answers: Vec<ApiAnswer>,
}

impl ApiQuestion {
    fn into_search_hit(mut self) -> Option<Hit> {
        let url = Url::parse(&self.link).ok()?;
        self.answers.sort_by(|left, right| {
            right
                .is_accepted
                .cmp(&left.is_accepted)
                .then_with(|| right.score.cmp(&left.score))
        });
        let mut details = Vec::new();
        if !self.tags.is_empty() {
            details.push(self.tags.join(", "));
        }
        details.push(format!("score {}", self.score));
        if self.is_answered {
            details.push("answered".into());
        }
        let mut snippet = details.join(" · ");
        let question_body = excerpt(&self.body_markdown, SEARCH_BODY_CHARACTERS);
        if !question_body.is_empty() {
            snippet.push_str("\n\n");
            write_attribution(
                &mut snippet,
                "Question",
                self.owner.as_ref(),
                self.score,
                &self.link,
                self.content_license.as_deref(),
            );
            snippet.push_str(&question_body);
        }
        if let Some(answer) = self.answers.first() {
            let answer_body = excerpt(&answer.body_markdown, SEARCH_ANSWER_CHARACTERS);
            if !answer_body.is_empty() {
                snippet.push_str("\n\n");
                let kind = if answer.is_accepted {
                    "Accepted answer"
                } else {
                    "Answer"
                };
                write_attribution(
                    &mut snippet,
                    kind,
                    answer.owner.as_ref(),
                    answer.score,
                    &answer.share_link,
                    answer.content_license.as_deref(),
                );
                snippet.push_str(&answer_body);
            }
        }
        Some(Hit {
            title: decode_title(&self.title),
            url,
            date: None,
            snippet,
        })
    }
}

fn excerpt(markdown: &str, max_characters: usize) -> String {
    let trimmed = markdown.trim();
    let mut excerpt = trimmed.chars().take(max_characters).collect::<String>();
    if trimmed.chars().count() > max_characters {
        excerpt.push('…');
    }
    excerpt
}

fn decode_title(title: &str) -> String {
    title
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[derive(Deserialize)]
struct ApiAnswer {
    #[serde(rename = "answer_id")]
    _answer_id: u64,
    #[serde(default)]
    body_markdown: String,
    #[serde(default)]
    score: i64,
    #[serde(default)]
    is_accepted: bool,
    owner: Option<ApiUser>,
    share_link: String,
    content_license: Option<String>,
}

#[derive(Deserialize)]
struct ApiUser {
    display_name: String,
    link: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_stack_exchange_question_urls() {
        let stack_overflow = question_ref(
            &Url::parse("https://stackoverflow.com/questions/12345/example#answer-1").unwrap(),
        )
        .unwrap();
        assert_eq!(stack_overflow.site, "stackoverflow");
        assert_eq!(stack_overflow.id, 12345);

        let unix = question_ref(&Url::parse("https://unix.stackexchange.com/q/67890/123").unwrap())
            .unwrap();
        assert_eq!(unix.site, "unix");
        assert_eq!(unix.id, 67890);

        let localized =
            question_ref(&Url::parse("https://ru.stackoverflow.com/questions/42/example").unwrap())
                .unwrap();
        assert_eq!(localized.site, "ru.stackoverflow");
        assert_eq!(localized.id, 42);

        assert!(
            question_ref(&Url::parse("https://stackoverflow.com/users/1/name").unwrap()).is_none()
        );
        assert!(
            question_ref(&Url::parse("https://example.com/questions/1/nope").unwrap()).is_none()
        );

        let printer = printer_url(
            &Url::parse("https://unix.stackexchange.com/questions/67890/example").unwrap(),
        )
        .unwrap();
        assert_eq!(printer.host_str(), Some("www.stackprinter.com"));
        assert!(printer.as_str().contains("question=67890"));
        assert!(printer.as_str().contains("service=unix.stackexchange"));
    }

    #[test]
    fn recognizes_stackprinter_failures_and_api_retry_windows() {
        assert!(printer_unavailable(
            "The StackExchange server is too busy at the moment. Please try again later."
        ));
        assert!(printer_unavailable(
            "..for the love it bears to fair maidens forgets its ferocity and wildness.. too many requests from this app/user pair"
        ));
        assert!(!printer_unavailable("# Useful question\n\nAccepted answer"));
        assert_eq!(
            retry_after_seconds(
                "too many requests from this IP, more requests available in 47924 seconds"
            ),
            Some(47_924)
        );
    }

    #[test]
    fn renders_accepted_answer_first_with_attribution() {
        let mut response: ApiResponse = serde_json::from_str(
            r#"{
                "items": [{
                    "question_id": 1,
                    "title": "How do I parse JSON?",
                    "body_markdown": "Use the right parser.",
                    "link": "https://stackoverflow.com/questions/1/example",
                    "score": 5,
                    "content_license": "CC BY-SA 4.0",
                    "owner": {"display_name": "Question Author", "link": "https://stackoverflow.com/users/1"},
                    "answers": [
                        {
                            "answer_id": 2,
                            "body_markdown": "Lower answer",
                            "score": 20,
                            "is_accepted": false,
                            "share_link": "https://stackoverflow.com/a/2",
                            "owner": {"display_name": "Other", "link": "https://stackoverflow.com/users/2"}
                        },
                        {
                            "answer_id": 3,
                            "body_markdown": "Accepted answer",
                            "score": 10,
                            "is_accepted": true,
                            "content_license": "CC BY-SA 4.0",
                            "share_link": "https://stackoverflow.com/a/3",
                            "owner": {"display_name": "Accepted [Author]", "link": "https://stackoverflow.com/users/3"}
                        }
                    ]
                }]
            }"#,
        )
        .unwrap();

        let markdown = render_question(&mut response.items[0]);
        assert!(markdown.contains("[Question Author](https://stackoverflow.com/users/1)"));
        assert!(markdown.contains("CC BY-SA 4.0"));
        assert!(markdown.contains("Accepted \\[Author\\]"));
        assert!(markdown.find("Accepted answer").unwrap() < markdown.find("Lower answer").unwrap());
    }

    #[test]
    fn turns_search_metadata_into_a_candidate_without_claiming_a_date() {
        let response: ApiResponse = serde_json::from_str(
            r#"{
                "items": [{
                    "question_id": 42,
                    "title": "Rust &quot;FromResidual&quot; error",
                    "link": "https://stackoverflow.com/questions/42/example",
                    "score": 12,
                    "tags": ["rust", "error-handling"],
                    "is_answered": true
                }]
            }"#,
        )
        .unwrap();

        let hit = response
            .items
            .into_iter()
            .next()
            .unwrap()
            .into_search_hit()
            .unwrap();

        assert_eq!(hit.title, "Rust \"FromResidual\" error");
        assert_eq!(
            hit.url.as_str(),
            "https://stackoverflow.com/questions/42/example"
        );
        assert_eq!(hit.snippet, "rust, error-handling · score 12 · answered");
        assert_eq!(hit.date, None);
    }

    #[test]
    fn search_candidates_include_an_attributed_answer_excerpt() {
        let response: ApiResponse = serde_json::from_str(
            r#"{
                "items": [{
                    "question_id": 42,
                    "title": "Convert a source error into my enum",
                    "body_markdown": "Why does the question mark operator reject my source error?",
                    "link": "https://stackoverflow.com/questions/42/example",
                    "score": 4,
                    "owner": {"display_name": "Questioner", "link": "https://stackoverflow.com/users/1"},
                    "content_license": "CC BY-SA 4.0",
                    "answers": [{
                        "answer_id": 7,
                        "body_markdown": "Implement From&lt;SourceError&gt; for your custom error enum.",
                        "score": 9,
                        "is_accepted": true,
                        "share_link": "https://stackoverflow.com/a/7",
                        "owner": {"display_name": "Answerer", "link": "https://stackoverflow.com/users/2"},
                        "content_license": "CC BY-SA 4.0"
                    }]
                }]
            }"#,
        )
        .unwrap();

        let hit = response
            .items
            .into_iter()
            .next()
            .unwrap()
            .into_search_hit()
            .unwrap();

        assert!(
            hit.snippet
                .contains("[Questioner](https://stackoverflow.com/users/1)")
        );
        assert!(
            hit.snippet
                .contains("[Answerer](https://stackoverflow.com/users/2)")
        );
        assert!(hit.snippet.contains("From&lt;SourceError&gt;"));
        assert!(hit.snippet.contains("CC BY-SA 4.0"));
    }
}
