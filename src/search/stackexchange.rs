use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::Mutex;
use tokio::time::{Instant, sleep_until};
use url::Url;

const API_BASE: &str = "https://api.stackexchange.com/2.3";
// Immutable filter created through /filters/create. It includes only the
// question, answer, attribution, quota, and backoff fields used below.
const API_FILTER: &str = "2hElTY3zn8AeeG)anHfU.obxyjY10FWBuiLATZQeMSje.t";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_CACHE_ENTRIES: usize = 256;
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ANSWERS: usize = 10;

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

        let next_request = self.state.lock().await.next_request;
        if next_request > Instant::now() {
            sleep_until(next_request).await;
        }

        let ids = ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";");
        let mut request = self
            .http
            .get(format!("{API_BASE}/questions/{ids}"))
            .query(&[("site", site), ("filter", API_FILTER)])
            .timeout(REQUEST_TIMEOUT);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request
            .send()
            .await
            .context("failed to call Stack Exchange API")?
            .error_for_status()
            .context("Stack Exchange API returned an error")?;
        let body = read_limited(response).await?;
        let mut response: ApiResponse =
            serde_json::from_slice(&body).context("invalid Stack Exchange API response")?;

        let now = Instant::now();
        let mut state = self.state.lock().await;
        if let Some(backoff) = response.backoff {
            state.next_request = now + Duration::from_secs(backoff);
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
struct ApiQuestion {
    question_id: u64,
    title: String,
    #[serde(default)]
    body_markdown: String,
    link: String,
    #[serde(default)]
    score: i64,
    owner: Option<ApiUser>,
    content_license: Option<String>,
    #[serde(default)]
    answers: Vec<ApiAnswer>,
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
}
