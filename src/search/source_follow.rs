use std::cmp::Ordering;
use std::collections::HashSet;

use url::Url;

use super::evidence::QueryEvidence;
use super::searxng::{self, Hit, SearchCandidate};

/// Recovers one explicitly requested first-party source from citations in the
/// pages we already fetched. This is intentionally not a general crawler.
pub(super) fn cited_candidate<'a>(
    query: &str,
    documents: impl IntoIterator<Item = (&'a Url, &'a str)>,
) -> Option<SearchCandidate> {
    if !searxng::has_strict_source_intent(&query.to_ascii_lowercase()) {
        return None;
    }
    let authority = searxng::named_authority_host(query)?;
    let documents = documents.into_iter().collect::<Vec<_>>();
    if documents
        .iter()
        .any(|(url, _)| belongs_to_authority(url, authority))
    {
        return None;
    }

    let evidence = QueryEvidence::new(query, documents.iter().map(|(_, markdown)| *markdown));
    let mut seen = HashSet::new();
    documents
        .into_iter()
        .flat_map(|(_, markdown)| markdown_links(markdown))
        .filter(|(_, url)| belongs_to_authority(url, authority))
        .filter(|(_, url)| seen.insert(url.clone()))
        .map(|(label, url)| {
            let score = citation_score(&evidence, &label, &url);
            (score, label, url)
        })
        .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
        .map(|(_, label, url)| SearchCandidate {
            hit: Hit {
                title: if label.is_empty() {
                    authority.to_owned()
                } else {
                    label.clone()
                },
                url: url.clone(),
                date: None,
                snippet: label,
            },
            source_priority: true,
            upstream_consensus: false,
            fetch_urls: vec![url],
        })
}

pub(super) fn recovered_title(content: &str, fallback: &str) -> String {
    let subject = content.lines().take(80).find_map(|line| {
        let line = line.trim().trim_start_matches(['#', '>', ' ']).trim();
        line.get(..8)
            .filter(|prefix| prefix.eq_ignore_ascii_case("subject:"))
            .map(|_| line[8..].trim())
            .filter(|subject| !subject.is_empty())
    });
    let heading = content.lines().take(20).find_map(|line| {
        line.trim()
            .strip_prefix('#')
            .map(|heading| heading.trim_start_matches('#').trim())
            .filter(|heading| !heading.is_empty())
    });
    subject
        .or(heading)
        .filter(|title| title.chars().count() <= 200)
        .unwrap_or(fallback)
        .to_owned()
}

fn markdown_links(markdown: &str) -> Vec<(String, Url)> {
    let mut links = Vec::new();
    let mut offset = 0;
    while let Some(relative_end) = markdown[offset..].find("](") {
        let label_end = offset + relative_end;
        let Some(label_start) = markdown[..label_end].rfind('[') else {
            offset = label_end + 2;
            continue;
        };
        let target_start = label_end + 2;
        let Some(relative_target_end) = markdown[target_start..].find(')') else {
            break;
        };
        let target_end = target_start + relative_target_end;
        let target = markdown[target_start..target_end]
            .split_ascii_whitespace()
            .next()
            .unwrap_or_default()
            .trim_matches(['<', '>']);
        if let Ok(url) = Url::parse(target)
            && matches!(url.scheme(), "http" | "https")
        {
            links.push((markdown[label_start + 1..label_end].trim().to_owned(), url));
        }
        offset = target_end + 1;
    }
    links
}

fn belongs_to_authority(url: &Url, authority: &str) -> bool {
    url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case(authority)
            || host.len() > authority.len()
                && host[..host.len() - authority.len()].ends_with('.')
                && host[host.len() - authority.len()..].eq_ignore_ascii_case(authority)
    })
}

fn citation_score(evidence: &QueryEvidence, label: &str, url: &Url) -> f32 {
    let description = format!("{label} {}", url.path());
    let path_segments = url
        .path_segments()
        .map(|segments| segments.filter(|segment| !segment.is_empty()).count())
        .unwrap_or_default()
        .min(6) as f32;
    evidence.score(&description) + path_segments * 0.025
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_a_cited_named_authority_for_an_original_source_request() {
        let secondary = Url::parse("https://security.example/xz").unwrap();
        let candidate = cited_candidate(
            "CVE-2024-3094 original oss-security disclosure xz backdoor",
            [(
                &secondary,
                "The disclosure appeared on the [Openwall mailing list](https://www.openwall.com/lists/oss-security/2024/03/29/4).",
            )],
        )
        .unwrap();

        assert_eq!(
            candidate.url.as_str(),
            "https://www.openwall.com/lists/oss-security/2024/03/29/4"
        );
        assert_eq!(candidate.title, "Openwall mailing list");
        assert!(candidate.source_priority);
    }

    #[test]
    fn does_not_crawl_without_strict_source_intent() {
        let secondary = Url::parse("https://security.example/xz").unwrap();
        assert!(
            cited_candidate(
                "CVE-2024-3094 xz backdoor analysis",
                [(
                    &secondary,
                    "See [Openwall](https://www.openwall.com/lists/oss-security/2024/03/29/4).",
                )],
            )
            .is_none()
        );
    }

    #[test]
    fn does_not_follow_a_citation_when_the_authority_is_already_present() {
        let original =
            Url::parse("https://www.openwall.com/lists/oss-security/2024/03/29/4").unwrap();
        let secondary = Url::parse("https://security.example/xz").unwrap();
        assert!(
            cited_candidate(
                "CVE-2024-3094 original oss-security disclosure xz backdoor",
                [
                    (&original, "Original disclosure"),
                    (
                        &secondary,
                        "[Mirror](https://www.openwall.com/lists/oss-security/2024/03/29/4)",
                    ),
                ],
            )
            .is_none()
        );
    }

    #[test]
    fn ignores_links_to_unrequested_hosts() {
        let secondary = Url::parse("https://security.example/xz").unwrap();
        assert!(
            cited_candidate(
                "CVE-2024-3094 original oss-security disclosure xz backdoor",
                [(
                    &secondary,
                    "[NVD](https://nvd.nist.gov/vuln/detail/CVE-2024-3094)",
                )],
            )
            .is_none()
        );
    }

    #[test]
    fn uses_a_mail_subject_as_the_recovered_result_title() {
        assert_eq!(
            recovered_title(
                "Message-ID: <example>\nSubject: backdoor in upstream xz/liblzma leading to ssh server compromise\n\nHi,",
                "Openwall mailing list",
            ),
            "backdoor in upstream xz/liblzma leading to ssh server compromise"
        );
        assert_eq!(
            recovered_title("# Primary specification\n\nText", "Official source"),
            "Primary specification"
        );
    }
}
