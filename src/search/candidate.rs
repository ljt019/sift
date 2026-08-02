use std::collections::{HashMap, HashSet};

use anyhow::{Result, ensure};

use super::evidence::{
    QueryEvidence, has_case_study_intent, has_full_text_intent, has_research_source_intent,
    is_as_of_query, normalized_terms,
};
use super::searxng::{Hit, SearchCandidate};
use crate::embeddings::EmbeddingWorker;

const EMBEDDING_DIMENSIONS: usize = 768;
const CONSENSUS_MAX_SCORE_GAP: f32 = 0.10;
// Candidate evidence only breaks close dense-score ties. Authority and exact
// document signals are intentionally several times stronger, and consensus
// head protection happens after scoring.
const QUERY_EVIDENCE_WEIGHT: f32 = 0.04;
const RECOMMENDATION_EVIDENCE_WEIGHT: f32 = 0.12;
const CASE_STUDY_FORM_WEIGHT: f32 = 0.10;

pub async fn select(
    embeddings: &EmbeddingWorker,
    query: &str,
    candidates: Vec<SearchCandidate>,
    limit: usize,
) -> Result<Vec<SearchCandidate>> {
    let metadata = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.url.as_str().to_owned(),
                (
                    candidate.source_priority,
                    candidate.upstream_consensus,
                    candidate.fetch_urls.clone(),
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let source_urls = metadata
        .iter()
        .filter(|(_, (source_priority, _, _))| *source_priority)
        .map(|(url, _)| url.clone())
        .collect::<HashSet<_>>();
    let consensus_urls = metadata
        .iter()
        .filter(|(_, (_, upstream_consensus, _))| *upstream_consensus)
        .map(|(url, _)| url.clone())
        .collect::<HashSet<_>>();
    let hits = candidates
        .into_iter()
        .map(|candidate| candidate.hit)
        .collect();
    let hits = prioritize_exact_title_hits(query, hits);
    if hits.len() <= limit {
        return Ok(wrap_candidates(hits, &metadata));
    }

    let query_embeddings = embeddings
        .embed_queries(vec![query.to_owned()], EMBEDDING_DIMENSIONS)
        .await?;
    ensure!(
        query_embeddings.len() == 1,
        "embedder returned {} candidate-query vectors",
        query_embeddings.len()
    );
    let documents = hits
        .iter()
        .map(|hit| format!("{}\n{}", hit.title, hit.snippet))
        .collect::<Vec<_>>();
    let evidence = QueryEvidence::new(query, documents.iter().map(String::as_str));
    let evidence_adjustments = documents
        .iter()
        .map(|document| candidate_evidence_adjustment(&evidence, document))
        .collect::<Vec<_>>();
    let recommendation_adjustments = documents
        .iter()
        .map(|document| RECOMMENDATION_EVIDENCE_WEIGHT * evidence.recommendation_score(document))
        .collect::<Vec<_>>();
    let year_adjustments = documents
        .iter()
        .map(|document| evidence.year_adjustment(document))
        .collect::<Vec<_>>();
    let document_embeddings = embeddings
        .embed_documents(documents, EMBEDDING_DIMENSIONS)
        .await?;
    ensure!(
        document_embeddings.len() == hits.len(),
        "embedder returned {} candidate vectors for {} hits",
        document_embeddings.len(),
        hits.len()
    );

    let query_embedding = &query_embeddings[0];
    ensure!(
        query_embedding.len() == EMBEDDING_DIMENSIONS,
        "candidate query embedding has {} dimensions, expected {EMBEDDING_DIMENSIONS}",
        query_embedding.len()
    );
    ensure!(
        query_embedding.iter().all(|value| value.is_finite()),
        "candidate query embedding is non-finite"
    );

    let recency_intent = has_recency_intent(query);
    let requested_date = query_date_key(query);
    let requested_period = requested_date.is_none().then(|| month_key(query)).flatten();
    let newest_date = recency_intent
        .then(|| hits.iter().filter_map(hit_date_key).max())
        .flatten();
    let mut scored = document_embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            ensure!(
                embedding.len() == query_embedding.len(),
                "candidate embedding {index} has {} dimensions, expected {}",
                embedding.len(),
                query_embedding.len()
            );
            ensure!(
                embedding.iter().all(|value| value.is_finite()),
                "candidate embedding {index} is non-finite"
            );
            let dense_score = query_embedding
                .iter()
                .zip(embedding)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let authority = source_authority_adjustment(query, &hits[index].url);
            let recency_adjustment = if recency_intent {
                staleness_penalty(newest_date, hit_date_key(&hits[index]))
            } else {
                0.0
            };
            let date_match_adjustment =
                explicit_date_adjustment(requested_date, hit_date_key(&hits[index]));
            let period_match_adjustment = if is_as_of_query(query) {
                0.0
            } else {
                explicit_period_adjustment(requested_period, hit_month_key(&hits[index]))
            };
            let title_match_adjustment = paper_title_adjustment(query, &hits[index]);
            let evidence_adjustment = evidence_adjustments[index];
            let recommendation_adjustment = recommendation_adjustments[index];
            let year_adjustment = year_adjustments[index];
            let case_study_form_adjustment = case_study_form_adjustment(query, &hits[index]);
            let score = dense_score
                + authority
                + recency_adjustment
                + date_match_adjustment
                + period_match_adjustment
                + title_match_adjustment
                + evidence_adjustment
                + recommendation_adjustment
                + year_adjustment
                + case_study_form_adjustment;
            ensure!(score.is_finite(), "candidate {index} score is non-finite");
            tracing::debug!(
                url = %hits[index].url,
                source_priority = source_urls.contains(hits[index].url.as_str()),
                date = hits[index].date.as_deref(),
                dense_score,
                authority,
                recency_adjustment,
                date_match_adjustment,
                period_match_adjustment,
                title_match_adjustment,
                evidence_adjustment,
                recommendation_adjustment,
                year_adjustment,
                case_study_form_adjustment,
                score,
                "scored search candidate"
            );
            Ok((index, score))
        })
        .collect::<Result<Vec<_>>>()?;
    scored.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_index.cmp(right_index))
    });

    let source_priority = hits
        .iter()
        .enumerate()
        .filter(|(_, hit)| source_urls.contains(hit.url.as_str()))
        .map(|(index, _)| index)
        .collect::<HashSet<_>>();
    let upstream_consensus = hits
        .iter()
        .enumerate()
        .filter(|(_, hit)| consensus_urls.contains(hit.url.as_str()))
        .map(|(index, _)| index)
        .collect::<HashSet<_>>();
    let selected = select_indices(
        hits.len(),
        &scored,
        limit,
        &source_priority,
        &upstream_consensus,
    );
    let selected = prioritize_exact_title_indices(query, &hits, selected, limit);
    let selected = reserve_requested_source(selected, &scored, &source_priority, limit);

    let mut hits = hits.into_iter().map(Some).collect::<Vec<_>>();
    let hits = selected
        .into_iter()
        .map(|index| hits[index].take().expect("candidate indices are unique"))
        .collect();
    Ok(wrap_candidates(hits, &metadata))
}

fn candidate_evidence_adjustment(evidence: &QueryEvidence, document: &str) -> f32 {
    QUERY_EVIDENCE_WEIGHT * evidence.score(document)
}

fn wrap_candidates(
    hits: Vec<Hit>,
    metadata: &HashMap<String, (bool, bool, Vec<url::Url>)>,
) -> Vec<SearchCandidate> {
    hits.into_iter()
        .map(|hit| {
            let (source_priority, upstream_consensus, fetch_urls) = metadata
                .get(hit.url.as_str())
                .cloned()
                .unwrap_or_else(|| (false, false, vec![hit.url.clone()]));
            SearchCandidate {
                hit,
                source_priority,
                upstream_consensus,
                fetch_urls,
            }
        })
        .collect()
}

fn paper_title_adjustment(query: &str, hit: &Hit) -> f32 {
    let Some(expected) = exact_title(query) else {
        return 0.0;
    };

    if exact_title_matches(query, &expected, hit) {
        0.20
    } else {
        0.0
    }
}

/// Preserve every manifestation of an explicitly requested document until
/// after fetching. A fixed pre-fetch cap can discard the only accessible copy.
fn prioritize_exact_title_hits(query: &str, hits: Vec<Hit>) -> Vec<Hit> {
    let Some(expected) = exact_title(query) else {
        return hits;
    };
    let (exact, related): (Vec<_>, Vec<_>) = hits
        .into_iter()
        .partition(|hit| exact_title_matches(query, &expected, hit));
    exact.into_iter().chain(related).collect()
}

fn prioritize_exact_title_indices(
    query: &str,
    hits: &[Hit],
    selected: Vec<usize>,
    limit: usize,
) -> Vec<usize> {
    let Some(expected) = exact_title(query) else {
        return selected;
    };
    let mut prioritized = hits
        .iter()
        .enumerate()
        .filter(|(_, hit)| exact_title_matches(query, &expected, hit))
        .map(|(index, _)| index)
        .take(limit)
        .collect::<Vec<_>>();
    if prioritized.is_empty() {
        return selected;
    }
    for index in selected {
        if prioritized.len() == limit {
            break;
        }
        if !prioritized.contains(&index) {
            prioritized.push(index);
        }
    }
    prioritized
}

pub(super) fn exact_document_match(query: &str, hit: &Hit) -> bool {
    exact_title(query).is_some_and(|expected| exact_title_matches(query, &expected, hit))
}

fn exact_title_matches(query: &str, expected: &str, hit: &Hit) -> bool {
    title_equivalent(expected, &hit.title)
        && (super::searxng::paper_title(query).is_none() || is_scholarly_manifestation(hit))
}

fn title_equivalent(expected: &str, candidate: &str) -> bool {
    let expected = normalized_title(expected);
    let candidate = normalized_title(candidate);
    if candidate == expected {
        return true;
    }

    let expected_words = expected.split_whitespace().collect::<Vec<_>>();
    let candidate_words = candidate.split_whitespace().collect::<Vec<_>>();
    if candidate_words.len() >= 4
        && candidate_words.len() < expected_words.len()
        && expected_words.starts_with(&candidate_words)
    {
        // Search engines often end long titles with an ellipsis.
        return candidate_words.len() * 3 >= expected_words.len() * 2;
    }
    if candidate_words.len() <= expected_words.len()
        || !candidate_words.starts_with(&expected_words)
    {
        return false;
    }

    candidate_words[expected_words.len()..].iter().all(|word| {
        word.chars().all(|character| character.is_ascii_digit())
            || matches!(
                *word,
                "abstract"
                    | "acm"
                    | "arxiv"
                    | "digital"
                    | "html"
                    | "library"
                    | "neurips"
                    | "nips"
                    | "org"
                    | "paper"
                    | "pdf"
                    | "proceedings"
            )
    })
}

fn is_scholarly_manifestation(hit: &Hit) -> bool {
    let host = hit.url.host_str().unwrap_or_default();
    [
        "aclanthology.org",
        "aclweb.org",
        "acm.org",
        "arxiv.org",
        "biorxiv.org",
        "cambridge.org",
        "cell.com",
        "core.ac.uk",
        "doi.org",
        "figshare.com",
        "frontiersin.org",
        "huggingface.co",
        "ieee.org",
        "jstor.org",
        "mdpi.com",
        "medrxiv.org",
        "nature.com",
        "ncbi.nlm.nih.gov",
        "neurips.cc",
        "nips.cc",
        "openreview.net",
        "osf.io",
        "oup.com",
        "paperswithcode.com",
        "plos.org",
        "researchsquare.com",
        "sagepub.com",
        "sciencedirect.com",
        "science.org",
        "semanticscholar.org",
        "springer.com",
        "tandfonline.com",
        "wiley.com",
        "zenodo.org",
    ]
    .iter()
    .any(|expected| host == *expected || host.ends_with(&format!(".{expected}")))
        || host.ends_with(".edu")
        || host.ends_with(".ac.uk")
        || host == "ai.meta.com"
        || host == "research.facebook.com"
        || host == "research.google"
        || host.ends_with(".research.google")
        || hit.url.path().contains("/doi/")
}

fn exact_title(query: &str) -> Option<String> {
    super::searxng::paper_title(query).or_else(|| {
        if !has_full_text_intent(query) {
            return None;
        }
        let lower = query.to_ascii_lowercase();
        let (start, marker_len) = lower
            .find("full text")
            .map(|start| (start, "full text".len()))
            .into_iter()
            .chain(
                lower
                    .find("full-text")
                    .map(|start| (start, "full-text".len())),
            )
            .min_by_key(|(start, _)| *start)?;
        let mut title = query.to_owned();
        title.replace_range(start..start + marker_len, "");
        let title = title.trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, ':' | '-' | '"' | '\'')
        });
        let title = title
            .get(..3)
            .filter(|prefix| prefix.eq_ignore_ascii_case("of "))
            .map_or(title, |_| &title[3..]);
        (normalized_title(title).split_whitespace().count() >= 2).then(|| title.to_owned())
    })
}

fn normalized_title(value: &str) -> String {
    let mut title = value.trim();
    while let Some(rest) = strip_leading_title_metadata(title) {
        title = rest.trim_start();
    }
    if let Some(rest) = strip_labeled_title_prefix(title, "pdf") {
        title = rest;
    } else if let Some(mut rest) = strip_labeled_title_prefix(title, "arxiv") {
        if rest
            .get(..4)
            .is_some_and(|suffix| suffix.eq_ignore_ascii_case(".org"))
        {
            rest = rest[4..].trim_start_matches([' ', ':', '-']);
        }
        if let Some((identifier, remainder)) = rest.split_once(char::is_whitespace)
            && identifier
                .chars()
                .any(|character| character.is_ascii_digit())
            && identifier
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.')
        {
            rest = remainder.trim_start();
        }
        while let Some(remainder) = strip_leading_title_metadata(rest) {
            rest = remainder.trim_start();
        }
        title = rest;
    }

    let lower = title.to_ascii_lowercase();
    let suffix = [
        " - arxiv",
        " – arxiv",
        " — arxiv",
        " | arxiv",
        " (arxiv",
        " - pubmed",
        " – pubmed",
        " — pubmed",
        " | pubmed",
        " - pmc",
        " | pmc",
        " - springerlink",
        " | springerlink",
    ]
    .iter()
    .filter_map(|marker| lower.find(marker))
    .min();
    if let Some(suffix) = suffix {
        title = title[..suffix].trim_end();
    }

    normalized_words(title)
}

fn strip_leading_title_metadata(title: &str) -> Option<&str> {
    let close = title.strip_prefix('[')?.find(']')? + 1;
    let metadata = &title[1..close];
    let normalized = metadata.to_ascii_lowercase();
    let is_metadata = normalized == "pdf"
        || normalized.starts_with("arxiv")
        || normalized
            .chars()
            .any(|character| character.is_ascii_digit())
        || normalized.contains('.');
    is_metadata.then(|| &title[close + 1..])
}

fn strip_labeled_title_prefix<'a>(title: &'a str, label: &str) -> Option<&'a str> {
    let prefix = title.get(..label.len())?;
    if !prefix.eq_ignore_ascii_case(label) {
        return None;
    }
    let rest = &title[label.len()..];
    rest.chars()
        .next()
        .is_some_and(|separator| matches!(separator, ' ' | ':' | '-' | '.'))
        .then(|| rest.trim_start_matches([' ', ':', '-']))
}

fn normalized_words(value: &str) -> String {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The URL-level authority signal used at both shortlist and full-text stages.
/// Project scope is mutually exclusive with the more general jurisdiction and
/// source-authority signals, matching the candidate scorer's original rules.
pub(super) fn source_authority_adjustment(query: &str, url: &url::Url) -> f32 {
    let project_scope = project_scope_adjustment(query, url);
    if project_scope != 0.0 {
        project_scope
    } else {
        jurisdiction_authority_bonus(query, url).max(authority_bonus(query, url))
    }
}

fn authority_bonus(query: &str, url: &url::Url) -> f32 {
    let host = url.host_str().unwrap_or_default();
    let named_authority = super::searxng::named_authority_host(query)
        .is_some_and(|authority| host == authority || host.ends_with(&format!(".{authority}")));
    if named_authority {
        // For an exact diagnostic the named project host is a fallback, not
        // necessarily the answer. A generic official page must not displace a
        // community post that explains this particular failure mode.
        let inferred_diagnostic = super::searxng::has_quoted_diagnostic(query)
            && !super::searxng::has_strict_source_intent(&query.to_ascii_lowercase());
        return if inferred_diagnostic { 0.04 } else { 0.12 } + descriptive_path_bonus(query, url);
    }
    let institutional = host.ends_with(".gov")
        || host.ends_with(".edu")
        || host.ends_with(".int")
        || host.ends_with(".ac.uk");
    if institutional || (has_research_source_intent(query) && is_scholarly_url(url)) {
        0.04
    } else {
        0.0
    }
}

/// Whether a URL is a direct scholarly record under the same deliberately
/// narrow definition used by candidate authority scoring.
pub(super) fn is_scholarly_url(url: &url::Url) -> bool {
    let host = url.host_str().unwrap_or_default();
    host == "arxiv.org"
        || host.ends_with(".arxiv.org")
        || host == "doi.org"
        || host == "dx.doi.org"
        || url.path().contains("/doi/")
}

/// A bounded document-form signal for explicit case-study requests. Matching
/// the requested words alone is insufficient: the title must also name a
/// non-generic subject from the query, which excludes case-study indexes and
/// generic comparison or buyer-guide pages.
pub(super) fn case_study_form_adjustment(query: &str, hit: &Hit) -> f32 {
    if !has_case_study_intent(query) || !has_case_study_form(&hit.title) {
        return 0.0;
    }

    let query_entities = case_study_query_entities(query);
    let title_terms = normalized_terms(&hit.title);
    if query_entities.iter().any(|term| title_terms.contains(term)) {
        CASE_STUDY_FORM_WEIGHT
    } else {
        0.0
    }
}

fn has_case_study_form(title: &str) -> bool {
    let title = normalized_words(title);
    let terms = title.split_whitespace().collect::<Vec<_>>();
    let explicitly_labeled = terms
        .windows(2)
        .any(|terms| terms == ["case", "study"] || terms == ["case", "studies"]);
    let describes_adoption = terms
        .iter()
        .any(|term| term.starts_with("adopt") || term.starts_with("migrat"));
    let reports_experience = terms
        .iter()
        .any(|term| matches!(*term, "experience" | "journey" | "lessons"));
    explicitly_labeled || describes_adoption && reports_experience
}

fn case_study_query_entities(query: &str) -> HashSet<String> {
    identity_terms(query)
        .into_iter()
        .filter(|term| !is_generic_case_study_term(term))
        .collect()
}

fn is_generic_case_study_term(term: &str) -> bool {
    matches!(
        term,
        "adopt"
            | "adopted"
            | "adopting"
            | "build"
            | "builds"
            | "case"
            | "cases"
            | "comparison"
            | "comparisons"
            | "experience"
            | "guide"
            | "guides"
            | "hermetic"
            | "journey"
            | "large"
            | "lessons"
            | "migration"
            | "migrated"
            | "migrating"
            | "monorepo"
            | "monorepos"
            | "polyglot"
            | "postmortem"
            | "postmortems"
            | "reproducible"
            | "reproducibility"
            | "study"
            | "studies"
            | "system"
            | "systems"
            | "tool"
            | "tools"
            | "tradeoff"
            | "tradeoffs"
    )
}

pub(super) fn project_scope_adjustment(query: &str, url: &url::Url) -> f32 {
    let host = url.host_str().unwrap_or_default();
    let query_terms = identity_terms(query);
    let source_intent = super::searxng::has_source_intent(&query.to_ascii_lowercase());

    if host == "github.com"
        && let Some(repository) = url.path_segments().and_then(|mut segments| {
            segments.next()?;
            segments.next()
        })
        && identity_matches(repository, &query_terms)
    {
        if source_intent {
            return 0.10;
        }
        if super::exact_identifier_query(query).is_some()
            && !is_generic_language_repository(repository)
        {
            return 0.08;
        }
    }

    if !source_intent {
        return 0.0;
    }

    if matches!(host, "docs.rs" | "crates.io" | "lib.rs") {
        let segments = url
            .path_segments()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let package = match (host, segments.as_slice()) {
            ("docs.rs", ["crate", package, ..]) => Some(*package),
            ("docs.rs", [package, ..]) => Some(*package),
            ("crates.io" | "lib.rs", ["crates", package, ..]) => Some(*package),
            _ => None,
        };
        return package.map_or(0.0, |package| {
            if identity_matches(package, &query_terms) {
                0.12
            } else {
                -0.18
            }
        });
    }

    host.split('.')
        .next()
        .filter(|label| identity_matches(label, &query_terms))
        .map_or(0.0, |_| 0.10)
}

fn is_generic_language_repository(repository: &str) -> bool {
    matches!(
        repository.to_ascii_lowercase().as_str(),
        "dart"
            | "go"
            | "kotlin"
            | "php-src"
            | "python"
            | "ruby"
            | "rust"
            | "scala"
            | "swift"
            | "typescript"
    )
}

fn identity_terms(value: &str) -> HashSet<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.len() >= 3)
        .map(str::to_ascii_lowercase)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "and"
                    | "current"
                    | "docs"
                    | "documentation"
                    | "example"
                    | "fix"
                    | "for"
                    | "from"
                    | "how"
                    | "official"
                    | "the"
                    | "use"
                    | "using"
                    | "with"
            )
        })
        .collect()
}

fn identity_matches(identity: &str, query_terms: &HashSet<String>) -> bool {
    let parts = identity
        .split(|character: char| !character.is_alphanumeric())
        .filter(|part| part.len() >= 2 && !matches!(*part, "crate" | "rs"))
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    !parts.is_empty() && parts.iter().all(|part| query_terms.contains(part))
}

fn jurisdiction_authority_bonus(query: &str, url: &url::Url) -> f32 {
    let Some(host) = url.host_str().filter(|host| host.ends_with(".gov")) else {
        return 0.0;
    };
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() >= 4)
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let public_information_intent = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "date"
                | "dates"
                | "deadline"
                | "election"
                | "license"
                | "permit"
                | "regulation"
                | "regulations"
                | "season"
                | "voting"
        )
    });
    let jurisdiction_match = host
        .split('.')
        .any(|label| tokens.iter().any(|token| token == label));

    if public_information_intent && jurisdiction_match {
        0.12
    } else {
        0.0
    }
}

fn descriptive_path_bonus(query: &str, url: &url::Url) -> f32 {
    let path = url.path().to_ascii_lowercase();
    let matching_terms = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| {
            term.len() >= 3
                && !matches!(
                    term.to_ascii_lowercase().as_str(),
                    "current" | "date" | "latest" | "news" | "official" | "recent"
                )
        })
        .filter(|term| path.contains(&term.to_ascii_lowercase()))
        .count();

    if matching_terms >= 2 { 0.05 } else { 0.0 }
}

fn has_recency_intent(query: &str) -> bool {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let explicit = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "breaking"
                | "headline"
                | "headlines"
                | "latest"
                | "news"
                | "newest"
                | "recent"
                | "recently"
                | "today"
        )
    });
    let current_temporal = tokens.iter().any(|token| token == "current")
        && tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "cpi"
                    | "data"
                    | "date"
                    | "advisory"
                    | "inflation"
                    | "mission"
                    | "price"
                    | "rate"
                    | "security"
                    | "status"
                    | "update"
                    | "version"
                    | "vulnerability"
                    | "weather"
            )
        });
    explicit || current_temporal
}

fn staleness_penalty(newest: Option<i32>, date: Option<i32>) -> f32 {
    let Some(age) = newest.zip(date).map(|(newest, date)| newest - date) else {
        return 0.0;
    };

    match age {
        ..=31 => 0.0,
        32..=183 => -0.015,
        184..=372 => -0.03,
        373..=744 => -0.06,
        _ => -0.10,
    }
}

fn date_key(date: &str) -> Option<i32> {
    let mut parts = date.get(..10)?.split('-');
    date_key_parts(
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    )
}

fn date_key_parts(year: i32, month: i32, day: i32) -> Option<i32> {
    ((1900..=2200).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day))
        .then_some(year * 372 + (month - 1) * 31 + day - 1)
}

fn query_date_key(query: &str) -> Option<i32> {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    tokens.windows(3).find_map(|parts| {
        let first = parts[0].parse::<i32>().ok();
        let second = parts[1].parse::<i32>().ok();
        let third = parts[2].parse::<i32>().ok();

        match (first, month_number(parts[0]), second, third) {
            (Some(year), _, Some(month), Some(day)) if year >= 1900 => {
                date_key_parts(year, month, day)
            }
            (Some(day), _, _, Some(year)) => date_key_parts(year, month_number(parts[1])?, day),
            (_, Some(month), Some(day), Some(year)) => date_key_parts(year, month, day),
            _ => None,
        }
    })
}

fn month_number(token: &str) -> Option<i32> {
    match token.to_ascii_lowercase().as_str() {
        "january" | "jan" => Some(1),
        "february" | "feb" => Some(2),
        "march" | "mar" => Some(3),
        "april" | "apr" => Some(4),
        "may" => Some(5),
        "june" | "jun" => Some(6),
        "july" | "jul" => Some(7),
        "august" | "aug" => Some(8),
        "september" | "sep" | "sept" => Some(9),
        "october" | "oct" => Some(10),
        "november" | "nov" => Some(11),
        "december" | "dec" => Some(12),
        _ => None,
    }
}

fn hit_date_key(hit: &Hit) -> Option<i32> {
    hit.date
        .as_deref()
        .and_then(date_key)
        .or_else(|| url_date_key(&hit.url))
}

fn hit_month_key(hit: &Hit) -> Option<i32> {
    month_key(&format!("{} {}", hit.title, hit.snippet))
        .or_else(|| hit.date.as_deref().and_then(date_key).map(|date| date / 31))
        .or_else(|| url_date_key(&hit.url).map(|date| date / 31))
}

fn month_key(value: &str) -> Option<i32> {
    let tokens = value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    tokens.windows(2).find_map(|parts| {
        let left_year = parts[0]
            .parse::<i32>()
            .ok()
            .filter(|year| (1900..=2200).contains(year));
        let right_year = parts[1]
            .parse::<i32>()
            .ok()
            .filter(|year| (1900..=2200).contains(year));
        let compact_month = parts[1]
            .strip_prefix(['m', 'M'])
            .and_then(|month| month.parse::<i32>().ok())
            .filter(|month| (1..=12).contains(month));
        let (year, month) = if let (Some(year), Some(month)) =
            (left_year, month_number(parts[1]).or(compact_month))
        {
            (year, month)
        } else if let (Some(month), Some(year)) = (month_number(parts[0]), right_year) {
            (year, month)
        } else {
            return None;
        };
        Some(year * 12 + month - 1)
    })
}

fn url_date_key(url: &url::Url) -> Option<i32> {
    let parts = url.path_segments()?.collect::<Vec<_>>();
    parts.windows(3).find_map(|parts| {
        date_key_parts(
            parts[0].parse().ok()?,
            parts[1].parse().ok()?,
            parts[2].parse().ok()?,
        )
    })
}

fn explicit_date_adjustment(requested: Option<i32>, candidate: Option<i32>) -> f32 {
    match requested
        .zip(candidate)
        .map(|(left, right)| left.abs_diff(right))
    {
        Some(0) => 0.12,
        Some(1) => 0.04,
        _ => 0.0,
    }
}

fn explicit_period_adjustment(requested: Option<i32>, candidate: Option<i32>) -> f32 {
    match requested
        .zip(candidate)
        .map(|(requested, candidate)| requested - candidate)
    {
        Some(0) => 0.12,
        Some(1) => 0.08,
        Some(2) => -0.04,
        Some(3..) => -0.10,
        Some(..0) => -0.03,
        None => 0.0,
    }
}

fn select_indices(
    hit_count: usize,
    dense_ranking: &[(usize, f32)],
    limit: usize,
    source_priority: &HashSet<usize>,
    upstream_consensus: &HashSet<usize>,
) -> Vec<usize> {
    let Some(best_score) = dense_ranking.first().map(|(_, score)| *score) else {
        return Vec::new();
    };
    let mut scores = vec![f32::NEG_INFINITY; hit_count];
    for &(index, score) in dense_ranking {
        if let Some(slot) = scores.get_mut(index) {
            *slot = score;
        }
    }
    // `source_priority` has already passed strict-intent and host verification
    // at the metasearch boundary. Reserve one result unconditionally; all
    // additional first-party candidates compete normally.
    let mut selected = dense_ranking
        .iter()
        .filter(|(index, _)| source_priority.contains(index))
        .take(usize::from(limit > 0))
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    let upstream_quota = limit.saturating_sub(selected.len()).div_ceil(2);
    selected.extend(
        (0..upstream_quota.min(hit_count))
            .filter(|&index| {
                upstream_consensus.contains(&index)
                    && !selected.contains(&index)
                    && best_score - scores[index] <= CONSENSUS_MAX_SCORE_GAP
            })
            .collect::<Vec<_>>(),
    );
    for &(index, _) in dense_ranking {
        if selected.len() == limit {
            break;
        }
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    selected
}

fn reserve_requested_source(
    mut selected: Vec<usize>,
    dense_ranking: &[(usize, f32)],
    source_priority: &HashSet<usize>,
    limit: usize,
) -> Vec<usize> {
    if limit == 0 {
        return Vec::new();
    }
    let Some(source) = dense_ranking
        .iter()
        .find(|(index, _)| source_priority.contains(index))
        .map(|(index, _)| *index)
    else {
        return selected;
    };

    selected.retain(|index| *index != source);
    selected.insert(0, source);
    selected.truncate(limit);
    selected
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        CASE_STUDY_FORM_WEIGHT, QUERY_EVIDENCE_WEIGHT, QueryEvidence, authority_bonus,
        candidate_evidence_adjustment, case_study_form_adjustment, date_key, exact_title,
        explicit_date_adjustment, explicit_period_adjustment, has_recency_intent, is_as_of_query,
        is_scholarly_manifestation, is_scholarly_url, jurisdiction_authority_bonus, month_key,
        paper_title_adjustment, prioritize_exact_title_hits, prioritize_exact_title_indices,
        project_scope_adjustment, query_date_key, reserve_requested_source, select_indices,
        source_authority_adjustment, staleness_penalty, title_equivalent, url_date_key,
    };
    use crate::search::searxng::Hit;
    use url::Url;

    fn assert_evidence_breaks_a_close_dense_tie(query: &str, generic: &str, answer: &str) {
        let documents = [generic, answer];
        let evidence = QueryEvidence::new(query, documents);
        let generic_adjustment = candidate_evidence_adjustment(&evidence, generic);
        let answer_adjustment = candidate_evidence_adjustment(&evidence, answer);

        assert!(
            answer_adjustment > generic_adjustment,
            "answer={answer_adjustment}, generic={generic_adjustment}"
        );
        assert!(
            0.70 + answer_adjustment > 0.71 + generic_adjustment,
            "the evidence signal did not recover a 0.01 dense-score deficit: answer={answer_adjustment}, generic={generic_adjustment}"
        );
        assert!(
            answer_adjustment <= QUERY_EVIDENCE_WEIGHT,
            "evidence adjustment exceeded its declared cap"
        );
    }

    #[test]
    fn exact_error_evidence_survives_candidate_shortlisting() {
        assert_evidence_breaks_a_close_dense_tie(
            "Rust E0277 FromResidual<Result<Infallible, _>> is not implemented using ? with a custom error enum",
            "Rust async error handling with the question-mark operator and custom enums",
            "Fix E0277: FromResidual<Result<Infallible, MyError>> is not implemented; convert the source error into the custom error enum before using ?.",
        );
    }

    #[test]
    fn exact_typescript_diagnostic_survives_candidate_shortlisting() {
        assert_evidence_breaks_a_close_dense_tie(
            "TypeScript TS2589 Type instantiation is excessively deep and possibly infinite Zod recursive schema workaround",
            "A guide to TypeScript types and recursive Zod schemas",
            "TS2589: Type instantiation is excessively deep and possibly infinite when declaring a recursive Zod schema, with a lazy-schema workaround.",
        );
    }

    #[test]
    fn implementation_case_studies_survive_candidate_shortlisting() {
        assert_evidence_breaks_a_close_dense_tie(
            "reproducible hermetic builds in large polyglot monorepos Bazel Nix Pants tradeoffs case studies",
            "An introduction to reproducible build tooling for monorepos",
            "A production case study comparing Bazel, Nix, and Pants tradeoffs while migrating a large polyglot monorepo to hermetic builds.",
        );
    }

    #[test]
    fn numerical_parity_evidence_survives_candidate_shortlisting() {
        assert_evidence_breaks_a_close_dense_tie(
            "detect silent numerical drift porting machine-learning inference from CPU to CUDA bitwise reproducibility tolerance testing",
            "How to deploy machine-learning inference on a CUDA GPU",
            "Detect silent numerical drift between CPU and CUDA inference with bitwise reproducibility checks plus absolute and relative tolerance testing.",
        );
    }

    #[test]
    fn protects_the_consensus_head_and_dense_ranks_the_tail() {
        let selected = select_indices(
            8,
            &[
                (7, 0.9),
                (0, 0.85),
                (1, 0.81),
                (6, 0.8),
                (5, 0.7),
                (4, 0.6),
                (3, 0.5),
                (2, 0.4),
            ],
            4,
            &HashSet::new(),
            &HashSet::from([0, 1]),
        );

        assert_eq!(selected, [0, 1, 7, 6]);
    }

    #[test]
    fn exact_paper_titles_beat_variants_on_scholarly_hosts() {
        let hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let query = "Original arXiv paper: Attention Is All You Need";

        assert_eq!(
            paper_title_adjustment(
                query,
                &hit(
                    "[1706.03762] Attention Is All You Need - arXiv.org",
                    "https://arxiv.org/abs/1706.03762"
                )
            ),
            0.20
        );
        assert_eq!(
            paper_title_adjustment(
                query,
                &hit(
                    "Tool Attention Is All You Need: Dynamic Tool Gating",
                    "https://arxiv.org/abs/2604.21816"
                )
            ),
            0.0
        );
        assert_eq!(
            paper_title_adjustment(
                query,
                &hit(
                    "PDF: Attention Is All You Need | arXiv",
                    "https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need"
                )
            ),
            0.20
        );
        assert_eq!(
            paper_title_adjustment(
                query,
                &hit(
                    "arXiv:1706.03762 [cs.CL] Attention Is All You Need",
                    "https://research.google/pubs/attention-is-all-you-need/"
                )
            ),
            0.20
        );
        assert_eq!(
            paper_title_adjustment(
                query,
                &hit(
                    "Attention Is All You Need",
                    "https://example.com/copied-title"
                )
            ),
            0.0
        );
    }

    #[test]
    fn exact_paper_queries_lead_with_exact_copies_and_backfill_the_tail() {
        let hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let hits = vec![
            hit(
                "[1706.03762] Attention Is All You Need - arXiv.org",
                "https://arxiv.org/abs/1706.03762",
            ),
            hit(
                "Tool Attention Is All You Need: Dynamic Tool Gating",
                "https://arxiv.org/abs/2604.21816",
            ),
            hit(
                "[PDF] Attention Is All You Need — arXiv",
                "https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need",
            ),
            hit(
                "An overview of transformer architectures",
                "https://example.com/transformers",
            ),
        ];
        let indices = prioritize_exact_title_indices(
            "Original paper PDF: Attention Is All You Need",
            &hits,
            vec![3, 1],
            4,
        );

        let selected =
            prioritize_exact_title_hits("Original paper PDF: Attention Is All You Need", hits);

        assert_eq!(indices, [0, 2, 3, 1]);
        assert_eq!(selected.len(), 4);
        assert_eq!(selected[0].url.host_str(), Some("arxiv.org"));
        assert_eq!(selected[1].url.host_str(), Some("proceedings.neurips.cc"));
        assert_eq!(selected[2].url.host_str(), Some("arxiv.org"));
        assert_eq!(selected[3].url.host_str(), Some("example.com"));
    }

    #[test]
    fn full_text_queries_recognize_exact_document_titles() {
        let hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let hits = vec![
            hit("RFC 9114: HTTP/3", "https://www.rfc-editor.org/rfc/rfc9114"),
            hit("What HTTP/3 means for the web", "https://example.com/http3"),
        ];

        let selected = prioritize_exact_title_hits("Full text of RFC 9114: HTTP/3", hits);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].url.host_str(), Some("www.rfc-editor.org"));
    }

    #[test]
    fn full_text_search_is_not_misread_as_a_document_title() {
        assert_eq!(
            exact_title("PostgreSQL full text search documentation"),
            None
        );
        assert_eq!(
            exact_title("Full-text of RFC 9114: HTTP/3"),
            Some("RFC 9114: HTTP/3".into())
        );
    }

    #[test]
    fn recognizes_common_biomedical_manifestations_and_title_suffixes() {
        let pubmed = Hit {
            title: "A randomized controlled trial - PubMed".into(),
            url: Url::parse("https://pubmed.ncbi.nlm.nih.gov/12345/").unwrap(),
            date: None,
            snippet: String::new(),
        };
        let pmc = Hit {
            title: "A randomized controlled trial | SpringerLink".into(),
            url: Url::parse("https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/").unwrap(),
            date: None,
            snippet: String::new(),
        };

        assert!(is_scholarly_manifestation(&pubmed));
        assert!(is_scholarly_manifestation(&pmc));
        assert!(title_equivalent(
            "A randomized controlled trial",
            &pubmed.title
        ));
        assert!(title_equivalent(
            "A randomized controlled trial",
            &pmc.title
        ));
    }

    #[test]
    fn exact_copies_are_not_discarded_before_fetching() {
        let hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let mut hits = (0..8)
            .map(|index| {
                hit(
                    "Attention Is All You Need",
                    &format!("https://arxiv.org/abs/1706.03762v{index}"),
                )
            })
            .collect::<Vec<_>>();
        hits.extend((0..8).map(|index| {
            hit(
                "A survey of transformer architectures",
                &format!("https://surveys{index}.example/transformers"),
            )
        }));

        let selected =
            prioritize_exact_title_hits("Original paper: Attention Is All You Need", hits);

        assert_eq!(selected.len(), 16);
        assert_eq!(
            selected
                .iter()
                .filter(|hit| hit.title == "Attention Is All You Need")
                .count(),
            8
        );
        assert!(
            selected[8..]
                .iter()
                .all(|hit| hit.title == "A survey of transformer architectures")
        );
    }

    #[test]
    fn exact_title_filter_is_inert_when_searxng_found_no_exact_copy() {
        let hits = vec![Hit {
            title: "Transformer architecture overview".into(),
            url: Url::parse("https://example.com/transformers").unwrap(),
            date: None,
            snippet: String::new(),
        }];

        let selected =
            prioritize_exact_title_hits("Original paper: Attention Is All You Need", hits);

        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn rejects_consensus_results_that_are_semantically_implausible() {
        let selected = select_indices(
            8,
            &[
                (7, 0.9),
                (6, 0.8),
                (5, 0.7),
                (4, 0.6),
                (3, 0.5),
                (2, 0.4),
                (1, 0.3),
                (0, 0.2),
            ],
            4,
            &HashSet::new(),
            &HashSet::from([0, 1]),
        );

        assert_eq!(selected, [7, 6, 5, 4]);
    }

    #[test]
    fn one_engine_head_does_not_receive_a_consensus_reservation() {
        let ranking = [(3, 0.90), (2, 0.89), (0, 0.88), (1, 0.70)];

        assert_eq!(
            select_indices(4, &ranking, 2, &HashSet::new(), &HashSet::new()),
            [3, 2]
        );
        assert_eq!(
            select_indices(4, &ranking, 2, &HashSet::new(), &HashSet::from([0])),
            [0, 3]
        );
    }

    #[test]
    fn preserves_a_relevant_deliberately_focused_first_party_result() {
        let selected = select_indices(
            4,
            &[(3, 0.9), (0, 0.82), (2, 0.8), (1, 0.7)],
            2,
            &HashSet::from([0]),
            &HashSet::new(),
        );

        assert_eq!(selected, [0, 3]);
    }

    #[test]
    fn reserves_one_verified_requested_source_even_outside_the_dense_score_gap() {
        let selected = select_indices(
            4,
            &[(3, 0.9), (2, 0.8), (1, 0.7), (0, 0.6)],
            2,
            &HashSet::from([0]),
            &HashSet::new(),
        );

        assert_eq!(selected, [0, 3]);
    }

    #[test]
    fn reasserts_the_requested_source_after_other_priority_passes() {
        assert_eq!(
            reserve_requested_source(
                vec![3, 2],
                &[(3, 0.9), (2, 0.8), (0, 0.6)],
                &HashSet::from([0]),
                2,
            ),
            [0, 3]
        );
    }

    #[test]
    fn recognizes_explicit_recency_in_events_facts_and_live_conditions() {
        assert!(has_recency_intent("latest OpenSSH vulnerability"));
        assert!(has_recency_intent("recent James Webb discovery"));
        assert!(has_recency_intent("OpenAI news"));
        assert!(has_recency_intent("current federal funds rate"));
        assert!(has_recency_intent("water temperature today"));
    }

    #[test]
    fn authority_bonus_is_narrow_and_query_aware() {
        let nvd = Url::parse("https://nvd.nist.gov/vuln/detail/CVE-2026-1").unwrap();
        let paper = Url::parse("https://www.pnas.org/doi/10.1073/example").unwrap();
        let doi = Url::parse("https://doi.org/10.1038/example").unwrap();
        let arxiv = Url::parse("https://arxiv.org/abs/2603.02871").unwrap();
        let blog = Url::parse("https://example.com/research/evidence").unwrap();
        let nasa = Url::parse("https://www.nasa.gov/mission/artemis-ii/").unwrap();
        let nasa_event = Url::parse("https://www.nasa.gov/event/artemis-ii-launch/").unwrap();
        let nasa_archive = Url::parse("https://ntrs.nasa.gov/citations/20260004348").unwrap();
        let rust_diagnostic =
            Url::parse("https://doc.rust-lang.org/std/convert/enum.Infallible.html").unwrap();

        assert_eq!(authority_bonus("latest vulnerability", &nvd), 0.04);
        assert_eq!(authority_bonus("NASA mission overview", &nasa), 0.12);
        assert_eq!(authority_bonus("NASA latest launch date", &nasa), 0.12);
        assert_eq!(
            authority_bonus("NASA Artemis II latest launch date", &nasa_event),
            0.17
        );
        assert_eq!(
            authority_bonus("NASA Artemis II latest launch date", &nasa_archive),
            0.12
        );
        assert_eq!(authority_bonus("latest research evidence", &paper), 0.04);
        assert_eq!(authority_bonus("latest research evidence", &doi), 0.04);
        assert_eq!(
            authority_bonus("cross-backend numerical reproducibility", &arxiv),
            0.04
        );
        assert_eq!(
            authority_bonus("Bazel Nix Pants tradeoffs case studies", &arxiv),
            0.0
        );
        assert_eq!(authority_bonus("latest release", &paper), 0.0);
        assert_eq!(authority_bonus("latest research evidence", &blog), 0.0);
        assert_eq!(
            authority_bonus(
                "Rust E0277 \"FromResidual<Result<Infallible, _>> is not implemented\" using ? with a custom error enum",
                &rust_diagnostic,
            ),
            0.09
        );
    }

    #[test]
    fn shared_authority_adjustment_matches_candidate_scoring() {
        let arxiv = Url::parse("https://arxiv.org/abs/2509.06977").unwrap();
        let tokio = Url::parse("https://docs.rs/tokio/latest/tokio/").unwrap();

        assert_eq!(
            source_authority_adjustment("cross-backend numerical reproducibility testing", &arxiv,),
            0.04
        );
        assert_eq!(
            source_authority_adjustment("Tokio official documentation", &tokio),
            project_scope_adjustment("Tokio official documentation", &tokio)
        );
    }

    #[test]
    fn scholarly_url_detection_is_narrow_and_host_aware() {
        for url in [
            "https://arxiv.org/abs/2509.06977",
            "https://doi.org/10.1145/example",
            "https://www.pnas.org/doi/10.1073/example",
        ] {
            assert!(is_scholarly_url(&Url::parse(url).unwrap()), "missed {url}");
        }
        assert!(!is_scholarly_url(
            &Url::parse("https://example.com/research/evidence").unwrap()
        ));
    }

    #[test]
    fn explicit_case_study_queries_prefer_firsthand_document_forms() {
        let query = "reproducible hermetic builds in large polyglot monorepos bazel nix pants tradeoffs case studies";
        let hit = |title: &str| Hit {
            title: title.into(),
            url: Url::parse("https://example.com/article").unwrap(),
            date: None,
            snippet: String::new(),
        };

        for title in [
            "Tweag case study: From adopting Pants to generalizing our CI",
            "Case Study: Incrementally migrating a Python monorepo from Bazel to Pants",
            "Bazel and Nix: A Migration Experience",
        ] {
            assert_eq!(
                case_study_form_adjustment(query, &hit(title)),
                CASE_STUDY_FORM_WEIGHT,
                "missed {title:?}"
            );
        }
    }

    #[test]
    fn case_study_form_signal_rejects_generic_or_unrequested_pages() {
        let query = "reproducible hermetic builds in large polyglot monorepos bazel nix pants tradeoffs case studies";
        let hit = |title: &str| Hit {
            title: title.into(),
            url: Url::parse("https://example.com/article").unwrap(),
            date: None,
            snippet: String::new(),
        };

        for title in [
            "Best Reproducible Build Tools: Buyer's Guide",
            "Bazel vs Nx: Which Monorepo Build Tool?",
            "Case Studies",
            "Monorepo Migration Case Studies",
        ] {
            assert_eq!(
                case_study_form_adjustment(query, &hit(title)),
                0.0,
                "accepted {title:?}"
            );
        }
        assert_eq!(
            case_study_form_adjustment(
                "compare Bazel Nix and Pants build systems",
                &hit("Tweag case study: From adopting Pants to generalizing our CI"),
            ),
            0.0
        );
    }

    #[test]
    fn project_scope_distinguishes_registry_lookalikes() {
        let tokio = Url::parse("https://docs.rs/tokio/latest/tokio/macro.select.html").unwrap();
        let fork =
            Url::parse("https://docs.rs/tokio_with_wasm/latest/tokio_with_wasm/macro.select.html")
                .unwrap();
        let source =
            Url::parse("https://github.com/tokio-rs/tokio/blob/main/tokio/src/lib.rs").unwrap();
        let zod_issue = Url::parse("https://github.com/colinhacks/zod/issues/6015").unwrap();
        let typescript_issue =
            Url::parse("https://github.com/microsoft/TypeScript/issues/34933").unwrap();

        assert_eq!(
            project_scope_adjustment("Rust tokio select official documentation", &tokio),
            0.12
        );
        assert_eq!(
            project_scope_adjustment("Rust tokio select official documentation", &fork),
            -0.18
        );
        assert_eq!(
            project_scope_adjustment("Rust tokio select official documentation", &source),
            0.10
        );
        assert_eq!(
            project_scope_adjustment(
                "TypeScript TS2589 \"Type instantiation is excessively deep\" Zod recursive schema workaround",
                &zod_issue,
            ),
            0.08
        );
        assert_eq!(
            project_scope_adjustment(
                "TypeScript TS2589 \"Type instantiation is excessively deep\" recursive schema workaround",
                &typescript_issue,
            ),
            0.0
        );
    }

    #[test]
    fn jurisdiction_authority_bonus_requires_both_place_and_public_information_intent() {
        let dnr = Url::parse("https://dnr.wisconsin.gov/topic/hunt/deer").unwrap();
        let unrelated = Url::parse("https://www.federalreserve.gov/monetarypolicy.htm").unwrap();
        let guide = Url::parse("https://huntingseason.com/states/wisconsin").unwrap();

        assert_eq!(
            jurisdiction_authority_bonus("Wisconsin deer hunting season dates 2026", &dnr),
            0.12
        );
        assert_eq!(
            jurisdiction_authority_bonus("Wisconsin economic outlook", &dnr),
            0.0
        );
        assert_eq!(
            jurisdiction_authority_bonus("Wisconsin deer hunting season dates 2026", &unrelated),
            0.0
        );
        assert_eq!(
            jurisdiction_authority_bonus("Wisconsin deer hunting season dates 2026", &guide),
            0.0
        );
    }

    #[test]
    fn freshness_signal_only_penalizes_explicitly_stale_results() {
        let newest = date_key("2026-08-02");

        assert_eq!(staleness_penalty(newest, date_key("2026-08-01")), 0.0);
        assert_eq!(staleness_penalty(newest, date_key("2026-07-01")), -0.015);
        assert_eq!(staleness_penalty(newest, date_key("2025-08-02")), -0.03);
        assert_eq!(staleness_penalty(newest, date_key("2023-11-21")), -0.10);
        assert_eq!(staleness_penalty(newest, None), 0.0);
    }

    #[test]
    fn parses_complete_query_dates_without_treating_years_as_dates() {
        assert_eq!(
            query_date_key("OpenAI news August 1, 2026"),
            date_key("2026-08-01")
        );
        assert_eq!(query_date_key("news 1 Aug 2026"), date_key("2026-08-01"));
        assert_eq!(query_date_key("news 2026-08-01"), date_key("2026-08-01"));
        assert_eq!(query_date_key("hypertension guideline 2026"), None);
        assert_eq!(query_date_key("release 2026.99.01"), None);
    }

    #[test]
    fn month_periods_prefer_the_nearest_available_release() {
        let july = month_key("current CPI July 2026");

        assert_eq!(july, month_key("2026 M07 Results"));
        assert_eq!(
            explicit_period_adjustment(july, month_key("June 2026")),
            0.08
        );
        assert_eq!(
            explicit_period_adjustment(july, month_key("2026 M05")),
            -0.04
        );
        assert_eq!(month_key("guideline 2026"), None);
    }

    #[test]
    fn extracts_dates_from_common_news_urls() {
        let npr =
            Url::parse("https://www.npr.org/2026/08/01/nx-s1-5914852/anthropic-openai-models")
                .unwrap();
        let undated = Url::parse("https://example.com/latest/story").unwrap();

        assert_eq!(url_date_key(&npr), date_key("2026-08-01"));
        assert_eq!(url_date_key(&undated), None);
    }

    #[test]
    fn exact_query_dates_receive_a_narrow_relevance_bonus() {
        let requested = date_key("2026-08-01");

        assert_eq!(
            explicit_date_adjustment(requested, date_key("2026-08-01")),
            0.12
        );
        assert_eq!(
            explicit_date_adjustment(requested, date_key("2026-07-31")),
            0.04
        );
        assert_eq!(
            explicit_date_adjustment(requested, date_key("2026-07-30")),
            0.0
        );
        assert_eq!(explicit_date_adjustment(None, date_key("2026-08-01")), 0.0);
    }

    #[test]
    fn distinguishes_as_of_queries_from_target_news_periods() {
        assert!(is_as_of_query(
            "recent Europa Clipper mission status August 2026"
        ));
        assert!(is_as_of_query(
            "latest NVIDIA security advisory August 2026"
        ));
        assert!(!is_as_of_query("OpenAI news August 2026"));
        assert!(!is_as_of_query("NASA mission update August 2026"));
    }
}
