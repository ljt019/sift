use std::collections::{HashMap, HashSet};
use std::time::Duration;

use serde::{Deserialize, Deserializer};
use tokio::time::{Instant, sleep_until};
use url::Url;

use super::cache::SearchOutcome;
use super::evidence::{
    has_case_study_intent, has_full_text_intent, has_machine_learning_inference_intent,
    has_numerical_parity_intent, is_as_of_query,
};
use super::exact_identifier_query;
use crate::error::{AppError, Result};
use crate::state::AppState;

const SUPPLEMENTAL_QUERY_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_SEARCH_CANDIDATES: usize = 32;

#[derive(Clone, Debug)]
pub struct Hit {
    pub title: String,
    pub url: Url,
    pub date: Option<String>,
    pub snippet: String,
}

#[derive(Clone, Debug)]
pub(crate) struct SearchCandidate {
    pub hit: Hit,
    pub source_priority: bool,
    pub upstream_consensus: bool,
    pub fetch_urls: Vec<Url>,
}

impl std::ops::Deref for SearchCandidate {
    type Target = Hit;

    fn deref(&self) -> &Self::Target {
        &self.hit
    }
}

pub(crate) async fn search(
    state: &AppState,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchCandidate>> {
    let mut candidates = state
        .search_cache
        .get_or_search(query, MAX_SEARCH_CANDIDATES, || {
            search_uncached(state, query, MAX_SEARCH_CANDIDATES)
        })
        .await?;
    candidates.truncate(limit);
    Ok(candidates)
}

async fn search_uncached(state: &AppState, query: &str, limit: usize) -> Result<SearchOutcome> {
    let auxiliary_allowed = allows_auxiliary_search(query);
    let stackexchange = async {
        if auxiliary_allowed && should_search_stackoverflow(query) {
            state.stackexchange.search(query, 8.min(limit)).await
        } else {
            Vec::new()
        }
    };
    let github = async {
        if auxiliary_allowed {
            super::github::search(state, query, 8.min(limit)).await
        } else {
            Vec::new()
        }
    };
    let (searxng, github, stackexchange) =
        tokio::join!(search_searxng(state, query, limit), github, stackexchange,);
    let mut auxiliary = github;
    auxiliary.extend(stackexchange);
    let SearchPool {
        candidates,
        healthy,
    } = combine_search_pools(searxng, auxiliary, limit)?;
    Ok(SearchOutcome::new(candidates, healthy))
}

fn allows_auxiliary_search(query: &str) -> bool {
    !has_strict_source_intent(&query.to_ascii_lowercase())
}

fn combine_search_pools(
    searxng: Result<SearchPool>,
    auxiliary: Vec<Hit>,
    limit: usize,
) -> Result<SearchPool> {
    match searxng {
        Ok(SearchPool {
            mut candidates,
            healthy,
        }) => {
            merge_auxiliary_hits(&mut candidates, auxiliary, limit);
            Ok(SearchPool {
                candidates,
                healthy,
            })
        }
        Err(error) if !auxiliary.is_empty() => {
            tracing::debug!(
                error = ?error,
                auxiliary_results = auxiliary.len(),
                "serving auxiliary results after SearXNG failed"
            );
            let mut candidates = Vec::new();
            merge_auxiliary_hits(&mut candidates, auxiliary, limit);
            Ok(SearchPool {
                candidates,
                healthy: false,
            })
        }
        Err(error) => Err(error),
    }
}

fn should_search_stackoverflow(query: &str) -> bool {
    if !has_it_intent(query) {
        return false;
    }
    exact_identifier_query(query).is_some()
        || quoted_phrase(query).is_some()
        || query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| {
                matches!(
                    token.to_ascii_lowercase().as_str(),
                    "bug"
                        | "error"
                        | "exception"
                        | "failed"
                        | "failure"
                        | "fix"
                        | "mismatch"
                        | "panic"
                        | "undefined"
                        | "unresolved"
                )
            })
}

struct SearchPool {
    candidates: Vec<SearchCandidate>,
    healthy: bool,
}

async fn search_searxng(state: &AppState, query: &str, limit: usize) -> Result<SearchPool> {
    let _permit = state
        .searxng_permits
        .acquire()
        .await
        .expect("SearXNG semaphore is never closed");
    wait_for_rate_limit(state).await;

    let route = route(query, &state.config.searxng_categories);
    let mut body = request(
        state,
        &route.query,
        &route.categories,
        None,
        state.config.searxng_timeout,
    )
    .await?;
    let supplemental_queries = supplemental_queries(
        query,
        &route,
        &body.results,
        &state.config.searxng_categories,
    );
    let mut observation_id = 1;
    tracing::debug!(
        original_query = query,
        routed_query = route.query,
        categories = route.categories,
        supplemental = ?supplemental_queries,
        "planned SearXNG search"
    );

    // A category is a fan-out over every engine assigned to it, not a ranking
    // hint. Most searches use at most one supplement. Explicit recommendation
    // queries may additionally use two disjoint precision lookups: a narrower
    // general-web query, then OpenAlex and Semantic Scholar for an accessible
    // manifestation of the exact document found by the broad search.
    for supplemental_query in supplemental_queries {
        wait_for_rate_limit(state).await;
        let timeout = state.config.searxng_timeout.min(SUPPLEMENTAL_QUERY_TIMEOUT);
        let time_range = match &supplemental_query {
            SupplementalQuery::Specialist(specialist)
                if has_category(&specialist.categories, "news") =>
            {
                freshness_window(query)
            }
            _ => None,
        };
        let supplemental = match &supplemental_query {
            SupplementalQuery::Manifestation(focused) => {
                request_engines(state, &focused.query, &focused.engines, time_range, timeout).await
            }
            _ => {
                request(
                    state,
                    supplemental_query.query(),
                    supplemental_query
                        .categories(&route.categories, &state.config.searxng_categories),
                    time_range,
                    timeout,
                )
                .await
            }
        };
        match supplemental {
            Ok(mut supplemental) => {
                mark_observation(&mut supplemental.results, observation_id);
                observation_id += 1;
                tracing::debug!(
                    query = supplemental_query.query(),
                    results = supplemental.results.len(),
                    "received supplemental SearXNG results"
                );
                match supplemental_query {
                    SupplementalQuery::Manifestation(manifestation) => {
                        merge_manifestation_result(&mut body, supplemental, &manifestation.query);
                    }
                    SupplementalQuery::Focused(_) => {
                        merge_discovery_results(&mut body, supplemental);
                    }
                    SupplementalQuery::FullText(_) => {
                        merge_discovery_results(&mut body, supplemental);
                    }
                    SupplementalQuery::Specialist(_) => {
                        merge_discovery_results(&mut body, supplemental);
                    }
                    SupplementalQuery::Local(_) => {
                        merge_local_results(&mut body, supplemental);
                    }
                    SupplementalQuery::Source(source) => {
                        if source.host.is_none()
                            && let Some(host) =
                                infer_official_host(&route.query, &supplemental.results)
                        {
                            merge_source_results(
                                &mut body,
                                supplemental,
                                Some(&host),
                                source.strict,
                            );
                        } else {
                            merge_source_results(
                                &mut body,
                                supplemental,
                                source.host.as_deref(),
                                source.strict,
                            );
                        }
                    }
                }
            }
            Err(error) => {
                tracing::debug!(
                    query = supplemental_query.query(),
                    error = ?error,
                    "supplemental SearXNG query failed"
                );
            }
        }
    }

    let candidates = into_hits(body, query, limit);
    // SearXNG is a fault-tolerant metasearch pool: individual engines are
    // expected to be suspended while healthy peers continue answering. Cache
    // any nonempty merged pool normally so partial outages do not increase
    // load on the engines that remain. Empty and transport-failed searches
    // still take the shorter retry path in SearchCache.
    let healthy = !candidates.is_empty();
    Ok(SearchPool {
        candidates,
        healthy,
    })
}

async fn request(
    state: &AppState,
    query: &str,
    categories: &str,
    time_range: Option<&str>,
    timeout: Duration,
) -> Result<SearxResponse> {
    request_target(
        state,
        query,
        SearchTarget::Categories(categories),
        time_range,
        timeout,
    )
    .await
}

async fn request_engines(
    state: &AppState,
    query: &str,
    engines: &str,
    time_range: Option<&str>,
    timeout: Duration,
) -> Result<SearxResponse> {
    request_target(
        state,
        query,
        SearchTarget::Engines(engines),
        time_range,
        timeout,
    )
    .await
}

#[derive(Clone, Copy, Debug)]
enum SearchTarget<'a> {
    Categories(&'a str),
    Engines(&'a str),
}

async fn request_target(
    state: &AppState,
    query: &str,
    target: SearchTarget<'_>,
    time_range: Option<&str>,
    timeout: Duration,
) -> Result<SearxResponse> {
    let url = search_url_target(&state.config.searxng_url, query, target, time_range);
    let response = state
        .http
        .get(url)
        .timeout(timeout)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(AppError::SearchBackend)?;

    let mut body: SearxResponse = response.json().await.map_err(|error| {
        AppError::SearchBackendResponse(format!("could not decode JSON: {error}"))
    })?;
    let mut contributors = body
        .results
        .iter()
        .flat_map(|result| result.engines.iter().cloned())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    contributors.sort_unstable();
    tracing::debug!(
        query,
        ?target,
        results = body.results.len(),
        contributors = ?contributors,
        unresponsive = ?body.unresponsive_engines,
        "received SearXNG response"
    );
    let target_label = match target {
        SearchTarget::Categories(categories) => format!("categories:{categories}"),
        SearchTarget::Engines(engines) => format!("engines:{engines}"),
    };
    super::debug::capture_search_response(
        state.config.sift_debug_dir.as_deref(),
        query,
        &target_label,
        body.results.len(),
        &contributors,
        &body.unresponsive_engines,
    )
    .await;
    // SearXNG score-sorts results, then rearranges them into UI-oriented
    // category/template blocks. Its JSON endpoint preserves that presentation
    // order, so restore the relevance order before Sift truncates candidates.
    sort_by_relevance(&mut body.results);
    Ok(body)
}

fn sort_by_relevance(results: &mut [SearxResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.engines.len().cmp(&left.engines.len()))
    });
}

fn mark_observation(results: &mut [SearxResult], observation_id: usize) {
    for result in results {
        result.observation_id = observation_id;
    }
}

async fn wait_for_rate_limit(state: &AppState) {
    let mut next_request = state.searxng_next_request.lock().await;
    let now = Instant::now();
    if *next_request > now {
        sleep_until(*next_request).await;
    }
    *next_request = Instant::now() + state.config.searxng_min_interval;
}

#[derive(Debug, PartialEq, Eq)]
struct Route {
    query: String,
    categories: String,
    supplemental: Option<SpecialistQuery>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SpecialistQuery {
    query: String,
    categories: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EngineQuery {
    query: String,
    engines: String,
}

type ManifestationQuery = EngineQuery;

#[derive(Debug, PartialEq, Eq)]
enum SupplementalQuery {
    Focused(SpecialistQuery),
    FullText(String),
    Manifestation(ManifestationQuery),
    Specialist(SpecialistQuery),
    Local(String),
    Source(SourceQuery),
}

impl SupplementalQuery {
    fn query(&self) -> &str {
        match self {
            Self::Focused(focused) => &focused.query,
            Self::FullText(query) => query,
            Self::Local(query) => query,
            Self::Manifestation(manifestation) => &manifestation.query,
            Self::Source(source) => &source.query,
            Self::Specialist(specialist) => &specialist.query,
        }
    }

    fn categories<'a>(&'a self, default: &'a str, configured: &'a str) -> &'a str {
        match self {
            Self::Local(_) => "map",
            Self::FullText(_) => source_categories(default, configured),
            Self::Manifestation(_) => {
                unreachable!("explicit engine queries do not use categories")
            }
            Self::Focused(focused) => &focused.categories,
            Self::Source(_) => source_categories(default, configured),
            Self::Specialist(specialist) => &specialist.categories,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SourceQuery {
    query: String,
    host: Option<String>,
    strict: bool,
}

fn supplemental_queries(
    query: &str,
    route: &Route,
    results: &[SearxResult],
    configured_categories: &str,
) -> Vec<SupplementalQuery> {
    // A manifestation lookup is useful for a known document, but harmful for
    // exploratory freshness queries: it turns the first broad result into the
    // subject of the only supplemental request and displaces the science
    // vertical that can discover a better document.
    let manifestation = (route.categories != "science" && !is_as_of_query(query))
        .then(|| scholarly_manifestation_query(query, results, configured_categories))
        .flatten();

    let source = source_aware_query(query, &route.query, results);
    if has_full_text_intent(query)
        && has_science_intent(query)
        && named_authority_host(query).is_none()
        && let Some(specialist) = route
            .supplemental
            .as_ref()
            .filter(|specialist| has_category(&specialist.categories, "science"))
            .cloned()
    {
        return vec![SupplementalQuery::Specialist(specialist)];
    }
    if full_text_locator_query(query).is_none()
        && source.as_ref().is_some_and(|source| source.strict)
    {
        return source.into_iter().map(SupplementalQuery::Source).collect();
    }

    // A project-specific host is useful evidence for an exact diagnostic, but
    // it is not a request to search only that project's docs. Preserve the
    // focused general supplement so search engines and the auxiliary GitHub /
    // Stack Exchange pools can still find the concrete failure and its fixes.
    // An explicit first-party request is handled by the strict-source branch
    // above.
    if has_quoted_diagnostic(query)
        && source.as_ref().is_some_and(|source| !source.strict)
        && let Some(specialist) = route
            .supplemental
            .as_ref()
            .filter(|specialist| has_category(&specialist.categories, "general"))
            .cloned()
    {
        return vec![SupplementalQuery::Specialist(specialist)];
    }

    if has_numerical_parity_intent(query)
        && has_machine_learning_inference_intent(query)
        && !has_scholarly_intent(query)
        && let Some(specialist) = route
            .supplemental
            .as_ref()
            .filter(|specialist| has_category(&specialist.categories, "science"))
            .cloned()
    {
        // One title-oriented query retrieves work on bit-level reproducible
        // inference; the complementary formulation retrieves empirical
        // studies of hardware-induced numerical deviations. These are
        // disjoint evidence forms in scholarly APIs, and neither formulation
        // reliably returns the other.
        return vec![
            SupplementalQuery::Specialist(specialist),
            SupplementalQuery::Specialist(SpecialistQuery {
                query: "numerical deviations neural network inference reproducibility".into(),
                categories: "science".into(),
            }),
        ];
    }

    if let Some(query) = focused_recommendation_query(query) {
        let mut supplemental = vec![SupplementalQuery::Focused(SpecialistQuery {
            query,
            categories: source_categories(&route.categories, configured_categories).into(),
        })];
        supplemental.extend(manifestation.map(SupplementalQuery::Manifestation));
        return supplemental;
    }

    if is_as_of_query(query)
        && let Some(specialist) = route
            .supplemental
            .as_ref()
            .filter(|specialist| has_category(&specialist.categories, "science"))
            .cloned()
    {
        return vec![SupplementalQuery::Specialist(specialist)];
    }

    manifestation
        .map(SupplementalQuery::Manifestation)
        .or_else(|| full_text_locator_query(query).map(SupplementalQuery::FullText))
        .or_else(|| source.map(SupplementalQuery::Source))
        .or_else(|| {
            has_category(configured_categories, "map")
                .then(|| local_map_query(query, &route.query))
                .flatten()
                .map(SupplementalQuery::Local)
        })
        .or_else(|| {
            route
                .supplemental
                .clone()
                .map(SupplementalQuery::Specialist)
        })
        .into_iter()
        .collect()
}

fn route(query: &str, configured_categories: &str) -> Route {
    let categories = configured_categories
        .split(',')
        .map(str::trim)
        .filter(|category| !category.is_empty())
        .collect::<Vec<_>>();
    let has = |category: &str| categories.contains(&category);

    let normalized_query = normalize_intent_modifiers(query);
    let paper_title = paper_title(query);
    let news_intent = has_news_intent(query);
    let it_intent = has_it_intent(query);
    let science_intent = paper_title.is_some() || has_science_intent(query);
    let prefer_science = science_intent && (!it_intent || has_scholarly_intent(query));
    let routed_category = if has("general") {
        // General web search is the independent recall baseline. Specialist
        // categories are queried separately so one brittle vertical cannot
        // remove the broad pool.
        "general"
    } else if has("news") && news_intent {
        "news"
    } else if has("science") && prefer_science {
        "science"
    } else if has("it") && it_intent {
        "it"
    } else if has("science") && science_intent {
        "science"
    } else if has("general") {
        "general"
    } else {
        categories.first().copied().unwrap_or_default()
    };
    let primary_query = paper_title
        .as_ref()
        .map(|title| format!("\"{title}\""))
        .unwrap_or_else(|| query.to_owned());
    let supplemental = if routed_category != "news" && has("news") && news_intent {
        Some(SpecialistQuery {
            query: normalize_news_query(query),
            // The broad primary already searched general engines. Keep the
            // one optional freshness lookup disjoint so it adds evidence
            // without charging the same fragile scrapers twice.
            categories: "news".into(),
        })
    } else if routed_category != "science" && has("science") && prefer_science {
        Some(SpecialistQuery {
            query: primary_query.clone(),
            categories: "science".into(),
        })
    } else if routed_category != "it" && has("it") && it_intent {
        let exact_query = exact_identifier_query(query);
        let technical_evidence = technical_evidence_kind(query);
        let identifier_focus = exact_query.is_some()
            && quoted_phrase(query).is_none()
            && compiler_error_code(query).is_none();
        Some(SpecialistQuery {
            query: exact_query
                .or_else(|| focused_technical_query(query))
                .unwrap_or(normalized_query),
            // A structural identifier can use the disjoint exact-token
            // vertical. Error diagnostics still use a focused general lookup:
            // package registries do not contain their answer passages.
            categories: if identifier_focus {
                "it".into()
            } else if technical_evidence == Some(TechnicalEvidence::NumericalParity)
                && has("science")
            {
                "science".into()
            } else if has("general") {
                "general".into()
            } else {
                "it".into()
            },
        })
    } else if routed_category != "science" && has("science") && science_intent {
        Some(SpecialistQuery {
            query: primary_query.clone(),
            categories: "science".into(),
        })
    } else {
        None
    };
    Route {
        supplemental,
        query: primary_query,
        categories: routed_category.to_owned(),
    }
}

/// Search engines often bury the actual recommendation chapter beneath
/// summaries and commentary when every intent word is sent verbatim. For an
/// explicit year + issuing-body query, keep that identifying phrase together
/// while retaining the subject terms for one precision-engine supplement.
fn focused_recommendation_query(query: &str) -> Option<String> {
    let tokens = query
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let year = tokens.iter().find(|token| {
        token.len() == 4
            && token
                .parse::<u16>()
                .is_ok_and(|year| (1900..=2200).contains(&year))
    })?;
    let authority = tokens
        .iter()
        .find_map(|token| super::evidence::recommendation_authority(token).map(|value| value.0))?;
    let document_kind = tokens.iter().find_map(|token| {
        if token.eq_ignore_ascii_case("standards") {
            Some("Standards")
        } else if token.eq_ignore_ascii_case("guideline")
            || token.eq_ignore_ascii_case("guidelines")
        {
            Some("Guidelines")
        } else if token.eq_ignore_ascii_case("recommendation")
            || token.eq_ignore_ascii_case("recommendations")
        {
            Some("Recommendations")
        } else {
            None
        }
    })?;
    let subject_terms = tokens
        .iter()
        .filter(|token| *token != year && !token.eq_ignore_ascii_case(authority))
        .filter(|token| {
            !matches!(
                token.to_ascii_lowercase().as_str(),
                "and"
                    | "adult"
                    | "adults"
                    | "current"
                    | "data"
                    | "for"
                    | "guideline"
                    | "guidelines"
                    | "official"
                    | "recommendation"
                    | "recommendations"
                    | "standards"
                    | "the"
            )
        })
        .take(2)
        .copied()
        .collect::<Vec<_>>();
    if subject_terms.is_empty() {
        return None;
    }

    Some(format!(
        "\"{year} {authority} {document_kind}\" {}",
        subject_terms.join(" ")
    ))
}

/// Produce a short, query-diverse technical lookup without weakening an exact
/// diagnostic into a bag of generic error words. This is deliberately used as
/// the one technical supplement rather than as another request.
fn focused_technical_query(query: &str) -> Option<String> {
    if let Some(phrase) = quoted_phrase(query) {
        if phrase.to_ascii_lowercase().contains("fromresidual")
            && query
                .split(|character: char| !character.is_alphanumeric())
                .any(|term| term.eq_ignore_ascii_case("rust"))
        {
            return Some(
                "Rust question mark operator custom error enum impl From source error".into(),
            );
        }
        let quoted = format!("\"{phrase}\"");
        let remaining = query.replacen(&quoted, " ", 1);
        let context = remaining
            .split_whitespace()
            .filter_map(clean_technical_term)
            .filter(|term| !is_technical_query_filler(term))
            .take(8)
            .collect::<Vec<_>>();
        return Some(if context.is_empty() {
            quoted
        } else {
            format!("{quoted} {}", context.join(" "))
        });
    }

    if let Some(evidence) = technical_evidence_kind(query) {
        return Some(match evidence {
            TechnicalEvidence::CaseStudy => focused_case_study_query(query),
            TechnicalEvidence::NumericalParity => focused_numerical_parity_query(query),
        });
    }

    let terms = query
        .split_whitespace()
        .filter_map(clean_technical_term)
        .filter(|term| !is_technical_query_filler(term))
        .take(12)
        .collect::<Vec<_>>();
    (terms.len() >= 3).then(|| terms.join(" "))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TechnicalEvidence {
    CaseStudy,
    NumericalParity,
}

fn technical_evidence_kind(query: &str) -> Option<TechnicalEvidence> {
    if has_case_study_intent(query) {
        return Some(TechnicalEvidence::CaseStudy);
    }
    if has_numerical_parity_intent(query) {
        return Some(TechnicalEvidence::NumericalParity);
    }

    None
}

fn focused_case_study_query(query: &str) -> String {
    let mut anchors = query
        .split_whitespace()
        .enumerate()
        .filter_map(|(index, word)| clean_technical_term(word).map(|term| (index, term)))
        .filter(|(index, term)| {
            is_strong_it_brand(&term.to_ascii_lowercase())
                || is_ambiguous_it_brand(&term.to_ascii_lowercase())
                || *index > 0 && term.chars().next().is_some_and(char::is_uppercase)
        })
        .map(|(_, term)| term)
        .take(4)
        .collect::<Vec<_>>();
    deduplicate_case_insensitive(&mut anchors);
    if anchors.is_empty() {
        anchors.extend(
            query
                .split_whitespace()
                .filter_map(clean_technical_term)
                .filter(|term| !is_technical_query_filler(term))
                .take(3),
        );
    }
    let contexts = query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .collect::<HashSet<_>>();
    let context = if contexts.contains("monorepo") || contexts.contains("monorepos") {
        Some("monorepo")
    } else if contexts.contains("build") || contexts.contains("builds") {
        Some("build")
    } else if contexts.contains("ci") {
        Some("CI")
    } else {
        None
    };
    let subjects = if anchors.len() == 1 {
        anchors[0].to_owned()
    } else {
        format!("({})", anchors.join(" OR "))
    };
    let context = context
        .filter(|context| {
            !anchors
                .iter()
                .any(|term| term.eq_ignore_ascii_case(context))
        })
        .map_or(String::new(), |context| format!(" {context}"));
    format!("{subjects}{context} (migration OR \"case study\" OR postmortem)")
}

fn focused_numerical_parity_query(query: &str) -> String {
    let query_terms = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<HashSet<_>>();
    let machine_learning = query_terms.contains("inference")
        || query_terms.contains("model")
        || query_terms.contains("models")
        || query_terms.contains("neural")
        || query_terms.contains("llm")
        || query_terms.contains("machine") && query_terms.contains("learning");
    let bitwise = query_terms.contains("bitwise") || query_terms.contains("bit");
    if machine_learning && bitwise {
        // Academic APIs rank title matches aggressively. This compact,
        // corpus-native formulation finds work on bit-level reproducible
        // training/inference; appending every concrete backend term instead
        // turns the lookup into an unrelated hardware-optimization search.
        return "bitwise reproducible deep learning inference".into();
    }

    if machine_learning {
        return "numerical deviations neural network inference reproducibility".into();
    }

    let mut terms = query_terms
        .iter()
        .filter(|term| {
            matches!(
                term.as_str(),
                "cpu" | "cuda" | "gpu" | "metal" | "mps" | "opencl" | "rocm" | "tpu"
            )
        })
        .map(String::as_str)
        .collect::<Vec<_>>();
    terms.sort_unstable();
    terms.extend(["numerical", "reproducibility", "testing"]);
    terms.join(" ")
}

fn deduplicate_case_insensitive(terms: &mut Vec<&str>) {
    let mut seen = HashSet::new();
    terms.retain(|term| seen.insert(term.to_ascii_lowercase()));
}

fn quoted_phrase(query: &str) -> Option<&str> {
    let start = query.find('"')? + 1;
    let end = query[start..].find('"')? + start;
    let phrase = query[start..end].trim();
    (phrase.chars().count() >= 4).then_some(phrase)
}

fn clean_technical_term(word: &str) -> Option<&str> {
    let term = word.trim_matches(|character: char| {
        !(character.is_alphanumeric() || matches!(character, '+' | '#' | '.' | ':' | '_' | '-'))
    });
    (!term.is_empty()).then_some(term)
}

fn is_technical_query_filler(term: &str) -> bool {
    matches!(
        term.to_ascii_lowercase().as_str(),
        "a" | "after"
            | "an"
            | "and"
            | "are"
            | "because"
            | "by"
            | "caused"
            | "do"
            | "does"
            | "for"
            | "from"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "porting"
            | "the"
            | "to"
            | "using"
            | "was"
            | "what"
            | "when"
            | "where"
            | "which"
            | "with"
    )
}

pub(super) fn has_it_intent(query: &str) -> bool {
    if exact_identifier_query(query).is_some() {
        return true;
    }
    let lower = query.to_ascii_lowercase();
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let strong_brand = tokens.iter().any(|token| is_strong_it_brand(token))
        || lower.contains("next.js")
        || lower.contains("c++")
        || lower.contains("c#")
        || lower.contains(".net");
    let ambiguous_brand = tokens.iter().any(|token| is_ambiguous_it_brand(token));
    let technical_context = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "api"
                | "async"
                | "await"
                | "borrow"
                | "class"
                | "cli"
                | "code"
                | "coding"
                | "config"
                | "configuration"
                | "context"
                | "coroutine"
                | "crate"
                | "database"
                | "dependency"
                | "deploy"
                | "docs"
                | "documentation"
                | "endpoint"
                | "enum"
                | "error"
                | "exception"
                | "framework"
                | "function"
                | "goroutine"
                | "hashmap"
                | "interface"
                | "library"
                | "method"
                | "middleware"
                | "module"
                | "package"
                | "pathlib"
                | "pointer"
                | "programming"
                | "request"
                | "response"
                | "runtime"
                | "sdk"
                | "serialization"
                | "server"
                | "sql"
                | "struct"
                | "test"
                | "trait"
                | "typeerror"
                | "useeffect"
        )
    });
    let code_shape = query.contains("::")
        || query.contains('_')
        || query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| {
                token
                    .chars()
                    .any(|character| character.is_ascii_lowercase())
                    && token
                        .chars()
                        .skip(1)
                        .any(|character| character.is_ascii_uppercase())
            });
    strong_brand || ambiguous_brand && (technical_context || code_shape)
}

fn is_strong_it_brand(token: &str) -> bool {
    matches!(
        token,
        "android"
            | "api"
            | "aspnet"
            | "aws"
            | "axum"
            | "azure"
            | "bazel"
            | "compiler"
            | "crdt"
            | "csharp"
            | "cuda"
            | "dart"
            | "database"
            | "docker"
            | "dotnet"
            | "ec2"
            | "flutter"
            | "gcp"
            | "github"
            | "golang"
            | "ios"
            | "javascript"
            | "kotlin"
            | "kubernetes"
            | "lambda"
            | "laravel"
            | "linux"
            | "mysql"
            | "nix"
            | "nixos"
            | "nodejs"
            | "pants"
            | "php"
            | "postgres"
            | "postgresql"
            | "redis"
            | "s3"
            | "scala"
            | "sdk"
            | "serde"
            | "sql"
            | "sqlite"
            | "terraform"
            | "tokio"
            | "typescript"
            | "valkey"
    )
}

fn is_ambiguous_it_brand(token: &str) -> bool {
    matches!(
        token,
        "go" | "java"
            | "node"
            | "python"
            | "r"
            | "rails"
            | "react"
            | "ruby"
            | "rust"
            | "spring"
            | "swift"
    )
}

fn has_science_intent(query: &str) -> bool {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let explicit_scholarly = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "arxiv" | "clinical" | "doi" | "journal" | "preprint" | "pubmed" | "trial" | "trials"
        )
    }) || tokens.windows(2).any(|tokens| {
        matches!(
            tokens,
            [left, right]
                if (left == "meta" && right == "analysis")
                    || (left == "systematic" && right == "review")
                    || (left == "randomized" && right == "trial")
        )
    });
    let distinctive_subject = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "astrophysics"
                | "cas9"
                | "crispr"
                | "epidemiology"
                | "exoplanet"
                | "genome"
                | "genomic"
                | "genomics"
                | "jwst"
                | "neuroscience"
                | "polymerase"
                | "proteomics"
                | "quantum"
                | "spectroscopy"
                | "stoichiometry"
                | "superconductor"
        )
    });
    let broad_subject = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "astronomy"
                | "biology"
                | "cancer"
                | "cardiology"
                | "chemistry"
                | "climate"
                | "diabetes"
                | "ecology"
                | "genetics"
                | "geology"
                | "hypertension"
                | "materials"
                | "medicine"
                | "medical"
                | "oncology"
                | "pharmacology"
                | "physics"
                | "protein"
        )
    });
    let research_cue = has_scholarly_intent(query)
        || tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "evidence"
                    | "experiment"
                    | "guideline"
                    | "guidelines"
                    | "mechanism"
                    | "recommendation"
                    | "recommendations"
                    | "standards"
            )
        });

    explicit_scholarly || distinctive_subject || broad_subject && research_cue
}

fn has_scholarly_intent(query: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "arxiv"
                    | "doi"
                    | "journal"
                    | "paper"
                    | "papers"
                    | "preprint"
                    | "pubmed"
                    | "research"
                    | "study"
                    | "studies"
            )
        })
}

fn has_category(configured_categories: &str, expected: &str) -> bool {
    configured_categories
        .split(',')
        .map(str::trim)
        .any(|category| category == expected)
}

fn local_map_query(original_query: &str, search_query: &str) -> Option<String> {
    let lower = original_query.to_ascii_lowercase();
    let local_intent =
        lower.contains(" near ") || lower.contains(" nearby") || lower.contains(" closest ");
    let unsuitable_for_map = lower
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| matches!(token, "hike" | "hikes" | "hiking" | "trail" | "trails"));
    if !local_intent || unsuitable_for_map {
        return None;
    }

    let words = search_query.split_whitespace().collect::<Vec<_>>();
    let separator = words
        .iter()
        .position(|word| matches!(local_word(word).as_str(), "near" | "nearby" | "closest"));
    let (subject, location) = separator.map_or((&words[..], &[][..]), |index| {
        (&words[..index], &words[index + 1..])
    });
    let subject = subject
        .iter()
        .filter(|word| !is_local_modifier(word))
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev();
    let location = location
        .iter()
        .take_while(|word| !is_local_constraint(word))
        .take(4);
    let query = subject
        .chain(location)
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    (!query.is_empty()).then_some(query)
}

fn local_word(word: &str) -> String {
    word.trim_matches(|character: char| !character.is_alphanumeric())
        .to_ascii_lowercase()
}

fn is_local_modifier(word: &&str) -> bool {
    matches!(
        local_word(word).as_str(),
        "best"
            | "closest"
            | "coworking"
            | "late"
            | "near"
            | "nearby"
            | "now"
            | "open"
            | "quiet"
            | "same-day"
    )
}

fn is_local_constraint(word: &&str) -> bool {
    matches!(
        local_word(word).as_str(),
        "after"
            | "before"
            | "diagnostic"
            | "fee"
            | "linux-friendly"
            | "open"
            | "outlets"
            | "recent"
            | "reviews"
            | "weekday"
            | "weekdays"
            | "wifi"
            | "wi-fi"
            | "with"
    )
}

fn has_news_intent(query: &str) -> bool {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let explicit_news = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "news" | "headline" | "headlines" | "breaking"
        )
    });

    let asks_for_freshness = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "current" | "latest" | "newest" | "recent" | "recently" | "today"
        )
    });
    if explicit_news {
        // `news` is also part of product names such as Hacker News. In an API
        // or other strong software query, treat it as that product unless the
        // user also asks for fresh news.
        return asks_for_freshness || !has_it_intent(query);
    }
    let asks_about_an_event = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "advisory"
                | "announcement"
                | "breach"
                | "incident"
                | "launch"
                | "launched"
                | "mission"
                | "outage"
                | "release"
                | "security"
                | "status"
                | "update"
                | "vulnerability"
        )
    });
    asks_for_freshness && asks_about_an_event
}

fn freshness_window(query: &str) -> Option<&'static str> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "current" | "latest" | "newest" | "recent" | "recently" | "today"
            )
        })
        .then_some("year")
}

fn normalize_intent_modifiers(query: &str) -> String {
    let normalized = query
        .split_whitespace()
        .filter(|word| {
            let token = word.trim_matches(|character: char| !character.is_alphanumeric());
            !matches!(
                token.to_ascii_lowercase().as_str(),
                "best"
                    | "a"
                    | "an"
                    | "are"
                    | "can"
                    | "could"
                    | "current"
                    | "currently"
                    | "did"
                    | "do"
                    | "documentation"
                    | "does"
                    | "how"
                    | "i"
                    | "is"
                    | "latest"
                    | "me"
                    | "my"
                    | "newest"
                    | "official"
                    | "recent"
                    | "recently"
                    | "should"
                    | "the"
                    | "today"
                    | "traditional"
                    | "was"
                    | "were"
                    | "what"
                    | "when"
                    | "where"
                    | "which"
                    | "why"
                    | "would"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    if normalized.is_empty() {
        query.to_owned()
    } else {
        normalized
    }
}

fn normalize_news_query(query: &str) -> String {
    let words = query.split_whitespace().collect::<Vec<_>>();
    let mut normalized = Vec::with_capacity(words.len());
    let mut index = 0;

    while index < words.len() {
        let token = words[index].trim_matches(|character: char| !character.is_alphanumeric());
        let lower = token.to_ascii_lowercase();
        if is_month(&lower) {
            let next_is_date = words
                .get(index + 1)
                .is_some_and(|word| is_day(word) || is_year(word));
            let introduced_as_date = index > 0
                && matches!(
                    words[index - 1]
                        .trim_matches(|character: char| !character.is_alphanumeric())
                        .to_ascii_lowercase()
                        .as_str(),
                    "as" | "during" | "from" | "in" | "on" | "since"
                );
            if next_is_date || introduced_as_date {
                index += 1;
                if index < words.len() && is_day(words[index]) {
                    index += 1;
                }
                if index < words.len() && is_year(words[index]) {
                    index += 1;
                }
                continue;
            }
        }
        if matches!(
            lower.as_str(),
            "news"
                | "headline"
                | "headlines"
                | "today"
                | "yesterday"
                | "breaking"
                | "latest"
                | "recent"
                | "recently"
                | "from"
                | "about"
        ) {
            index += 1;
            continue;
        }
        normalized.push(words[index]);
        index += 1;
    }

    let normalized = normalized.join(" ");
    if normalized.trim().is_empty() {
        query.to_owned()
    } else {
        normalized
    }
}

fn full_text_locator_query(query: &str) -> Option<String> {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if !has_full_text_intent(query) {
        return None;
    }

    let (locator_index, locator) = tokens.iter().enumerate().find_map(|(index, token)| {
        let locator = if token.eq_ignore_ascii_case("article") || token.eq_ignore_ascii_case("art")
        {
            "Article"
        } else if token.eq_ignore_ascii_case("section") {
            "Section"
        } else if token.eq_ignore_ascii_case("clause") {
            "Clause"
        } else if token.eq_ignore_ascii_case("chapter") {
            "Chapter"
        } else {
            return None;
        };
        Some((index, locator))
    })?;
    let identifier = *tokens.get(locator_index + 1)?;
    if identifier.len() > 12 {
        return None;
    }

    let common_locator =
        locator_index > 0 && tokens[locator_index - 1].eq_ignore_ascii_case("common");
    let useful_instrument_term = |token: &str| {
        !matches!(
            token.to_ascii_lowercase().as_str(),
            "full" | "official" | "text"
        )
    };
    let prefix_end = locator_index.saturating_sub(usize::from(common_locator));
    let mut instrument = tokens[..prefix_end]
        .iter()
        .copied()
        .filter(|token| useful_instrument_term(token))
        .collect::<Vec<_>>();
    if instrument
        .iter()
        .filter(|token| !matches!(token.to_ascii_lowercase().as_str(), "of" | "the"))
        .count()
        < 2
    {
        instrument = tokens[locator_index + 2..]
            .iter()
            .copied()
            .filter(|token| useful_instrument_term(token))
            .collect();
    }
    if instrument
        .iter()
        .filter(|token| !matches!(token.to_ascii_lowercase().as_str(), "of" | "the"))
        .count()
        < 2
    {
        return None;
    }

    let locator = if common_locator {
        format!("Common {locator} {identifier}")
    } else {
        format!("{locator} {identifier}")
    };
    Some(format!("\"{}\" \"{locator}\"", instrument.join(" ")))
}

fn source_aware_query(
    original_query: &str,
    search_query: &str,
    results: &[SearxResult],
) -> Option<SourceQuery> {
    let lower = original_query.to_ascii_lowercase();
    let strict = has_strict_source_intent(&lower);
    // The quoted general query used by `route` already returns repository and
    // publisher manifestations. A second site-scoped query narrows that useful
    // pool back to one host and spends metasearch quota for worse recall.
    if paper_title(original_query).is_some() {
        return None;
    }
    if let Some(host) = named_authority_host(original_query) {
        let terms = if has_it_intent(original_query) {
            focused_technical_query(search_query)
                .unwrap_or_else(|| focused_source_terms(search_query))
        } else {
            focused_source_terms(search_query)
        };
        let query = format!("{terms} site:{host}");
        return Some(SourceQuery {
            query,
            host: Some(host.into()),
            strict,
        });
    }
    if lower.contains("primary source") {
        let subject = original_query
            .split_whitespace()
            .filter(|word| {
                !matches!(
                    word.trim_matches(|character: char| !character.is_alphanumeric())
                        .to_ascii_lowercase()
                        .as_str(),
                    "primary" | "source" | "sources"
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        return (!subject.is_empty()).then(|| SourceQuery {
            query: format!("{subject} archival document transcript"),
            host: None,
            strict: true,
        });
    }
    if let Some(code) = compiler_error_code(original_query) {
        let compact_error_lookup = original_query
            .split_whitespace()
            .filter(|token| {
                !matches!(
                    token
                        .trim_matches(|character: char| !character.is_alphanumeric())
                        .to_ascii_lowercase()
                        .as_str(),
                    "error" | "rust"
                )
            })
            .count()
            <= 2;
        let documentation_intent = has_documentation_intent(&lower);
        if !compact_error_lookup && !documentation_intent {
            return None;
        }
        if query_contains_token(original_query, "rust") && code.starts_with('E') {
            return Some(SourceQuery {
                query: format!("{code} site:doc.rust-lang.org"),
                host: Some("doc.rust-lang.org".into()),
                strict,
            });
        }
        if let Some(host) = infer_official_host(search_query, results) {
            let search_host = registrable_host(&host);
            return Some(SourceQuery {
                query: format!("{code} site:{search_host}"),
                host: Some(host),
                strict,
            });
        }
    }
    if !has_source_intent(&lower) {
        return None;
    }

    if let Some(host) = infer_official_host(search_query, results) {
        let search_host = registrable_host(&host);
        return Some(SourceQuery {
            query: format!("{} site:{search_host}", focused_source_terms(search_query)),
            host: Some(host),
            strict,
        });
    }

    has_documentation_intent(&lower).then(|| SourceQuery {
        query: original_query
            .split_whitespace()
            .map(|word| {
                if word.eq_ignore_ascii_case("documentation") {
                    "docs"
                } else {
                    word
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        host: None,
        strict,
    })
}

fn scholarly_manifestation_query(
    query: &str,
    results: &[SearxResult],
    configured_categories: &str,
) -> Option<ManifestationQuery> {
    if !has_category(configured_categories, "science")
        || !has_science_intent(query)
        || !has_scholarly_manifestation_intent(query)
    {
        return None;
    }

    let result = results.iter().find(|result| {
        result.title.as_deref().is_some_and(|title| {
            title.split_whitespace().count() >= 4 && !is_generic_scholarly_title(query, title)
        }) && is_scholarly_article_url(&result.url)
    })?;
    let mut title = result
        .title
        .as_deref()?
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| {
                !(character.is_alphanumeric() || matches!(character, '-' | '—' | '–'))
            })
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    for year in query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| {
            token.len() == 4
                && token
                    .parse::<u16>()
                    .is_ok_and(|year| (1900..=2200).contains(&year))
        })
    {
        if !query_contains_token(&title, year) {
            title.push(' ');
            title.push_str(year);
        }
    }
    (!title.is_empty()).then_some(ManifestationQuery {
        query: title,
        engines: "openalex,semantic scholar".into(),
    })
}

fn is_generic_scholarly_title(query: &str, title: &str) -> bool {
    ["editorial", "introduction", "overview", "summary"]
        .into_iter()
        .any(|term| query_contains_token(title, term) && !query_contains_token(query, term))
}

fn has_scholarly_manifestation_intent(query: &str) -> bool {
    paper_title(query).is_some()
        || focused_recommendation_query(query).is_some()
        || query_contains_token(query, "doi")
        || query
            .split_whitespace()
            .map(|token| {
                token.trim_matches(|character: char| {
                    !character.is_alphanumeric() && character != '.'
                })
            })
            .any(is_arxiv_identifier)
}

fn is_arxiv_identifier(token: &str) -> bool {
    let mut parts = token.split('.');
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(year_month), Some(number), None)
            if year_month.len() == 4
                && number.len() >= 4
                && year_month.chars().all(|character| character.is_ascii_digit())
                && number.chars().all(|character| character.is_ascii_digit())
    )
}

fn is_scholarly_article_url(value: &str) -> bool {
    let Some(url) = parse_web_url(value) else {
        return false;
    };
    let host = url.host_str().unwrap_or_default();
    let path = url.path().to_ascii_lowercase();
    host == "doi.org"
        || host == "arxiv.org"
        || host.ends_with(".arxiv.org")
        || host == "aclanthology.org"
        || host.ends_with(".aclanthology.org")
        || host == "pmc.ncbi.nlm.nih.gov"
        || host == "pubmed.ncbi.nlm.nih.gov"
        || path.contains("/article/")
        || path.contains("/doi/")
        || path.contains("/paper/")
}

fn focused_source_terms(query: &str) -> String {
    let normalized = normalize_intent_modifiers(query);
    let query_terms = query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .collect::<HashSet<_>>();
    let original_disclosure =
        query_terms.contains("original") && query_terms.contains("disclosure");
    normalized
        .split_whitespace()
        .filter(|word| {
            let token = word
                .trim_matches(|character: char| !character.is_alphanumeric())
                .to_ascii_lowercase();
            !matches!(
                token.as_str(),
                "disclosure"
                    | "docs"
                    | "documentation"
                    | "official"
                    | "original"
                    | "primary"
                    | "source"
                    | "sources"
            ) && (!original_disclosure || !is_cve_identifier(&token))
        })
        .take(8)
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_cve_identifier(token: &str) -> bool {
    let mut parts = token.split('-');
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some("cve"), Some(year), Some(number), None)
            if year.len() == 4
                && year.chars().all(|character| character.is_ascii_digit())
                && number.len() >= 4
                && number.chars().all(|character| character.is_ascii_digit())
    )
}

pub(super) fn has_strict_source_intent(lower_query: &str) -> bool {
    let original_document = query_contains_token(lower_query, "original")
        && (query_contains_token(lower_query, "disclosure")
            || query_contains_token(lower_query, "paper"));

    query_contains_token(lower_query, "arxiv")
        || has_full_text_intent(lower_query)
        || original_document
        || lower_query.contains("primary source")
        || lower_query.contains("release notes")
        || query_contains_token(lower_query, "official")
        || lower_query.contains("first-party")
        || lower_query.contains("first party")
        || lower_query.contains("upstream documentation")
        || lower_query.contains("upstream docs")
        || lower_query.contains("vendor documentation")
        || lower_query.contains("vendor docs")
}

pub(super) fn has_quoted_diagnostic(query: &str) -> bool {
    quoted_phrase(query).is_some()
        && (compiler_error_code(query).is_some()
            || query
                .split(|character: char| !character.is_alphanumeric())
                .any(|token| {
                    matches!(
                        token.to_ascii_lowercase().as_str(),
                        "error"
                            | "exception"
                            | "failed"
                            | "failure"
                            | "invalid"
                            | "not"
                            | "unable"
                            | "unauthorized"
                    )
                }))
}

pub(super) fn named_authority_host(query: &str) -> Option<&'static str> {
    let lower_query = query.to_ascii_lowercase();
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();

    if lower_query.contains("next.js") {
        return Some("nextjs.org");
    }
    if tokens.iter().any(|token| token == "pip")
        && lower_query.contains("externally-managed-environment")
    {
        return Some("packaging.python.org");
    }

    // An exact diagnostic plus an unambiguous project name identifies a useful
    // first-party fallback. The planner preserves the focused general lookup
    // unless the query explicitly asks for that source, so this host mapping
    // does not displace community fixes. Keeping the check behind an exact
    // diagnostic also avoids affecting casual branded queries such as the Rust
    // game or the Amazon rainforest.
    if has_quoted_diagnostic(query) {
        if tokens
            .iter()
            .any(|token| matches!(token.as_str(), "postgres" | "postgresql"))
        {
            return Some("postgresql.org");
        }
        if tokens.iter().any(|token| token == "rust") {
            return Some("doc.rust-lang.org");
        }
        if tokens.iter().any(|token| token == "aws") {
            return Some("docs.aws.amazon.com");
        }
        if tokens.iter().any(|token| token == "kubernetes") {
            return Some("kubernetes.io");
        }
        if tokens.iter().any(|token| token == "zod") {
            return Some("zod.dev");
        }
    }

    if tokens.iter().any(|token| token == "whatwg") {
        return Some(if tokens.iter().any(|token| token == "dom") {
            "dom.spec.whatwg.org"
        } else {
            "whatwg.org"
        });
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["oss", "security"])
    {
        return Some("openwall.com");
    }
    if tokens.iter().any(|token| token == "nist") {
        return Some("nist.gov");
    }
    if tokens.iter().any(|token| token == "cdc") {
        return Some("cdc.gov");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["copyright", "office"])
    {
        return Some("copyright.gov");
    }
    if tokens.iter().any(|token| token == "webauthn")
        && tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "official" | "spec" | "specification" | "standard"
            )
        })
    {
        return Some("w3.org");
    }
    if tokens.iter().any(|token| token == "node")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "lts" | "release" | "releases" | "schedule"))
    {
        return Some("nodejs.org");
    }
    if tokens.iter().any(|token| token == "rust")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "release" | "releases"))
        && tokens.iter().any(|token| token == "notes")
    {
        return Some("doc.rust-lang.org");
    }
    if tokens.iter().any(|token| token == "npm")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "incident" | "incidents" | "status"))
    {
        return Some("status.npmjs.org");
    }
    if tokens.iter().any(|token| token == "github")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "docs" | "documentation" | "official"))
    {
        return Some("docs.github.com");
    }
    if tokens.iter().any(|token| token == "aws")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "docs" | "documentation" | "official"))
    {
        return Some("docs.aws.amazon.com");
    }
    if tokens.iter().any(|token| token == "nvidia")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "advisory" | "security" | "vulnerability"))
    {
        return Some("nvidia.com");
    }

    if tokens
        .windows(3)
        .any(|tokens| tokens == ["federal", "funds", "rate"])
    {
        return Some("federalreserve.gov");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["geneva", "convention"] || tokens == ["geneva", "conventions"])
    {
        return Some("icrc.org");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["mortgage", "rate"] || tokens == ["mortgage", "rates"])
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "data" | "official"))
    {
        return Some("freddiemac.com");
    }
    if tokens.iter().any(|token| token == "cpi")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "data" | "official"))
    {
        return Some("bls.gov");
    }
    if tokens
        .windows(2)
        .any(|tokens| tokens == ["united", "nations"])
    {
        return Some("un.org");
    }
    if tokens.iter().any(|token| token == "acc")
        && tokens.iter().any(|token| token == "aha")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "guideline" | "guidelines"))
    {
        return Some("jacc.org");
    }
    if tokens.iter().any(|token| token == "ada") && tokens.iter().any(|token| token == "diabetes") {
        return Some("diabetesjournals.org");
    }
    if tokens.iter().any(|token| token == "postgresql")
        && tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "docs" | "documentation" | "official" | "release" | "releases"
            )
        })
    {
        return Some("postgresql.org");
    }
    if tokens.iter().any(|token| token == "sqlite")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "docs" | "documentation" | "official"))
    {
        return Some("sqlite.org");
    }
    if tokens.iter().any(|token| token == "python")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "docs" | "documentation" | "official"))
    {
        return Some("docs.python.org");
    }
    if tokens.iter().any(|token| token == "react")
        && tokens
            .iter()
            .any(|token| matches!(token.as_str(), "docs" | "documentation" | "official"))
    {
        return Some("react.dev");
    }
    if tokens.iter().any(|token| token == "rfc")
        && tokens
            .iter()
            .any(|token| token.chars().all(|character| character.is_ascii_digit()))
    {
        return Some("rfc-editor.org");
    }
    for token in &tokens {
        if token == "fda" {
            return Some("fda.gov");
        }
        if token == "nasa" {
            return Some("nasa.gov");
        }
        if token == "nixos" {
            return Some("nixos.org");
        }
        if token == "tokio" {
            return Some("docs.rs");
        }
        if token == "arxiv" {
            return Some("arxiv.org");
        }
    }
    let says_world_health_organization = tokens.windows(3).any(|tokens| {
        tokens == ["world", "health", "organization"]
            || tokens == ["world", "health", "organisation"]
    });
    if says_world_health_organization
        || query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| token == "WHO")
    {
        return Some("who.int");
    }
    None
}

fn query_contains_token(query: &str, expected: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case(expected))
}

pub(super) fn paper_title(query: &str) -> Option<String> {
    let lower = query.to_ascii_lowercase();
    let arxiv_known_item = query_contains_token(query, "arxiv")
        && query_contains_token(query, "paper")
        && !query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| {
                matches!(
                    token.to_ascii_lowercase().as_str(),
                    "about" | "latest" | "new" | "newest" | "papers" | "recent"
                )
            });
    if !(lower.contains("original paper") || arxiv_known_item) {
        return None;
    }
    let title = query
        .split_whitespace()
        .filter(|word| {
            !matches!(
                word.trim_matches(|character: char| !character.is_alphanumeric())
                    .to_ascii_lowercase()
                    .as_str(),
                "arxiv" | "original" | "paper" | "papers" | "pdf"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    (title.split_whitespace().count() >= 3).then_some(title)
}

fn source_categories<'a>(default: &'a str, configured: &'a str) -> &'a str {
    if has_category(configured, "general") {
        "general"
    } else {
        default
    }
}

fn compiler_error_code(query: &str) -> Option<&str> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .find(|token| {
            let letters = token
                .chars()
                .take_while(|character| character.is_ascii_alphabetic())
                .count();
            let digits = token.chars().count().saturating_sub(letters);
            (1..=4).contains(&letters)
                && (3..=5).contains(&digits)
                && token
                    .chars()
                    .skip(letters)
                    .all(|character| character.is_ascii_digit())
        })
}

fn merge_source_results(
    primary: &mut SearxResponse,
    supplemental: SearxResponse,
    host: Option<&str>,
    strict: bool,
) {
    match host {
        Some(host) => {
            let (mut focused, _ignored_filter): (Vec<_>, Vec<_>) = supplemental
                .results
                .into_iter()
                .partition(|result| result_matches_host(result, host));
            let (mut matching_primary, remaining_primary): (Vec<_>, Vec<_>) =
                std::mem::take(&mut primary.results)
                    .into_iter()
                    .partition(|result| result_matches_host(result, host));
            if strict {
                mark_source_priority(&mut focused);
                mark_source_priority(&mut matching_primary);
            }
            focused.extend(matching_primary);
            primary.results = interleave_results(focused, remaining_primary);
        }
        None => {
            primary.results =
                interleave_results(supplemental.results, std::mem::take(&mut primary.results));
        }
    }
}

fn mark_source_priority(results: &mut [SearxResult]) {
    for result in results {
        result.source_priority = true;
    }
}

fn merge_local_results(primary: &mut SearxResponse, supplemental: SearxResponse) {
    let mut local = supplemental.results.into_iter();
    let promoted = local.by_ref().take(4).collect::<Vec<_>>();
    let mut results = Vec::with_capacity(promoted.len() + primary.results.len() + local.len());
    results.extend(promoted);
    results.append(&mut primary.results);
    results.extend(local);
    primary.results = results;
}

fn merge_discovery_results(primary: &mut SearxResponse, supplemental: SearxResponse) {
    primary.results =
        interleave_results(std::mem::take(&mut primary.results), supplemental.results);
}

fn merge_manifestation_result(
    primary: &mut SearxResponse,
    supplemental: SearxResponse,
    lookup_query: &str,
) {
    let manifestations = best_manifestations(supplemental.results, lookup_query);
    primary.results = interleave_results(std::mem::take(&mut primary.results), manifestations);
}

fn best_manifestations(results: Vec<SearxResult>, lookup_query: &str) -> Vec<SearxResult> {
    let reference_key = normalized_title_key(lookup_query);
    let reference_terms = reference_key.split_whitespace().count();
    let best_coverage = results
        .iter()
        .map(|result| manifestation_title_coverage(&reference_key, result))
        .max()
        .unwrap_or_default();

    if best_coverage == 0 {
        return results.into_iter().take(1).collect();
    }

    let mut matching = results
        .into_iter()
        .enumerate()
        .filter(|(_, result)| manifestation_title_coverage(&reference_key, result) == best_coverage)
        .collect::<Vec<_>>();
    matching.sort_by_key(|(index, result)| {
        (
            std::cmp::Reverse(manifestation_metadata_score(result)),
            *index,
        )
    });

    // A weak best match is merely the least-bad search result. Keep one in
    // that case. When the title match is strong, retain a few distinct
    // manifestations (publisher, archive, HTML/PDF) just as a document search
    // should; ordinary queries still collapse equivalent titles later.
    let limit = if best_coverage >= 4 && best_coverage * 3 >= reference_terms * 2 {
        4
    } else {
        1
    };
    matching
        .into_iter()
        .take(limit)
        .map(|(_, result)| result)
        .collect()
}

fn manifestation_title_coverage(reference_key: &str, result: &SearxResult) -> usize {
    let Some(title) = result.title.as_deref() else {
        return 0;
    };
    let candidate = normalized_title_key(title)
        .split_whitespace()
        .map(str::to_owned)
        .collect::<HashSet<_>>();
    reference_key
        .split_whitespace()
        .filter(|token| candidate.contains(*token))
        .count()
}

fn manifestation_metadata_score(result: &SearxResult) -> usize {
    usize::from(result.pdf_url.as_deref().and_then(parse_web_url).is_some()) * 4
        + usize::from(result.html_url.as_deref().and_then(parse_web_url).is_some()) * 2
        + usize::from(result.doi.as_deref().and_then(normalized_doi).is_some())
}

fn interleave_results(left: Vec<SearxResult>, right: Vec<SearxResult>) -> Vec<SearxResult> {
    let capacity = left.len() + right.len();
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    let mut results = Vec::with_capacity(capacity);

    loop {
        let left_result = left.next();
        let right_result = right.next();
        if left_result.is_none() && right_result.is_none() {
            break;
        }
        results.extend(left_result);
        results.extend(right_result);
    }
    results
}

fn result_matches_host(result: &SearxResult, expected: &str) -> bool {
    Url::parse(&result.url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            if expected.starts_with('.') {
                host.ends_with(expected)
            } else {
                host == expected || host.ends_with(&format!(".{expected}"))
            }
        })
}

pub(super) fn has_source_intent(lower_query: &str) -> bool {
    query_contains_token(lower_query, "arxiv")
        || has_full_text_intent(lower_query)
        || lower_query.contains("original paper")
        || lower_query.contains("primary source")
        || lower_query.contains("release notes")
        || lower_query.split_whitespace().any(|word| {
            matches!(
                word.trim_matches(|character: char| !character.is_alphanumeric()),
                "documentation" | "docs" | "guideline" | "guidelines" | "official"
            )
        })
}

fn has_documentation_intent(lower_query: &str) -> bool {
    lower_query.split_whitespace().any(|word| {
        matches!(
            word.trim_matches(|character: char| !character.is_alphanumeric()),
            "documentation" | "docs" | "official"
        )
    })
}

fn infer_official_host(search_query: &str, results: &[SearxResult]) -> Option<String> {
    let query_tokens = distinctive_tokens(search_query);
    let query_words = search_query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let public_authority_intent = query_words
        .iter()
        .any(|word| matches!(word.as_str(), "election" | "voter" | "voting"))
        && query_words
            .iter()
            .any(|word| matches!(word.as_str(), "deadline" | "registration"));
    results
        .iter()
        .take(12)
        .enumerate()
        .filter_map(|(index, result)| {
            let url = Url::parse(&result.url).ok()?;
            let host = url.host_str()?;
            let host = host.strip_prefix("www.").unwrap_or(host);
            if is_generic_host(host) {
                return None;
            }
            let title = result
                .title
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            let host_labels = host
                .split('.')
                .map(|label| {
                    label
                        .chars()
                        .filter(|character| character.is_alphanumeric())
                        .flat_map(char::to_lowercase)
                        .collect::<String>()
                })
                .collect::<Vec<_>>();
            let stable_brand_match = query_tokens.iter().any(|token| {
                let brand = brand_token(token);
                host_labels.iter().any(|label| label == brand)
            });
            let title_matches = query_tokens
                .iter()
                .filter(|token| title.contains(token.as_str()))
                .take(2)
                .count();
            let institutional = host.ends_with(".gov")
                || host.ends_with(".edu")
                || host.ends_with(".int")
                || host.ends_with(".ac.uk");
            if public_authority_intent && !host.ends_with(".gov") {
                return None;
            }
            let documentation_host = host.starts_with("doc.")
                || host.starts_with("docs.")
                || host == "docs.rs"
                || host.ends_with(".readthedocs.io");
            let documented_identity = documentation_host
                && url.path().split('/').any(|segment| {
                    let segment = brand_token(segment);
                    query_tokens
                        .iter()
                        .any(|token| brand_token(token) == segment)
                });
            let credible = if public_authority_intent {
                host.ends_with(".gov") && title_matches >= 1
            } else {
                (stable_brand_match || documented_identity) && title_matches >= 1
                    || institutional && title_matches >= 2
            };
            let score = usize::from(stable_brand_match) * 4
                + usize::from(documented_identity) * 3
                + usize::from(institutional) * 2
                + title_matches
                + result.engines.len().min(2);
            credible.then(|| (score, index, official_search_host(host)))
        })
        .max_by(|left, right| left.0.cmp(&right.0).then_with(|| right.1.cmp(&left.1)))
        .map(|(_, _, host)| host)
}

pub(super) fn registrable_host(host: &str) -> String {
    let labels = host.split('.').collect::<Vec<_>>();
    if labels.len() <= 2 {
        return host.to_owned();
    }
    let suffix = labels[labels.len() - 2..].join(".");
    let labels_to_keep = if matches!(
        suffix.as_str(),
        "ac.uk" | "co.jp" | "co.uk" | "com.au" | "com.br" | "com.cn" | "org.uk"
    ) {
        3
    } else {
        2
    };
    labels[labels.len() - labels_to_keep..].join(".")
}

fn official_search_host(host: &str) -> String {
    if host.starts_with("doc.") || host.starts_with("docs.") {
        host.to_owned()
    } else {
        registrable_host(host)
    }
}

fn distinctive_tokens(query: &str) -> Vec<String> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|token| token.len() >= 3)
        .filter(|token| !token.chars().all(|character| character.is_ascii_digit()))
        .filter(|token| {
            !matches!(
                token.as_str(),
                "and"
                    | "documentation"
                    | "example"
                    | "for"
                    | "full"
                    | "guideline"
                    | "guidelines"
                    | "how"
                    | "notes"
                    | "official"
                    | "release"
                    | "text"
                    | "the"
                    | "using"
                    | "what"
                    | "with"
            )
        })
        .collect()
}

fn brand_token(token: &str) -> &str {
    let stem = token.trim_end_matches(|character: char| character.is_ascii_digit());
    if stem.len() >= 3 { stem } else { token }
}

fn is_generic_host(host: &str) -> bool {
    [
        "github.com",
        "medium.com",
        "reddit.com",
        "stackoverflow.com",
        "wikipedia.org",
        "youtube.com",
    ]
    .iter()
    .any(|generic| host == *generic || host.ends_with(&format!(".{generic}")))
}

fn is_month(token: &str) -> bool {
    matches!(
        token,
        "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
            | "jan"
            | "feb"
            | "mar"
            | "apr"
            | "jun"
            | "jul"
            | "aug"
            | "sep"
            | "sept"
            | "oct"
            | "nov"
            | "dec"
    )
}

fn is_day(word: &str) -> bool {
    word.trim_matches(|character: char| !character.is_ascii_digit())
        .parse::<u8>()
        .is_ok_and(|day| (1..=31).contains(&day))
}

fn is_year(word: &str) -> bool {
    let word = word.trim_matches(|character: char| !character.is_ascii_digit());
    word.len() == 4
        && word
            .parse::<u16>()
            .is_ok_and(|year| (1900..=2200).contains(&year))
}

fn search_url_target(
    base: &Url,
    query: &str,
    target: SearchTarget<'_>,
    time_range: Option<&str>,
) -> Url {
    let mut url = base.clone();
    url.set_path("/search");
    let mut parameters = url.query_pairs_mut();
    parameters
        .append_pair("q", query)
        .append_pair("format", "json")
        .append_pair("language", "en")
        .append_pair("safesearch", "0");
    match target {
        SearchTarget::Categories(categories) if !categories.is_empty() => {
            parameters.append_pair("categories", categories);
        }
        SearchTarget::Engines(engines) if !engines.is_empty() => {
            parameters.append_pair("engines", engines);
        }
        SearchTarget::Categories(_) | SearchTarget::Engines(_) => {}
    }
    if let Some(time_range) = time_range {
        parameters.append_pair("time_range", time_range);
    }
    drop(parameters);
    url
}

fn into_hits(body: SearxResponse, query: &str, limit: usize) -> Vec<SearchCandidate> {
    if limit == 0 {
        return Vec::new();
    }
    let mut by_url = HashMap::<String, usize>::new();
    let mut by_title = Vec::<(String, usize)>::new();
    let mut by_doi = HashMap::<String, usize>::new();
    let lower_query = query.to_ascii_lowercase();
    let reserve_named_authority = reserves_named_authority(query);
    let keep_alternate_versions = paper_title(query).is_some()
        || has_full_text_intent(query)
        || lower_query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| matches!(token, "docs" | "documentation"));
    let mut hits = Vec::<SearchCandidate>::with_capacity(limit.min(body.results.len()));
    // Kept as a private sidecar so canonical URL variants observed by multiple
    // engines in the same SearXNG request establish consensus without leaking
    // engine names into domain or API types. Seeing the URL again only after a
    // query reformulation is not independent same-query consensus.
    let mut engine_observations =
        Vec::<HashMap<usize, HashSet<String>>>::with_capacity(hits.capacity());

    for result in body.results {
        let preferred_url = preferred_result_url(&result);
        let Some(url) = parse_web_url(preferred_url) else {
            tracing::debug!(url = preferred_url, "discarding invalid search result URL");
            continue;
        };
        let fetch_urls = result_fetch_urls(&result, &url);
        let has_explicit_fetch_url = fetch_urls
            .iter()
            .any(|fetch_url| canonical_key(fetch_url) != canonical_key(&url));

        let snippet = result
            .content
            .clone()
            .filter(|content| !content.trim().is_empty())
            .or_else(|| format_map_details(result.address.clone(), result.data.clone()))
            .unwrap_or_default();
        let title = result.title.clone().unwrap_or_else(|| "Untitled".into());
        let published_date = result
            .published_date
            .as_deref()
            .map(|date| date.chars().take(10).collect::<String>());
        let title_key = normalized_title_key(&title);
        let doi = result.doi.as_deref().and_then(normalized_doi);
        let existing = (if keep_alternate_versions {
            by_url.get(&canonical_key(&url)).copied()
        } else {
            fetch_urls
                .iter()
                .find_map(|url| by_url.get(&canonical_key(url)).copied())
        })
        .or_else(|| {
            (!keep_alternate_versions)
                .then(|| doi.as_ref().and_then(|doi| by_doi.get(doi).copied()))
                .flatten()
        })
        .or_else(|| {
            (!keep_alternate_versions && title != "Untitled")
                .then(|| matching_title_index(&title_key, &by_title))
                .flatten()
        });
        if let Some(index) = existing {
            merge_fetch_urls(
                &mut hits[index].fetch_urls,
                fetch_urls,
                has_explicit_fetch_url,
            );
            if hits[index].hit.snippet.is_empty() && !snippet.is_empty() {
                hits[index].hit.snippet = snippet;
            }
            if published_date.as_deref() > hits[index].hit.date.as_deref() {
                hits[index].hit.date = published_date;
            }
            hits[index].source_priority |= (result.source_priority
                || (has_strict_source_intent(&lower_query) || reserve_named_authority)
                    && matches_named_authority(query, &url))
                && source_scope_matches(query, &url);
            engine_observations[index]
                .entry(result.observation_id)
                .or_default()
                .extend(normalized_engines(&result.engines));
            hits[index].upstream_consensus = engine_observations[index]
                .values()
                .any(|engines| engines.len() >= 2);
            if keep_alternate_versions {
                by_url.insert(canonical_key(&hits[index].hit.url), index);
            } else {
                for fetch_url in &hits[index].fetch_urls {
                    by_url.insert(canonical_key(fetch_url), index);
                }
            }
            if let Some(doi) = doi {
                by_doi.insert(doi, index);
            }
            continue;
        }
        if hits.len() == limit {
            continue;
        }

        let hit = Hit {
            title,
            url,
            date: published_date,
            snippet,
        };
        let source_priority = (result.source_priority
            || (has_strict_source_intent(&lower_query) || reserve_named_authority)
                && matches_named_authority(query, &hit.url))
            && source_scope_matches(query, &hit.url);
        let index = hits.len();
        let engines = normalized_engines(&result.engines);
        let upstream_consensus = engines.len() >= 2;
        hits.push(SearchCandidate {
            fetch_urls,
            hit,
            source_priority,
            upstream_consensus,
        });
        engine_observations.push(HashMap::from([(result.observation_id, engines)]));
        if keep_alternate_versions {
            by_url.insert(canonical_key(&hits[index].hit.url), index);
        } else {
            for fetch_url in &hits[index].fetch_urls {
                by_url.insert(canonical_key(fetch_url), index);
            }
        }
        if title_key != "untitled" {
            by_title.push((title_key, index));
        }
        if let Some(doi) = doi {
            by_doi.insert(doi, index);
        }
    }

    hits
}

/// Reserve one canonical result when a query names both a known authority and
/// the kind of first-party document it wants. Unlike strict source intent,
/// this does not suppress community or specialist searches; it only keeps the
/// canonical document from being buried beneath implementations and mirrors.
fn reserves_named_authority(query: &str) -> bool {
    named_authority_host(query).is_some()
        && !has_quoted_diagnostic(query)
        && query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| {
                matches!(
                    token.to_ascii_lowercase().as_str(),
                    "advisory"
                        | "algorithm"
                        | "documentation"
                        | "docs"
                        | "guidance"
                        | "manual"
                        | "policy"
                        | "reference"
                        | "schedule"
                        | "spec"
                        | "specification"
                        | "standard"
                        | "status"
                )
            })
}

fn preferred_result_url(result: &SearxResult) -> &str {
    result
        .links
        .iter()
        .find(|link| link.label.eq_ignore_ascii_case("official website"))
        .map(|link| link.url.as_str())
        .unwrap_or(&result.url)
}

#[cfg(test)]
fn has_usable_result(results: &[SearxResult]) -> bool {
    results
        .iter()
        .any(|result| parse_web_url(preferred_result_url(result)).is_some())
}

fn normalized_engines(engines: &[String]) -> HashSet<String> {
    engines
        .iter()
        .map(|engine| engine.trim().to_ascii_lowercase())
        .filter(|engine| !engine.is_empty())
        .collect()
}

fn parse_web_url(value: &str) -> Option<Url> {
    let mut url = Url::parse(value.trim()).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    url.set_fragment(None);
    Some(url)
}

fn result_fetch_urls(result: &SearxResult, primary: &Url) -> Vec<Url> {
    let pdf = result.pdf_url.as_deref().and_then(parse_web_url);
    let html = result.html_url.as_deref().and_then(parse_web_url);
    let prefer_html = pdf
        .as_ref()
        .is_some_and(|url| url.path().to_ascii_lowercase().ends_with(".pdf"))
        && html
            .as_ref()
            .is_some_and(|url| !url.path().to_ascii_lowercase().ends_with(".pdf"));
    let alternates = if prefer_html {
        html.into_iter().chain(pdf).collect::<Vec<_>>()
    } else {
        pdf.into_iter().chain(html).collect()
    };
    let mut urls = alternates
        .into_iter()
        .chain(std::iter::once(primary.clone()))
        .collect::<Vec<_>>();
    let mut seen = HashSet::new();
    urls.retain(|url| seen.insert(canonical_key(url)));
    urls
}

fn merge_fetch_urls(existing: &mut Vec<Url>, discovered: Vec<Url>, prioritize_discovered: bool) {
    let current = std::mem::take(existing);
    let (first, second) = if prioritize_discovered {
        (discovered, current)
    } else {
        (current, discovered)
    };
    let mut seen = HashSet::new();
    *existing = first
        .into_iter()
        .chain(second)
        .filter(|url| seen.insert(canonical_key(url)))
        .collect();
}

fn normalized_doi(value: &str) -> Option<String> {
    let doi = value
        .trim()
        .trim_start_matches("doi:")
        .trim_start_matches("https://doi.org/")
        .trim_start_matches("http://doi.org/")
        .trim()
        .to_ascii_lowercase();
    (!doi.is_empty()).then_some(doi)
}

fn matching_title_index(title: &str, titles: &[(String, usize)]) -> Option<usize> {
    titles
        .iter()
        .filter(|(candidate, _)| title_keys_equivalent(title, candidate))
        .map(|(_, index)| *index)
        .min()
}

fn title_keys_equivalent(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let left = left.split_whitespace().collect::<Vec<_>>();
    let right = right.split_whitespace().collect::<Vec<_>>();
    let (shorter, longer) = if left.len() <= right.len() {
        (&left, &right)
    } else {
        (&right, &left)
    };
    shorter.len() >= 4 && shorter.len() * 3 >= longer.len() * 2 && longer.starts_with(shorter)
}

fn source_scope_matches(query: &str, url: &Url) -> bool {
    if url.host_str() != Some("docs.rs") {
        return true;
    }

    let mut segments = url.path_segments().into_iter().flatten();
    let first = segments.next();
    let package = if first == Some("crate") {
        segments.next()
    } else {
        first
    };
    package.is_some_and(|package| {
        query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| token.eq_ignore_ascii_case(package))
    })
}

fn matches_named_authority(query: &str, url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if query_contains_token(query, "tokio") && host == "docs.rs" {
        return true;
    }
    named_authority_host(query)
        .is_some_and(|authority| host == authority || host.ends_with(&format!(".{authority}")))
}

fn merge_auxiliary_hits(primary: &mut Vec<SearchCandidate>, auxiliary: Vec<Hit>, limit: usize) {
    let mut urls = primary
        .iter()
        .map(|hit| canonical_key(&hit.url))
        .collect::<HashSet<_>>();
    let mut titles = primary
        .iter()
        .map(|hit| normalized_title_key(&hit.title))
        .collect::<HashSet<_>>();
    let auxiliary = auxiliary
        .into_iter()
        .filter(|hit| {
            urls.insert(canonical_key(&hit.url)) && titles.insert(normalized_title_key(&hit.title))
        })
        .map(|hit| SearchCandidate {
            fetch_urls: vec![hit.url.clone()],
            hit,
            source_priority: false,
            upstream_consensus: false,
        });
    let insert_at = primary.len().min(4);
    primary.splice(insert_at..insert_at, auxiliary);
    primary.truncate(limit);
}

fn normalized_title_key(title: &str) -> String {
    let words = title
        .replace("%20", " ")
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let mut observed_numbers = HashSet::new();
    let repeated_numbers = words
        .iter()
        .filter(|word| word.len() >= 4 && word.chars().all(|character| character.is_ascii_digit()))
        .filter(|word| !observed_numbers.insert((*word).clone()))
        .cloned()
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    words
        .into_iter()
        .filter(|word| {
            !(word.chars().all(|character| character.is_ascii_digit())
                && repeated_numbers.contains(word))
        })
        .filter(|word| seen.insert(word.clone()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn canonical_key(url: &Url) -> String {
    let mut key = url.clone();
    if key.scheme() == "http" {
        let _ = key.set_scheme("https");
    }
    if let Some(host) = key.host_str().map(str::to_owned)
        && let Some(host) = host.strip_prefix("www.")
    {
        let _ = key.set_host(Some(host));
    }
    if key.path().len() > 1 {
        let trimmed = key.path().trim_end_matches('/').to_owned();
        key.set_path(&trimmed);
    }

    let retained = key
        .query_pairs()
        .filter(|(name, _)| {
            let name = name.to_ascii_lowercase();
            !name.starts_with("utm_")
                && !matches!(
                    name.as_str(),
                    "fbclid" | "gclid" | "ref" | "referrer" | "source"
                )
        })
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    key.set_query(None);
    if !retained.is_empty() {
        key.query_pairs_mut().extend_pairs(retained);
    }
    key.to_string()
}

/// SearXNG fields other than `url` are optional in real responses.
#[derive(Deserialize)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<SearxResult>,
    #[serde(default)]
    unresponsive_engines: Vec<Vec<String>>,
}

#[derive(Clone, Deserialize)]
struct SearxResult {
    url: String,
    title: Option<String>,
    content: Option<String>,
    #[serde(rename = "publishedDate", alias = "published_date")]
    published_date: Option<String>,
    address: Option<SearxAddress>,
    #[serde(default)]
    data: Vec<SearxData>,
    #[serde(default, deserialize_with = "deserialize_links")]
    links: Vec<SearxLink>,
    #[serde(default)]
    score: f32,
    #[serde(default)]
    engines: Vec<String>,
    #[serde(skip)]
    observation_id: usize,
    doi: Option<String>,
    pdf_url: Option<String>,
    html_url: Option<String>,
    #[serde(skip)]
    source_priority: bool,
}

#[derive(Clone, Deserialize)]
struct SearxAddress {
    house_number: Option<String>,
    road: Option<String>,
    locality: Option<String>,
    postcode: Option<String>,
    country: Option<String>,
}

#[derive(Clone, Deserialize)]
struct SearxData {
    key: String,
    value: String,
}

#[derive(Clone, Deserialize)]
struct SearxLink {
    label: String,
    url: String,
}

fn deserialize_links<'de, D>(deserializer: D) -> std::result::Result<Vec<SearxLink>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Links {
        List(Vec<SearxLink>),
        Map(HashMap<String, String>),
    }

    Ok(match Links::deserialize(deserializer)? {
        Links::List(links) => links,
        Links::Map(links) => links
            .into_iter()
            .map(|(label, url)| SearxLink { label, url })
            .collect(),
    })
}

fn format_map_details(address: Option<SearxAddress>, data: Vec<SearxData>) -> Option<String> {
    let mut details = address
        .map(format_address)
        .filter(|address| !address.is_empty())
        .into_iter()
        .collect::<Vec<_>>();
    details.extend(
        data.into_iter()
            .filter(|item| item.key == "opening_hours")
            .map(|item| format!("Hours: {}", item.value)),
    );
    (!details.is_empty()).then(|| details.join(". "))
}

fn format_address(address: SearxAddress) -> String {
    let street = [address.house_number, address.road]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    let address = [
        (!street.is_empty()).then_some(street),
        address.locality,
        address.postcode,
        address.country,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(", ");
    if address.is_empty() {
        String::new()
    } else {
        format!("Address: {address}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hit(title: &str, url: &str) -> Hit {
        Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: title.into(),
        }
    }

    fn test_candidate(title: &str, url: &str) -> SearchCandidate {
        let hit = test_hit(title, url);
        SearchCandidate {
            fetch_urls: vec![hit.url.clone()],
            hit,
            source_priority: false,
            upstream_consensus: false,
        }
    }

    #[test]
    fn partial_metasearch_keeps_every_available_pool() {
        let combined = combine_search_pools(
            Ok(SearchPool {
                candidates: vec![test_candidate("SearX result", "https://search.example/")],
                healthy: true,
            }),
            vec![test_hit("GitHub result", "https://github.com/example/repo")],
            8,
        )
        .unwrap();

        assert!(combined.healthy);
        assert_eq!(combined.candidates.len(), 2);
    }

    #[test]
    fn auxiliary_results_survive_a_total_searxng_failure() {
        let combined = combine_search_pools(
            Err(AppError::SearchBackendResponse(
                "SearXNG unavailable".into(),
            )),
            vec![test_hit("GitHub result", "https://github.com/example/repo")],
            8,
        )
        .unwrap();

        assert!(!combined.healthy);
        assert_eq!(combined.candidates.len(), 1);
        assert_eq!(combined.candidates[0].title, "GitHub result");
    }

    #[test]
    fn total_retrieval_failure_still_returns_the_backend_error() {
        let error = combine_search_pools(
            Err(AppError::SearchBackendResponse(
                "SearXNG unavailable".into(),
            )),
            Vec::new(),
            8,
        )
        .err()
        .expect("an empty auxiliary pool cannot hide the backend failure");

        assert!(matches!(
            error,
            AppError::SearchBackendResponse(message) if message == "SearXNG unavailable"
        ));
    }

    #[test]
    fn accepts_template_specific_link_shapes() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://crates.io/crates/example","links":{"Source code":"https://github.com/example/example"}},
                {"url":"https://example.com/place","links":[{"label":"Official website","url":"https://official.example/"}]}
            ]}"#,
        )
        .unwrap();

        assert_eq!(body.results[0].links[0].label, "Source code");
        assert_eq!(body.results[1].links[0].url, "https://official.example/");
    }

    #[test]
    fn normalizes_deduplicates_and_limits_results() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://example.com/a#one","title":"A","content":"first"},
                    {"url":"https://example.com/a#two","title":"duplicate"},
                    {"url":"http://www.example.com/a/?utm_source=test","title":"canonical duplicate"},
                    {"url":"https://mirror.example/a","title":"  A  "},
                    {"url":"ftp://example.com/file"},
                    {"url":"not a URL"},
                    {"url":"https://example.com/b","publishedDate":"2026-08-01T12:00:00"},
                    {"url":"https://example.com/c"}
                ]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, "ordinary query", 2);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url.as_str(), "https://example.com/a");
        assert_eq!(hits[0].snippet, "first");
        assert_eq!(hits[1].date.as_deref(), Some("2026-08-01"));
    }

    #[test]
    fn duplicate_results_preserve_the_newest_known_date() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/status","title":"Mission status"},
                {"url":"https://example.com/status","title":"Mission status","publishedDate":"2026-08-01T12:00:00"},
                {"url":"https://example.com/status","title":"Mission status","publishedDate":"2026-07-31T12:00:00"}
            ]}"#,
        )
        .unwrap();

        let hits = into_hits(body, "recent mission status", 8);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].date.as_deref(), Some("2026-08-01"));
    }

    #[test]
    fn same_query_duplicates_union_private_engine_provenance() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/answer","title":"The answer","engines":["Bing"]},
                {"url":"http://www.example.com/answer/?utm_source=yep","title":"The answer","engines":["yep"]}
            ]}"#,
        )
        .unwrap();

        let hits = into_hits(body, "ordinary query", 1);

        assert_eq!(hits.len(), 1);
        assert!(hits[0].upstream_consensus);
    }

    #[test]
    fn query_reformulation_does_not_manufacture_engine_consensus() {
        let mut body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/answer","title":"The answer","engines":["Bing"]},
                {"url":"http://www.example.com/answer/?utm_source=yep","title":"The answer","engines":["yep"]}
            ]}"#,
        )
        .unwrap();
        body.results[1].observation_id = 1;

        let hits = into_hits(body, "ordinary query", 1);

        assert_eq!(hits.len(), 1);
        assert!(!hits[0].upstream_consensus);
    }

    #[test]
    fn invalid_only_responses_have_no_usable_search_result() {
        let invalid: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"not a URL","title":"Malformed"},
                {"url":"ftp://example.com/file","title":"Unsupported scheme"}
            ]}"#,
        )
        .unwrap();
        let valid: SearxResponse = serde_json::from_str(
            r#"{"results":[{"url":"https://example.com/answer","title":"Usable"}]}"#,
        )
        .unwrap();

        assert!(!has_usable_result(&invalid.results));
        assert!(into_hits(invalid, "ordinary query", 8).is_empty());
        assert!(has_usable_result(&valid.results));
    }

    #[test]
    fn deduplicates_noisy_numeric_title_variants() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://lake.example/a","title":"Lake Mendota water temperature 2742 2742 | LakeMonster"},
                    {"url":"https://lake.example/b","title":"Lake%20Mendota Water Temperature | LakeMonster"},
                    {"url":"https://lake.example/c","title":"Mendota research buoy"}
                ]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, "ordinary query", 3);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].title, "Mendota research buoy");
        assert_ne!(
            normalized_title_key("PostgreSQL 11 Released"),
            normalized_title_key("PostgreSQL 18 Released")
        );
        assert_ne!(
            normalized_title_key("RFC 9110: HTTP Semantics"),
            normalized_title_key("RFC 9111: HTTP Caching")
        );
    }

    #[test]
    fn exact_document_queries_keep_alternate_urls_with_the_same_title() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://arxiv.org/abs/1706.03762","title":"Attention Is All You Need"},
                {"url":"https://proceedings.neurips.cc/paper.pdf","title":"Attention Is All You Need"}
            ]}"#,
        )
        .unwrap();

        let hits = into_hits(body, "Attention Is All You Need original paper PDF", 8);

        assert_eq!(hits.len(), 2);
        assert_eq!(paper_title("convert a paper form to PDF"), None);
    }

    #[test]
    fn documentation_queries_keep_versioned_pages_with_the_same_title() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://docs.rs/tokio/latest/tokio/macro.select.html","title":"select in tokio - Rust"},
                {"url":"https://docs.rs/tokio/1.46.1/tokio/macro.select.html","title":"select in tokio - Rust"}
            ]}"#,
        )
        .unwrap();

        let hits = into_hits(body, "tokio select official documentation", 8);

        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn paper_metadata_merges_an_accessible_full_text_manifestation() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {
                    "url":"https://diabetesjournals.org/care/article/49/Supplement_1/S216/163933/chapter",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in ..."
                },
                {
                    "url":"https://doi.org/10.2337/dc26-s010",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in Diabetes—2026",
                    "doi":"10.2337/dc26-S010",
                    "pdf_url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12690187/",
                    "html_url":"https://doi.org/10.2337/dc26-s010"
                },
                {
                    "url":"https://pubmed.ncbi.nlm.nih.gov/41358899/",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in Diabetes—2026"
                }
            ]}"#,
        )
        .unwrap();

        let hits = into_hits(
            body,
            "2026 ADA diabetes standards statin recommendations official",
            8,
        );

        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0]
                .fetch_urls
                .iter()
                .map(Url::as_str)
                .collect::<Vec<_>>(),
            [
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC12690187/",
                "https://doi.org/10.2337/dc26-s010",
                "https://diabetesjournals.org/care/article/49/Supplement_1/S216/163933/chapter",
                "https://pubmed.ncbi.nlm.nih.gov/41358899/",
            ]
        );
        assert!(hits[0].source_priority);
    }

    #[test]
    fn literal_pdf_alternates_do_not_preempt_clean_html() {
        let mut body: SearxResponse = serde_json::from_str(
            r#"{"results":[{
                "url":"https://example.edu/paper",
                "title":"A Paper",
                "pdf_url":"https://example.edu/paper.pdf",
                "html_url":"https://example.edu/paper.html"
            }]}"#,
        )
        .unwrap();
        let result = body.results.remove(0);
        let primary = Url::parse(&result.url).unwrap();

        assert_eq!(
            result_fetch_urls(&result, &primary)
                .iter()
                .map(Url::as_str)
                .collect::<Vec<_>>(),
            [
                "https://example.edu/paper.html",
                "https://example.edu/paper.pdf",
                "https://example.edu/paper",
            ]
        );
    }

    #[test]
    fn derives_a_narrow_scholarly_lookup_from_a_primary_result() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {
                    "url":"https://professional.diabetes.org/standards-of-care",
                    "title":"Standards of Care in Diabetes | ADA Clinical Guidelines"
                },
                {
                    "url":"https://diabetesjournals.org/care/article/49/Supplement_1/S6/163930/summary",
                    "title":"Summary of Revisions: Standards of Care in Diabetes—2026"
                },
                {
                    "url":"https://diabetesjournals.org/care/article/49/Supplement_1/S216/163933/chapter",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in ..."
                }
            ]}"#,
        )
        .unwrap();

        assert_eq!(
            scholarly_manifestation_query(
                "2026 ADA diabetes standards statin recommendations adults official",
                &body.results,
                "general,it,science",
            ),
            Some(ManifestationQuery {
                query: "10 Cardiovascular Disease and Risk Management Standards of Care in 2026"
                    .into(),
                engines: "openalex,semantic scholar".into(),
            })
        );
        assert_eq!(
            scholarly_manifestation_query(
                "how is diabetes diagnosed",
                &body.results,
                "general,it,science",
            ),
            None
        );
    }

    #[test]
    fn explicit_recommendations_prefer_the_named_first_party_source() {
        let query = "2026 ADA diabetes standards statin recommendations adults official";
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {
                    "url":"https://diabetesjournals.org/care/article/49/Supplement_1/S216/163933/chapter",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in ..."
                }
            ]}"#,
        )
        .unwrap();
        let route = route(query, "general,science");

        assert_eq!(
            supplemental_queries(query, &route, &body.results, "general,science"),
            vec![SupplementalQuery::Source(SourceQuery {
                query: "2026 ADA diabetes standards statin recommendations adults site:diabetesjournals.org".into(),
                host: Some("diabetesjournals.org".into()),
                strict: true,
            })]
        );
    }

    #[test]
    fn exploratory_fresh_paper_queries_keep_the_science_supplement() {
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[{
                "url":"https://arxiv.org/abs/2608.00001",
                "title":"A claimed room temperature superconductor"
            }]}"#,
        )
        .unwrap();

        for query in [
            "latest room temperature superconductor paper",
            "latest hypertension treatment guideline",
        ] {
            let route = route(query, "general,science");
            assert_eq!(
                supplemental_queries(query, &route, &body.results, "general,science"),
                vec![SupplementalQuery::Specialist(SpecialistQuery {
                    query: query.into(),
                    categories: "science".into(),
                })],
                "{query}"
            );
        }

        let query = "current ADA diabetes standards";
        let routed = route(query, "general,science");
        assert_eq!(
            supplemental_queries(query, &routed, &body.results, "general,science"),
            vec![SupplementalQuery::Specialist(SpecialistQuery {
                query: query.into(),
                categories: "science".into(),
            })]
        );

        let query = "paper about CRISPR base editing delivery";
        let routed = route(query, "general,science");
        assert_eq!(
            supplemental_queries(query, &routed, &body.results, "general,science"),
            vec![SupplementalQuery::Specialist(SpecialistQuery {
                query: query.into(),
                categories: "science".into(),
            })]
        );

        let query = "creatine sleep deprivation cognitive performance randomized controlled trial full text";
        let route = route(query, "general,science");
        assert_eq!(
            supplemental_queries(query, &route, &body.results, "general,science"),
            vec![SupplementalQuery::Specialist(SpecialistQuery {
                query: query.into(),
                categories: "science".into(),
            })]
        );
    }

    #[test]
    fn restores_score_order_after_searxng_category_grouping() {
        let mut body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/vertical","score":0.4,"engines":["mdn"]},
                {"url":"https://example.com/consensus","score":2.5,"engines":["bing","yep"]},
                {"url":"https://example.com/single","score":2.5,"engines":["bing"]}
            ]}"#,
        )
        .unwrap();

        sort_by_relevance(&mut body.results);

        assert_eq!(
            body.results
                .iter()
                .map(|result| result.url.as_str())
                .collect::<Vec<_>>(),
            [
                "https://example.com/consensus",
                "https://example.com/single",
                "https://example.com/vertical",
            ]
        );
    }

    #[test]
    fn routes_explicit_news_broadly_and_preserves_dates_for_discovery() {
        assert_eq!(
            route("OpenAI news August 1 2026", "general,it,science,news"),
            Route {
                query: "OpenAI news August 1 2026".into(),
                categories: "general".into(),
                supplemental: Some(SpecialistQuery {
                    query: "OpenAI".into(),
                    categories: "news".into(),
                }),
            }
        );
        assert_eq!(
            route("what causes the aurora borealis", "general,it,science,news"),
            Route {
                query: "what causes the aurora borealis".into(),
                categories: "general".into(),
                supplemental: None,
            }
        );
        assert_eq!(
            route("May Mobility news", "general,news").supplemental,
            Some(SpecialistQuery {
                query: "May Mobility".into(),
                categories: "news".into(),
            })
        );
        assert_eq!(
            route("March Madness news", "general,news").supplemental,
            Some(SpecialistQuery {
                query: "March Madness".into(),
                categories: "news".into(),
            })
        );
    }

    #[test]
    fn routes_fresh_advisories_and_mission_status_to_news() {
        assert_eq!(freshness_window("latest NVIDIA advisory"), Some("year"));
        assert_eq!(freshness_window("OpenAI news August 1 2026"), None);
        assert_eq!(
            route(
                "latest NVIDIA Blackwell CUDA driver security advisory August 2026",
                "general,news",
            )
            .supplemental,
            Some(SpecialistQuery {
                query: "NVIDIA Blackwell CUDA driver security advisory".into(),
                categories: "news".into(),
            })
        );
        assert_eq!(
            route(
                "recent Europa Clipper mission status August 2026",
                "general,news",
            )
            .supplemental,
            Some(SpecialistQuery {
                query: "Europa Clipper mission status".into(),
                categories: "news".into(),
            })
        );
        assert_eq!(
            route("current US CPI inflation official data", "general,news").supplemental,
            None
        );
    }

    #[test]
    fn product_names_containing_news_keep_strong_api_intent() {
        let routed = route("Hacker News API official documentation", "general,it,news");

        assert!(!has_news_intent("Hacker News API official documentation"));
        assert_eq!(
            routed.supplemental,
            Some(SpecialistQuery {
                query: "Hacker News API official documentation".into(),
                categories: "general".into(),
            })
        );
        assert!(has_news_intent("latest Hacker News API outage"));
    }

    #[test]
    fn normalizes_search_intent_without_misrouting_current_conditions() {
        assert_eq!(
            route("Lake Mendota water temperature today", "general,news"),
            Route {
                query: "Lake Mendota water temperature today".into(),
                categories: "general".into(),
                supplemental: None,
            }
        );
        assert_eq!(
            route(
                "What is the US federal funds rate in August 2026?",
                "general,news"
            ),
            Route {
                query: "What is the US federal funds rate in August 2026?".into(),
                categories: "general".into(),
                supplemental: None,
            }
        );
        assert_eq!(
            route(
                "current FDA approved Alzheimer's drugs 2026",
                "general,news"
            ),
            Route {
                query: "current FDA approved Alzheimer's drugs 2026".into(),
                categories: "general".into(),
                supplemental: None,
            }
        );
    }

    #[test]
    fn focuses_clear_local_intent_for_map_search() {
        assert_eq!(
            local_map_query(
                "best coffee shops near Madison Wisconsin open late",
                "coffee shops near Madison Wisconsin open late"
            ),
            Some("coffee shops Madison Wisconsin".into())
        );
        assert_eq!(
            local_map_query(
                "best coworking cafes near Austin Texas open late",
                "coworking cafes Austin Texas"
            ),
            Some("cafes Austin Texas".into())
        );
        assert_eq!(
            local_map_query(
                "coffee shops near Madison Wisconsin",
                "coffee shops near Madison Wisconsin"
            ),
            Some("coffee shops Madison Wisconsin".into())
        );
        assert_eq!(
            local_map_query(
                "quiet coffee shop near Logan Square Chicago open after 8 pm weekdays outlets Wi-Fi recent reviews",
                "quiet coffee shop near Logan Square Chicago open after 8 pm weekdays outlets Wi-Fi recent reviews"
            ),
            Some("coffee shop Logan Square Chicago".into())
        );
        assert_eq!(
            local_map_query(
                "same-day ThinkPad repair near 60647 Linux-friendly diagnostic fee recent reviews",
                "same-day ThinkPad repair near 60647 Linux-friendly diagnostic fee recent reviews"
            ),
            Some("ThinkPad repair 60647".into())
        );
        assert_eq!(
            local_map_query(
                "Lake Mendota water temperature today",
                "Lake Mendota water temperature"
            ),
            None
        );
        assert_eq!(
            local_map_query(
                "best hiking trails near Madison Wisconsin",
                "hiking trails near Madison Wisconsin"
            ),
            None
        );
        assert!(has_category("general,news,map", "map"));
    }

    #[test]
    fn respects_restricted_category_configuration() {
        assert_eq!(
            route("axum timeout", "it,science"),
            Route {
                query: "axum timeout".into(),
                categories: "it".into(),
                supplemental: None,
            }
        );
    }

    #[test]
    fn supplements_temporal_queries_without_the_date_constraint() {
        assert_eq!(
            route(
                "2026 hypertension first line treatment guideline adults",
                "general,science"
            ),
            Route {
                query: "2026 hypertension first line treatment guideline adults".into(),
                categories: "general".into(),
                supplemental: Some(SpecialistQuery {
                    query: "2026 hypertension first line treatment guideline adults".into(),
                    categories: "science".into(),
                }),
            }
        );
        assert_eq!(
            route("latest room temperature superconductor", "general,science").supplemental,
            Some(SpecialistQuery {
                query: "latest room temperature superconductor".into(),
                categories: "science".into(),
            })
        );
        assert_eq!(
            route("latest room temperature superconductor", "general,science").categories,
            "general"
        );
    }

    #[test]
    fn focuses_explicit_recommendation_issuer_and_year_without_more_fanout() {
        assert_eq!(
            focused_recommendation_query(
                "2026 ADA diabetes standards statin recommendations adults official"
            ),
            Some("\"2026 ADA Standards\" diabetes statin".into())
        );
        assert_eq!(
            focused_recommendation_query(
                "2026 ada diabetes standards statin recommendations adults official"
            ),
            Some("\"2026 ADA Standards\" diabetes statin".into())
        );
        assert_eq!(
            focused_recommendation_query("2026 HTTP API retry recommendations"),
            None
        );
        assert_eq!(
            route(
                "2026 ADA diabetes standards statin recommendations adults official",
                "general,science"
            ),
            Route {
                query: "2026 ADA diabetes standards statin recommendations adults official".into(),
                categories: "general".into(),
                supplemental: Some(SpecialistQuery {
                    query: "2026 ADA diabetes standards statin recommendations adults official"
                        .into(),
                    categories: "science".into(),
                }),
            }
        );
        assert_eq!(
            focused_recommendation_query("hypertension treatment guideline adults"),
            None
        );
    }

    #[test]
    fn adds_specialist_categories_only_for_matching_intent() {
        assert_eq!(
            route(
                "Attention Is All You Need arXiv paper",
                "general,it,science"
            )
            .categories,
            "general"
        );
        assert_eq!(
            route("Rust E0502 borrow error", "general,it,science").categories,
            "general"
        );
        assert_eq!(
            route("traditional shakshuka recipe", "general,it,science").categories,
            "general"
        );
        assert!(has_it_intent("Rust E0502 borrow error"));
        assert!(has_it_intent(
            "Python pathlib Path.walk official documentation"
        ));
        assert!(!has_it_intent("remove rust from cast iron"));
        assert!(!has_it_intent("python snake habitat"));
        assert!(!has_it_intent("how should I react"));
        assert!(!has_it_intent("swollen lymph node"));
    }

    #[test]
    fn routes_diverse_software_and_science_queries_without_ambiguous_false_positives() {
        for query in [
            "Java HashMap iteration order official documentation",
            "Go context cancellation best practices",
            "AWS S3 bucket lifecycle Terraform configuration",
            "C# ASP.NET Core dependency injection",
            "Kotlin coroutine Flow exception handling",
            "Swift async await actor isolation",
            "Ruby on Rails ActiveRecord transaction",
            "PHP Laravel middleware authentication",
            "Dart Flutter widget lifecycle",
            "Python library for CRISPR guide design",
        ] {
            assert_eq!(
                route(query, "general,it,science")
                    .supplemental
                    .as_ref()
                    .map(|supplemental| supplemental.categories.as_str()),
                Some("general"),
                "{query}"
            );
        }
        assert_eq!(
            route("C++ std::unique_ptr move semantics", "general,it,science")
                .supplemental
                .as_ref()
                .map(|supplemental| supplemental.categories.as_str()),
            Some("it")
        );

        for query in [
            "CRISPR-Cas9 off-target effects in human cells",
            "JWST exoplanet atmosphere spectroscopy paper",
            "organic chemistry SN2 solvent reaction mechanism study",
            "quantum entanglement Bell inequality experiment",
            "genome-wide association study type 2 diabetes",
            "ecology biodiversity meta-analysis DOI",
            "Python protein folding research paper",
        ] {
            assert_eq!(
                route(query, "general,it,science")
                    .supplemental
                    .as_ref()
                    .map(|supplemental| supplemental.categories.as_str()),
                Some("science"),
                "{query}"
            );
        }

        for query in [
            "best coffee in Java Indonesia",
            "rules for the board game Go",
            "remove rust from cast iron",
            "Ruby gemstone clarity guide",
            "Spring vegetable planting calendar",
            "Swift bird migration",
            "science fiction books like Dune",
            "primary sources causes of the 2008 financial crisis",
            "traditional shakshuka recipe",
            "Amazon rainforest rainfall trends",
        ] {
            assert_eq!(
                route(query, "general,it,science").supplemental,
                None,
                "{query}"
            );
        }
    }

    #[test]
    fn supplements_code_identifiers_with_a_focused_exact_query() {
        assert_eq!(
            route(
                "invalid_reference_casting tokenizers Rust 1.73",
                "general,it"
            )
            .supplemental,
            Some(SpecialistQuery {
                query: "\"invalid_reference_casting\" tokenizers".into(),
                categories: "it".into(),
            })
        );
        assert!(should_search_stackoverflow(
            "Next.js hydration error caused by Date.now"
        ));
        assert!(!should_search_stackoverflow(
            "GitHub Actions matrix jobs limit official documentation"
        ));
        assert!(!should_search_stackoverflow(
            "reproducible hermetic builds Bazel Nix Pants case studies"
        ));
        assert_eq!(
            route("rust unresolved import tower_http timeout", "general,it").supplemental,
            Some(SpecialistQuery {
                query: "\"tower_http\" timeout".into(),
                categories: "it".into(),
            })
        );
        assert_eq!(
            route(
                "Kubernetes pod \"exec user process caused: exec format error\" arm64 amd64 multi-arch image",
                "general,it"
            )
            .supplemental,
            Some(SpecialistQuery {
                query: "\"exec user process caused: exec format error\" Kubernetes pod arm64 amd64 multi-arch image".into(),
                categories: "general".into(),
            })
        );
        assert_eq!(
            route(
                "detect silent numerical drift porting machine-learning inference from CPU to CUDA bitwise reproducibility tolerance testing",
                "general,it,science"
            )
            .supplemental,
            Some(SpecialistQuery {
                query: "bitwise reproducible deep learning inference".into(),
                categories: "science".into(),
            })
        );
        assert_eq!(
            route(
                "reproducible hermetic builds in large polyglot monorepos Bazel Nix Pants tradeoffs case studies",
                "general,it"
            )
            .supplemental,
            Some(SpecialistQuery {
                query: "(Bazel OR Nix OR Pants) monorepo (migration OR \"case study\" OR postmortem)"
                    .into(),
                categories: "general".into(),
            })
        );
    }

    #[test]
    fn case_study_focus_uses_boolean_subject_and_evidence_alternatives() {
        assert_eq!(
            focused_case_study_query(
                "reproducible hermetic builds in large polyglot monorepos Bazel Nix Pants tradeoffs case studies"
            ),
            "(Bazel OR Nix OR Pants) monorepo (migration OR \"case study\" OR postmortem)"
        );
    }

    #[test]
    fn numerical_parity_focus_uses_terms_native_to_the_evidence_corpus() {
        assert_eq!(
            focused_numerical_parity_query(
                "detect silent drift in bitwise CPU to CUDA machine-learning inference"
            ),
            "bitwise reproducible deep learning inference"
        );
        assert_eq!(
            focused_numerical_parity_query(
                "compare neural network inference numerical deviations across GPU backends"
            ),
            "numerical deviations neural network inference reproducibility"
        );
        assert_eq!(
            focused_numerical_parity_query("CUDA CPU kernel numerical parity testing"),
            "cpu cuda numerical reproducibility testing"
        );

        let query = "detect silent numerical drift porting machine-learning inference from CPU to CUDA bitwise reproducibility tolerance testing";
        let route = route(query, "general,it,science");
        assert_eq!(
            supplemental_queries(query, &route, &[], "general,it,science"),
            vec![
                SupplementalQuery::Specialist(SpecialistQuery {
                    query: "bitwise reproducible deep learning inference".into(),
                    categories: "science".into(),
                }),
                SupplementalQuery::Specialist(SpecialistQuery {
                    query: "numerical deviations neural network inference reproducibility".into(),
                    categories: "science".into(),
                }),
            ]
        );
    }

    #[test]
    fn source_descriptors_without_exclusivity_keep_auxiliary_searches() {
        for query in [
            "npm registry service status incidents",
            "WHATWG DOM standard AbortSignal.any algorithm",
            "Node.js release schedule active LTS",
            "NIST post-quantum migration guidance",
            "Rust tokio select docs",
            "Rust error E0502 documentation examples",
            "security vulnerability disclosure timeline",
            "WebAuthn Level 3 specification conditional mediation",
        ] {
            assert!(allows_auxiliary_search(query), "{query}");
            assert!(
                !has_strict_source_intent(&query.to_ascii_lowercase()),
                "{query}"
            );
        }

        for query in [
            "official npm registry status",
            "React useEffect official documentation",
            "Attention Is All You Need original paper",
            "CVE-2024-3094 original disclosure",
            "primary source for the 2008 financial crisis",
            "Node.js release notes",
            "Geneva Convention Common Article 3 full text",
        ] {
            assert!(!allows_auxiliary_search(query), "{query}");
            assert!(
                has_strict_source_intent(&query.to_ascii_lowercase()),
                "{query}"
            );
        }
    }

    #[test]
    fn quoted_diagnostics_keep_general_recall_unless_first_party_is_explicit() {
        assert_eq!(
            focused_technical_query(
                "Rust E0277 \"FromResidual<Result<Infallible, _>> is not implemented\" using ? with a custom error enum"
            ),
            Some("Rust question mark operator custom error enum impl From source error".into())
        );

        let query = "TypeScript TS2589 \"Type instantiation is excessively deep\" Zod";
        let routed = route(query, "general,it");
        let focused = routed
            .supplemental
            .clone()
            .expect("an exact diagnostic gets a focused supplement");
        assert_eq!(focused.categories, "general");
        assert_eq!(routed.query, query);
        assert!(allows_auxiliary_search(query));
        assert!(should_search_stackoverflow(query));
        assert_eq!(
            source_aware_query(query, &routed.query, &[]),
            Some(SourceQuery {
                query:
                    "\"Type instantiation is excessively deep\" TypeScript TS2589 Zod site:zod.dev"
                        .into(),
                host: Some("zod.dev".into()),
                strict: false,
            })
        );
        assert_eq!(
            supplemental_queries(query, &routed, &[], "general,it"),
            vec![SupplementalQuery::Specialist(focused)]
        );

        let explicit = format!("{query} official documentation");
        let routed = route(&explicit, "general,it");
        let supplemental = supplemental_queries(&explicit, &routed, &[], "general,it");
        assert!(!allows_auxiliary_search(&explicit));
        assert!(matches!(
            supplemental.as_slice(),
            [SupplementalQuery::Source(SourceQuery {
                host: Some(host),
                strict: true,
                ..
            })] if host == "zod.dev"
        ));
    }

    #[test]
    fn expands_primary_source_intent_to_institutional_reports() {
        assert_eq!(
            source_aware_query(
                "primary sources causes of the 2008 financial crisis",
                "primary sources causes of the 2008 financial crisis",
                &[]
            ),
            Some(SourceQuery {
                query: "causes of the 2008 financial crisis archival document transcript".into(),
                host: None,
                strict: true,
            })
        );
        assert!(has_source_intent(
            "primary sources causes of the 2008 financial crisis"
        ));
    }

    #[test]
    fn named_authorities_receive_a_first_party_supplement() {
        assert_eq!(
            source_aware_query(
                "current FDA approved Alzheimer's drugs 2026",
                "FDA approved Alzheimer's drugs 2026",
                &[]
            ),
            Some(SourceQuery {
                query: "FDA approved Alzheimer's drugs 2026 site:fda.gov".into(),
                host: Some("fda.gov".into()),
                strict: false,
            })
        );
        assert_eq!(
            named_authority_host("FDA-approved medication"),
            Some("fda.gov")
        );
        assert_eq!(named_authority_host("cve database"), None);
        assert_eq!(
            named_authority_host("NASA mission status"),
            Some("nasa.gov")
        );
        assert_eq!(
            named_authority_host("WHATWG DOM Standard AbortSignal.any algorithm"),
            Some("dom.spec.whatwg.org")
        );
        assert_eq!(
            named_authority_host("CVE-2024-3094 original oss-security disclosure"),
            Some("openwall.com")
        );
        assert_eq!(
            named_authority_host("Node.js release schedule active LTS"),
            Some("nodejs.org")
        );
        assert_eq!(
            named_authority_host("Next.js hydration error caused by Date.now"),
            Some("nextjs.org")
        );
        assert_eq!(
            named_authority_host("pip error externally-managed-environment"),
            Some("packaging.python.org")
        );
        assert_eq!(
            named_authority_host(
                "PostgreSQL \"cached plan must not change result type\" after ALTER TABLE"
            ),
            Some("postgresql.org")
        );
        assert_eq!(
            named_authority_host(
                "GitHub Actions AWS OIDC \"Not authorized to perform sts:AssumeRoleWithWebIdentity\""
            ),
            Some("docs.aws.amazon.com")
        );
        assert_eq!(
            named_authority_host(
                "Kubernetes pod \"exec user process caused: exec format error\" arm64 amd64"
            ),
            Some("kubernetes.io")
        );
        assert_eq!(
            named_authority_host(
                "TypeScript TS2589 \"Type instantiation is excessively deep\" Zod"
            ),
            Some("zod.dev")
        );
        assert_eq!(named_authority_host("Amazon rainforest rainfall"), None);
        assert_eq!(named_authority_host("Rust game server crash"), None);
        assert_eq!(
            named_authority_host("latest stable Rust release release notes"),
            Some("doc.rust-lang.org")
        );
        assert_eq!(
            named_authority_host("npm registry service status incidents"),
            Some("status.npmjs.org")
        );
        assert_eq!(
            named_authority_host("NIST post-quantum migration guidance"),
            Some("nist.gov")
        );
        assert_eq!(named_authority_host("CDC rabies guidance"), Some("cdc.gov"));
        assert_eq!(
            named_authority_host("U.S. Copyright Office guidance"),
            Some("copyright.gov")
        );
        assert_eq!(
            named_authority_host("NixOS Docker setup"),
            Some("nixos.org")
        );
        assert_eq!(
            named_authority_host("WHO adult physical activity guidelines"),
            Some("who.int")
        );
        assert_eq!(
            named_authority_host("who should use a Pin projection"),
            None
        );
        assert_eq!(
            named_authority_host("US federal funds rate August 2026"),
            Some("federalreserve.gov")
        );
        assert_eq!(
            named_authority_host("Geneva Convention Common Article 3 full text"),
            Some("icrc.org")
        );
        assert_eq!(
            named_authority_host("current US mortgage rate official data"),
            Some("freddiemac.com")
        );
        assert_eq!(named_authority_host("best mortgage rates"), None);
        assert_eq!(
            named_authority_host("current US CPI inflation official data"),
            Some("bls.gov")
        );
        assert_eq!(
            named_authority_host("Attention Is All You Need arXiv paper"),
            Some("arxiv.org")
        );
        assert_eq!(
            named_authority_host("United Nations Charter Article 51 full text"),
            Some("un.org")
        );
        assert_eq!(
            named_authority_host("2026 ACC AHA cholesterol guideline"),
            Some("jacc.org")
        );
        assert_eq!(
            named_authority_host("2026 ADA diabetes standards"),
            Some("diabetesjournals.org")
        );
        assert_eq!(
            named_authority_host("Rust tokio select documentation"),
            Some("docs.rs")
        );
        assert_eq!(named_authority_host("ACC conference schedule"), None);
        assert!(has_source_intent("attention is all you need arxiv paper"));
        assert_eq!(
            source_aware_query(
                "Attention Is All You Need arXiv paper",
                "Attention Is All You Need arXiv paper",
                &[]
            ),
            None
        );
        assert_eq!(
            source_aware_query(
                "What is the US federal funds rate in August 2026?",
                "US federal funds rate in August 2026?",
                &[]
            ),
            Some(SourceQuery {
                query: "US federal funds rate in August 2026? site:federalreserve.gov".into(),
                host: Some("federalreserve.gov".into()),
                strict: false,
            })
        );
        assert_eq!(
            source_aware_query(
                "NixOS Docker NVIDIA container toolkit setup",
                "NixOS Docker NVIDIA container toolkit setup",
                &[]
            ),
            Some(SourceQuery {
                query: "NixOS Docker NVIDIA container toolkit setup site:nixos.org".into(),
                host: Some("nixos.org".into()),
                strict: false,
            })
        );
        assert_eq!(named_authority_host("approved medication"), None);
    }

    #[test]
    fn named_standard_reserves_its_canonical_document_without_becoming_strict() {
        let query = "WHATWG DOM Standard AbortSignal.any() algorithm";
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {
                    "url":"https://github.com/example/abort-controller",
                    "title":"A compliant AbortSignal implementation"
                },
                {
                    "url":"https://dom.spec.whatwg.org/",
                    "title":"DOM Standard"
                }
            ]}"#,
        )
        .unwrap();

        let hits = into_hits(body, query, 8);

        assert!(!has_strict_source_intent(&query.to_ascii_lowercase()));
        assert!(!hits[0].source_priority);
        assert!(hits[1].source_priority);
    }

    #[test]
    fn inferred_diagnostic_authority_remains_a_fallback() {
        let query = "Rust E0277 \"FromResidual is not implemented\" custom error";
        let body: SearxResponse = serde_json::from_str(
            r#"{"results":[{
                "url":"https://doc.rust-lang.org/std/ops/trait.FromResidual.html",
                "title":"FromResidual in std::ops"
            }]}"#,
        )
        .unwrap();

        let hits = into_hits(body, query, 8);

        assert!(!hits[0].source_priority);
    }

    #[test]
    fn original_disclosure_lookup_uses_pre_identifier_subject_terms() {
        let query = "CVE-2024-3094 original oss-security disclosure xz backdoor";
        assert_eq!(focused_source_terms(query), "oss-security xz backdoor");
        assert!(is_cve_identifier("cve-2024-3094"));
        assert!(!is_cve_identifier("cve-2024-xz"));

        assert_eq!(
            source_aware_query(query, query, &[]),
            Some(SourceQuery {
                query: "oss-security xz backdoor site:openwall.com".into(),
                host: Some("openwall.com".into()),
                strict: true,
            })
        );
    }

    #[test]
    fn broadens_full_text_locator_queries_without_losing_the_instrument() {
        assert_eq!(
            full_text_locator_query("Geneva Convention Common Article 3 full text"),
            Some("\"Geneva Convention\" \"Common Article 3\"".into())
        );
        assert_eq!(
            full_text_locator_query("United Nations Charter Article 51 full text"),
            Some("\"United Nations Charter\" \"Article 51\"".into())
        );
        assert_eq!(
            full_text_locator_query("United Nations Charter Article 51 full-text"),
            Some("\"United Nations Charter\" \"Article 51\"".into())
        );
        assert_eq!(
            full_text_locator_query("Common Article 3 Geneva Convention full text"),
            Some("\"Geneva Convention\" \"Common Article 3\"".into())
        );
        assert_eq!(full_text_locator_query("Article 3 full text"), None);

        let route = route(
            "Geneva Convention Common Article 3 full text",
            "general,it,science",
        );
        let body: SearxResponse = serde_json::from_str(r#"{"results":[]}"#).unwrap();
        assert_eq!(
            supplemental_queries(
                "Geneva Convention Common Article 3 full text",
                &route,
                &body.results,
                "general,it,science",
            ),
            vec![SupplementalQuery::FullText(
                "\"Geneva Convention\" \"Common Article 3\"".into()
            )]
        );
    }

    #[test]
    fn compiler_errors_use_a_compact_first_party_query() {
        let results: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://rustfaq.org/e0502","title":"Rust E0502 guide"},
                    {"url":"https://doc.rust-lang.org/error_codes/E0502.html","title":"E0502 - Error codes index - Learn Rust"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            source_aware_query(
                "Rust E0502 cannot borrow mutable because immutable example",
                "Rust E0502 cannot borrow mutable because immutable example",
                &results.results
            ),
            None
        );
        assert_eq!(
            source_aware_query("Rust E0502", "Rust E0502", &[]),
            Some(SourceQuery {
                query: "E0502 site:doc.rust-lang.org".into(),
                host: Some("doc.rust-lang.org".into()),
                strict: false,
            })
        );
        assert_eq!(compiler_error_code("HTTP 404 response"), None);
    }

    #[test]
    fn infers_first_party_documentation_hosts() {
        let results: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://stackoverflow.com/questions/1","title":"React useEffect loop"},
                    {"url":"https://react.dev/reference/react/useEffect","title":"useEffect - React"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            source_aware_query(
                "React useEffect infinite loop official documentation",
                "React useEffect infinite loop",
                &results.results
            ),
            Some(SourceQuery {
                query: "React useEffect infinite loop site:react.dev".into(),
                host: Some("react.dev".into()),
                strict: true,
            })
        );
        assert_eq!(registrable_host("publications.iarc.who.int"), "who.int");
        assert_eq!(registrable_host("docs.nvidia.com"), "nvidia.com");
        assert_eq!(registrable_host("law.ox.ac.uk"), "ox.ac.uk");
        assert_eq!(official_search_host("docs.nvidia.com"), "docs.nvidia.com");
        assert_eq!(official_search_host("download.nvidia.com"), "nvidia.com");
    }

    #[test]
    fn first_party_inference_rejects_branded_third_party_docs() {
        let sqlite: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://coddy.tech/docs/sqlite/wal-mode","title":"SQLite WAL mode and concurrency"},
                    {"url":"https://sqlite.org/index.html","title":"SQLite Home Page"},
                    {"url":"https://sqlite.org/wal.html","title":"Write-Ahead Logging"}
                ]
            }"#,
        )
        .unwrap();
        let qwen: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://qwen3.app/","title":"Qwen3 model guide"},
                    {"url":"https://qwen.ai/blog?id=qwen3-coder","title":"Qwen3-Coder: Agentic Coding in the World"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            infer_official_host(
                "SQLite WAL mode concurrent readers writers",
                &sqlite.results
            ),
            Some("sqlite.org".into())
        );
        assert_eq!(
            infer_official_host("Qwen3 Coder model context length", &qwen.results),
            Some("qwen.ai".into())
        );

        let election: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://industry.visitcalifornia.com/meeting","title":"California 2026 registration"},
                    {"url":"https://www.sos.ca.gov/elections/upcoming-elections/general-election-november-3-2026","title":"California 2026 General Election"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(
            infer_official_host(
                "California 2026 election voter registration deadline",
                &election.results
            ),
            Some("ca.gov".into())
        );
    }

    #[test]
    fn falls_back_to_a_compact_official_docs_query() {
        assert_eq!(
            source_aware_query(
                "React useEffect infinite loop official documentation",
                "React useEffect infinite loop",
                &[]
            ),
            Some(SourceQuery {
                query: "React useEffect infinite loop site:react.dev".into(),
                host: Some("react.dev".into()),
                strict: true,
            })
        );
    }

    #[test]
    fn source_results_promote_all_matching_results_and_discard_ignored_site_filters() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://example.com/react","title":"Generic guide"},
                    {"url":"https://react.dev/reference/react/useEffect","title":"useEffect - React"}
                ]
            }"#,
        )
        .unwrap();
        let supplemental: SearxResponse = serde_json::from_str(
            r#"{
                "results": [
                    {"url":"https://github.com/facebook/react","title":"React source"},
                    {"url":"https://react.dev/learn/synchronizing-with-effects","title":"Synchronizing with Effects"},
                    {"url":"https://react.dev/learn/removing-effect-dependencies","title":"Removing Effect Dependencies"},
                    {"url":"https://react.dev/learn/lifecycle-of-reactive-effects","title":"Lifecycle of Reactive Effects"}
                ]
            }"#,
        )
        .unwrap();

        merge_source_results(&mut primary, supplemental, Some("react.dev"), false);

        assert_eq!(
            primary.results[0].url,
            "https://react.dev/learn/synchronizing-with-effects"
        );
        assert_eq!(primary.results[1].url, "https://example.com/react");
        assert_eq!(
            primary.results[2].url,
            "https://react.dev/learn/removing-effect-dependencies"
        );
        assert_eq!(
            primary.results[3].url,
            "https://react.dev/learn/lifecycle-of-reactive-effects"
        );
        assert_eq!(
            primary.results[4].url,
            "https://react.dev/reference/react/useEffect"
        );
        assert_eq!(primary.results.len(), 5);
    }

    #[test]
    fn strict_source_results_keep_the_independent_pool_but_drop_ignored_site_results() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/react","title":"Generic guide"},
                {"url":"https://react.dev/reference/react/useEffect","title":"useEffect - React"}
            ]}"#,
        )
        .unwrap();
        let supplemental: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://developer.mozilla.org/en-US/docs/Web/JavaScript","title":"Ignored site filter"},
                {"url":"https://react.dev/learn/synchronizing-with-effects","title":"Synchronizing with Effects"}
            ]}"#,
        )
        .unwrap();

        merge_source_results(&mut primary, supplemental, Some("react.dev"), true);

        assert_eq!(primary.results.len(), 3);
        assert_eq!(
            primary.results[0].url,
            "https://react.dev/learn/synchronizing-with-effects"
        );
        assert_eq!(primary.results[1].url, "https://example.com/react");
        assert_eq!(
            primary.results[2].url,
            "https://react.dev/reference/react/useEffect"
        );
    }

    #[test]
    fn source_queries_use_general_engines_when_available() {
        let query = SupplementalQuery::Source(SourceQuery {
            query: "Pin projection site:doc.rust-lang.org".into(),
            host: Some("doc.rust-lang.org".into()),
            strict: true,
        });

        assert_eq!(query.categories("science", "general,it,science"), "general");
        assert_eq!(query.categories("science", "science"), "science");
    }

    #[test]
    fn discovery_results_are_interleaved_before_candidate_truncation() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://primary.example/1"},
                {"url":"https://primary.example/2"},
                {"url":"https://primary.example/3"}
            ]}"#,
        )
        .unwrap();
        let discovery: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://discovery.example/1"},
                {"url":"https://discovery.example/2"}
            ]}"#,
        )
        .unwrap();

        merge_discovery_results(&mut primary, discovery);

        assert_eq!(
            primary
                .results
                .iter()
                .map(|result| result.url.as_str())
                .collect::<Vec<_>>(),
            [
                "https://primary.example/1",
                "https://discovery.example/1",
                "https://primary.example/2",
                "https://discovery.example/2",
                "https://primary.example/3",
            ]
        );
    }

    #[test]
    fn manifestation_lookup_only_adds_its_exact_head() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://primary.example/1"},
                {"url":"https://primary.example/2"}
            ]}"#,
        )
        .unwrap();
        let lookup: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://doi.org/10.1/exact"},
                {"url":"https://doi.org/10.1/adjacent"}
            ]}"#,
        )
        .unwrap();

        merge_manifestation_result(&mut primary, lookup, "exact");

        assert_eq!(
            primary
                .results
                .iter()
                .map(|result| result.url.as_str())
                .collect::<Vec<_>>(),
            [
                "https://primary.example/1",
                "https://doi.org/10.1/exact",
                "https://primary.example/2",
            ]
        );
    }

    #[test]
    fn manifestation_lookup_prefers_accessible_metadata_and_keeps_exact_alternates() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://primary.example/1"},
                {"url":"https://primary.example/2"}
            ]}"#,
        )
        .unwrap();
        let lookup: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {
                    "url":"https://pubmed.ncbi.nlm.nih.gov/1234/",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in Diabetes—2026"
                },
                {
                    "url":"https://doi.org/10.2337/dc26-s010",
                    "title":"10. Cardiovascular Disease and Risk Management: Standards of Care in Diabetes—2026",
                    "doi":"10.2337/dc26-S010",
                    "pdf_url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12690187/"
                },
                {
                    "url":"https://doi.org/10.2337/dc26-s011",
                    "title":"11. Chronic Kidney Disease and Risk Management: Standards of Care in Diabetes—2026",
                    "doi":"10.2337/dc26-S011",
                    "pdf_url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12690188/"
                }
            ]}"#,
        )
        .unwrap();

        merge_manifestation_result(
            &mut primary,
            lookup,
            "10 Cardiovascular Disease and Risk Management Standards of Care in Diabetes 2026",
        );

        assert_eq!(
            primary
                .results
                .iter()
                .map(|result| result.url.as_str())
                .collect::<Vec<_>>(),
            [
                "https://primary.example/1",
                "https://doi.org/10.2337/dc26-s010",
                "https://primary.example/2",
                "https://pubmed.ncbi.nlm.nih.gov/1234/",
            ]
        );
    }

    #[test]
    fn local_results_promote_places_without_discarding_web_results() {
        let mut primary: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://example.com/late-hours","title":"Late-night guide"},
                {"url":"https://example.com/reviews","title":"Local reviews"}
            ]}"#,
        )
        .unwrap();
        let local: SearxResponse = serde_json::from_str(
            r#"{"results":[
                {"url":"https://www.openstreetmap.org/node/1","title":"Cafe 1"},
                {"url":"https://www.openstreetmap.org/node/2","title":"Cafe 2"},
                {"url":"https://www.openstreetmap.org/node/3","title":"Cafe 3"},
                {"url":"https://www.openstreetmap.org/node/4","title":"Cafe 4"},
                {"url":"https://www.openstreetmap.org/node/5","title":"Cafe 5"}
            ]}"#,
        )
        .unwrap();

        merge_local_results(&mut primary, local);

        assert_eq!(primary.results[0].title.as_deref(), Some("Cafe 1"));
        assert_eq!(primary.results[3].title.as_deref(), Some("Cafe 4"));
        assert_eq!(
            primary.results[4].title.as_deref(),
            Some("Late-night guide")
        );
        assert_eq!(primary.results[6].title.as_deref(), Some("Cafe 5"));
    }

    #[test]
    fn exact_identifier_candidates_join_after_the_consensus_head() {
        let make_hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let mut primary = (0..8)
            .map(|index| SearchCandidate {
                hit: make_hit(
                    &format!("Web result {index}"),
                    &format!("https://example.com/{index}"),
                ),
                source_priority: false,
                upstream_consensus: false,
                fetch_urls: vec![Url::parse(&format!("https://example.com/{index}")).unwrap()],
            })
            .collect::<Vec<_>>();
        let auxiliary = vec![
            make_hit(
                "Fix invalid_reference_casting",
                "https://github.com/example/project/issues/1",
            ),
            make_hit(
                "Web result 2",
                "https://github.com/example/project/issues/2",
            ),
        ];

        merge_auxiliary_hits(&mut primary, auxiliary, 8);

        assert_eq!(primary.len(), 8);
        assert_eq!(primary[3].title, "Web result 3");
        assert_eq!(primary[4].title, "Fix invalid_reference_casting");
        assert_eq!(primary[5].title, "Web result 4");
    }

    #[test]
    fn map_addresses_become_fallback_snippets() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [{
                    "url":"https://openstreetmap.org/node/1",
                    "title":"Colectivo Coffee",
                    "content":"",
                    "address": {
                        "house_number":"25",
                        "road":"South Pinckney Street",
                        "locality":"Madison",
                        "postcode":"53703",
                        "country":"United States"
                    }
                }]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, "ordinary query", 1);

        assert_eq!(
            hits[0].snippet,
            "Address: 25 South Pinckney Street, Madison, 53703, United States"
        );
    }

    #[test]
    fn map_results_keep_hours_and_prefer_the_official_website() {
        let body: SearxResponse = serde_json::from_str(
            r#"{
                "results": [{
                    "url":"https://openstreetmap.org/node/1",
                    "title":"Cuppa Austin Coffee",
                    "address": {
                        "house_number":"9225",
                        "road":"West Parmer Lane",
                        "locality":"Austin",
                        "postcode":"78717",
                        "country":"United States"
                    },
                    "data": [{
                        "key":"opening_hours",
                        "label":"open days",
                        "value":"Mo-Fr 06:30-21:00; Sa,Su 07:00-21:00"
                    }],
                    "links": [{
                        "label":"official website",
                        "url":"https://www.cuppaaustin.com/"
                    }]
                }]
            }"#,
        )
        .unwrap();

        let hits = into_hits(body, "ordinary query", 1);

        assert_eq!(hits[0].url.as_str(), "https://www.cuppaaustin.com/");
        assert_eq!(
            hits[0].snippet,
            "Address: 9225 West Parmer Lane, Austin, 78717, United States. Hours: Mo-Fr 06:30-21:00; Sa,Su 07:00-21:00"
        );
    }

    #[test]
    fn requests_selected_categories_without_overriding_engines() {
        let url = search_url_target(
            &Url::parse("http://localhost:8080").unwrap(),
            "axum timeout",
            SearchTarget::Categories("general,it,science,news"),
            None,
        );
        let parameters = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(
            parameters.get("q").map(|value| value.as_ref()),
            Some("axum timeout")
        );
        assert_eq!(
            parameters.get("categories").map(|value| value.as_ref()),
            Some("general,it,science,news")
        );
        assert!(!parameters.contains_key("engines"));
    }

    #[test]
    fn requests_one_explicit_engine_without_category_fanout() {
        let url = search_url_target(
            &Url::parse("http://localhost:8080").unwrap(),
            "Cardiovascular Disease and Risk Management 2026",
            SearchTarget::Engines("openalex"),
            None,
        );
        let parameters = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(
            parameters.get("engines").map(|value| value.as_ref()),
            Some("openalex")
        );
        assert!(!parameters.contains_key("categories"));
    }

    #[test]
    fn requests_freshness_window_without_changing_the_category() {
        let url = search_url_target(
            &Url::parse("http://localhost:8080").unwrap(),
            "latest OpenSSH vulnerability",
            SearchTarget::Categories("general"),
            Some("year"),
        );
        let parameters = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(
            parameters.get("categories").map(|value| value.as_ref()),
            Some("general")
        );
        assert_eq!(
            parameters.get("time_range").map(|value| value.as_ref()),
            Some("year")
        );
    }
}
