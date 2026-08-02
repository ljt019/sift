use anyhow::{Result, ensure};

use super::searxng::Hit;
use crate::embeddings::EmbeddingWorker;

const EMBEDDING_DIMENSIONS: usize = 768;
const CONSENSUS_MAX_SCORE_GAP: f32 = 0.10;

pub async fn select(
    embeddings: &EmbeddingWorker,
    query: &str,
    hits: Vec<Hit>,
    limit: usize,
) -> Result<Vec<Hit>> {
    if hits.len() <= limit {
        return Ok(hits);
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
            let authority = jurisdiction_authority_bonus(query, &hits[index].url)
                + authority_bonus(query, &hits[index].url);
            let recency_adjustment = if recency_intent {
                staleness_penalty(newest_date, hit_date_key(&hits[index]))
            } else {
                0.0
            };
            let date_match_adjustment =
                explicit_date_adjustment(requested_date, hit_date_key(&hits[index]));
            let title_match_adjustment = arxiv_title_adjustment(query, &hits[index]);
            let status_change_adjustment = recent_status_adjustment(query, &hits[index]);
            let score = dense_score
                + authority
                + recency_adjustment
                + date_match_adjustment
                + title_match_adjustment
                + status_change_adjustment;
            ensure!(score.is_finite(), "candidate {index} score is non-finite");
            tracing::debug!(
                url = %hits[index].url,
                date = hits[index].date.as_deref(),
                dense_score,
                authority,
                recency_adjustment,
                date_match_adjustment,
                title_match_adjustment,
                status_change_adjustment,
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

    let selected = if recency_intent {
        scored.iter().take(limit).map(|(index, _)| *index).collect()
    } else {
        select_indices(
            hits.len(),
            &scored,
            limit,
            super::searxng::has_source_intent(&query.to_ascii_lowercase())
                && super::searxng::named_authority_host(query) != Some("arxiv.org"),
        )
    };

    let mut hits = hits.into_iter().map(Some).collect::<Vec<_>>();
    Ok(selected
        .into_iter()
        .map(|index| hits[index].take().expect("candidate indices are unique"))
        .collect())
}

fn arxiv_title_adjustment(query: &str, hit: &Hit) -> f32 {
    let Some(expected) = super::searxng::arxiv_title(query) else {
        return 0.0;
    };
    let host = hit.url.host_str().unwrap_or_default();
    if host != "arxiv.org" && !host.ends_with(".arxiv.org") {
        return 0.0;
    }

    let title = hit
        .title
        .strip_prefix('[')
        .and_then(|title| title.split_once(']').map(|(_, title)| title))
        .unwrap_or(&hit.title)
        .split(" - arXiv")
        .next()
        .unwrap_or(&hit.title);
    if normalized_words(title) == normalized_words(&expected) {
        0.20
    } else {
        0.0
    }
}

fn normalized_words(value: &str) -> String {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn recent_status_adjustment(query: &str, hit: &Hit) -> f32 {
    if !has_recency_intent(query)
        || !query
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| token.eq_ignore_ascii_case("status"))
    {
        return 0.0;
    }

    let candidate = format!("{} {}", hit.title, hit.snippet).to_ascii_lowercase();
    let reports_a_change = [
        "approved",
        "canceled",
        "cancelled",
        "completed",
        "dead",
        "delayed",
        "discontinued",
        "halted",
        "launched",
        "postponed",
        "released",
        "resumed",
        "suspended",
        "terminated",
    ]
    .iter()
    .any(|term| {
        candidate
            .split(|character: char| !character.is_alphanumeric())
            .any(|token| token == *term)
    });

    if reports_a_change { 0.08 } else { 0.0 }
}

fn authority_bonus(query: &str, url: &url::Url) -> f32 {
    let host = url.host_str().unwrap_or_default();
    let named_authority = super::searxng::named_authority_host(query)
        .is_some_and(|authority| host == authority || host.ends_with(&format!(".{authority}")));
    if named_authority {
        return 0.12 + descriptive_path_bonus(query, url);
    }
    let institutional = host.ends_with(".gov")
        || host.ends_with(".edu")
        || host.ends_with(".int")
        || host.ends_with(".ac.uk");
    let research_intent = query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "evidence" | "paper" | "research" | "study"
            )
        });
    let scholarly = host == "arxiv.org"
        || host.ends_with(".arxiv.org")
        || host == "doi.org"
        || host == "dx.doi.org"
        || url.path().contains("/doi/");

    if institutional || (research_intent && scholarly) {
        0.04
    } else {
        0.0
    }
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
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "breaking"
                    | "headline"
                    | "headlines"
                    | "latest"
                    | "news"
                    | "newest"
                    | "recent"
                    | "recently"
            )
        })
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

fn select_indices(
    hit_count: usize,
    dense_ranking: &[(usize, f32)],
    limit: usize,
    preserve_first_source: bool,
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
    let upstream_quota = limit.div_ceil(2);
    let mut selected = (0..upstream_quota.min(hit_count))
        .filter(|&index| {
            (preserve_first_source && index == 0)
                || best_score - scores[index] <= CONSENSUS_MAX_SCORE_GAP
        })
        .collect::<Vec<_>>();
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

#[cfg(test)]
mod tests {
    use super::{
        arxiv_title_adjustment, authority_bonus, date_key, explicit_date_adjustment,
        has_recency_intent, jurisdiction_authority_bonus, query_date_key, recent_status_adjustment,
        select_indices, staleness_penalty, url_date_key,
    };
    use crate::search::searxng::Hit;
    use url::Url;

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
            false,
        );

        assert_eq!(selected, [0, 1, 7, 6]);
    }

    #[test]
    fn exact_arxiv_titles_beat_papers_with_title_variants() {
        let hit = |title: &str, url: &str| Hit {
            title: title.into(),
            url: Url::parse(url).unwrap(),
            date: None,
            snippet: String::new(),
        };
        let query = "Original arXiv paper: Attention Is All You Need";

        assert_eq!(
            arxiv_title_adjustment(
                query,
                &hit(
                    "[1706.03762] Attention Is All You Need - arXiv.org",
                    "https://arxiv.org/abs/1706.03762"
                )
            ),
            0.20
        );
        assert_eq!(
            arxiv_title_adjustment(
                query,
                &hit(
                    "Tool Attention Is All You Need: Dynamic Tool Gating",
                    "https://arxiv.org/abs/2604.21816"
                )
            ),
            0.0
        );
        assert_eq!(
            arxiv_title_adjustment(
                query,
                &hit(
                    "Attention Is All You Need",
                    "https://en.wikipedia.org/wiki/Attention_Is_All_You_Need"
                )
            ),
            0.0
        );
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
            false,
        );

        assert_eq!(selected, [7, 6, 5, 4]);
    }

    #[test]
    fn preserves_a_deliberately_focused_first_party_result() {
        let selected = select_indices(4, &[(3, 0.9), (2, 0.8), (1, 0.7), (0, 0.6)], 2, true);

        assert_eq!(selected, [0, 3]);
    }

    #[test]
    fn recognizes_explicit_recency_without_misclassifying_current_state() {
        assert!(has_recency_intent("latest OpenSSH vulnerability"));
        assert!(has_recency_intent("recent James Webb discovery"));
        assert!(has_recency_intent("OpenAI news"));
        assert!(!has_recency_intent("current federal funds rate"));
        assert!(!has_recency_intent("water temperature today"));
    }

    #[test]
    fn authority_bonus_is_narrow_and_query_aware() {
        let nvd = Url::parse("https://nvd.nist.gov/vuln/detail/CVE-2026-1").unwrap();
        let paper = Url::parse("https://www.pnas.org/doi/10.1073/example").unwrap();
        let doi = Url::parse("https://doi.org/10.1038/example").unwrap();
        let blog = Url::parse("https://example.com/research/evidence").unwrap();
        let nasa = Url::parse("https://www.nasa.gov/mission/artemis-ii/").unwrap();
        let nasa_event = Url::parse("https://www.nasa.gov/event/artemis-ii-launch/").unwrap();
        let nasa_archive = Url::parse("https://ntrs.nasa.gov/citations/20260004348").unwrap();

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
        assert_eq!(authority_bonus("latest release", &paper), 0.0);
        assert_eq!(authority_bonus("latest research evidence", &blog), 0.0);
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
    fn recent_status_queries_favor_results_that_report_a_change() {
        let hit = |title: &str, snippet: &str| Hit {
            title: title.into(),
            url: Url::parse("https://example.com").unwrap(),
            date: None,
            snippet: snippet.into(),
        };

        assert_eq!(
            recent_status_adjustment(
                "recent Mars mission status",
                &hit("NASA mission is dead", "Congress canceled the program")
            ),
            0.08
        );
        assert_eq!(
            recent_status_adjustment(
                "recent Mars mission status",
                &hit("Mars mission overview", "The mission is proposed")
            ),
            0.0
        );
        assert_eq!(
            recent_status_adjustment(
                "Mars mission status",
                &hit("NASA mission delayed", "A new schedule is expected")
            ),
            0.0
        );
        assert_eq!(
            recent_status_adjustment(
                "recent Mars mission news",
                &hit("NASA mission delayed", "A new schedule is expected")
            ),
            0.0
        );
    }
}
