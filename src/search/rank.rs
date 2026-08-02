use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail, ensure};

use super::Raw;
use super::chunk::{self, Chunk};
use super::evidence::QueryEvidence;
use crate::embeddings::{DOCUMENT_CHUNK_TOKENS, EmbeddingWorker};

const EMBEDDING_DIMENSIONS: usize = 768;
const CHUNK_OVERLAP_TOKENS: usize = 64;
const MAX_CANDIDATE_CHUNKS_PER_DOCUMENT: usize = 8;
const MAX_CHUNKS: usize = 24;
const MAX_CHUNKS_PER_DOCUMENT: usize = 3;
const CONTENT_CONSENSUS_MAX_SCORE_GAP: f32 = 0.10;
const LEXICAL_EVIDENCE_WEIGHT: f32 = 0.10;
const RECOMMENDATION_EVIDENCE_WEIGHT: f32 = 0.12;
const MAX_NUMERIC_CONSENSUS_BONUS: f32 = 0.05;
const MIN_NUMERIC_CONSENSUS_DOMAINS: usize = 3;
const MAX_SEARCH_EXCERPT_CHARACTERS: usize = 800;
const EVIDENCE_ANCHOR_MAX_SCORE_GAP: f32 = 0.10;
const NUMERICAL_PARITY_MISSING_CONTEXT_PENALTY: f32 = 0.08;
const NUMERICAL_PARITY_ISSUE_MAX_SCORE_GAP: f32 = 0.20;
const DIVERSITY_MAX_SCORE_GAP: f32 = 0.04;
const DIVERSITY_MIN_SHARED_TITLE_TERMS: usize = 3;
const DIVERSITY_TITLE_CONTAINMENT_PERCENT: usize = 60;

pub struct RankedDocument {
    pub raw: Raw,
    pub content: String,
    pub truncated: bool,
}

#[derive(Clone, Copy)]
struct ChunkSelectionLimits {
    total: usize,
    per_document: usize,
}

pub async fn select(
    embeddings: &EmbeddingWorker,
    query: &str,
    raw: Vec<Raw>,
    document_limit: usize,
) -> Result<Vec<RankedDocument>> {
    if document_limit == 0 {
        return Ok(Vec::new());
    }

    let evidence_documents = raw
        .iter()
        .map(|raw| format!("{}\n{}\n{}", raw.hit.title, raw.hit.snippet, raw.content))
        .collect::<Vec<_>>();
    let evidence = QueryEvidence::new(query, evidence_documents.iter().map(String::as_str));
    let numeric_consensus = NumericConsensus::new(query, &raw);
    let numerical_parity_adjustments = evidence_documents
        .iter()
        .map(|document| numerical_parity_content_adjustment(query, document))
        .collect::<Vec<_>>();
    let chunks = build_chunks(embeddings, &evidence, &raw)?;

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let dense_scores = score_dense(embeddings, query, &raw, &chunks).await?;
    let mut scores = dense_scores
        .into_iter()
        .enumerate()
        .map(|(chunk, dense_score)| {
            (
                chunk,
                combined_score(dense_score, evidence.score(&chunks[chunk].text))
                    + super::candidate::source_authority_adjustment(
                        query,
                        &raw[chunks[chunk].document].hit.url,
                    )
                    + super::candidate::case_study_form_adjustment(
                        query,
                        &raw[chunks[chunk].document].hit,
                    )
                    + RECOMMENDATION_EVIDENCE_WEIGHT
                        * recommendation_score(&evidence, &raw, &chunks[chunk])
                    + numeric_consensus.score_text(&chunks[chunk].text)
                    + evidence.year_adjustment(&chunks[chunk].text)
                    + numerical_parity_adjustments[chunks[chunk].document],
            )
        })
        .collect::<Vec<_>>();
    apply_document_evidence(&raw, &chunks, &evidence, &numeric_consensus, &mut scores);
    let selected_documents = select_documents(
        query,
        &raw,
        &chunks,
        &scores,
        document_limit.min(raw.len()),
        &evidence,
        &numeric_consensus,
    )?;
    let selected_set = selected_documents.iter().copied().collect::<HashSet<_>>();
    scores.retain(|(chunk, _)| selected_set.contains(&chunks[*chunk].document));

    select_scored(
        raw,
        chunks,
        scores,
        &selected_documents,
        ChunkSelectionLimits {
            total: MAX_CHUNKS,
            per_document: MAX_CHUNKS_PER_DOCUMENT,
        },
        &evidence,
        &numeric_consensus,
    )
}

fn combined_score(dense_score: f32, evidence_score: f32) -> f32 {
    dense_score + LEXICAL_EVIDENCE_WEIGHT * evidence_score
}

fn numerical_parity_content_adjustment(query: &str, document: &str) -> f32 {
    if !super::evidence::has_numerical_parity_intent(query)
        || !super::evidence::has_machine_learning_inference_intent(query)
    {
        return 0.0;
    }

    let lower = document.to_ascii_lowercase();
    let has_machine_learning = contains_machine_learning_context(&lower);
    let has_discrepancy = contains_numerical_discrepancy(&lower);
    -NUMERICAL_PARITY_MISSING_CONTEXT_PENALTY
        * (!has_machine_learning as u8 + !has_discrepancy as u8) as f32
}

fn contains_machine_learning_context(lower: &str) -> bool {
    lower.contains("inference")
        || lower.contains("machine learning")
        || lower.contains("deep learning")
        || lower.contains("neural")
        || lower.contains("ai model")
        || lower.contains("language model")
        || lower.contains("transformer")
        || lower.contains("pytorch")
        || lower.contains("tensorflow")
}

fn contains_numerical_discrepancy(lower: &str) -> bool {
    [
        "corrupt",
        "deviat",
        "differ",
        "diverg",
        "drift",
        "inconsisten",
        "incorrect",
        "mismatch",
        "nondetermin",
        "non-determin",
        "reproduc",
        "tolerance",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn apply_document_evidence(
    raw: &[Raw],
    chunks: &[Chunk],
    evidence: &QueryEvidence,
    numeric_consensus: &NumericConsensus,
    scores: &mut [(usize, f32)],
) {
    let mut best_by_document = HashMap::<usize, usize>::new();
    for (position, &(chunk, score)) in scores.iter().enumerate() {
        let document = chunks[chunk].document;
        best_by_document
            .entry(document)
            .and_modify(|best| {
                if score > scores[*best].1 {
                    *best = position;
                }
            })
            .or_insert(position);
    }

    for (document, position) in best_by_document {
        let excerpt_score = evidence.score(&format!(
            "{}\n{}",
            raw[document].hit.title, raw[document].hit.snippet
        ));
        let chunk_score = evidence.score(&chunks[scores[position].0].text);
        if excerpt_score > chunk_score {
            scores[position].1 += LEXICAL_EVIDENCE_WEIGHT * (excerpt_score - chunk_score);
        }
        scores[position].1 += (numeric_consensus.score(document)
            - numeric_consensus.score_text(&chunks[scores[position].0].text))
        .max(0.0);
    }
}

fn recommendation_score(evidence: &QueryEvidence, raw: &[Raw], chunk: &Chunk) -> f32 {
    evidence.recommendation_score(&format!(
        "{}\n{}\n{}",
        raw[chunk.document].hit.title, raw[chunk.document].hit.snippet, chunk.text
    ))
}

#[derive(Debug)]
struct NumericConsensus {
    scores: Vec<f32>,
    fact_scores: HashMap<i32, f32>,
    context: Option<NumericQueryContext>,
}

impl NumericConsensus {
    fn new(query: &str, raw: &[Raw]) -> Self {
        let Some(context) = NumericQueryContext::new(query) else {
            return Self {
                scores: vec![0.0; raw.len()],
                fact_scores: HashMap::new(),
                context: None,
            };
        };

        let facts = raw
            .iter()
            .map(|raw| context.percentage_facts(&format!("{}\n{}", raw.hit.title, raw.hit.snippet)))
            .collect::<Vec<_>>();
        let mut support = HashMap::<i32, HashSet<String>>::new();
        for (document, document_facts) in facts.iter().enumerate() {
            let Some(host) = raw[document].hit.url.host_str() else {
                continue;
            };
            let root = super::searxng::registrable_host(host);
            for &fact in document_facts {
                support.entry(fact).or_default().insert(root.clone());
            }
        }
        let max_support = support.values().map(HashSet::len).max().unwrap_or_default();
        let fact_scores = support
            .iter()
            .filter(|(_, roots)| roots.len() >= MIN_NUMERIC_CONSENSUS_DOMAINS)
            .map(|(&fact, roots)| {
                (
                    fact,
                    MAX_NUMERIC_CONSENSUS_BONUS * roots.len() as f32 / max_support as f32,
                )
            })
            .collect::<HashMap<_, _>>();
        let scores = facts
            .iter()
            .map(|document| {
                document
                    .iter()
                    .filter_map(|fact| fact_scores.get(fact))
                    .copied()
                    .max_by(f32::total_cmp)
                    .unwrap_or_default()
            })
            .collect();
        Self {
            scores,
            fact_scores,
            context: Some(context),
        }
    }

    fn score(&self, document: usize) -> f32 {
        self.scores.get(document).copied().unwrap_or_default()
    }

    fn score_text(&self, text: &str) -> f32 {
        self.context
            .as_ref()
            .map(|context| context.percentage_facts(text))
            .unwrap_or_default()
            .iter()
            .filter_map(|fact| self.fact_scores.get(fact))
            .copied()
            .max_by(f32::total_cmp)
            .unwrap_or_default()
    }

    fn has_consensus(&self) -> bool {
        !self.fact_scores.is_empty()
    }

    fn adds_evidence(&self, excerpt: &str, content: &str) -> bool {
        let Some(context) = &self.context else {
            return false;
        };
        let excerpt_facts = context.percentage_facts(excerpt);
        let content_facts = context.percentage_facts(content);
        self.fact_scores
            .iter()
            .any(|(fact, _)| excerpt_facts.contains(fact) && !content_facts.contains(fact))
    }
}

#[derive(Debug)]
struct NumericQueryContext {
    topic_terms: HashSet<String>,
    relation_terms: HashSet<&'static str>,
}

impl NumericQueryContext {
    fn new(query: &str) -> Option<Self> {
        let query_terms = normalized_numeric_terms(query).collect::<HashSet<_>>();
        let explicit_percentage = query_terms.contains("percent")
            || query_terms.contains("percentage")
            || query.contains('%');
        let has_percentage_measure = query_terms.iter().any(|term| {
            matches!(
                term.as_str(),
                "apr"
                    | "growth"
                    | "inflation"
                    | "rate"
                    | "rates"
                    | "unemployment"
                    | "yield"
                    | "yields"
            )
        });
        if !explicit_percentage && !has_percentage_measure {
            return None;
        }

        let mut relation_terms = HashSet::new();
        for term in &query_terms {
            match term.as_str() {
                "rate" | "rates" | "apr" | "interest" => {
                    relation_terms.extend(["rate", "rates", "apr", "apy", "interest"]);
                }
                "inflation" | "cpi" => {
                    relation_terms.extend([
                        "inflation",
                        "inflationary",
                        "cpi",
                        "price",
                        "prices",
                        "rose",
                        "risen",
                    ]);
                }
                "growth" => {
                    relation_terms.extend(["growth", "grew", "grown", "expanded"]);
                }
                "unemployment" => {
                    relation_terms.extend(["unemployment", "jobless", "joblessness"]);
                }
                "yield" | "yields" => {
                    relation_terms.extend(["yield", "yields"]);
                }
                _ => {}
            }
        }

        let topic_terms = query_terms
            .into_iter()
            .filter(|term| is_numeric_topic_term(term))
            .collect();
        Some(Self {
            topic_terms,
            relation_terms,
        })
    }

    fn percentage_facts(&self, text: &str) -> HashSet<i32> {
        let words = text.split_whitespace().collect::<Vec<_>>();
        let mut facts = HashSet::new();
        for (index, word) in words.iter().enumerate() {
            let Some(fact) = percentage_fact(&words, index, word) else {
                continue;
            };
            let relation_start = index.saturating_sub(6);
            let relation_end = (index + 7).min(words.len());
            let relation_context = normalized_word_slice(&words[relation_start..relation_end]);
            if !self.relation_terms.is_empty()
                && !relation_context
                    .iter()
                    .any(|term| self.relation_terms.contains(term.as_str()))
            {
                continue;
            }

            let topic_start = index.saturating_sub(14);
            let topic_end = (index + 15).min(words.len());
            let topic_context = normalized_word_slice(&words[topic_start..topic_end]);
            if !self.topic_terms.is_empty()
                && !topic_context
                    .iter()
                    .any(|term| self.topic_terms.contains(term))
            {
                continue;
            }
            if is_unrequested_down_payment(&relation_context, &self.topic_terms) {
                continue;
            }
            facts.insert(fact);
        }
        facts
    }
}

fn percentage_fact(words: &[&str], index: usize, word: &str) -> Option<i32> {
    let word = word
        .trim_matches(|character: char| {
            !(character.is_ascii_digit() || matches!(character, '.' | '-' | '+' | '%'))
        })
        .trim_end_matches('.');
    let (number, explicit_percent) = match word.strip_suffix('%') {
        Some(number) => (number.trim_end_matches('.'), true),
        None => (word, false),
    };
    let followed_by_percent = words.get(index + 1).is_some_and(|word| {
        matches!(
            word.trim_matches(|character: char| !character.is_alphabetic())
                .to_ascii_lowercase()
                .as_str(),
            "percent" | "percentage"
        )
    });
    if !(explicit_percent || followed_by_percent) {
        return None;
    }
    let value = number.parse::<f32>().ok()?;
    if value.is_finite() && value.abs() <= 100.0 {
        Some((value * 1_000.0).round() as i32)
    } else {
        None
    }
}

fn normalized_numeric_terms(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_ascii_lowercase)
}

fn normalized_word_slice(words: &[&str]) -> HashSet<String> {
    words
        .iter()
        .flat_map(|word| normalized_numeric_terms(word))
        .collect()
}

fn is_numeric_topic_term(term: &str) -> bool {
    term.len() > 2
        && !term.chars().all(|character| character.is_ascii_digit())
        && !matches!(
            term,
            "about"
                | "and"
                | "annual"
                | "apr"
                | "are"
                | "average"
                | "change"
                | "changes"
                | "cpi"
                | "current"
                | "did"
                | "does"
                | "for"
                | "from"
                | "growth"
                | "how"
                | "inflation"
                | "interest"
                | "latest"
                | "monthly"
                | "much"
                | "now"
                | "percent"
                | "percentage"
                | "price"
                | "prices"
                | "rate"
                | "rates"
                | "than"
                | "that"
                | "the"
                | "their"
                | "this"
                | "today"
                | "unemployment"
                | "was"
                | "were"
                | "what"
                | "when"
                | "where"
                | "which"
                | "with"
                | "year"
                | "yield"
                | "yields"
        )
}

fn is_unrequested_down_payment(context: &HashSet<String>, query_topics: &HashSet<String>) -> bool {
    context.contains("down") && context.contains("payment") && !query_topics.contains("payment")
}

#[allow(clippy::too_many_arguments)]
fn evidence_anchor_document(
    query: &str,
    raw: &[Raw],
    best_by_document: &[Option<f32>],
    best_score: f32,
    evidence: &QueryEvidence,
    exact_matches: &[bool],
    prioritize_exact: bool,
    require_extracted: bool,
    require_recommendation: bool,
    recommendation_scores: &[f32],
) -> Option<usize> {
    let research_intent = super::evidence::has_research_source_intent(query);

    raw.iter()
        .enumerate()
        .filter_map(|(document, raw)| {
            let score = best_by_document[document]?;
            if best_score - score > EVIDENCE_ANCHOR_MAX_SCORE_GAP
                || prioritize_exact && !exact_matches[document]
                || require_extracted && raw.from_snippet
                || require_recommendation && recommendation_scores[document] == 0.0
            {
                return None;
            }
            let case_study = super::candidate::case_study_form_adjustment(query, &raw.hit) > 0.0;
            let research = research_intent && super::candidate::is_scholarly_url(&raw.hit.url);
            if !case_study && !research {
                return None;
            }
            let text = format!("{}\n{}\n{}", raw.hit.title, raw.hit.snippet, raw.content);
            Some((document, evidence.score(&text), score))
        })
        .max_by(
            |(left_document, left_evidence, left_score),
             (right_document, right_evidence, right_score)| {
                left_score
                    .total_cmp(right_score)
                    .then_with(|| left_evidence.total_cmp(right_evidence))
                    .then_with(|| right_document.cmp(left_document))
            },
        )
        .map(|(document, _, _)| document)
}

fn numerical_parity_issue_document(
    query: &str,
    raw: &[Raw],
    best_by_document: &[Option<f32>],
    best_score: f32,
) -> Option<usize> {
    if !super::evidence::has_numerical_parity_intent(query)
        || !super::evidence::has_machine_learning_inference_intent(query)
    {
        return None;
    }

    raw.iter()
        .enumerate()
        .filter_map(|(document, raw)| {
            let score = best_by_document[document]?;
            if best_score - score > NUMERICAL_PARITY_ISSUE_MAX_SCORE_GAP
                || !is_github_issue(&raw.hit.url)
            {
                return None;
            }
            let text = format!("{}\n{}\n{}", raw.hit.title, raw.hit.snippet, raw.content);
            let lower = text.to_ascii_lowercase();
            if numerical_parity_content_adjustment(query, &text) != 0.0
                || !contains_token(&lower, "cpu")
                || !contains_gpu_backend(&lower)
            {
                return None;
            }
            Some((document, parity_issue_title_strength(&raw.hit.title), score))
        })
        .max_by(
            |(left_document, left_strength, left_score),
             (right_document, right_strength, right_score)| {
                left_strength
                    .cmp(right_strength)
                    .then_with(|| left_score.total_cmp(right_score))
                    .then_with(|| right_document.cmp(left_document))
            },
        )
        .map(|(document, _, _)| document)
}

fn is_github_issue(url: &url::Url) -> bool {
    url.host_str().is_some_and(|host| host == "github.com")
        && url
            .path_segments()
            .is_some_and(|segments| segments.collect::<Vec<_>>().get(2) == Some(&"issues"))
}

fn parity_issue_title_strength(title: &str) -> u8 {
    let lower = title.to_ascii_lowercase();
    u8::from(contains_gpu_backend(&lower)) * 2
        + u8::from(contains_numerical_discrepancy(&lower)) * 2
        + u8::from(contains_machine_learning_context(&lower))
        + u8::from(contains_token(&lower, "cpu"))
}

fn contains_gpu_backend(lower: &str) -> bool {
    ["cuda", "gpu", "metal", "mps", "opencl", "rocm", "tpu"]
        .iter()
        .any(|backend| contains_token(lower, backend))
}

fn contains_token(lower: &str, needle: &str) -> bool {
    lower
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| token == needle)
}

fn source_identity(raw: &Raw) -> Option<String> {
    let url = &raw.hit.url;
    let host = url
        .host_str()?
        .trim_start_matches("www.")
        .to_ascii_lowercase();
    let segments = url
        .path_segments()
        .into_iter()
        .flatten()
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();

    let scoped = match (host.as_str(), segments.as_slice()) {
        ("github.com" | "gitlab.com" | "codeberg.org", [owner, repository, ..])
            if !matches!(*owner, "orgs" | "search" | "topics" | "users") =>
        {
            Some(format!(
                "{host}/{}/{}",
                owner.to_ascii_lowercase(),
                repository.trim_end_matches(".git").to_ascii_lowercase()
            ))
        }
        ("docs.rs", ["crate", package, ..])
        | ("crates.io" | "lib.rs", ["crates", package, ..])
        | ("docs.rs", [package, ..]) => Some(format!("{host}/{}", package.to_ascii_lowercase())),
        _ => None,
    };

    scoped.or_else(|| Some(super::searxng::registrable_host(&host)))
}

fn title_subject_terms(title: &str) -> HashSet<String> {
    if url::Url::parse(title).is_ok() {
        return HashSet::new();
    }

    let subject = [" | ", " — ", " - "]
        .into_iter()
        .fold(title, |subject, separator| {
            subject
                .rsplit_once(separator)
                .map_or(subject, |(head, _)| head)
        });
    subject
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|term| term.len() >= 2)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "a" | "an"
                    | "and"
                    | "against"
                    | "at"
                    | "by"
                    | "for"
                    | "from"
                    | "how"
                    | "in"
                    | "into"
                    | "of"
                    | "on"
                    | "or"
                    | "the"
                    | "to"
                    | "using"
                    | "with"
            )
        })
        .collect()
}

fn same_source_subject(left: &Raw, right: &Raw) -> bool {
    let (Some(left_source), Some(right_source)) = (source_identity(left), source_identity(right))
    else {
        return false;
    };
    if left_source != right_source {
        return false;
    }
    let left = title_subject_terms(&left.hit.title);
    let right = title_subject_terms(&right.hit.title);
    let shared = left.intersection(&right).count();
    let smaller = left.len().min(right.len());
    shared >= DIVERSITY_MIN_SHARED_TITLE_TERMS
        && shared * 100 >= smaller * DIVERSITY_TITLE_CONTAINMENT_PERCENT
}

#[allow(clippy::too_many_arguments)]
fn diversify_documents(
    raw: &[Raw],
    mut selected: Vec<usize>,
    dense_ranking: &[(usize, f32)],
    best_by_document: &[Option<f32>],
    exact_matches: &[bool],
    evidence_anchor: Option<usize>,
    numerical_parity_issue: Option<usize>,
    require_extracted: bool,
    require_recommendation: bool,
    recommendation_scores: &[f32],
) -> Vec<usize> {
    let mut selected_set = selected.iter().copied().collect::<HashSet<_>>();
    for position in 0..selected.len() {
        let document = selected[position];
        let protected = raw[document].source_priority
            || exact_matches[document]
            || evidence_anchor == Some(document)
            || numerical_parity_issue == Some(document);
        if protected
            || !selected[..position]
                .iter()
                .any(|previous| same_source_subject(&raw[*previous], &raw[document]))
        {
            continue;
        }
        let Some(score) = best_by_document[document] else {
            continue;
        };
        let alternative = dense_ranking
            .iter()
            .find_map(|(candidate, candidate_score)| {
                if selected_set.contains(candidate)
                    || score - candidate_score > DIVERSITY_MAX_SCORE_GAP
                    || require_extracted && raw[*candidate].from_snippet
                    || require_recommendation && recommendation_scores[*candidate] == 0.0
                    || selected
                        .iter()
                        .enumerate()
                        .any(|(selected_position, selected)| {
                            selected_position != position
                                && same_source_subject(&raw[*selected], &raw[*candidate])
                        })
                {
                    None
                } else {
                    Some(*candidate)
                }
            });
        if let Some(alternative) = alternative {
            selected_set.remove(&document);
            selected_set.insert(alternative);
            selected[position] = alternative;
        }
    }
    selected
}

/// Chooses documents by their best answer-bearing passage, while retaining
/// upstream head results that are close enough to the best content score.
/// Requested sources, exact documents, and an explicitly requested evidence
/// form lead the otherwise-upstream order. Passage ranking uses the same chunk
/// scores, so full-content selection does not require a second forward pass.
fn select_documents(
    query: &str,
    raw: &[Raw],
    chunks: &[Chunk],
    scores: &[(usize, f32)],
    limit: usize,
    evidence: &QueryEvidence,
    numeric_consensus: &NumericConsensus,
) -> Result<Vec<usize>> {
    let document_count = raw.len();
    if limit == 0 || document_count == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        scores.iter().all(|(index, _)| *index < chunks.len()),
        "chunk score index is out of range"
    );

    let mut best_by_document = vec![None::<f32>; document_count];
    for &(chunk, score) in scores {
        let document = chunks[chunk].document;
        ensure!(
            document < document_count,
            "chunk document index is out of range"
        );
        let best = &mut best_by_document[document];
        *best = Some(best.map_or(score, |current| current.max(score)));
    }

    let mut recommendation_scores = vec![0.0_f32; raw.len()];
    for chunk in chunks {
        recommendation_scores[chunk.document] =
            recommendation_scores[chunk.document].max(recommendation_score(evidence, raw, chunk));
    }
    let numeric_scores = (0..raw.len())
        .map(|document| numeric_consensus.score(document))
        .collect::<Vec<_>>();
    let exact_matches = raw
        .iter()
        .map(|document| super::candidate::exact_document_match(query, &document.hit))
        .collect::<Vec<_>>();
    let prioritize_exact = exact_matches.iter().any(|matches| *matches);
    // Repeated numeric facts are only a bounded score adjustment. They never
    // become a categorical ordering or eligibility rule: consensus can be
    // confidently repeated while still answering a different question.
    let has_numeric_consensus = numeric_consensus.has_consensus();
    let prioritize_recommendations = evidence.has_recommendation_intent()
        && recommendation_scores.iter().any(|score| *score > 0.0);
    let prioritize_extracted =
        evidence.has_full_text_intent() && raw.iter().any(|document| !document.from_snippet);
    let require_extracted = prioritize_extracted
        && (!prioritize_exact
            || exact_matches
                .iter()
                .enumerate()
                .any(|(document, exact)| *exact && !raw[document].from_snippet));
    let has_joint_priority = prioritize_recommendations
        && has_numeric_consensus
        && recommendation_scores
            .iter()
            .enumerate()
            .any(|(document, score)| *score > 0.0 && numeric_scores[document] > 0.0);
    let require_recommendation = !prioritize_extracted
        && prioritize_recommendations
        && (!has_numeric_consensus || has_joint_priority);
    let mut dense_ranking = best_by_document
        .iter()
        .enumerate()
        .filter_map(|(document, score)| score.map(|score| (document, score)))
        .collect::<Vec<_>>();
    dense_ranking.sort_by(
        |(left_document, left_score), (right_document, right_score)| {
            let exact_order = if prioritize_exact {
                exact_matches[*right_document].cmp(&exact_matches[*left_document])
            } else {
                std::cmp::Ordering::Equal
            };
            let extraction_order = if prioritize_extracted {
                raw[*left_document]
                    .from_snippet
                    .cmp(&raw[*right_document].from_snippet)
            } else {
                std::cmp::Ordering::Equal
            };
            let recommendation_order = if prioritize_recommendations {
                recommendation_scores[*right_document]
                    .total_cmp(&recommendation_scores[*left_document])
            } else {
                std::cmp::Ordering::Equal
            };
            exact_order
                .then(extraction_order)
                .then(recommendation_order)
                .then_with(|| right_score.total_cmp(left_score))
                .then_with(|| left_document.cmp(right_document))
        },
    );
    let Some(best_score) = dense_ranking
        .iter()
        .filter(|(document, _)| !prioritize_exact || exact_matches[*document])
        .filter(|(document, _)| !require_extracted || !raw[*document].from_snippet)
        .filter(|(document, _)| !require_recommendation || recommendation_scores[*document] > 0.0)
        .map(|(_, score)| *score)
        .max_by(f32::total_cmp)
    else {
        return Ok(Vec::new());
    };
    let evidence_anchor = evidence_anchor_document(
        query,
        raw,
        &best_by_document,
        best_score,
        evidence,
        &exact_matches,
        prioritize_exact,
        require_extracted,
        require_recommendation,
        &recommendation_scores,
    );
    let numerical_parity_issue =
        numerical_parity_issue_document(query, raw, &best_by_document, best_score);

    // `source_priority` is only set for an explicitly requested source whose
    // host was verified. Reserve one chunkable source regardless of its dense
    // score; all additional first-party pages compete normally.
    let mut selected = dense_ranking
        .iter()
        .filter(|(document, _)| raw[*document].source_priority)
        .take(1)
        .map(|(document, _)| *document)
        .collect::<Vec<_>>();
    if let Some(anchor) = evidence_anchor
        && selected.len() < limit
        && !selected.contains(&anchor)
    {
        selected.push(anchor);
    }
    if let Some(issue) = numerical_parity_issue
        && selected.len() < limit
        && !selected.contains(&issue)
    {
        selected.push(issue);
    }
    let upstream_quota = limit.saturating_sub(selected.len()).div_ceil(2);
    let upstream = (0..upstream_quota.min(document_count))
        .filter(|&document| {
            !selected.contains(&document)
                && raw[document].upstream_consensus
                && (!prioritize_exact || exact_matches[document])
                && (!require_extracted || !raw[document].from_snippet)
                && (!require_recommendation || recommendation_scores[document] > 0.0)
                && best_by_document[document]
                    .is_some_and(|score| best_score - score <= CONTENT_CONSENSUS_MAX_SCORE_GAP)
        })
        .collect::<Vec<_>>();
    selected.extend(upstream);
    for &(document, _) in &dense_ranking {
        if selected.len() == limit {
            break;
        }
        if !selected.contains(&document) {
            selected.push(document);
        }
    }
    // Keep the requested source first so even a deliberately tiny response
    // budget exposes it instead of silently spending the whole budget on a
    // later third-party result.
    selected.sort_by_key(|document| {
        (
            !raw[*document].source_priority,
            !exact_matches[*document],
            evidence_anchor != Some(*document),
            *document,
        )
    });
    let mut selected = diversify_documents(
        raw,
        selected,
        &dense_ranking,
        &best_by_document,
        &exact_matches,
        evidence_anchor,
        numerical_parity_issue,
        require_extracted,
        require_recommendation,
        &recommendation_scores,
    );
    selected.sort_by_key(|document| {
        (
            !raw[*document].source_priority,
            !exact_matches[*document],
            evidence_anchor != Some(*document),
            *document,
        )
    });

    for (document, score) in best_by_document.iter().enumerate() {
        if let Some(score) = score {
            tracing::debug!(
                url = %raw[document].hit.url,
                source_priority = raw[document].source_priority,
                upstream_consensus = raw[document].upstream_consensus,
                from_snippet = raw[document].from_snippet,
                extracted_priority = prioritize_extracted && !raw[document].from_snippet,
                exact_document = exact_matches[document],
                recommendation_evidence = recommendation_scores[document],
                numeric_consensus = numeric_scores[document],
                evidence_anchor = evidence_anchor == Some(document),
                numerical_parity_issue = numerical_parity_issue == Some(document),
                score,
                selected = selected.contains(&document),
                "scored resolved search result"
            );
        }
    }

    Ok(selected)
}

fn build_chunks(
    embeddings: &EmbeddingWorker,
    evidence: &QueryEvidence,
    raw: &[Raw],
) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    for (document, raw) in raw.iter().enumerate() {
        let spans =
            embeddings.document_spans(&raw.content, DOCUMENT_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)?;
        let document_chunks = chunk::split(document, &raw.content, &spans)?;
        for chunk in &document_chunks {
            let input_tokens = embeddings.document_input_tokens(&chunk.text)?;
            ensure!(
                input_tokens <= DOCUMENT_CHUNK_TOKENS,
                "chunk tokenizer produced a {input_tokens}-token document input from {} content tokens, exceeding the {}-token limit",
                chunk.tokens,
                DOCUMENT_CHUNK_TOKENS
            );
        }
        chunks.extend(limit_candidates(
            document_chunks,
            MAX_CANDIDATE_CHUNKS_PER_DOCUMENT,
            evidence,
        ));
    }

    Ok(chunks)
}

pub(super) async fn score_dense(
    embeddings: &EmbeddingWorker,
    query: &str,
    raw: &[Raw],
    chunks: &[Chunk],
) -> Result<Vec<f32>> {
    // Query and documents use different EmbeddingGemma task prompts, so they
    // intentionally remain two worker submissions. All document chunks share
    // one submission and are split into device batches inside the worker.
    let query_embeddings = embeddings
        .embed_queries(vec![query.to_owned()], EMBEDDING_DIMENSIONS)
        .await?;
    ensure!(
        query_embeddings.len() == 1,
        "embedder returned {} query vectors",
        query_embeddings.len()
    );
    let document_embeddings = embeddings
        .embed_documents(
            chunks.iter().map(|chunk| chunk.text.clone()).collect(),
            EMBEDDING_DIMENSIONS,
        )
        .await?;
    ensure!(
        document_embeddings.len() == chunks.len(),
        "embedder returned {} vectors for {} chunks",
        document_embeddings.len(),
        chunks.len()
    );

    let query_embedding = &query_embeddings[0];
    ensure!(
        query_embedding.len() == EMBEDDING_DIMENSIONS,
        "query embedding has {} dimensions, expected {EMBEDDING_DIMENSIONS}",
        query_embedding.len()
    );
    if let Some((dimension, value)) = query_embedding
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("query embedding is non-finite at dimension {dimension}: {value}");
    }

    let mut scores = Vec::with_capacity(chunks.len());
    for (index, embedding) in document_embeddings.iter().enumerate() {
        ensure!(
            embedding.len() == query_embedding.len(),
            "document embedding {index} has {} dimensions, expected {}",
            embedding.len(),
            query_embedding.len()
        );
        if let Some((dimension, value)) = embedding
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            let chunk = &chunks[index];
            bail!(
                "document embedding {index} is non-finite at dimension {dimension}: {value}; \
                 source={} chunk={} characters={}",
                raw[chunk.document].hit.url,
                chunk.ordinal,
                chunk.text.chars().count(),
            );
        }
        let score = dot(query_embedding, embedding);
        ensure!(
            score.is_finite(),
            "document embedding {index} produced a non-finite score"
        );
        scores.push(score);
    }

    Ok(scores)
}

fn limit_candidates(chunks: Vec<Chunk>, limit: usize, evidence: &QueryEvidence) -> Vec<Chunk> {
    if chunks.len() <= limit {
        return chunks;
    }
    if limit == 1 {
        return chunks.into_iter().take(1).collect();
    }

    let mut lexical = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| (index, evidence.score(&chunk.text)))
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    lexical.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_index.cmp(right_index))
    });

    // Keep the local neighborhood of an exact compound identifier. The
    // defining algorithm often starts in the next tokenizer window, and a
    // purely lexical top-k can otherwise retain the heading while discarding
    // the steps it introduces.
    let lexical_quota = limit.div_ceil(2);
    let identifier_seeds = lexical
        .iter()
        .filter(|(index, _)| evidence.has_exact_identifier(&chunks[*index].text))
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    let mut selected = identifier_seeds
        .iter()
        .copied()
        .take(lexical_quota)
        .collect::<HashSet<_>>();
    for distance in 1..chunks.len() {
        if selected.len() == lexical_quota {
            break;
        }
        for &seed in &identifier_seeds {
            for neighbor in [seed.checked_add(distance), seed.checked_sub(distance)] {
                if selected.len() == lexical_quota {
                    break;
                }
                if let Some(neighbor) = neighbor.filter(|index| *index < chunks.len()) {
                    selected.insert(neighbor);
                }
            }
        }
    }
    for (index, _) in lexical {
        if selected.len() == lexical_quota {
            break;
        }
        selected.insert(index);
    }
    let last = chunks.len() - 1;
    let remaining = limit.saturating_sub(selected.len());
    for sample in 0..remaining {
        if selected.len() == limit {
            break;
        }
        let index = if remaining == 1 {
            last
        } else {
            sample * last / (remaining - 1)
        };
        selected.insert(index);
    }
    for index in 0..chunks.len() {
        if selected.len() == limit {
            break;
        }
        selected.insert(index);
    }

    let mut chunks = chunks.into_iter().map(Some).collect::<Vec<_>>();
    let mut selected = selected.into_iter().collect::<Vec<_>>();
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|index| chunks[index].take().expect("sample indices are unique"))
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn select_scored(
    raw: Vec<Raw>,
    chunks: Vec<Chunk>,
    mut scored: Vec<(usize, f32)>,
    document_order: &[usize],
    limits: ChunkSelectionLimits,
    evidence: &QueryEvidence,
    numeric_consensus: &NumericConsensus,
) -> Result<Vec<RankedDocument>> {
    ensure!(!scored.is_empty(), "received no chunk scores");
    ensure!(
        scored.iter().all(|(index, _)| *index < chunks.len()),
        "chunk score index is out of range"
    );
    let unique = scored
        .iter()
        .map(|(index, _)| *index)
        .collect::<HashSet<_>>();
    ensure!(
        unique.len() == scored.len(),
        "received duplicate chunk scores"
    );
    ensure!(
        document_order.iter().all(|document| *document < raw.len()),
        "document order index is out of range"
    );
    ensure!(
        document_order.iter().copied().collect::<HashSet<_>>().len() == document_order.len(),
        "received duplicate document order indices"
    );

    scored.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| {
                chunks[*left_index]
                    .document
                    .cmp(&chunks[*right_index].document)
            })
            .then_with(|| {
                chunks[*left_index]
                    .ordinal
                    .cmp(&chunks[*right_index].ordinal)
            })
    });

    let mut selected = HashMap::<usize, Vec<(usize, f32)>>::new();
    let mut best_by_document = HashMap::<usize, (usize, f32)>::new();
    for &(chunk_index, score) in &scored {
        let document = chunks[chunk_index].document;
        best_by_document
            .entry(document)
            .and_modify(|(best_index, best_score)| {
                let exact = evidence.has_exact_identifier(&chunks[chunk_index].text);
                let best_exact = evidence.has_exact_identifier(&chunks[*best_index].text);
                if (exact && !best_exact) || (exact == best_exact && score > *best_score) {
                    (*best_index, *best_score) = (chunk_index, score);
                }
            })
            .or_insert((chunk_index, score));
    }
    let mut best_by_document = best_by_document.into_values().collect::<Vec<_>>();
    best_by_document.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score.total_cmp(left_score).then_with(|| {
            chunks[*left_index]
                .document
                .cmp(&chunks[*right_index].document)
        })
    });
    for (chunk_index, score) in best_by_document.into_iter().take(limits.total) {
        selected
            .entry(chunks[chunk_index].document)
            .or_default()
            .push((chunk_index, score));
    }

    // Once a document wins on an exact compound identifier, use its remaining
    // quota for the neighboring tokenizer windows. This keeps a method's
    // definition and algorithm together instead of pairing the signature with
    // a distant paragraph that only shares generic query words.
    let coherent_documents = selected
        .iter()
        .filter(|(_, document_chunks)| {
            document_chunks
                .iter()
                .any(|(index, _)| evidence.has_exact_identifier(&chunks[*index].text))
        })
        .map(|(document, _)| *document)
        .collect::<HashSet<_>>();
    let mut coherent_order = coherent_documents.iter().copied().collect::<Vec<_>>();
    coherent_order.sort_by(|left, right| {
        let left_score = selected[left][0].1;
        let right_score = selected[right][0].1;
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    for document in coherent_order {
        while selected.values().map(Vec::len).sum::<usize>() < limits.total
            && selected[&document].len() < limits.per_document
        {
            let Some(&(chunk_index, score)) = scored
                .iter()
                .filter(|(index, _)| chunks[*index].document == document)
                .filter(|(index, _)| {
                    !selected[&document]
                        .iter()
                        .any(|(selected_index, _)| selected_index == index)
                })
                .filter(|(index, _)| {
                    selected[&document].iter().any(|(selected_index, _)| {
                        chunks[*selected_index]
                            .ordinal
                            .abs_diff(chunks[*index].ordinal)
                            == 1
                    })
                })
                .max_by(|(left_index, left_score), (right_index, right_score)| {
                    let left_exact = evidence.has_exact_identifier(&chunks[*left_index].text);
                    let right_exact = evidence.has_exact_identifier(&chunks[*right_index].text);
                    left_exact
                        .cmp(&right_exact)
                        .then_with(|| left_score.total_cmp(right_score))
                        .then_with(|| {
                            chunks[*right_index]
                                .ordinal
                                .cmp(&chunks[*left_index].ordinal)
                        })
                })
            else {
                break;
            };
            selected
                .get_mut(&document)
                .expect("coherent document is selected")
                .push((chunk_index, score));
        }
    }

    for (chunk_index, score) in scored {
        if selected.values().map(Vec::len).sum::<usize>() >= limits.total {
            break;
        }
        let document = chunks[chunk_index].document;
        let document_chunks = selected.entry(document).or_default();
        if document_chunks.len() == limits.per_document
            || document_chunks
                .iter()
                .any(|(selected_index, _)| *selected_index == chunk_index)
        {
            continue;
        }
        if coherent_documents.contains(&document)
            && !document_chunks.iter().any(|(selected_index, _)| {
                chunks[*selected_index]
                    .ordinal
                    .abs_diff(chunks[chunk_index].ordinal)
                    == 1
            })
        {
            continue;
        }
        document_chunks.push((chunk_index, score));
    }

    let mut raw = raw.into_iter().map(Some).collect::<Vec<_>>();
    let ranked = document_order
        .iter()
        .filter_map(|&document| {
            let selected = selected.remove(&document)?;
            let raw = raw[document]
                .take()
                .expect("ordered document indices are unique");
            let selected_chunks = selected
                .iter()
                .map(|(index, _)| &chunks[*index])
                .collect::<Vec<_>>();
            let mut assembled = chunk::assemble(&raw.content, &selected_chunks);
            if useful_search_excerpt(
                evidence,
                numeric_consensus,
                document,
                &raw.hit.snippet,
                &assembled.content,
            ) {
                let excerpt = truncate_excerpt(&raw.hit.snippet, MAX_SEARCH_EXCERPT_CHARACTERS);
                assembled.content = if assembled.content.is_empty() {
                    format!("Search excerpt:\n\n{excerpt}")
                } else {
                    format!("Search excerpt:\n\n{excerpt}\n\n{}", assembled.content)
                };
            }
            Some(RankedDocument {
                raw,
                content: assembled.content,
                truncated: assembled.truncated,
            })
        })
        .collect::<Vec<_>>();

    Ok(ranked)
}

fn useful_search_excerpt(
    evidence: &QueryEvidence,
    numeric_consensus: &NumericConsensus,
    document: usize,
    excerpt: &str,
    content: &str,
) -> bool {
    let excerpt = excerpt.trim();
    if excerpt.is_empty()
        || normalized_whitespace(content).contains(&normalized_whitespace(excerpt))
    {
        return false;
    }

    evidence.score(excerpt) > evidence.score(content)
        || evidence.adds_terms(excerpt, content)
        || (numeric_consensus.score(document) > 0.0
            && numeric_consensus.adds_evidence(excerpt, content))
}

fn normalized_whitespace(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn truncate_excerpt(excerpt: &str, max_characters: usize) -> String {
    let excerpt = excerpt.trim();
    if excerpt.chars().count() <= max_characters {
        return excerpt.to_owned();
    }
    let mut truncated = excerpt.chars().take(max_characters - 1).collect::<String>();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::search::searxng::Hit;

    fn raw(url: &str, content: &str) -> Raw {
        raw_with_snippet(url, "", content)
    }

    fn raw_with_snippet(url: &str, snippet: &str, content: &str) -> Raw {
        Raw {
            hit: Hit {
                title: url.into(),
                url: Url::parse(url).unwrap(),
                date: None,
                snippet: snippet.into(),
            },
            source_priority: false,
            upstream_consensus: false,
            content: content.into(),
            full_length: content.chars().count(),
            from_snippet: false,
        }
    }

    fn raw_with_title(url: &str, title: &str, content: &str) -> Raw {
        let mut raw = raw(url, content);
        raw.hit.title = title.into();
        raw
    }

    fn one_chunk_per_document(raw: &[Raw]) -> Vec<Chunk> {
        raw.iter()
            .enumerate()
            .map(|(document, raw)| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: raw.content.len(),
                tokens: 8,
                text: raw.content.clone(),
            })
            .collect()
    }

    fn query_evidence(query: &str, raw: &[Raw]) -> QueryEvidence {
        let documents = raw
            .iter()
            .map(|raw| format!("{}\n{}\n{}", raw.hit.title, raw.hit.snippet, raw.content))
            .collect::<Vec<_>>();
        QueryEvidence::new(query, documents.iter().map(String::as_str))
    }

    #[test]
    fn rare_query_terms_carry_more_evidence_than_ubiquitous_terms() {
        let raw = vec![
            raw("https://example.com/0", "postgresql jsonb containment"),
            raw("https://example.com/1", "postgresql jsonb"),
            raw("https://example.com/2", "postgresql jsonb"),
            raw("https://example.com/3", "postgresql jsonb"),
        ];
        let evidence = query_evidence("postgresql jsonb containment", &raw);

        assert!(evidence.score("jsonb containment") > evidence.score("postgresql jsonb"));
    }

    #[test]
    fn repeated_percentage_facts_break_numeric_answer_ties() {
        let raw = vec![
            raw_with_snippet(
                "https://official.example/release",
                "Inflation was 3.5 percent over the last 12 months.",
                "blocked",
            ),
            raw_with_snippet(
                "https://news.example/report",
                "The annual inflation rate fell to 3.5%.",
                "article",
            ),
            raw_with_snippet(
                "https://analysis.example.org/report",
                "Headline inflation measured 3.5 percent.",
                "article",
            ),
            raw_with_snippet(
                "https://bad.example/value",
                "The CPI index is 333.95% and rose 11.39%.",
                "article",
            ),
        ];

        let consensus = NumericConsensus::new("current US CPI inflation rate", &raw);

        assert_eq!(consensus.score(0), MAX_NUMERIC_CONSENSUS_BONUS);
        assert_eq!(consensus.score(1), MAX_NUMERIC_CONSENSUS_BONUS);
        assert_eq!(consensus.score(2), MAX_NUMERIC_CONSENSUS_BONUS);
        assert_eq!(consensus.score(3), 0.0);
    }

    #[test]
    fn numeric_consensus_only_scores_query_aligned_percentage_context() {
        let raw = ["a.com", "b.net", "c.org"]
            .into_iter()
            .map(|host| {
                raw_with_snippet(
                    &format!("https://{host}/"),
                    "Annual inflation was 3.5%.",
                    "article",
                )
            })
            .collect::<Vec<_>>();
        let consensus = NumericConsensus::new("current inflation rate", &raw);

        assert_eq!(
            consensus.score_text("Annual inflation was 3.5%."),
            MAX_NUMERIC_CONSENSUS_BONUS
        );
        assert_eq!(consensus.score_text("Save 3.5% with this discount."), 0.0);
        assert_eq!(consensus.score_text("Annual inflation was 4.2%."), 0.0);
    }

    #[test]
    fn mortgage_down_payments_do_not_become_the_answer_rate() {
        let mut raw = ["lender-a.com", "lender-b.net", "lender-c.org"]
            .into_iter()
            .map(|host| {
                raw_with_snippet(
                    &format!("https://{host}/mortgages"),
                    "A 20% down payment avoids private mortgage insurance.",
                    "Mortgage guide",
                )
            })
            .collect::<Vec<_>>();
        raw.push(raw_with_snippet(
            "https://rates.example/mortgage",
            "The average 30-year fixed mortgage rate is 6.625% today.",
            "Current mortgage rates",
        ));
        let chunks = raw
            .iter()
            .enumerate()
            .map(|(document, raw)| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: raw.content.len(),
                tokens: 6,
                text: raw.content.clone(),
            })
            .collect::<Vec<_>>();
        let query = "current 30-year fixed mortgage rate";
        let consensus = NumericConsensus::new(query, &raw);

        assert!(!consensus.has_consensus());
        assert!(consensus.scores.iter().all(|score| *score == 0.0));
        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.79), (1, 0.79), (2, 0.79), (3, 0.80)],
            1,
            &query_evidence(query, &raw),
            &consensus,
        )
        .unwrap();

        assert_eq!(selected, [3]);
    }

    #[test]
    fn numeric_consensus_counts_independent_domains() {
        let raw = vec![
            raw_with_snippet("https://a.publisher.com/one", "Inflation was 3.5%", "a"),
            raw_with_snippet("https://b.publisher.com/two", "Inflation was 3.5%", "b"),
            raw_with_snippet("https://independent.net/three", "Inflation was 3.5%", "c"),
        ];

        let consensus = NumericConsensus::new("current inflation rate", &raw);

        assert!(consensus.scores.iter().all(|score| *score == 0.0));
    }

    #[test]
    fn stronger_numeric_consensus_receives_more_bounded_evidence() {
        let mut raw = ["a.com", "b.net", "c.org", "d.dev"]
            .into_iter()
            .map(|host| raw_with_snippet(&format!("https://{host}/"), "Inflation was 3.5%", "x"))
            .collect::<Vec<_>>();
        raw.extend(["e.com", "f.net", "g.org"].into_iter().map(|host| {
            raw_with_snippet(&format!("https://{host}/"), "Core inflation was 2.6%", "x")
        }));

        let consensus = NumericConsensus::new("current inflation rate", &raw);

        assert_eq!(consensus.score(0), MAX_NUMERIC_CONSENSUS_BONUS);
        assert_eq!(consensus.score(4), MAX_NUMERIC_CONSENSUS_BONUS * 0.75);
        assert!(
            consensus
                .scores
                .iter()
                .all(|score| { (0.0..=MAX_NUMERIC_CONSENSUS_BONUS).contains(score) })
        );
    }

    #[test]
    fn percentage_consensus_is_inert_for_non_numeric_queries() {
        let raw = vec![
            raw_with_snippet("https://example.com/a", "3.5%", "a"),
            raw_with_snippet("https://example.com/b", "3.5 percent", "b"),
        ];

        let consensus = NumericConsensus::new("tokio cancellation safety", &raw);

        assert_eq!(consensus.score(0), 0.0);
        assert_eq!(consensus.score(1), 0.0);
    }

    #[test]
    fn broad_price_change_queries_do_not_assume_a_percentage_answer() {
        let raw = ["a.com", "b.net", "c.org"]
            .into_iter()
            .map(|host| {
                raw_with_snippet(
                    &format!("https://{host}/housing"),
                    "Save 20% with a larger down payment.",
                    "Housing market report",
                )
            })
            .collect::<Vec<_>>();

        let consensus = NumericConsensus::new("current home price changes", &raw);

        assert!(!consensus.has_consensus());
        assert!(consensus.scores.iter().all(|score| *score == 0.0));
    }

    #[test]
    fn lexical_evidence_only_breaks_a_close_dense_contest() {
        let generic = combined_score(0.80, 0.0);

        assert!(combined_score(0.76, 1.0) > generic);
        assert!(combined_score(0.60, 1.0) < generic);
    }

    #[test]
    fn decisive_rare_terms_outweigh_generic_query_coverage() {
        let raw = vec![
            raw("https://generic.example/1", "2026 ADA diabetes standards"),
            raw("https://generic.example/2", "2026 ADA diabetes standards"),
            raw("https://generic.example/3", "2026 ADA diabetes standards"),
            raw(
                "https://answer.example/",
                "statin therapy recommendations for adults",
            ),
        ];
        let evidence = query_evidence(
            "2026 ADA diabetes standards statin recommendations adults official",
            &raw,
        );

        let generic = evidence.score("2026 ADA diabetes standards");
        let answer = evidence.score("statin therapy recommendations for adults");

        assert!(answer > generic, "answer={answer}, generic={generic}");
        assert!((0.0..=1.0).contains(&generic));
        assert!((0.0..=1.0).contains(&answer));
    }

    #[test]
    fn selection_is_global_but_caps_each_document() {
        let raw = vec![
            raw("https://example.com/a", "aaaabbbbcccc"),
            raw("https://example.com/b", "dddd"),
        ];
        let chunks = vec![
            Chunk {
                document: 0,
                ordinal: 0,
                start: 0,
                end: 4,
                tokens: 4,
                text: "aaaa".into(),
            },
            Chunk {
                document: 0,
                ordinal: 1,
                start: 4,
                end: 8,
                tokens: 4,
                text: "bbbb".into(),
            },
            Chunk {
                document: 0,
                ordinal: 2,
                start: 8,
                end: 12,
                tokens: 4,
                text: "cccc".into(),
            },
            Chunk {
                document: 1,
                ordinal: 0,
                start: 0,
                end: 4,
                tokens: 4,
                text: "dddd".into(),
            },
        ];
        let evidence = query_evidence("", &raw);
        let numeric_consensus = NumericConsensus::new("", &raw);

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)],
            &[0, 1],
            ChunkSelectionLimits {
                total: 3,
                per_document: 2,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].raw.hit.url.as_str(), "https://example.com/a");
        assert_eq!(ranked[0].content, "aaaabbbb\n\n…");
        assert!(ranked[0].truncated);
        assert_eq!(ranked[1].content, "dddd");
        assert!(!ranked[1].truncated);
    }

    #[test]
    fn candidate_limit_samples_the_whole_document() {
        let chunks = (0..10)
            .map(|index| Chunk {
                document: 0,
                ordinal: index,
                start: index * 2,
                end: index * 2 + 2,
                tokens: 2,
                text: index.to_string(),
            })
            .collect();
        let evidence = QueryEvidence::new("unmatched query", std::iter::empty());
        let chunks = limit_candidates(chunks, 3, &evidence);

        assert_eq!(
            chunks.iter().map(|chunk| chunk.ordinal).collect::<Vec<_>>(),
            [0, 4, 9]
        );
    }

    #[test]
    fn candidate_limit_keeps_query_bearing_chunks() {
        let chunks = (0..10)
            .map(|index| Chunk {
                document: 0,
                ordinal: index,
                start: index * 2,
                end: index * 2 + 2,
                tokens: 2,
                text: if index == 5 {
                    "jsonb containment gin"
                } else {
                    "unrelated prose"
                }
                .into(),
            })
            .collect();
        let evidence = QueryEvidence::new("PostgreSQL JSONB GIN containment", std::iter::empty());
        let chunks = limit_candidates(chunks, 3, &evidence);

        assert!(chunks.iter().any(|chunk| chunk.ordinal == 5));
    }

    #[test]
    fn candidate_limit_prefers_the_chunk_with_the_rare_query_term() {
        let chunks = (0..10)
            .map(|index| Chunk {
                document: 0,
                ordinal: index,
                start: index * 2,
                end: index * 2 + 2,
                tokens: 2,
                text: if index == 7 {
                    "postgresql jsonb containment"
                } else {
                    "postgresql jsonb"
                }
                .into(),
            })
            .collect();
        let documents = vec![
            raw("https://example.com/0", "postgresql jsonb containment"),
            raw("https://example.com/1", "postgresql jsonb"),
            raw("https://example.com/2", "postgresql jsonb"),
        ];
        let evidence = query_evidence("PostgreSQL JSONB containment", &documents);
        let chunks = limit_candidates(chunks, 3, &evidence);

        assert!(chunks.iter().any(|chunk| chunk.ordinal == 7));
    }

    #[test]
    fn candidate_limit_keeps_a_compound_identifiers_neighborhood() {
        let chunks = (0..20)
            .map(|index| Chunk {
                document: 0,
                ordinal: index,
                start: index * 2,
                end: index * 2 + 2,
                tokens: 2,
                text: if index == 10 {
                    "Widget.parse() algorithm"
                } else {
                    "unrelated prose"
                }
                .into(),
            })
            .collect();
        let evidence = QueryEvidence::new("Widget.parse() algorithm", std::iter::empty());
        let chunks = limit_candidates(chunks, 8, &evidence);
        let ordinals = chunks
            .iter()
            .map(|chunk| chunk.ordinal)
            .collect::<HashSet<_>>();

        assert!(ordinals.contains(&10));
        assert!(ordinals.contains(&9));
        assert!(ordinals.contains(&11));
    }

    #[test]
    fn candidate_limit_keeps_the_document_tail_after_lexical_seeding() {
        let chunks = (0..20)
            .map(|index| Chunk {
                document: 0,
                ordinal: index,
                start: index * 2,
                end: index * 2 + 2,
                tokens: 2,
                text: if index < 4 {
                    format!("jsonb containment {index}")
                } else {
                    "unrelated prose".into()
                },
            })
            .collect();
        let evidence = QueryEvidence::new("PostgreSQL JSONB containment", std::iter::empty());
        let chunks = limit_candidates(chunks, 8, &evidence);

        assert!(chunks.iter().any(|chunk| chunk.ordinal == 19));
    }

    #[test]
    fn full_content_selection_protects_only_plausible_upstream_results() {
        let raw = (0..5)
            .map(|document| raw(&format!("https://example.com/{document}"), "content"))
            .collect::<Vec<_>>();
        let chunks = (0..5)
            .map(|document| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: 1,
                tokens: 1,
                text: document.to_string(),
            })
            .collect::<Vec<_>>();

        let selected = select_documents(
            "",
            &raw,
            &chunks,
            &[(0, 0.85), (1, 0.65), (2, 0.60), (3, 0.80), (4, 0.90)],
            3,
            &query_evidence("", &raw),
            &NumericConsensus::new("", &raw),
        )
        .unwrap();

        // The first upstream result remains because its best passage is close
        // to the global best. The weak second result does not consume the
        // other upstream-head reservation, so stronger tail documents enter.
        assert_eq!(selected, [0, 3, 4]);
    }

    #[test]
    fn explicit_case_study_queries_reserve_one_close_evidence_anchor() {
        let raw = vec![
            raw_with_title(
                "https://overview.example/builds",
                "Bazel Nix and Pants build-system overview",
                "A feature comparison of three build systems.",
            ),
            raw_with_title(
                "https://generic.example/hermetic",
                "Hermetic builds for polyglot monorepos",
                "General advice for hermetic build configuration.",
            ),
            raw_with_title(
                "https://engineering.example/migration",
                "Production migration case study: Bazel to Nix",
                "A team migrated its large polyglot monorepo and reports the tradeoffs.",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);
        let query = "Bazel Nix Pants polyglot monorepo case studies";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.87), (2, 0.84)],
            2,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [2, 0]);
    }

    #[test]
    fn case_study_body_words_do_not_turn_a_generic_title_into_an_anchor() {
        let raw = vec![
            raw_with_title(
                "https://overview.example/builds",
                "Bazel Nix and Pants build-system overview",
                "A feature comparison of three build systems.",
            ),
            raw_with_title(
                "https://generic.example/hermetic",
                "Hermetic builds for polyglot monorepos",
                "General advice for hermetic build configuration.",
            ),
            raw_with_title(
                "https://index.example/case-studies",
                "Build engineering resources",
                "An index containing production migration case study links.",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);
        let query = "Bazel Nix Pants polyglot monorepo case studies";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.87), (2, 0.84)],
            2,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 1]);
    }

    #[test]
    fn research_queries_reserve_one_close_scholarly_source() {
        let raw = vec![
            raw_with_title(
                "https://overview.example/reproducibility",
                "Reproducible machine learning overview",
                "A broad introduction.",
            ),
            raw_with_title(
                "https://blog.example/numerics",
                "Numerical reproducibility techniques",
                "A practical blog post.",
            ),
            raw_with_title(
                "https://arxiv.org/abs/2601.12345",
                "Measuring numerical reproducibility across accelerators",
                "A controlled evaluation of CPU and GPU inference drift.",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);
        let query = "research paper numerical reproducibility CPU GPU";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.87), (2, 0.82)],
            2,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [2, 0]);
    }

    #[test]
    fn scholarly_anchor_uses_full_content_relevance_before_lexical_coverage() {
        let raw = vec![
            raw_with_title(
                "https://overview.example/reproducibility",
                "Testing numerical drift across inference backends",
                "Compare CPU and CUDA outputs with explicit tolerance gates.",
            ),
            raw_with_title(
                "https://arxiv.org/abs/2603.02871",
                "Floating-point-consistent cross-verification of DDA solvers",
                "A reproducibility benchmark for an unrelated numerical solver.",
            ),
            raw_with_title(
                "https://arxiv.org/abs/2509.06977",
                "Reproducible cross-backend compatibility for deep learning",
                "Three-tier verification detects CPU and CUDA inference drift.",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);
        let query = "research paper numerical reproducibility CPU CUDA inference";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.84), (2, 0.88)],
            2,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [2, 0]);
    }

    #[test]
    fn machine_learning_parity_queries_demote_missing_evidence_axes() {
        let query = "detect CPU to CUDA machine-learning inference numerical drift";
        assert_eq!(
            numerical_parity_content_adjustment(
                query,
                "A neural-network inference output diverges between CPU and CUDA."
            ),
            0.0
        );
        assert_eq!(
            numerical_parity_content_adjustment(
                query,
                "A reproducibility benchmark for CPU and GPU DDA solvers."
            ),
            -NUMERICAL_PARITY_MISSING_CONTEXT_PENALTY
        );
        assert_eq!(
            numerical_parity_content_adjustment(
                query,
                "Optimize CUDA matrix multiplication for model inference throughput."
            ),
            -NUMERICAL_PARITY_MISSING_CONTEXT_PENALTY
        );
        assert_eq!(
            numerical_parity_content_adjustment(query, "CUDA matrix multiplication throughput."),
            -2.0 * NUMERICAL_PARITY_MISSING_CONTEXT_PENALTY
        );
        assert_eq!(
            numerical_parity_content_adjustment(
                "CPU and CUDA numerical parity for a fluid solver",
                "unrelated"
            ),
            0.0
        );
    }

    #[test]
    fn numerical_parity_queries_reserve_a_concrete_cross_backend_issue() {
        let raw = vec![
            raw_with_title(
                "https://overview.example/numerics",
                "Cross-backend inference reproducibility",
                "A broad overview of CPU and CUDA numerical drift.",
            ),
            raw_with_title(
                "https://github.com/example/audio/issues/15",
                "CPU bf16 causes divergent model output",
                "CPU inference differs by precision; the loader also defaults to CUDA.",
            ),
            raw_with_title(
                "https://github.com/example/runtime/issues/26027",
                "CUDA inference produces subtly corrupted transformer output",
                "CPU-only inference is correct, but the same model corrupts output on GPU.",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);
        let query = "detect silent numerical drift porting machine-learning inference from CPU to CUDA bitwise reproducibility tolerance testing";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.82), (2, 0.71)],
            2,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 2]);
    }

    #[test]
    fn close_same_source_subjects_yield_to_an_independent_result() {
        let raw = vec![
            raw_with_title(
                "https://www.milik.ai/articles/first",
                "CUDA Kernel Validation Against Silent AI Model Corruption | Milik",
                "Validate generated CUDA kernels against a CPU reference.",
            ),
            raw_with_title(
                "https://www.milik.ai/articles/second",
                "CUDA Kernel Validation for AI-Generated Code | Milik",
                "Test generated CUDA kernels for subtle numerical errors.",
            ),
            raw_with_title(
                "https://independent.example/gpu-numerics",
                "Testing CPU and GPU numerical differences",
                "Measure ULP, relative error, and absolute error across backends.",
            ),
            raw_with_title(
                "https://docs.nvidia.com/numerical-errors",
                "Numerical errors in CUDA workloads",
                "NVIDIA documents techniques for finding silent numerical failures.",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);

        let selected = select_documents(
            "detect silent numerical drift between CPU and CUDA inference",
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.88), (2, 0.87), (3, 0.86)],
            3,
            &query_evidence("CPU CUDA numerical drift", &raw),
            &NumericConsensus::new("CPU CUDA numerical drift", &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 2, 3]);
    }

    #[test]
    fn a_materially_stronger_same_source_result_is_not_sacrificed_for_diversity() {
        let raw = vec![
            raw_with_title(
                "https://publisher.example/first",
                "CUDA Kernel Validation Against Silent Model Corruption",
                "first",
            ),
            raw_with_title(
                "https://publisher.example/second",
                "CUDA Kernel Validation for Silent Model Corruption",
                "second",
            ),
            raw_with_title(
                "https://independent.example/alternative",
                "GPU testing overview",
                "alternative",
            ),
        ];
        let chunks = one_chunk_per_document(&raw);

        let selected = select_documents(
            "CUDA model corruption validation",
            &raw,
            &chunks,
            &[(0, 0.90), (1, 0.88), (2, 0.80)],
            2,
            &query_evidence("CUDA model corruption validation", &raw),
            &NumericConsensus::new("CUDA model corruption validation", &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 1]);
    }

    #[test]
    fn diversity_scopes_code_hosts_by_repository_and_keeps_distinct_docs_subjects() {
        let raw = vec![
            raw_with_title(
                "https://github.com/team/kernel-one/issues/1",
                "Fix silent CUDA numerical drift",
                "one",
            ),
            raw_with_title(
                "https://github.com/team/kernel-two/issues/2",
                "Fix silent CUDA numerical drift",
                "two",
            ),
            raw_with_title(
                "https://docs.nvidia.com/cuda/capture.html",
                "CUDA Graph capture lifecycle",
                "capture",
            ),
            raw_with_title(
                "https://docs.nvidia.com/cuda/reductions.html",
                "Deterministic reductions and floating-point tolerance",
                "reductions",
            ),
            raw_with_title(
                "https://alternative.example/article",
                "GPU correctness overview",
                "alternative",
            ),
        ];
        let best = vec![Some(0.90), Some(0.89), Some(0.88), Some(0.87), Some(0.86)];
        let exact = vec![false; raw.len()];
        let recommendations = vec![0.0; raw.len()];

        let selected = diversify_documents(
            &raw,
            vec![0, 1, 2, 3],
            &[(0, 0.90), (1, 0.89), (2, 0.88), (3, 0.87), (4, 0.86)],
            &best,
            &exact,
            None,
            None,
            false,
            false,
            &recommendations,
        );

        assert_eq!(selected, [0, 1, 2, 3]);
    }

    #[test]
    fn diversity_never_displaces_a_protected_result() {
        let documents = || {
            vec![
                raw_with_title(
                    "https://publisher.example/first",
                    "CUDA Kernel Validation Against Silent Model Corruption",
                    "first",
                ),
                raw_with_title(
                    "https://publisher.example/second",
                    "CUDA Kernel Validation for Silent Model Corruption",
                    "second",
                ),
                raw_with_title(
                    "https://independent.example/alternative",
                    "Independent GPU numerical testing",
                    "alternative",
                ),
            ]
        };
        let ranking = [(0, 0.90), (1, 0.88), (2, 0.87)];
        let best = vec![Some(0.90), Some(0.88), Some(0.87)];
        let recommendations = vec![0.0; 3];

        let mut source = documents();
        source[1].source_priority = true;
        assert_eq!(
            diversify_documents(
                &source,
                vec![0, 1],
                &ranking,
                &best,
                &[false; 3],
                None,
                None,
                false,
                false,
                &recommendations,
            ),
            [0, 1]
        );
        let base = documents();
        assert_eq!(
            diversify_documents(
                &base,
                vec![0, 1],
                &ranking,
                &best,
                &[false, true, false],
                None,
                None,
                false,
                false,
                &recommendations,
            ),
            [0, 1]
        );
        assert_eq!(
            diversify_documents(
                &base,
                vec![0, 1],
                &ranking,
                &best,
                &[false; 3],
                Some(1),
                None,
                false,
                false,
                &recommendations,
            ),
            [0, 1]
        );
    }

    #[test]
    fn full_text_queries_prefer_extracted_documents_over_snippet_fallbacks() {
        let mut raw = vec![
            raw(
                "https://official.example/article-3",
                "Article 3 opening search excerpt",
            ),
            raw(
                "https://archive.example/article-3",
                "Article 3 complete extracted text",
            ),
        ];
        raw[0].from_snippet = true;
        let chunks = raw
            .iter()
            .enumerate()
            .map(|(document, raw)| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: raw.content.len(),
                tokens: 5,
                text: raw.content.clone(),
            })
            .collect::<Vec<_>>();
        let query = "Convention Article 3 full text";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.99), (1, 0.50)],
            1,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [1]);
    }

    #[test]
    fn exact_document_manifestations_displace_related_papers() {
        let title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks";
        let mut raw = vec![
            raw("https://arxiv.org/abs/2005.11401", "original abstract"),
            raw("https://arxiv.org/abs/2506.00054", "related survey"),
            raw(
                "https://huggingface.co/papers/2005.11401",
                "original abstract",
            ),
            raw(
                "https://ai.meta.com/research/publications/rag/",
                "original publication page",
            ),
        ];
        raw[0].hit.title = title.into();
        raw[1].hit.title = "Retrieval-Augmented Generation: A Comprehensive Survey".into();
        raw[2].hit.title = title.into();
        raw[3].hit.title = title.into();
        let chunks = raw
            .iter()
            .enumerate()
            .map(|(document, raw)| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: raw.content.len(),
                tokens: 4,
                text: raw.content.clone(),
            })
            .collect::<Vec<_>>();
        let query =
            "arXiv original paper Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.60), (1, 0.99), (2, 0.55), (3, 0.50)],
            3,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 2, 3]);
    }

    #[test]
    fn numeric_consensus_does_not_categorically_override_dense_relevance() {
        let raw = vec![
            raw_with_snippet(
                "https://a.example/release",
                "Annual inflation was 3.5%.",
                "Annual inflation was 3.5%.",
            ),
            raw_with_snippet(
                "https://b.example/report",
                "The CPI inflation rate was 3.5 percent.",
                "The CPI inflation rate was 3.5 percent.",
            ),
            raw_with_snippet(
                "https://c.example/news",
                "Prices rose 3.5% over twelve months.",
                "Prices rose 3.5% over twelve months.",
            ),
            raw_with_snippet(
                "https://table.example/cpi",
                "Current consumer price data table.",
                "Current consumer price data table.",
            ),
            raw_with_snippet(
                "https://stale.example/cpi",
                "Annual inflation was 4.2%.",
                "Annual inflation was 4.2%.",
            ),
        ];
        let chunks = raw
            .iter()
            .enumerate()
            .map(|(document, raw)| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: raw.content.len(),
                tokens: 5,
                text: raw.content.clone(),
            })
            .collect::<Vec<_>>();
        let query = "current CPI inflation rate";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.80), (1, 0.80), (2, 0.80), (3, 0.10), (4, 0.99)],
            4,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 1, 2, 4]);
    }

    #[test]
    fn disjoint_recommendation_and_numeric_evidence_still_select_documents() {
        let mut raw = ["a.example", "b.example", "c.example"]
            .into_iter()
            .map(|host| {
                raw_with_snippet(
                    &format!("https://{host}/rate"),
                    "ADA inflation rate was 3.5%.",
                    "ADA inflation rate was 3.5%.",
                )
            })
            .collect::<Vec<_>>();
        raw.extend(
            ["guidance.example", "practice.example"]
                .into_iter()
                .map(|host| {
                    raw_with_snippet(
                        &format!("https://{host}/guidance"),
                        "ADA adults should use preventive therapy.",
                        "ADA adults should use preventive therapy.",
                    )
                }),
        );
        let chunks = raw
            .iter()
            .enumerate()
            .map(|(document, raw)| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: raw.content.len(),
                tokens: 6,
                text: raw.content.clone(),
            })
            .collect::<Vec<_>>();
        let query = "current ADA inflation rate recommendations adults";

        let selected = select_documents(
            query,
            &raw,
            &chunks,
            &[(0, 0.85), (1, 0.85), (2, 0.85), (3, 0.90), (4, 0.90)],
            4,
            &query_evidence(query, &raw),
            &NumericConsensus::new(query, &raw),
        )
        .unwrap();

        assert_eq!(selected.len(), 4);
        assert!(selected.iter().any(|document| *document <= 2));
        assert!(selected.iter().any(|document| *document >= 3));
    }

    #[test]
    fn full_content_selection_only_prefers_relevant_first_party_results() {
        let mut raw = (0..4)
            .map(|document| raw(&format!("https://example.com/{document}"), "content"))
            .collect::<Vec<_>>();
        raw[0].source_priority = true;
        raw[1].source_priority = true;
        let chunks = (0..4)
            .map(|document| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: 1,
                tokens: 1,
                text: document.to_string(),
            })
            .collect::<Vec<_>>();

        let selected = select_documents(
            "",
            &raw,
            &chunks,
            &[(0, 0.89), (1, 0.60), (2, 0.90), (3, 0.80)],
            2,
            &query_evidence("", &raw),
            &NumericConsensus::new("", &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 2]);
    }

    #[test]
    fn full_content_selection_reserves_one_verified_requested_source() {
        let mut raw = (0..4)
            .map(|document| raw(&format!("https://example.com/{document}"), "content"))
            .collect::<Vec<_>>();
        raw[0].source_priority = true;
        raw[1].source_priority = true;
        let chunks = (0..4)
            .map(|document| Chunk {
                document,
                ordinal: 0,
                start: 0,
                end: 1,
                tokens: 1,
                text: document.to_string(),
            })
            .collect::<Vec<_>>();

        let selected = select_documents(
            "",
            &raw,
            &chunks,
            &[(0, 0.60), (1, 0.55), (2, 0.90), (3, 0.80)],
            2,
            &query_evidence("", &raw),
            &NumericConsensus::new("", &raw),
        )
        .unwrap();

        assert_eq!(selected, [0, 2]);
    }

    #[test]
    fn full_content_selection_uses_a_documents_best_passage() {
        let raw = vec![
            raw("https://example.com/0", "ab"),
            raw("https://example.com/1", "c"),
        ];
        let chunks = vec![
            Chunk {
                document: 0,
                ordinal: 0,
                start: 0,
                end: 1,
                tokens: 1,
                text: "a".into(),
            },
            Chunk {
                document: 0,
                ordinal: 1,
                start: 1,
                end: 2,
                tokens: 1,
                text: "b".into(),
            },
            Chunk {
                document: 1,
                ordinal: 0,
                start: 0,
                end: 1,
                tokens: 1,
                text: "c".into(),
            },
        ];

        let selected = select_documents(
            "",
            &raw,
            &chunks,
            &[(0, 0.10), (1, 0.90), (2, 0.80)],
            1,
            &query_evidence("", &raw),
            &NumericConsensus::new("", &raw),
        )
        .unwrap();

        assert_eq!(selected, [0]);
    }

    #[test]
    fn selection_reserves_one_chunk_per_document() {
        let raw = vec![
            raw("https://example.com/a", "aaaa"),
            raw("https://example.com/b", "bbbb"),
            raw("https://example.com/c", "cccc"),
        ];
        let chunks = vec![
            Chunk {
                document: 0,
                ordinal: 0,
                start: 0,
                end: 2,
                tokens: 2,
                text: "aa".into(),
            },
            Chunk {
                document: 0,
                ordinal: 1,
                start: 2,
                end: 4,
                tokens: 2,
                text: "aa".into(),
            },
            Chunk {
                document: 1,
                ordinal: 0,
                start: 0,
                end: 4,
                tokens: 4,
                text: "bbbb".into(),
            },
            Chunk {
                document: 2,
                ordinal: 0,
                start: 0,
                end: 4,
                tokens: 4,
                text: "cccc".into(),
            },
        ];
        let evidence = query_evidence("", &raw);
        let numeric_consensus = NumericConsensus::new("", &raw);

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)],
            &[0, 1, 2],
            ChunkSelectionLimits {
                total: 3,
                per_document: 3,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].raw.hit.url.as_str(), "https://example.com/a");
        assert_eq!(ranked[1].raw.hit.url.as_str(), "https://example.com/b");
        assert_eq!(ranked[2].raw.hit.url.as_str(), "https://example.com/c");
    }

    #[test]
    fn compound_identifier_selection_keeps_the_local_algorithm_together() {
        let content = "generic introduction\nWidget.parse() definition\nstep one\nstep two\nunrelated insertion algorithm";
        let raw = vec![raw("https://example.com/widget", content)];
        let chunks = content
            .split_inclusive('\n')
            .enumerate()
            .scan(0, |start, (ordinal, text)| {
                let end = *start + text.len();
                let chunk = Chunk {
                    document: 0,
                    ordinal,
                    start: *start,
                    end,
                    tokens: 2,
                    text: text.trim().into(),
                };
                *start = end;
                Some(chunk)
            })
            .collect::<Vec<_>>();
        let evidence = query_evidence("Widget.parse() algorithm", &raw);
        let numeric_consensus = NumericConsensus::new("Widget.parse() algorithm", &raw);

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.10), (1, 0.40), (2, 0.30), (3, 0.20), (4, 0.99)],
            &[0],
            ChunkSelectionLimits {
                total: 3,
                per_document: 3,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert!(ranked[0].content.contains("Widget.parse() definition"));
        assert!(ranked[0].content.contains("step one"));
        assert!(ranked[0].content.contains("step two"));
        assert!(!ranked[0].content.contains("unrelated insertion algorithm"));
    }

    #[test]
    fn selected_document_order_survives_chunk_assembly() {
        let raw = vec![
            raw("https://example.com/first", "first"),
            raw("https://example.com/second", "second"),
        ];
        let chunks = vec![
            Chunk {
                document: 0,
                ordinal: 0,
                start: 0,
                end: 5,
                tokens: 1,
                text: "first".into(),
            },
            Chunk {
                document: 1,
                ordinal: 0,
                start: 0,
                end: 6,
                tokens: 1,
                text: "second".into(),
            },
        ];

        let evidence = query_evidence("", &raw);
        let numeric_consensus = NumericConsensus::new("", &raw);
        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.1), (1, 0.9)],
            &[1, 0],
            ChunkSelectionLimits {
                total: 2,
                per_document: 1,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert_eq!(ranked[0].raw.hit.url.as_str(), "https://example.com/second");
        assert_eq!(ranked[1].raw.hit.url.as_str(), "https://example.com/first");
    }

    #[test]
    fn preserves_a_query_bearing_search_excerpt_beside_extracted_content() {
        let raw = vec![raw_with_snippet(
            "https://bls.gov/cpi",
            "CPI rose 3.5 percent in July 2026.",
            "Bureau of Labor Statistics consumer price landing page.",
        )];
        let evidence = query_evidence("current CPI inflation July 2026", &raw);
        let numeric_consensus = NumericConsensus::new("current CPI inflation July 2026", &raw);
        let chunks = vec![Chunk {
            document: 0,
            ordinal: 0,
            start: 0,
            end: raw[0].content.len(),
            tokens: 8,
            text: raw[0].content.clone(),
        }];

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.8)],
            &[0],
            ChunkSelectionLimits {
                total: 1,
                per_document: 1,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert!(
            ranked[0]
                .content
                .starts_with("Search excerpt:\n\nCPI rose 3.5 percent in July 2026.")
        );
        assert!(ranked[0].content.contains("Bureau of Labor Statistics"));
    }

    #[test]
    fn does_not_repeat_an_excerpt_already_present_in_the_selected_passage() {
        let content = "CPI rose 3.5 percent in July 2026.";
        let raw = vec![raw_with_snippet(
            "https://bls.gov/cpi",
            "CPI rose 3.5 percent in July 2026.",
            content,
        )];
        let evidence = query_evidence("CPI inflation July 2026", &raw);
        let numeric_consensus = NumericConsensus::new("CPI inflation July 2026", &raw);
        let chunks = vec![Chunk {
            document: 0,
            ordinal: 0,
            start: 0,
            end: content.len(),
            tokens: 8,
            text: content.into(),
        }];

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.8)],
            &[0],
            ChunkSelectionLimits {
                total: 1,
                per_document: 1,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert_eq!(ranked[0].content, content);
    }

    #[test]
    fn ignores_non_query_bearing_search_excerpts() {
        let raw = vec![raw_with_snippet(
            "https://example.com/page",
            "Welcome to our home page.",
            "JSONB containment uses the at-sign greater-than operator.",
        )];
        let evidence = query_evidence("PostgreSQL JSONB containment", &raw);
        let numeric_consensus = NumericConsensus::new("PostgreSQL JSONB containment", &raw);
        let chunks = vec![Chunk {
            document: 0,
            ordinal: 0,
            start: 0,
            end: raw[0].content.len(),
            tokens: 8,
            text: raw[0].content.clone(),
        }];

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.8)],
            &[0],
            ChunkSelectionLimits {
                total: 1,
                per_document: 1,
            },
            &evidence,
            &numeric_consensus,
        )
        .unwrap();

        assert!(!ranked[0].content.contains("Search excerpt:"));
    }

    #[test]
    fn caps_search_excerpts_on_character_boundaries() {
        let excerpt = "é".repeat(MAX_SEARCH_EXCERPT_CHARACTERS + 10);
        let truncated = truncate_excerpt(&excerpt, MAX_SEARCH_EXCERPT_CHARACTERS);

        assert_eq!(truncated.chars().count(), MAX_SEARCH_EXCERPT_CHARACTERS);
        assert!(truncated.ends_with('…'));
    }
}
