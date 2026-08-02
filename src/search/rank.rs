use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail, ensure};

use super::Raw;
use super::chunk::{self, Chunk};
use crate::embeddings::{DOCUMENT_CHUNK_TOKENS, EmbeddingWorker};

const EMBEDDING_DIMENSIONS: usize = 768;
const CHUNK_OVERLAP_TOKENS: usize = 64;
const MAX_CANDIDATE_CHUNKS_PER_DOCUMENT: usize = 24;
const MAX_CHUNKS: usize = 24;
const MAX_CHUNKS_PER_DOCUMENT: usize = 3;

pub struct RankedDocument {
    pub raw: Raw,
    pub content: String,
    pub truncated: bool,
    pub score: f32,
}

pub async fn select(
    embeddings: &EmbeddingWorker,
    query: &str,
    raw: Vec<Raw>,
) -> Result<Vec<RankedDocument>> {
    let chunks = build_chunks(embeddings, &raw)?;

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let scores = score_dense(embeddings, query, &raw, &chunks)
        .await?
        .into_iter()
        .enumerate()
        .collect();

    select_scored(raw, chunks, scores, MAX_CHUNKS, MAX_CHUNKS_PER_DOCUMENT)
}

pub(super) fn build_chunks(embeddings: &EmbeddingWorker, raw: &[Raw]) -> Result<Vec<Chunk>> {
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

fn limit_candidates(chunks: Vec<Chunk>, limit: usize) -> Vec<Chunk> {
    if chunks.len() <= limit {
        return chunks;
    }
    if limit == 1 {
        return chunks.into_iter().take(1).collect();
    }

    let last = chunks.len() - 1;
    let mut chunks = chunks.into_iter().map(Some).collect::<Vec<_>>();
    (0..limit)
        .map(|index| index * last / (limit - 1))
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
    max_chunks: usize,
    max_chunks_per_document: usize,
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
    for (chunk_index, score) in scored {
        let document = chunks[chunk_index].document;
        let document_chunks = selected.entry(document).or_default();
        if document_chunks.len() == max_chunks_per_document {
            continue;
        }
        document_chunks.push((chunk_index, score));
        if selected.values().map(Vec::len).sum::<usize>() == max_chunks {
            break;
        }
    }

    let mut ranked = raw
        .into_iter()
        .enumerate()
        .filter_map(|(document, raw)| {
            let selected = selected.remove(&document)?;
            let score = selected
                .iter()
                .map(|(_, score)| *score)
                .max_by(f32::total_cmp)
                .expect("selected document has at least one chunk");
            let selected_chunks = selected
                .iter()
                .map(|(index, _)| &chunks[*index])
                .collect::<Vec<_>>();
            let assembled = chunk::assemble(&raw.content, &selected_chunks);
            Some((
                document,
                RankedDocument {
                    raw,
                    content: assembled.content,
                    truncated: assembled.truncated,
                    score,
                },
            ))
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|(left_index, left), (right_index, right)| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left_index.cmp(right_index))
    });
    Ok(ranked.into_iter().map(|(_, document)| document).collect())
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::search::searxng::Hit;

    fn raw(url: &str, content: &str) -> Raw {
        Raw {
            hit: Hit {
                title: url.into(),
                url: Url::parse(url).unwrap(),
                date: None,
                snippet: String::new(),
            },
            content: content.into(),
            full_length: content.chars().count(),
            from_snippet: false,
        }
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

        let ranked = select_scored(
            raw,
            chunks,
            vec![(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)],
            3,
            2,
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
        let chunks = limit_candidates(chunks, 3);

        assert_eq!(
            chunks.iter().map(|chunk| chunk.ordinal).collect::<Vec<_>>(),
            [0, 4, 9]
        );
    }
}
