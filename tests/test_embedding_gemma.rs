use anyhow::Result;
#[cfg(feature = "cuda")]
use sift::embeddings::EmbeddingBackend;
use sift::embeddings::EmbeddingGemma;
use sift::embeddings::EmbeddingWorker;

#[path = "common/embedding_gemma_goldens.rs"]
mod fixtures;

use fixtures::{EmbeddingGemmaGoldens, GoldenEmbedding, golden_by_name};

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());

    let dot_product = left.iter().zip(right).map(|(x, y)| x * y).sum::<f32>();
    let left_norm = left.iter().map(|x| x * x).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot_product / (left_norm * right_norm)
}

fn assert_matches_golden(actual: &[f32], golden: &GoldenEmbedding) {
    let similarity = cosine_similarity(actual, &golden.embedding);
    let (max_absolute_error, squared_error) = actual.iter().zip(&golden.embedding).fold(
        (0.0f32, 0.0f32),
        |(maximum, sum), (actual, expected)| {
            let error = (actual - expected).abs();
            (maximum.max(error), sum + error * error)
        },
    );
    let root_mean_square_error = (squared_error / actual.len() as f32).sqrt();
    assert!(
        similarity > 0.999_999_5 && max_absolute_error < 3e-7 && root_mean_square_error < 1e-7,
        "{} differed from its golden: cosine={similarity}, max_abs={max_absolute_error:e}, \
         rmse={root_mean_square_error:e}",
        golden.name,
    );
}

fn assert_model_matches_goldens(model: &EmbeddingGemma) -> Result<()> {
    let goldens = EmbeddingGemmaGoldens::load();
    let query_rust = golden_by_name(&goldens.queries_768, "rust_json");
    let query_france = golden_by_name(&goldens.queries_768, "capital_of_france");
    let rust = golden_by_name(&goldens.documents_768, "serde_json");
    let france = golden_by_name(&goldens.documents_768, "paris");

    let queries = model.embed_queries(&[&query_rust.text, &query_france.text], 768)?;
    assert_eq!(queries.len(), 2);
    assert_matches_golden(&queries[0], query_rust);
    assert_matches_golden(&queries[1], query_france);

    let documents = model.embed_documents(&[&rust.text, &france.text, &rust.text], 768)?;
    assert_eq!(documents.len(), 3);
    assert_matches_golden(&documents[0], rust);
    assert_matches_golden(&documents[1], france);
    assert_matches_golden(&documents[2], rust);

    let document_goldens_128 = goldens
        .documents_128
        .iter()
        .filter(|golden| golden.name != "over_context_window")
        .collect::<Vec<_>>();
    let document_texts_128 = document_goldens_128
        .iter()
        .map(|golden| golden.text.as_str())
        .collect::<Vec<_>>();
    let documents_128 = model.embed_documents(&document_texts_128, 128)?;
    for (actual, golden) in documents_128.iter().zip(document_goldens_128) {
        assert_matches_golden(actual, golden);
    }

    let single = model.embed_documents(&[&rust.text], 768)?;
    assert_eq!(single.len(), 1);
    assert_matches_golden(&single[0], rust);
    assert!(cosine_similarity(&documents[0], &single[0]) > 0.999);
    assert!(cosine_similarity(&documents[2], &single[0]) > 0.999);

    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn test_embedding_gemma() -> Result<()> {
    dotenvy::dotenv().ok();

    assert_model_matches_goldens(&EmbeddingGemma::load()?)?;

    #[cfg(feature = "cuda")]
    assert_model_matches_goldens(&EmbeddingGemma::load_on(EmbeddingBackend::Cuda(0))?)?;

    let goldens = EmbeddingGemmaGoldens::load();
    let query_rust = golden_by_name(&goldens.queries_768, "rust_json");
    let query_france = golden_by_name(&goldens.queries_768, "capital_of_france");
    let rust = golden_by_name(&goldens.documents_768, "serde_json");
    let france = golden_by_name(&goldens.documents_768, "paris");
    let worker = EmbeddingWorker::spawn(sift::embeddings::EmbeddingBackend::Cpu)?;

    let chunk_text = format!(
        "🙂 Unicode and Rust code: `serde_json::from_str::<Value>(input)?;` {}",
        "Token-aware chunks should remain predictable across code and prose. ".repeat(40)
    );
    let spans = worker.document_spans(&chunk_text, 32, 8)?;
    assert!(spans.len() > 1);
    assert!(spans.windows(2).all(|pair| pair[1].start < pair[0].end));
    for span in &spans {
        let chunk = &chunk_text[span.start..span.end];
        assert!(span.tokens > 0);
        assert!(worker.document_input_tokens(chunk)? <= 32);
    }

    let queries = worker
        .embed_queries(
            vec![query_rust.text.clone(), query_france.text.clone()],
            768,
        )
        .await?;
    assert_matches_golden(&queries[0], query_rust);
    assert_matches_golden(&queries[1], query_france);

    let documents = worker
        .embed_documents(
            vec![rust.text.clone(), france.text.clone(), rust.text.clone()],
            768,
        )
        .await?;
    assert_matches_golden(&documents[0], rust);
    assert_matches_golden(&documents[1], france);
    assert_matches_golden(&documents[2], rust);

    Ok(())
}
