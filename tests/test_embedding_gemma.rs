use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use sift::embeddings::EmbeddingBackend;
use sift::embeddings::EmbeddingGemma;
use sift::embeddings::EmbeddingWorker;

#[path = "common/embedding_gemma_goldens.rs"]
mod fixtures;

use fixtures::{EmbeddingGemmaGoldens, GoldenEmbedding, golden_by_name};

#[derive(Clone, Copy)]
struct ErrorBounds {
    max_absolute: f32,
    root_mean_square: f32,
}

const PORTABLE_BOUNDS: ErrorBounds = ErrorBounds {
    max_absolute: 3e-7,
    root_mean_square: 1e-7,
};
const LONG_CONTEXT_BOUNDS: ErrorBounds = ErrorBounds {
    max_absolute: 2e-6,
    root_mean_square: 1e-6,
};
#[cfg(feature = "rocm")]
const ROCM_BOUNDS: ErrorBounds = ErrorBounds {
    max_absolute: 1e-6,
    root_mean_square: 2e-7,
};
const BATCH_VARIANT_BOUNDS: ErrorBounds = ErrorBounds {
    max_absolute: 1e-6,
    root_mean_square: 3e-7,
};

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());

    let dot_product = left.iter().zip(right).map(|(x, y)| x * y).sum::<f32>();
    let left_norm = left.iter().map(|x| x * x).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot_product / (left_norm * right_norm)
}

fn error_metrics(actual: &[f32], golden: &GoldenEmbedding) -> (f32, f32, f32) {
    let similarity = cosine_similarity(actual, &golden.embedding);
    let (max_absolute_error, squared_error) = actual.iter().zip(&golden.embedding).fold(
        (0.0f32, 0.0f32),
        |(maximum, sum), (actual, expected)| {
            let error = (actual - expected).abs();
            (maximum.max(error), sum + error * error)
        },
    );
    let root_mean_square_error = (squared_error / actual.len() as f32).sqrt();
    (similarity, max_absolute_error, root_mean_square_error)
}

fn assert_matches_golden(actual: &[f32], golden: &GoldenEmbedding, bounds: ErrorBounds) {
    let (similarity, max_absolute_error, root_mean_square_error) = error_metrics(actual, golden);
    assert!(
        similarity > 0.999_999_5
            && max_absolute_error < bounds.max_absolute
            && root_mean_square_error < bounds.root_mean_square,
        "{} differed from its golden: cosine={similarity}, max_abs={max_absolute_error:e}, \
         rmse={root_mean_square_error:e}",
        golden.name,
    );
}

fn assert_model_matches_reference_batches(
    model: &EmbeddingGemma,
    backend: &str,
    bounds: ErrorBounds,
) -> Result<()> {
    let goldens = EmbeddingGemmaGoldens::load();
    let mut minimum_cosine = f32::INFINITY;
    let mut maximum_absolute = 0.0_f32;
    let mut maximum_rmse = 0.0_f32;

    let mut check = |actual: &[f32], golden: &GoldenEmbedding| {
        let (cosine, max_absolute, rmse) = error_metrics(actual, golden);
        minimum_cosine = minimum_cosine.min(cosine);
        maximum_absolute = maximum_absolute.max(max_absolute);
        maximum_rmse = maximum_rmse.max(rmse);
        let case_bounds = if golden.name == "over_context_window" {
            LONG_CONTEXT_BOUNDS
        } else {
            bounds
        };
        assert_matches_golden(actual, golden, case_bounds);
    };

    let query_texts = goldens
        .queries_768
        .iter()
        .map(|golden| golden.text.as_str())
        .collect::<Vec<_>>();
    let queries = model.embed_queries(&query_texts, 768)?;
    for (actual, golden) in queries.iter().zip(&goldens.queries_768) {
        check(actual, golden);
    }

    let document_texts = goldens
        .documents_768
        .iter()
        .map(|golden| golden.text.as_str())
        .collect::<Vec<_>>();
    let documents = model.embed_documents(&document_texts, 768)?;
    for (actual, golden) in documents.iter().zip(&goldens.documents_768) {
        check(actual, golden);
    }

    let document_texts = goldens
        .documents_128
        .iter()
        .map(|golden| golden.text.as_str())
        .collect::<Vec<_>>();
    let documents = model.embed_documents(&document_texts, 128)?;
    for (actual, golden) in documents.iter().zip(&goldens.documents_128) {
        check(actual, golden);
    }

    println!(
        "GOLDEN=PASS backend={backend} batches=reference min_cosine={minimum_cosine} \
         max_abs={maximum_absolute:e} max_rmse={maximum_rmse:e}"
    );
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn test_embedding_gemma() -> Result<()> {
    dotenvy::dotenv().ok();

    assert_model_matches_reference_batches(&EmbeddingGemma::load()?, "cpu", PORTABLE_BOUNDS)?;

    #[cfg(feature = "cuda")]
    assert_model_matches_reference_batches(
        &EmbeddingGemma::load_on(EmbeddingBackend::Cuda(0))?,
        "cuda",
        PORTABLE_BOUNDS,
    )?;

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
    assert_matches_golden(&queries[0], query_rust, PORTABLE_BOUNDS);
    assert_matches_golden(&queries[1], query_france, PORTABLE_BOUNDS);

    let documents = worker
        .embed_documents(
            vec![rust.text.clone(), france.text.clone(), rust.text.clone()],
            768,
        )
        .await?;
    assert_matches_golden(&documents[0], rust, BATCH_VARIANT_BOUNDS);
    assert_matches_golden(&documents[1], france, BATCH_VARIANT_BOUNDS);
    assert_matches_golden(&documents[2], rust, BATCH_VARIANT_BOUNDS);

    Ok(())
}

#[cfg(feature = "rocm")]
#[tokio::test(flavor = "current_thread")]
async fn test_embedding_gemma_rocm() -> Result<()> {
    dotenvy::dotenv().ok();
    let model = EmbeddingGemma::load_on(EmbeddingBackend::Rocm(0))?;
    assert_model_matches_reference_batches(&model, "rocm", ROCM_BOUNDS)
}
