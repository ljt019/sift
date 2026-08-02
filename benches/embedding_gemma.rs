use std::hint::black_box;
use std::time::Instant;

use anyhow::{Result, bail};
use sift::embeddings::{EmbeddingBackend, EmbeddingGemma};

#[path = "../tests/common/embedding_gemma_inputs.rs"]
mod support;

use support::DOCUMENTS;

fn embed(model: &EmbeddingGemma, documents: &[&str], batched: bool) -> Result<()> {
    if batched {
        black_box(model.embed_documents(documents, 768)?);
    } else {
        for document in documents {
            black_box(model.embed_documents(&[document], 768)?);
        }
    }
    Ok(())
}

fn measure(model: &EmbeddingGemma, documents: &[&str], batched: bool) -> Result<f64> {
    let start = Instant::now();
    embed(model, black_box(documents), batched)?;
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

fn statistics(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let Some(backend) = std::env::args().nth(1) else {
        // `cargo test --all-targets` launches custom benchmark harnesses. It
        // supplies no backend, so leave model loading to explicit bench runs.
        return Ok(());
    };
    let backend = match backend.as_str() {
        "cpu" => EmbeddingBackend::Cpu,
        "cuda" => EmbeddingBackend::Cuda(0),
        "rocm" => EmbeddingBackend::Rocm(0),
        other => bail!("unsupported backend {other:?}; expected cpu, cuda, or rocm"),
    };
    let model = EmbeddingGemma::load_on(backend)?;

    for batch_size in [1, 2, 4, 8, 16, 32] {
        let documents = DOCUMENTS
            .iter()
            .copied()
            .cycle()
            .take(batch_size)
            .collect::<Vec<_>>();
        embed(&model, &documents, false)?;
        embed(&model, &documents, true)?;

        let serial = (0..5)
            .map(|_| measure(&model, &documents, false))
            .collect::<Result<Vec<_>>>()?;
        let batched = (0..5)
            .map(|_| measure(&model, &documents, true))
            .collect::<Result<Vec<_>>>()?;
        let (serial_mean, serial_stdev) = statistics(&serial);
        let (batched_mean, batched_stdev) = statistics(&batched);
        println!(
            "batch={batch_size} serial_mean_ms={serial_mean:.3} \
             serial_stdev_ms={serial_stdev:.3} batched_mean_ms={batched_mean:.3} \
             batched_stdev_ms={batched_stdev:.3}"
        );
    }

    Ok(())
}
