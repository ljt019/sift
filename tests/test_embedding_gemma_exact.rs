use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use sift::embeddings::{EmbeddingBackend, EmbeddingGemma};

#[path = "common/embedding_gemma_inputs.rs"]
mod support;

use support::DOCUMENTS;

const MODEL_REVISION: &str = "57c266a740f537b4dc058e1b0cda161fd15afa75";
const QUERIES: [&str; 2] = ["how do I parse json in rust", "capital of france"];

struct Scenario<'a> {
    name: &'static str,
    kind: InputKind,
    documents: &'a [&'a str],
    dimensions: usize,
}

#[derive(Clone, Copy)]
enum InputKind {
    Query,
    Document,
}

struct Capture<'a> {
    scenario: Scenario<'a>,
    embeddings: Vec<Vec<f32>>,
}

fn capture<'a>(model: &EmbeddingGemma, scenarios: Vec<Scenario<'a>>) -> Result<Vec<Capture<'a>>> {
    scenarios
        .into_iter()
        .map(|scenario| {
            let embeddings = match scenario.kind {
                InputKind::Query => model.embed_queries(scenario.documents, scenario.dimensions)?,
                InputKind::Document => {
                    model.embed_documents(scenario.documents, scenario.dimensions)?
                }
            };
            Ok(Capture {
                scenario,
                embeddings,
            })
        })
        .collect()
}

fn raw_bytes(captures: &[Capture<'_>]) -> Vec<u8> {
    captures
        .iter()
        .flat_map(|capture| &capture.embeddings)
        .flatten()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(output, "{byte:02x}").expect("writing to a string cannot fail");
            output
        })
}

fn check_exact(expected: &[u8], captures: &[Capture<'_>]) -> Result<()> {
    let actual = raw_bytes(captures);
    ensure!(
        expected.len() == actual.len(),
        "EXACT=FAIL expected_bytes={} actual_bytes={}",
        expected.len(),
        actual.len()
    );

    let mut offset = 0;
    for capture in captures {
        for (document, embedding) in capture.embeddings.iter().enumerate() {
            for (dimension, &value) in embedding.iter().enumerate() {
                let expected_bits = u32::from_le_bytes(
                    expected[offset..offset + 4]
                        .try_into()
                        .expect("four-byte f32 fixture chunk"),
                );
                let actual_bits = value.to_bits();
                if expected_bits != actual_bits {
                    let expected_value = f32::from_bits(expected_bits);
                    bail!(
                        "EXACT=FAIL scenario={} document={} dimension={} \
                         expected_bits=0x{expected_bits:08x} actual_bits=0x{actual_bits:08x} \
                         expected={expected_value:e} actual={value:e}",
                        capture.scenario.name,
                        document,
                        dimension,
                    );
                }
                offset += 4;
            }
        }
    }
    Ok(())
}

fn fixture_path(filename: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/exact")
        .join(filename)
}

fn scenarios() -> Vec<Scenario<'static>> {
    vec![
        Scenario {
            name: "single-768",
            kind: InputKind::Document,
            documents: &DOCUMENTS[..1],
            dimensions: 768,
        },
        Scenario {
            name: "batch-8-768",
            kind: InputKind::Document,
            documents: &DOCUMENTS,
            dimensions: 768,
        },
        Scenario {
            name: "batch-8-128",
            kind: InputKind::Document,
            documents: &DOCUMENTS,
            dimensions: 128,
        },
        Scenario {
            name: "queries-2-768",
            kind: InputKind::Query,
            documents: &QUERIES,
            dimensions: 768,
        },
    ]
}

fn check_backend(backend_name: &str, backend: EmbeddingBackend, fixture_name: &str) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "exactness tests must run with --release"
    );
    dotenvy::dotenv().ok();

    let model = EmbeddingGemma::load_on(backend)?;
    let captures = capture(&model, scenarios())?;
    let actual = raw_bytes(&captures);
    let fixture = fixture_path(fixture_name);

    if std::env::var("SIFT_BLESS_EXACT").as_deref() == Ok("1") {
        fs::write(&fixture, &actual)
            .with_context(|| format!("failed to write {}", fixture.display()))?;
        println!(
            "EXACT=BLESSED backend={backend_name} bytes={} sha256={} model_revision={MODEL_REVISION}",
            actual.len(),
            digest(&actual),
        );
        return Ok(());
    }

    let expected = fs::read(&fixture).with_context(|| {
        format!(
            "missing exact fixture {}; rerun with SIFT_BLESS_EXACT=1",
            fixture.display()
        )
    })?;
    check_exact(&expected, &captures)?;
    println!(
        "EXACT=PASS backend={backend_name} elements={} sha256={} model_revision={MODEL_REVISION}",
        actual.len() / 4,
        digest(&actual),
    );
    Ok(())
}

#[test]
#[ignore = "loads EmbeddingGemma and checks machine-specific CPU output"]
fn embedding_gemma_cpu_is_bit_exact() -> Result<()> {
    check_backend(
        "cpu",
        EmbeddingBackend::Cpu,
        "embedding_gemma_cpu_zen5_f32.bin",
    )
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads EmbeddingGemma and checks machine-specific CUDA output"]
fn embedding_gemma_cuda_is_bit_exact() -> Result<()> {
    check_backend(
        "cuda",
        EmbeddingBackend::Cuda(0),
        "embedding_gemma_cuda_sm86_cuda12.9_f32.bin",
    )
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "loads EmbeddingGemma and checks gfx1151 hipBLAS output"]
fn embedding_gemma_rocm_hipblas_is_bit_exact() -> Result<()> {
    ensure!(
        std::env::var("SIFT_ROCM_FORCE_HIPRTC_MATMUL").as_deref() != Ok("1"),
        "hipBLAS exactness requires SIFT_ROCM_FORCE_HIPRTC_MATMUL to be unset"
    );
    check_backend(
        "rocm-hipblas",
        EmbeddingBackend::Rocm(0),
        "embedding_gemma_rocm_gfx1151_hipblas_rocm7.2_f32.bin",
    )
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "loads EmbeddingGemma and checks gfx1151 HIPRTC matmul output"]
fn embedding_gemma_rocm_hiprtc_is_bit_exact() -> Result<()> {
    ensure!(
        std::env::var("SIFT_ROCM_FORCE_HIPRTC_MATMUL").as_deref() == Ok("1"),
        "HIPRTC exactness requires SIFT_ROCM_FORCE_HIPRTC_MATMUL=1"
    );
    check_backend(
        "rocm-hiprtc",
        EmbeddingBackend::Rocm(0),
        "embedding_gemma_rocm_gfx1151_hiprtc_rocm7.2_f32.bin",
    )
}

#[test]
fn exact_comparison_accepts_identical_bits() {
    let captures = vec![diagnostic_capture()];
    check_exact(&raw_bytes(&captures), &captures).unwrap();
}

#[test]
fn mismatch_identifies_the_exact_element_and_bits() {
    let captures = vec![diagnostic_capture()];
    let mut expected = raw_bytes(&captures);
    expected[4] ^= 1;

    let error = check_exact(&expected, &captures).unwrap_err().to_string();
    assert!(error.contains("scenario=diagnostic document=0 dimension=1"));
    assert!(error.contains("expected_bits=0x40000001 actual_bits=0x40000000"));
}

fn diagnostic_capture() -> Capture<'static> {
    Capture {
        scenario: Scenario {
            name: "diagnostic",
            kind: InputKind::Document,
            documents: &["one"],
            dimensions: 2,
        },
        embeddings: vec![vec![1.0, 2.0]],
    }
}
