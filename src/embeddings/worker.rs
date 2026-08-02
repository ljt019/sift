use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, oneshot};

use super::embedding_gemma::EmbeddingBatchLimits;
use super::{DocumentTokenSpan, EmbeddingBackend, EmbeddingGemma, EmbeddingTokenizer};

const QUEUE_CAPACITY: usize = 32;

#[derive(Clone)]
pub struct EmbeddingWorker {
    sender: mpsc::Sender<Request>,
    tokenizer: Arc<EmbeddingTokenizer>,
    _thread: Arc<WorkerThread>,
}

struct WorkerThread(std::sync::Mutex<Option<std::thread::JoinHandle<()>>>);

impl Drop for WorkerThread {
    fn drop(&mut self) {
        if let Some(thread) = self.0.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}

enum InputKind {
    Query,
    Document,
}

struct Request {
    kind: InputKind,
    inputs: Vec<String>,
    dimensions: usize,
    response: oneshot::Sender<Result<Vec<Vec<f32>>>>,
}

impl EmbeddingWorker {
    pub fn spawn(backend: EmbeddingBackend) -> Result<Self> {
        let (sender, mut receiver) = mpsc::channel::<Request>(QUEUE_CAPACITY);
        let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);

        let thread = std::thread::Builder::new()
            .name("sift-embeddings".into())
            .spawn(move || {
                let model = match EmbeddingGemma::load_on(backend) {
                    Ok(model) => model,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                };
                let limits = match model.automatic_document_batch_tokens() {
                    Ok(max_tokens) => EmbeddingBatchLimits { max_tokens },
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                };
                tracing::info!(
                    ?backend,
                    max_batch_tokens = limits.max_tokens,
                    "sized EmbeddingGemma document batches"
                );
                let tokenizer = match model.document_tokenizer() {
                    Ok(tokenizer) => tokenizer,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                };
                if ready_sender.send(Ok(tokenizer)).is_err() {
                    return;
                }

                while let Some(request) = receiver.blocking_recv() {
                    if request.response.is_closed() {
                        continue;
                    }
                    let inputs = request
                        .inputs
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>();
                    let result = match request.kind {
                        InputKind::Query => model.embed_queries(&inputs, request.dimensions),
                        InputKind::Document => {
                            model.embed_documents_with_limits(&inputs, request.dimensions, limits)
                        }
                    };
                    let _ = request.response.send(result);
                }
            })
            .context("failed to spawn embedding worker thread")?;

        let tokenizer = ready_receiver
            .recv()
            .context("embedding worker stopped during startup")?
            .context("failed to load EmbeddingGemma")?;
        Ok(Self {
            sender,
            tokenizer: Arc::new(tokenizer),
            _thread: Arc::new(WorkerThread(std::sync::Mutex::new(Some(thread)))),
        })
    }

    pub fn document_spans(
        &self,
        text: &str,
        max_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Vec<DocumentTokenSpan>> {
        self.tokenizer
            .document_spans(text, max_tokens, overlap_tokens)
    }

    pub fn document_input_tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer.document_input_tokens(text)
    }

    pub async fn embed_queries(
        &self,
        queries: Vec<String>,
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.submit(InputKind::Query, queries, dimensions).await
    }

    pub async fn embed_documents(
        &self,
        documents: Vec<String>,
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.submit(InputKind::Document, documents, dimensions)
            .await
    }

    async fn submit(
        &self,
        kind: InputKind,
        inputs: Vec<String>,
        dimensions: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Request {
                kind,
                inputs,
                dimensions,
                response,
            })
            .await
            .context("embedding worker has stopped")?;
        receiver
            .await
            .context("embedding worker dropped a request")?
    }
}
