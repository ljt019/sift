mod embedding_gemma;
mod worker;

pub(crate) const DOCUMENT_CHUNK_TOKENS: usize = 512;

pub use embedding_gemma::{
    DocumentTokenSpan, EmbeddingBackend, EmbeddingGemma, EmbeddingTokenizer,
};
pub use worker::EmbeddingWorker;
