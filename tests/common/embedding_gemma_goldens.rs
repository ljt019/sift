use serde::Deserialize;

const GOLDENS_JSON: &str = include_str!("../fixtures/embedding_gemma_goldens.json");

#[derive(Debug, Deserialize)]
pub struct EmbeddingGemmaGoldens {
    pub queries_768: Vec<GoldenEmbedding>,
    pub documents_768: Vec<GoldenEmbedding>,
    pub documents_128: Vec<GoldenEmbedding>,
}

impl EmbeddingGemmaGoldens {
    pub fn load() -> Self {
        serde_json::from_str(GOLDENS_JSON).expect("EmbeddingGemma goldens should be valid JSON")
    }
}

#[derive(Debug, Deserialize)]
pub struct GoldenEmbedding {
    pub name: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub fn golden_by_name<'a>(goldens: &'a [GoldenEmbedding], name: &str) -> &'a GoldenEmbedding {
    goldens
        .iter()
        .find(|golden| golden.name == name)
        .unwrap_or_else(|| panic!("golden embedding {name:?} should exist"))
}
