use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use hf_hub::HFClientSync;
use serde::Deserialize;
use sift_embedding_runtime::nn::{Embedding, Linear, VarBuilder, embedding, linear_no_bias};
use sift_embedding_runtime::{D, DType, Device, DeviceLocation, MemoryProfile, Module, Tensor};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

use super::DOCUMENT_CHUNK_TOKENS;

const MODEL_OWNER: &str = "google";
const MODEL_NAME: &str = "embeddinggemma-300m";
const MODEL_REVISION: &str = "57c266a740f537b4dc058e1b0cda161fd15afa75";
const DOCUMENT_PROMPT: &str = "title: none | text: ";
const QUERY_PROMPT: &str = "task: search result | query: ";
const GPU_MEMORY_UTILIZATION_NUMERATOR: usize = 3;
const GPU_MEMORY_UTILIZATION_DENOMINATOR: usize = 4;

pub struct EmbeddingGemma {
    tokenizer: Tokenizer,
    encoder: TextEncoder,
    projection_in: Linear,
    projection_out: Linear,
    device: Device,
    pad_token_id: u32,
    max_input_tokens: usize,
    max_batch_length_difference: usize,
}

#[derive(Clone)]
pub struct EmbeddingTokenizer {
    tokenizer: Tokenizer,
    max_input_tokens: usize,
    document_prompt_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocumentTokenSpan {
    pub start: usize,
    pub end: usize,
    pub tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingBackend {
    Cpu,
    Cuda(usize),
    Rocm(usize),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("expected cpu, cuda, cuda:N, rocm, or rocm:N")]
pub struct ParseEmbeddingBackendError;

impl std::str::FromStr for EmbeddingBackend {
    type Err = ParseEmbeddingBackendError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            "rocm" => Ok(Self::Rocm(0)),
            _ => {
                let (backend, ordinal) = value.split_once(':').ok_or(ParseEmbeddingBackendError)?;
                let ordinal = ordinal.parse().map_err(|_| ParseEmbeddingBackendError)?;
                match backend {
                    "cuda" => Ok(Self::Cuda(ordinal)),
                    "rocm" => Ok(Self::Rocm(ordinal)),
                    _ => Err(ParseEmbeddingBackendError),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddingBatchLimits {
    pub max_tokens: usize,
}

impl EmbeddingGemma {
    pub fn load() -> Result<Self> {
        Self::load_on(EmbeddingBackend::Cpu)
    }

    pub fn load_on(backend: EmbeddingBackend) -> Result<Self> {
        let files = ModelFiles::download()?;
        let config: Config = serde_json::from_slice(
            &fs::read(&files.config).context("failed to read EmbeddingGemma config")?,
        )
        .context("failed to parse EmbeddingGemma config")?;
        config.validate()?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|error| anyhow::anyhow!("failed to load EmbeddingGemma tokenizer: {error}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_position_embeddings,
                ..TruncationParams::default()
            }))
            .map_err(|error| {
                anyhow::anyhow!("failed to configure EmbeddingGemma tokenizer: {error}")
            })?;
        let device = match backend {
            EmbeddingBackend::Cpu => Device::Cpu,
            EmbeddingBackend::Cuda(ordinal) => Device::new_cuda(ordinal)
                .with_context(|| format!("failed to initialize CUDA device {ordinal}"))?,
            EmbeddingBackend::Rocm(ordinal) => Device::new_rocm(ordinal)
                .with_context(|| format!("failed to initialize ROCm device {ordinal}"))?,
        };
        let model_weights = mmap_weights(&files.model, &device)?;
        let projection_in_weights = mmap_weights(&files.projection_in, &device)?;
        let projection_out_weights = mmap_weights(&files.projection_out, &device)?;

        let encoder = TextEncoder::load(&config, model_weights)?;
        let projection_in = linear_no_bias(
            config.hidden_size,
            config.projection_size,
            projection_in_weights.pp("linear"),
        )?;
        let projection_out = linear_no_bias(
            config.projection_size,
            config.hidden_size,
            projection_out_weights.pp("linear"),
        )?;
        // A padded query position outside every valid token's sliding window
        // would softmax an all-masked row and produce NaNs. Keep each batch's
        // token-length spread strictly inside the local attention window.
        let max_batch_length_difference =
            if config.layer_types.contains(&LayerType::SlidingAttention) {
                config.bidirectional_window() - 1
            } else {
                usize::MAX
            };

        Ok(Self {
            tokenizer,
            encoder,
            projection_in,
            projection_out,
            device,
            pad_token_id: config.pad_token_id,
            max_input_tokens: config.max_position_embeddings,
            max_batch_length_difference,
        })
    }

    pub fn document_tokenizer(&self) -> Result<EmbeddingTokenizer> {
        EmbeddingTokenizer::new(self.tokenizer.clone(), self.max_input_tokens)
    }

    pub(crate) fn automatic_document_batch_tokens(&self) -> Result<usize> {
        match self.device.location() {
            DeviceLocation::Cpu => Ok(std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .saturating_mul(DOCUMENT_CHUNK_TOKENS)),
            DeviceLocation::Cuda { .. } | DeviceLocation::Rocm { .. } => {
                self.profile_gpu_document_batch_tokens()
            }
        }
    }

    fn profile_gpu_document_batch_tokens(&self) -> Result<usize> {
        let source = "sift embedding batch calibration ".repeat(DOCUMENT_CHUNK_TOKENS);
        let tokenizer = self.document_tokenizer()?;
        let span = tokenizer
            .document_spans(&source, DOCUMENT_CHUNK_TOKENS, 0)?
            .into_iter()
            .next()
            .context("batch calibration produced no document chunk")?;
        let document = &source[span.start..span.end];
        let input_tokens = tokenizer.document_input_tokens(document)?;

        ensure!(
            self.device.begin_memory_profile()?,
            "GPU memory profiling is unavailable"
        );
        let embedding = self.embed_documents(&[document], self.encoder.hidden_size);
        let profile = self.device.end_memory_profile();
        embedding.context("EmbeddingGemma batch calibration failed")?;
        let profile = profile?.context("GPU memory profiling did not produce a measurement")?;

        Ok(document_batch_tokens_from_profile(profile, input_tokens))
    }

    pub fn embed_documents(&self, documents: &[&str], dimensions: usize) -> Result<Vec<Vec<f32>>> {
        self.embed(documents, DOCUMENT_PROMPT, dimensions, None)
    }

    pub fn embed_queries(&self, queries: &[&str], dimensions: usize) -> Result<Vec<Vec<f32>>> {
        self.embed(queries, QUERY_PROMPT, dimensions, None)
    }

    pub(crate) fn embed_documents_with_limits(
        &self,
        documents: &[&str],
        dimensions: usize,
        limits: EmbeddingBatchLimits,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed(documents, DOCUMENT_PROMPT, dimensions, Some(limits))
    }

    fn embed(
        &self,
        texts: &[&str],
        prompt: &str,
        dimensions: usize,
        limits: Option<EmbeddingBatchLimits>,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(!texts.is_empty(), "at least one input is required");
        ensure!(
            (1..=self.encoder.hidden_size).contains(&dimensions),
            "embedding dimensions must be between 1 and {}",
            self.encoder.hidden_size
        );

        let inputs = texts
            .iter()
            .map(|text| format!("{prompt}{text}"))
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|error| anyhow::anyhow!("failed to tokenize embedding inputs: {error}"))?;
        ensure!(
            encodings.iter().all(|encoding| !encoding.is_empty()),
            "tokenizer produced no tokens"
        );

        let ranges = split_batch_ranges(
            &encodings.iter().map(Encoding::len).collect::<Vec<_>>(),
            limits.unwrap_or(EmbeddingBatchLimits {
                max_tokens: usize::MAX,
            }),
            self.max_batch_length_difference,
        )?;
        let mut embeddings = Vec::with_capacity(texts.len());
        for range in ranges {
            embeddings.extend(self.embed_batch(&encodings[range], dimensions)?);
        }
        Ok(embeddings)
    }

    fn embed_batch(&self, encodings: &[Encoding], dimensions: usize) -> Result<Vec<Vec<f32>>> {
        let batch_size = encodings.len();
        let sequence_length = encodings
            .iter()
            .map(Encoding::len)
            .max()
            .expect("embedding batches are non-empty");
        let mut input_ids = Vec::with_capacity(batch_size * sequence_length);
        let mut valid_tokens = Vec::with_capacity(batch_size * sequence_length);
        let mut token_counts = Vec::with_capacity(batch_size);
        for encoding in encodings {
            input_ids.extend_from_slice(encoding.get_ids());
            input_ids.resize(
                input_ids.len() + sequence_length - encoding.len(),
                self.pad_token_id,
            );

            valid_tokens.extend(
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&value| value as f32),
            );
            valid_tokens.resize(valid_tokens.len() + sequence_length - encoding.len(), 0.0);
            token_counts.push(encoding.get_attention_mask().iter().sum::<u32>() as f32);
        }
        let attention_bias = valid_tokens
            .iter()
            .map(|&value| if value == 0.0 { f32::NEG_INFINITY } else { 0.0 })
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_vec(input_ids, (batch_size, sequence_length), &self.device)?;
        let attention_bias = Tensor::from_vec(
            attention_bias,
            (batch_size, 1, 1, sequence_length),
            &self.device,
        )?;
        let pooling_mask =
            Tensor::from_vec(valid_tokens, (batch_size, sequence_length, 1), &self.device)?;
        let token_counts = Tensor::from_vec(token_counts, (batch_size, 1), &self.device)?;

        let token_embeddings = self.encoder.forward(&input_ids, &attention_bias)?;
        let pooled = token_embeddings
            .broadcast_mul(&pooling_mask)?
            .sum(1)?
            .broadcast_div(&token_counts)?;
        let projected = pooled
            .apply(&self.projection_in)?
            .apply(&self.projection_out)?
            .narrow(D::Minus1, 0, dimensions)?;
        let norm = projected.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let embeddings = projected
            .broadcast_div(&norm)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("failed to read EmbeddingGemma output")?;
        Ok(embeddings
            .chunks_exact(dimensions)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl EmbeddingTokenizer {
    fn new(mut tokenizer: Tokenizer, max_input_tokens: usize) -> Result<Self> {
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!("failed to configure chunk tokenizer: {error}"))?;
        let document_prompt_tokens = tokenizer
            .encode(DOCUMENT_PROMPT, true)
            .map_err(|error| anyhow::anyhow!("failed to tokenize document prompt: {error}"))?
            .len();
        Ok(Self {
            tokenizer,
            max_input_tokens,
            document_prompt_tokens,
        })
    }

    pub fn document_spans(
        &self,
        text: &str,
        max_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Vec<DocumentTokenSpan>> {
        ensure!(
            max_tokens <= self.max_input_tokens,
            "document chunk size {max_tokens} exceeds the model limit of {} tokens",
            self.max_input_tokens
        );
        ensure!(
            max_tokens > self.document_prompt_tokens,
            "document chunk size must exceed the {}-token document prompt",
            self.document_prompt_tokens
        );
        let mut content_tokens = max_tokens - self.document_prompt_tokens;

        loop {
            ensure!(
                overlap_tokens < content_tokens,
                "document chunk overlap must be smaller than the available content window"
            );
            let spans = self.spans_with_content_limit(text, content_tokens, overlap_tokens)?;
            let excess = spans.iter().try_fold(0, |largest, span| {
                let input_tokens = self.document_input_tokens(&text[span.start..span.end])?;
                Ok::<_, anyhow::Error>(largest.max(input_tokens.saturating_sub(max_tokens)))
            })?;
            if excess == 0 {
                return Ok(spans);
            }

            // Prefixing can change the tokenization at the prompt/content seam,
            // so the prompt's standalone token count is only an initial bound.
            // Tighten it until every emitted slice fits the actual model input.
            content_tokens = content_tokens.checked_sub(excess).filter(|&limit| {
                limit > overlap_tokens
            }).ok_or_else(|| {
                anyhow::anyhow!(
                    "document chunk size {max_tokens} leaves no room beyond the {overlap_tokens}-token overlap"
                )
            })?;
        }
    }

    pub fn document_input_tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer
            .encode(format!("{DOCUMENT_PROMPT}{text}"), true)
            .map(|encoding| encoding.len())
            .map_err(|error| anyhow::anyhow!("failed to tokenize document input: {error}"))
    }

    fn spans_with_content_limit(
        &self,
        text: &str,
        content_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Vec<DocumentTokenSpan>> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: content_tokens,
                stride: overlap_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|error| anyhow::anyhow!("failed to configure chunk tokenizer: {error}"))?;
        let encoding = tokenizer.encode(text, false).map_err(|error| {
            anyhow::anyhow!("failed to tokenize document for chunking: {error}")
        })?;

        std::iter::once(&encoding)
            .chain(encoding.get_overflowing())
            .filter(|encoding| !encoding.is_empty())
            .map(|encoding| {
                let (start, end) = encoding
                    .get_offsets()
                    .iter()
                    .copied()
                    .filter(|(start, end)| start < end)
                    .fold((usize::MAX, 0), |(minimum, maximum), (start, end)| {
                        (minimum.min(start), maximum.max(end))
                    });
                ensure!(
                    start != usize::MAX,
                    "chunk tokenizer produced no text offsets"
                );
                ensure!(
                    start <= end
                        && end <= text.len()
                        && text.is_char_boundary(start)
                        && text.is_char_boundary(end),
                    "chunk tokenizer produced invalid text offsets {start}..{end}"
                );
                Ok(DocumentTokenSpan {
                    start,
                    end,
                    tokens: encoding.len(),
                })
            })
            .collect()
    }
}

fn split_batch_ranges(
    lengths: &[usize],
    limits: EmbeddingBatchLimits,
    max_length_difference: usize,
) -> Result<Vec<Range<usize>>> {
    ensure!(
        limits.max_tokens > 0,
        "embedding token limit must be positive"
    );

    let mut ranges = Vec::new();
    let mut start = 0;
    let mut longest = 0;
    let mut shortest = usize::MAX;
    for (index, &length) in lengths.iter().enumerate() {
        ensure!(
            length <= limits.max_tokens,
            "input {index} has {length} tokens, exceeding the per-batch limit of {}",
            limits.max_tokens
        );
        let candidate_longest = longest.max(length);
        let candidate_shortest = shortest.min(length);
        let candidate_count = index - start + 1;
        let exceeds_tokens = candidate_longest > limits.max_tokens / candidate_count;
        let exceeds_length_spread = candidate_longest - candidate_shortest > max_length_difference;
        if exceeds_tokens || exceeds_length_spread {
            ranges.push(start..index);
            start = index;
            longest = length;
            shortest = length;
        } else {
            longest = candidate_longest;
            shortest = candidate_shortest;
        }
    }
    ranges.push(start..lengths.len());
    Ok(ranges)
}

fn document_batch_tokens_from_profile(profile: MemoryProfile, input_tokens: usize) -> usize {
    if profile.peak_bytes == 0 || input_tokens == 0 {
        return DOCUMENT_CHUNK_TOKENS;
    }

    // Attention memory is quadratic in sequence length. Scaling the measured
    // peak by the squared length ratio safely covers a calibration chunk that
    // happened to tokenize a few tokens short of the production limit.
    let target_squared = (DOCUMENT_CHUNK_TOKENS as u128).pow(2);
    let input_squared = (input_tokens as u128).pow(2);
    let peak_at_limit = (profile.peak_bytes as u128)
        .saturating_mul(target_squared)
        .div_ceil(input_squared)
        .min(usize::MAX as u128) as usize;
    let usable_memory = profile
        .free_bytes
        .saturating_mul(GPU_MEMORY_UTILIZATION_NUMERATOR)
        / GPU_MEMORY_UTILIZATION_DENOMINATOR;
    let inputs = (usable_memory / peak_at_limit).max(1);
    inputs.saturating_mul(DOCUMENT_CHUNK_TOKENS)
}

fn mmap_weights<'a>(path: &Path, device: &Device) -> Result<VarBuilder<'a>> {
    // SAFETY: sift-embedding owns the read-only mapping for the lifetime of the returned
    // VarBuilder. The cached model files are not modified while the model is loaded.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
        .with_context(|| format!("failed to map weights from {}", path.display()))
}

struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    model: PathBuf,
    projection_in: PathBuf,
    projection_out: PathBuf,
}

impl ModelFiles {
    fn download() -> Result<Self> {
        let client = HFClientSync::new().context("failed to create Hugging Face client")?;
        let repository = client.model(MODEL_OWNER, MODEL_NAME);
        let get = |filename: &str| {
            repository
                .download_file()
                .filename(filename)
                .revision(MODEL_REVISION)
                .send()
                .with_context(|| format!("failed to download {filename}"))
        };

        Ok(Self {
            config: get("config.json")?,
            tokenizer: get("tokenizer.json")?,
            model: get("model.safetensors")?,
            projection_in: get("2_Dense/model.safetensors")?,
            projection_out: get("3_Dense/model.safetensors")?,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
enum Activation {
    #[serde(rename = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

impl Module for Activation {
    fn forward(&self, input: &Tensor) -> sift_embedding_runtime::Result<Tensor> {
        match self {
            Self::GeluPytorchTanh => input.gelu(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Config {
    attention_bias: bool,
    head_dim: usize,
    hidden_activation: Activation,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<LayerType>,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    pad_token_id: u32,
    query_pre_attn_scalar: usize,
    rms_norm_eps: f64,
    rope_local_base_freq: f64,
    rope_theta: f64,
    sliding_window: usize,
    use_bidirectional_attention: bool,
    vocab_size: usize,
    #[serde(default = "default_projection_size")]
    projection_size: usize,
}

const fn default_projection_size() -> usize {
    3072
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(!self.attention_bias, "attention bias is not supported");
        ensure!(
            self.use_bidirectional_attention,
            "EmbeddingGemma must use bidirectional attention"
        );
        ensure!(
            self.num_hidden_layers == self.layer_types.len(),
            "layer type count does not match hidden layer count"
        );
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "attention heads must be divisible by key/value heads"
        );
        ensure!(
            self.num_attention_heads * self.head_dim == self.hidden_size,
            "attention dimensions do not match hidden size"
        );
        ensure!(self.sliding_window >= 2, "sliding window is too small");
        Ok(())
    }

    fn bidirectional_window(&self) -> usize {
        self.sliding_window / 2 + 1
    }
}

struct TextEncoder {
    token_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    hidden_size: usize,
}

impl TextEncoder {
    fn load(config: &Config, weights: VarBuilder<'_>) -> Result<Self> {
        let token_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            weights.pp("embed_tokens"),
        )?;
        let layer_weights = weights.pp("layers");
        let layers = config
            .layer_types
            .iter()
            .enumerate()
            .map(|(index, layer_type)| {
                DecoderLayer::load(config, *layer_type, layer_weights.pp(index))
            })
            .collect::<sift_embedding_runtime::Result<Vec<_>>>()?;
        let norm = GemmaRmsNorm::load(config.hidden_size, config.rms_norm_eps, weights.pp("norm"))?;

        Ok(Self {
            token_embeddings,
            layers,
            norm,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_bias: &Tensor,
    ) -> sift_embedding_runtime::Result<Tensor> {
        let mut hidden_states = self.token_embeddings.forward(input_ids)?;
        hidden_states = (hidden_states * (self.hidden_size as f64).sqrt())?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_bias)?;
        }

        self.norm.forward(&hidden_states)
    }
}

struct DecoderLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl DecoderLayer {
    fn load(
        config: &Config,
        layer_type: LayerType,
        weights: VarBuilder<'_>,
    ) -> sift_embedding_runtime::Result<Self> {
        Ok(Self {
            attention: Attention::load(config, layer_type, weights.pp("self_attn"))?,
            mlp: Mlp::load(config, weights.pp("mlp"))?,
            input_layernorm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                weights.pp("input_layernorm"),
            )?,
            post_attention_layernorm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                weights.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_layernorm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                weights.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_layernorm: GemmaRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                weights.pp("post_feedforward_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        attention_bias: &Tensor,
    ) -> sift_embedding_runtime::Result<Tensor> {
        let attention = self
            .attention
            .forward(&self.input_layernorm.forward(input)?, attention_bias)?;
        let hidden_states = (input + self.post_attention_layernorm.forward(&attention)?)?;
        let feedforward = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&hidden_states)?)?;
        hidden_states + self.post_feedforward_layernorm.forward(&feedforward)?
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    rotary: RotaryEmbedding,
    sliding_window: Option<usize>,
}

impl Attention {
    fn load(
        config: &Config,
        layer_type: LayerType,
        weights: VarBuilder<'_>,
    ) -> sift_embedding_runtime::Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                weights.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                weights.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                weights.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                weights.pp("o_proj"),
            )?,
            q_norm: GemmaRmsNorm::load(config.head_dim, config.rms_norm_eps, weights.pp("q_norm"))?,
            k_norm: GemmaRmsNorm::load(config.head_dim, config.rms_norm_eps, weights.pp("k_norm"))?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.query_pre_attn_scalar as f64).sqrt(),
            rotary: RotaryEmbedding::new(
                config.head_dim,
                match layer_type {
                    LayerType::SlidingAttention => config.rope_local_base_freq,
                    LayerType::FullAttention => config.rope_theta,
                },
            ),
            sliding_window: (layer_type == LayerType::SlidingAttention)
                .then(|| config.bidirectional_window()),
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        attention_bias: &Tensor,
    ) -> sift_embedding_runtime::Result<Tensor> {
        let (batch_size, sequence_length, _) = input.dims3()?;
        let query = input
            .apply(&self.q_proj)?
            .reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = input
            .apply(&self.k_proj)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_kv_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?;
        let value = input
            .apply(&self.v_proj)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_kv_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?;
        let query = self.q_norm.forward(&query)?;
        let key = self.k_norm.forward(&key)?;
        let (query, key) = self.rotary.apply(&query, &key)?;
        let groups = self.num_heads / self.num_kv_heads;
        let key = repeat_key_values(&key, groups)?;
        let value = repeat_key_values(&value, groups)?;

        let mut scores =
            (query.matmul(&key.transpose(2, 3)?)? * self.scale)?.broadcast_add(attention_bias)?;
        if let Some(window) = self.sliding_window
            && sequence_length > window
        {
            scores = scores.broadcast_add(&sliding_attention_mask(
                sequence_length,
                window,
                scores.device(),
            )?)?;
        }
        let probabilities = sift_embedding_runtime::nn::ops::softmax_last_dim(&scores)?;
        probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch_size, sequence_length, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

fn repeat_key_values(input: &Tensor, groups: usize) -> sift_embedding_runtime::Result<Tensor> {
    if groups == 1 {
        return Ok(input.clone());
    }
    let (batch_size, key_value_heads, sequence_length, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((
            batch_size,
            key_value_heads,
            groups,
            sequence_length,
            head_dim,
        ))?
        .reshape((
            batch_size,
            key_value_heads * groups,
            sequence_length,
            head_dim,
        ))
}

fn sliding_attention_mask(
    sequence_length: usize,
    window: usize,
    device: &Device,
) -> sift_embedding_runtime::Result<Tensor> {
    let values = (0..sequence_length)
        .flat_map(|query| {
            (0..sequence_length).map(move |key| {
                if query.abs_diff(key) < window {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, sequence_length, sequence_length), device)
}

struct RotaryEmbedding {
    head_dim: usize,
    base: f64,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, base: f64) -> Self {
        Self { head_dim, base }
    }

    fn apply(
        &self,
        query: &Tensor,
        key: &Tensor,
    ) -> sift_embedding_runtime::Result<(Tensor, Tensor)> {
        let sequence_length = query.dim(2)?;
        let inverse_frequencies = (0..self.head_dim)
            .step_by(2)
            .map(|index| 1.0f32 / self.base.powf(index as f64 / self.head_dim as f64) as f32)
            .collect::<Vec<_>>();
        let inverse_frequencies =
            Tensor::from_vec(inverse_frequencies, (1, self.head_dim / 2), query.device())?;
        let positions = Tensor::arange(0u32, sequence_length as u32, query.device())?
            .to_dtype(DType::F32)?
            .reshape((sequence_length, 1))?;
        let frequencies = positions.matmul(&inverse_frequencies)?;
        let cos = frequencies.cos()?.contiguous()?;
        let sin = frequencies.sin()?.contiguous()?;

        Ok((
            sift_embedding_runtime::nn::rotary_emb::rope(&query.contiguous()?, &cos, &sin)?,
            sift_embedding_runtime::nn::rotary_emb::rope(&key.contiguous()?, &cos, &sin)?,
        ))
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl Mlp {
    fn load(config: &Config, weights: VarBuilder<'_>) -> sift_embedding_runtime::Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                weights.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                weights.pp("up_proj"),
            )?,
            down_proj: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                weights.pp("down_proj"),
            )?,
            activation: config.hidden_activation,
        })
    }

    fn forward(&self, input: &Tensor) -> sift_embedding_runtime::Result<Tensor> {
        let gate = input.apply(&self.gate_proj)?.apply(&self.activation)?;
        (gate * input.apply(&self.up_proj)?)?.apply(&self.down_proj)
    }
}

struct GemmaRmsNorm {
    weight: Tensor,
    epsilon: f64,
}

impl GemmaRmsNorm {
    fn load(
        size: usize,
        epsilon: f64,
        weights: VarBuilder<'_>,
    ) -> sift_embedding_runtime::Result<Self> {
        Ok(Self {
            weight: weights.get(size, "weight")?,
            epsilon,
        })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, input: &Tensor) -> sift_embedding_runtime::Result<Tensor> {
        let hidden_size = input.dim(D::Minus1)?;
        let normalized = input.broadcast_div(
            &((input.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)? + self.epsilon)?
                .sqrt()?,
        )?;
        normalized.broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_tokens: usize) -> EmbeddingBatchLimits {
        EmbeddingBatchLimits { max_tokens }
    }

    #[test]
    fn batch_ranges_charge_for_padding() {
        assert_eq!(
            split_batch_ranges(&[2, 8, 3, 4], limits(16), usize::MAX).unwrap(),
            [0..2, 2..4]
        );
    }

    #[test]
    fn batch_ranges_keep_padding_inside_the_sliding_attention_window() {
        assert_eq!(
            split_batch_ranges(&[40, 400, 390], limits(2_000), 256).unwrap(),
            [0..1, 1..3]
        );
    }

    #[test]
    fn batch_ranges_reject_an_input_over_the_token_limit() {
        let error = split_batch_ranges(&[9], limits(8), usize::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("input 0 has 9 tokens"));
    }

    #[test]
    fn gpu_batch_capacity_keeps_memory_headroom() {
        let tokens = document_batch_tokens_from_profile(
            MemoryProfile {
                free_bytes: 1_000,
                peak_bytes: 100,
            },
            DOCUMENT_CHUNK_TOKENS,
        );
        assert_eq!(tokens, 7 * DOCUMENT_CHUNK_TOKENS);
    }

    #[test]
    fn gpu_batch_capacity_accounts_for_shorter_calibration_input() {
        let tokens = document_batch_tokens_from_profile(
            MemoryProfile {
                free_bytes: 1_000,
                peak_bytes: 100,
            },
            DOCUMENT_CHUNK_TOKENS / 2,
        );
        assert_eq!(tokens, DOCUMENT_CHUNK_TOKENS);
    }

    #[test]
    fn embedding_backend_parses_device_ordinals() {
        assert_eq!("cpu".parse(), Ok(EmbeddingBackend::Cpu));
        assert_eq!("cuda".parse(), Ok(EmbeddingBackend::Cuda(0)));
        assert_eq!("cuda:2".parse(), Ok(EmbeddingBackend::Cuda(2)));
        assert_eq!("rocm".parse(), Ok(EmbeddingBackend::Rocm(0)));
        assert_eq!("rocm:3".parse(), Ok(EmbeddingBackend::Rocm(3)));
        assert!("rocm:".parse::<EmbeddingBackend>().is_err());
    }
}
