use crate::{Module, Result, Tensor};

#[derive(Clone, Debug)]
pub struct Embedding {
    weights: Tensor,
    hidden_size: usize,
}

impl Module for Embedding {
    fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let mut output_shape = token_ids.dims().to_vec();
        output_shape.push(self.hidden_size);
        self.weights
            .index_select(&token_ids.flatten_all()?, 0)?
            .reshape(output_shape)
    }
}

pub fn embedding(
    vocabulary_size: usize,
    hidden_size: usize,
    weights: super::VarBuilder<'_>,
) -> Result<Embedding> {
    Ok(Embedding {
        weights: weights.get((vocabulary_size, hidden_size), "weight")?,
        hidden_size,
    })
}
