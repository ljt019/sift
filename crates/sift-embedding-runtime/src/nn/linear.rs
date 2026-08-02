use crate::{Module, Result, Tensor};

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.weight.t()?;
        match *input.dims() {
            [batch, sequence, input_size] if input.is_contiguous() => input
                .reshape((batch * sequence, input_size))?
                .matmul(&weight)?
                .reshape((batch, sequence, ())),
            _ => input.matmul(&weight),
        }
    }
}

pub fn linear_no_bias(
    input_size: usize,
    output_size: usize,
    weights: super::VarBuilder<'_>,
) -> Result<Linear> {
    Ok(Linear {
        weight: weights.get((output_size, input_size), "weight")?,
    })
}
