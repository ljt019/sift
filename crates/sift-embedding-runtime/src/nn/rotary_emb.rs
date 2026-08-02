use crate::{CpuStorage, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

struct RotaryEmbedding;

impl crate::CustomOp3 for RotaryEmbedding {
    fn name(&self) -> &'static str {
        "rotary-embedding"
    }

    fn cpu_fwd(
        &self,
        source: &CpuStorage,
        source_layout: &Layout,
        cosine: &CpuStorage,
        cosine_layout: &Layout,
        sine: &CpuStorage,
        sine_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn apply(
            source: &[f32],
            source_layout: &Layout,
            cosine: &[f32],
            cosine_layout: &Layout,
            sine: &[f32],
            sine_layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            fn contiguous<'a, T>(values: &'a [T], layout: &Layout, name: &str) -> Result<&'a [T]> {
                match layout.contiguous_offsets() {
                    Some((start, end)) => Ok(&values[start..end]),
                    None => crate::bail!("{name} must be contiguous"),
                }
            }
            let source = contiguous(source, source_layout, "RoPE input")?;
            let cosine = contiguous(cosine, cosine_layout, "RoPE cosine")?;
            let sine = contiguous(sine, sine_layout, "RoPE sine")?;
            let (batch, heads, tokens, dimension) = source_layout.shape().dims4()?;
            let batched_frequencies =
                cosine_layout.dims().len() == 3 && sine_layout.dims().len() == 3;
            let mut output = vec![0.0; batch * heads * tokens * dimension];

            source
                .par_chunks(tokens * dimension)
                .zip(output.par_chunks_mut(tokens * dimension))
                .enumerate()
                .for_each(|(batch_head, (source, output))| {
                    for token in 0..tokens {
                        for index in 0..dimension / 2 {
                            let left = token * dimension + index;
                            let right = left + dimension / 2;
                            let mut frequency = token * dimension / 2 + index;
                            if batched_frequencies {
                                frequency += (batch_head / heads) * tokens * dimension / 2;
                            }
                            output[left] =
                                source[left] * cosine[frequency] - source[right] * sine[frequency];
                            output[right] =
                                source[left] * sine[frequency] + source[right] * cosine[frequency];
                        }
                    }
                });

            Ok((
                CpuStorage::F32(output),
                (batch, heads, tokens, dimension).into(),
            ))
        }

        match (source, cosine, sine) {
            (CpuStorage::F32(x), CpuStorage::F32(c), CpuStorage::F32(s)) => {
                apply(x, source_layout, c, cosine_layout, s, sine_layout)
            }
            _ => crate::bail!("RoPE only supports f32"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        source: &crate::CudaStorage,
        source_layout: &Layout,
        cosine: &crate::CudaStorage,
        cosine_layout: &Layout,
        sine: &crate::CudaStorage,
        sine_layout: &Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
        use crate::cuda::{kernels, WrapErr};
        use crate::CudaDevice;

        fn apply(
            source: &CudaSlice<f32>,
            source_layout: &Layout,
            cosine: &CudaSlice<f32>,
            cosine_layout: &Layout,
            sine: &CudaSlice<f32>,
            sine_layout: &Layout,
            device: &CudaDevice,
        ) -> Result<CudaSlice<f32>> {
            let source = match source_layout.contiguous_offsets() {
                Some((start, end)) => source.slice(start..end),
                None => crate::bail!("RoPE input must be contiguous"),
            };
            let cosine = match cosine_layout.contiguous_offsets() {
                Some((start, end)) => cosine.slice(start..end),
                None => crate::bail!("RoPE cosine must be contiguous"),
            };
            let sine = match sine_layout.contiguous_offsets() {
                Some((start, end)) => sine.slice(start..end),
                None => crate::bail!("RoPE sine must be contiguous"),
            };
            let (batch, heads, tokens, dimension) = source_layout.shape().dims4()?;
            let frequency_batch_stride =
                if cosine_layout.dims().len() == 3 && sine_layout.dims().len() == 3 {
                    (heads * tokens * dimension) as u32
                } else {
                    0
                };
            let elements = batch * heads * tokens * dimension;
            let config = LaunchConfig::for_num_elems((elements / 2) as u32);
            let function = device.get_or_load_func("rope_f32", &kernels::REDUCE)?;
            // SAFETY: The launched kernel initializes every output element.
            let output = unsafe { device.alloc::<f32>(elements)? };
            let mut builder = function.builder();
            builder.arg(&source);
            builder.arg(&cosine);
            builder.arg(&sine);
            builder.arg(&output);
            crate::builder_arg!(
                builder,
                (batch * heads) as u32,
                (tokens * dimension) as u32,
                dimension as u32,
                frequency_batch_stride
            );
            // SAFETY: Kernel arguments and launch dimensions match reduce.cu's RoPE kernel.
            unsafe { builder.launch(config) }.w()?;
            Ok(output)
        }

        use crate::cuda::CudaStorageSlice::F32;
        let device = source.device();
        let slice = match (&source.slice, &cosine.slice, &sine.slice) {
            (F32(x), F32(c), F32(s)) => F32(apply(
                x,
                source_layout,
                c,
                cosine_layout,
                s,
                sine_layout,
                device,
            )?),
            _ => crate::bail!("RoPE only supports f32"),
        };
        Ok((
            crate::cuda::CudaStorage {
                slice,
                device: device.clone(),
            },
            source_layout.shape().clone(),
        ))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        source: &crate::RocmStorage,
        source_layout: &Layout,
        cosine: &crate::RocmStorage,
        cosine_layout: &Layout,
        sine: &crate::RocmStorage,
        sine_layout: &Layout,
    ) -> Result<(crate::RocmStorage, Shape)> {
        crate::rocm::rope_fwd(
            source,
            source_layout,
            cosine,
            cosine_layout,
            sine,
            sine_layout,
        )
    }
}

fn frequency_shape(frequencies: &Tensor, batch: usize) -> Result<(usize, usize)> {
    match *frequencies.dims() {
        [tokens, dimension] => Ok((tokens, dimension)),
        [frequency_batch, tokens, dimension] if frequency_batch == batch => Ok((tokens, dimension)),
        _ => crate::bail!("invalid RoPE frequency shape {:?}", frequencies.shape()),
    }
}

pub fn rope(input: &Tensor, cosine: &Tensor, sine: &Tensor) -> Result<Tensor> {
    let (batch, _, tokens, dimension) = input.dims4()?;
    let cosine_shape = frequency_shape(cosine, batch)?;
    let sine_shape = frequency_shape(sine, batch)?;
    if cosine_shape.1 * 2 != dimension
        || sine_shape.1 * 2 != dimension
        || tokens > cosine_shape.0
        || tokens > sine_shape.0
    {
        crate::bail!(
            "inconsistent RoPE shapes {:?}, {:?}, {:?}",
            input.shape(),
            cosine.shape(),
            sine.shape()
        )
    }
    if !input.is_contiguous() || !cosine.is_contiguous() || !sine.is_contiguous() {
        crate::bail!("RoPE input and frequencies must be contiguous")
    }
    input.apply_op3_no_bwd(cosine, sine, &RotaryEmbedding)
}
