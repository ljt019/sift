use crate::{CpuStorage, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

struct SoftmaxLastDim;

impl crate::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax(source: &[f32], layout: &Layout) -> Result<(CpuStorage, Shape)> {
            let source = match layout.contiguous_offsets() {
                Some((start, end)) => &source[start..end],
                None => crate::bail!("softmax input must be contiguous"),
            };
            let shape = layout.shape();
            let columns = shape.dims()[shape.rank() - 1];
            let mut output = vec![0.0f32; shape.elem_count()];
            source
                .par_chunks(columns)
                .zip(output.par_chunks_mut(columns))
                .for_each(|(source, output)| {
                    let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for (source, output) in source.iter().zip(output.iter_mut()) {
                        *output = (*source - maximum).exp();
                    }
                    let sum: f32 = output.iter().sum();
                    output.iter_mut().for_each(|value| *value /= sum);
                });
            Ok((CpuStorage::F32(output), shape.clone()))
        }

        match storage {
            CpuStorage::F32(values) => softmax(values, layout),
            _ => crate::bail!("unsupported dtype for softmax {storage:?}"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
        use crate::cuda::{kernels, Map1, WrapErr};
        use crate::CudaDevice;

        struct Softmax;
        impl Map1 for Softmax {
            fn f(
                &self,
                source: &CudaSlice<f32>,
                device: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<f32>> {
                let source = match layout.contiguous_offsets() {
                    Some((start, end)) => source.slice(start..end),
                    None => crate::bail!("softmax input must be contiguous"),
                };
                let elements = layout.shape().elem_count();
                let columns = layout.dims()[layout.dims().len() - 1];
                let config = LaunchConfig {
                    grid_dim: ((elements / columns) as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                };
                let function = device.get_or_load_func("softmax_f32", &kernels::REDUCE)?;
                // SAFETY: The launched kernel initializes every output element.
                let output = unsafe { device.alloc::<f32>(elements)? };
                let mut builder = function.builder();
                builder.arg(&source);
                builder.arg(&output);
                crate::builder_arg!(builder, columns as i32);
                // SAFETY: Kernel arguments and launch dimensions match softmax.cu.
                unsafe { builder.launch(config) }.w()?;
                Ok(output)
            }
        }

        let device = storage.device();
        let slice = Softmax.map(&storage.slice, device, layout)?;
        Ok((
            crate::cuda::CudaStorage {
                slice,
                device: device.clone(),
            },
            layout.shape().clone(),
        ))
    }
}

pub fn softmax_last_dim(input: &Tensor) -> Result<Tensor> {
    input.apply_op1_no_bwd(&SoftmaxLastDim)
}
