//! Read-only F32 safetensors loading for EmbeddingGemma weights.

use crate::{Device, Error, Result, Tensor};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::path::Path;

pub struct MmapedSafetensors {
    files: Vec<memmap2::Mmap>,
    routing: HashMap<String, usize>,
}

impl MmapedSafetensors {
    /// Memory-map model shards and index every tensor by name.
    ///
    /// # Safety
    ///
    /// The mapped files must not be modified while this value exists.
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut routing = HashMap::new();

        for (index, path) in paths.iter().enumerate() {
            let path = path.as_ref();
            let file =
                std::fs::File::open(path).map_err(|error| Error::from(error).with_path(path))?;
            let mapped = memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|error| Error::from(error).with_path(path))?;
            let tensors = SafeTensors::deserialize(&mapped)
                .map_err(|error| Error::from(error).with_path(path))?;
            for name in tensors.names() {
                routing.insert(name.to_owned(), index);
            }
            files.push(mapped);
        }

        Ok(Self { files, routing })
    }

    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let index = self.routing.get(name).copied().ok_or_else(|| {
            Error::CannotFindTensor {
                path: name.to_owned(),
            }
            .bt()
        })?;
        let tensors = SafeTensors::deserialize(&self.files[index])?;
        load_f32(tensors.tensor(name)?, device)
    }
}

fn load_f32(view: TensorView<'_>, device: &Device) -> Result<Tensor> {
    if view.dtype() != Dtype::F32 {
        return Err(Error::UnsupportedSafeTensorDtype(view.dtype()).bt());
    }

    let bytes = view.data();
    debug_assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);
    let mut values = vec![0.0f32; bytes.len() / std::mem::size_of::<f32>()];
    // SAFETY: `values` owns exactly `bytes.len()` writable bytes and the regions do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            values.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    Tensor::from_vec(values, view.shape(), device)
}
