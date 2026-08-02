use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use crate::safetensors::MmapedSafetensors;
use crate::{DType, Device, Error, Result, Shape, Tensor};

/// Read-only access to the model's memory-mapped safetensor weights.
pub struct VarBuilder<'a> {
    weights: Arc<MmapedSafetensors>,
    path: Vec<String>,
    dtype: DType,
    device: Device,
    lifetime: PhantomData<&'a ()>,
}

impl Clone for VarBuilder<'_> {
    fn clone(&self) -> Self {
        Self {
            weights: Arc::clone(&self.weights),
            path: self.path.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            lifetime: PhantomData,
        }
    }
}

impl<'a> VarBuilder<'a> {
    /// Memory-map one or more safetensor files.
    ///
    /// # Safety
    ///
    /// The mapped files must not be modified while this builder or tensors loaded
    /// from it remain alive.
    pub unsafe fn from_mmaped_safetensors<P: AsRef<Path>>(
        paths: &[P],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // SAFETY: The caller upholds MmapedSafetensors' file-lifetime contract.
        let weights = unsafe { MmapedSafetensors::multi(paths)? };
        Ok(Self {
            weights: Arc::new(weights),
            path: Vec::new(),
            dtype,
            device: device.clone(),
            lifetime: PhantomData,
        })
    }

    pub fn pp(&self, component: impl ToString) -> Self {
        let mut path = self.path.clone();
        path.push(component.to_string());
        Self {
            weights: Arc::clone(&self.weights),
            path,
            dtype: self.dtype,
            device: self.device.clone(),
            lifetime: PhantomData,
        }
    }

    pub fn get(&self, shape: impl Into<Shape>, name: &str) -> Result<Tensor> {
        let path = if self.path.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{}", self.path.join("."), name)
        };
        let expected = shape.into();
        let tensor = self
            .weights
            .load(&path, &self.device)?
            .to_dtype(self.dtype)?;
        if tensor.shape() != &expected {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }
}
