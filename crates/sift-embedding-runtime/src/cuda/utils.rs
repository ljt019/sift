use crate::{DType, Layout, Result};
use cudarc::driver::CudaSlice;

use super::{CudaDevice, CudaError, CudaStorageSlice};

pub trait Map1 {
    fn f(
        &self,
        source: &CudaSlice<f32>,
        device: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<f32>>;

    fn map(
        &self,
        storage: &CudaStorageSlice,
        device: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaStorageSlice> {
        match storage {
            CudaStorageSlice::F32(source) => {
                Ok(CudaStorageSlice::F32(self.f(source, device, layout)?))
            }
            CudaStorageSlice::U32(_) => Err(CudaError::UnsupportedDtype {
                dtype: DType::U32,
                op: "floating-point operation",
            }
            .into()),
        }
    }
}

pub trait Map2 {
    fn f(
        &self,
        left: &CudaSlice<f32>,
        left_layout: &Layout,
        right: &CudaSlice<f32>,
        right_layout: &Layout,
        device: &CudaDevice,
    ) -> Result<CudaSlice<f32>>;

    fn map(
        &self,
        left: &CudaStorageSlice,
        left_layout: &Layout,
        right: &CudaStorageSlice,
        right_layout: &Layout,
        device: &CudaDevice,
    ) -> Result<CudaStorageSlice> {
        match (left, right) {
            (CudaStorageSlice::F32(left), CudaStorageSlice::F32(right)) => Ok(
                CudaStorageSlice::F32(self.f(left, left_layout, right, right_layout, device)?),
            ),
            _ => Err(CudaError::InternalError("binary operation requires f32 tensors").into()),
        }
    }
}
