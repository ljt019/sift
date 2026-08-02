use crate::tensor::from_storage;
use crate::{CpuStorage, CudaStorage, Layout, Result, RocmStorage, Shape, Tensor};

pub trait CustomOp1 {
    fn name(&self) -> &'static str;

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)>;

    fn cuda_fwd(&self, _: &CudaStorage, _: &Layout) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no CUDA implementation for {}", self.name()).into(),
        ))
    }

    fn rocm_fwd(&self, _: &RocmStorage, _: &Layout) -> Result<(RocmStorage, Shape)> {
        Err(crate::Error::Rocm(
            format!("no ROCm implementation for {}", self.name()).into(),
        ))
    }
}

pub trait CustomOp3 {
    fn name(&self) -> &'static str;

    fn cpu_fwd(
        &self,
        first: &CpuStorage,
        first_layout: &Layout,
        second: &CpuStorage,
        second_layout: &Layout,
        third: &CpuStorage,
        third_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no CUDA implementation for {}", self.name()).into(),
        ))
    }

    fn rocm_fwd(
        &self,
        _: &RocmStorage,
        _: &Layout,
        _: &RocmStorage,
        _: &Layout,
        _: &RocmStorage,
        _: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        Err(crate::Error::Rocm(
            format!("no ROCm implementation for {}", self.name()).into(),
        ))
    }
}

impl Tensor {
    pub fn apply_op1_no_bwd(&self, operation: &impl CustomOp1) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op1(self.layout(), operation)?;
        Ok(from_storage(storage, shape))
    }

    pub fn apply_op3_no_bwd(
        &self,
        second: &Self,
        third: &Self,
        operation: &impl CustomOp3,
    ) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op3(
            self.layout(),
            second.storage(),
            second.layout(),
            third.storage(),
            third.layout(),
            operation,
        )?;
        Ok(from_storage(storage, shape))
    }
}
