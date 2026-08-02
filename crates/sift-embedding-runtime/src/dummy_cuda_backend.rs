//! Implementation of the Cuda backend when Cuda support has not been compiled in.
//!
#![allow(dead_code)]
use crate::op::{BinaryOpT, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct CudaDevice;

#[derive(Debug)]
pub struct CudaStorage;

macro_rules! fail {
    () => {
        unimplemented!("cuda support has not been enabled, add `cuda` feature to enable.")
    };
}
impl CudaDevice {}

impl crate::backend::BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn reduce_sum(&self, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

impl crate::backend::BackendDevice for CudaDevice {
    type Storage = CudaStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}
