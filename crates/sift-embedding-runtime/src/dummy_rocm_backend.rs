//! Placeholder ROCm backend used when the `rocm` feature is disabled.
#![allow(dead_code)]

use crate::op::{BinaryOpT, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct RocmDevice;

#[derive(Debug)]
pub struct RocmStorage;

macro_rules! fail {
    () => {
        unimplemented!("ROCm support has not been enabled; add the `rocm` feature")
    };
}

impl crate::backend::BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn reduce_sum(&self, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}

impl crate::backend::BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn zeros_impl(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    unsafe fn alloc_uninit(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}
