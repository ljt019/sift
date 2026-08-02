//! Traits to Define Backend Behavior
//!
use crate::op::{BinaryOpT, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

pub trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn dtype(&self) -> DType;

    fn device(&self) -> &Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self>;

    fn reduce_sum(&self, _: &Layout, _: &[usize]) -> Result<Self>;

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self>;

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self>;
    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self>;

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()>;
}

pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    // TODO: Make the usize generic and part of a generic DeviceLocation.
    fn new(_: usize) -> Result<Self>;

    fn location(&self) -> crate::DeviceLocation;

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage>;

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage>;
}
