use crate::backend::BackendStorage;
use crate::{CpuStorage, CpuStorageRef, Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    U32,
    F32,
}

pub trait WithDType:
    Sized
    + Copy
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Mul<Output = Self>
    + PartialOrd
    + std::fmt::Display
    + Send
    + Sync
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_>;
    fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage;
    fn cpu_storage_as_slice(storage: &CpuStorage) -> Result<&[Self]>;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }
}

macro_rules! with_dtype {
    ($type:ty, $variant:ident) => {
        impl WithDType for $type {
            fn zero() -> Self {
                0 as Self
            }

            fn one() -> Self {
                1 as Self
            }

            fn from_f64(value: f64) -> Self {
                value as Self
            }

            fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_> {
                CpuStorageRef::$variant(data)
            }

            fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage {
                CpuStorage::$variant(data)
            }

            fn cpu_storage_as_slice(storage: &CpuStorage) -> Result<&[Self]> {
                match storage {
                    CpuStorage::$variant(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$variant,
                        got: storage.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }
        }
    };
}

with_dtype!(u32, U32);
with_dtype!(f32, F32);
