use crate::backend::BackendDevice;
use crate::cpu::CpuDevice;
use crate::{DType, Result, Shape, Storage, WithDType};

/// A `DeviceLocation` represents a physical device whereas multiple `Device`
/// can live on the same location (typically for cuda devices).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProfile {
    pub free_bytes: usize,
    pub peak_bytes: usize,
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => device.location(),
        }
    }

    pub fn begin_memory_profile(&self) -> Result<bool> {
        match self {
            Self::Cpu => Ok(false),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => {
                device.begin_memory_profile()?;
                Ok(true)
            }
            #[cfg(not(feature = "cuda"))]
            Self::Cuda(_) => Err(crate::Error::NotCompiledWithCudaSupport),
        }
    }

    pub fn end_memory_profile(&self) -> Result<Option<MemoryProfile>> {
        match self {
            Self::Cpu => Ok(None),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => {
                let (free_bytes, peak_bytes) = device.end_memory_profile()?;
                Ok(Some(MemoryProfile {
                    free_bytes,
                    peak_bytes,
                }))
            }
            #[cfg(not(feature = "cuda"))]
            Self::Cuda(_) => Err(crate::Error::NotCompiledWithCudaSupport),
        }
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn storage_from_slice<D: WithDType>(&self, data: &[D]) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(D::to_cpu_storage(data))),
            Device::Cuda(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
        }
    }
}
