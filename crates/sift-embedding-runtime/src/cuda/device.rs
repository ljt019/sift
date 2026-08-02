use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Result, Shape};
pub use cudarc;
use cudarc::driver::CudaFunction;
pub use sift_embedding_kernels as kernels;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use super::{CudaStorage, CudaStorageSlice, WrapErr};

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

#[derive(Clone)]
pub struct CudaDevice {
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    memory_profile: Arc<MemoryProfileState>,
}

#[derive(Debug)]
struct MemoryProfileState {
    active: AtomicBool,
    free_before: AtomicUsize,
    minimum_free: AtomicUsize,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({})", self.context.ordinal())
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let allocation = self.stream.alloc::<T>(len).w()?;
        self.record_memory_usage();
        Ok(allocation)
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let allocation = self.stream.alloc_zeros::<T>(len).w()?;
        self.record_memory_usage();
        Ok(allocation)
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn clone_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let allocation = self.stream.clone_htod(src).w()?;
        self.record_memory_usage();
        Ok(allocation)
    }

    pub fn begin_memory_profile(&self) -> Result<()> {
        let (free, _) = self.context.mem_get_info().w()?;
        self.memory_profile
            .free_before
            .store(free, Ordering::Relaxed);
        self.memory_profile
            .minimum_free
            .store(free, Ordering::Relaxed);
        self.memory_profile.active.store(true, Ordering::Release);
        Ok(())
    }

    pub fn end_memory_profile(&self) -> Result<(usize, usize)> {
        self.record_memory_usage();
        self.memory_profile.active.store(false, Ordering::Release);
        let free_before = self.memory_profile.free_before.load(Ordering::Relaxed);
        let minimum_free = self.memory_profile.minimum_free.load(Ordering::Relaxed);
        Ok((free_before, free_before.saturating_sub(minimum_free)))
    }

    fn record_memory_usage(&self) {
        if !self.memory_profile.active.load(Ordering::Acquire) {
            return;
        }
        if let Ok((free, _)) = self.context.mem_get_info() {
            self.memory_profile
                .minimum_free
                .fetch_min(free, Ordering::Relaxed);
        }
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    fn from_context_and_stream(
        context: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            context,
            stream,
            blas: Arc::new(blas),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            memory_profile: Arc::new(MemoryProfileState {
                active: AtomicBool::new(false),
                free_before: AtomicUsize::new(0),
                minimum_free: AtomicUsize::new(0),
            }),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.per_thread_stream();
        Self::from_context_and_stream(context, stream)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }
}
