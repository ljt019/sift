use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, DeviceLocation, Shape, WithDType};

use super::sys::{
    Api, HipBlasHandle, HipFunction, HipModule, HipStream, MEMCPY_DEVICE_TO_DEVICE,
    MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE,
};
use super::{kernels, Result, RocmError, RocmStorage, RocmStorageSlice};

/// A plain value that can be copied between host and ROCm device memory.
///
/// # Safety
///
/// Implementors must have no invalid bit patterns, pointers, references, or
/// other host-only state. Their in-memory representation must also match the
/// corresponding type used by the HIP kernels.
pub unsafe trait DeviceRepr: Copy + Send + Sync + 'static {}

unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for usize {}

pub struct RocmBuffer<T: DeviceRepr> {
    pointer: *mut T,
    len: usize,
    ordinal: usize,
    api: Arc<Api>,
    marker: PhantomData<T>,
}

unsafe impl<T: DeviceRepr> Send for RocmBuffer<T> {}
unsafe impl<T: DeviceRepr> Sync for RocmBuffer<T> {}

impl<T: DeviceRepr> std::fmt::Debug for RocmBuffer<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RocmBuffer")
            .field("pointer", &self.pointer)
            .field("len", &self.len)
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl<T: DeviceRepr> RocmBuffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn pointer(&self) -> *mut T {
        self.pointer
    }

    pub fn pointer_at(&self, offset: usize) -> Result<*mut T> {
        if offset > self.len {
            return Err(RocmError::new(format!(
                "ROCm buffer offset {offset} exceeds length {}",
                self.len
            )));
        }
        // SAFETY: The checked offset is within or one past the allocation.
        Ok(unsafe { self.pointer.add(offset) })
    }
}

impl<T: DeviceRepr> Drop for RocmBuffer<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        if let Err(error) = self.api.set_device(self.ordinal) {
            tracing_fallback(&format!(
                "failed to select ROCm device while freeing memory: {error}"
            ));
            return;
        }
        // SAFETY: This buffer uniquely owns the live hipMalloc allocation.
        if let Err(error) = unsafe { self.api.free(self.pointer.cast()) } {
            tracing_fallback(&format!("failed to free ROCm memory: {error}"));
        }
    }
}

fn tracing_fallback(message: &str) {
    eprintln!("sift ROCm warning: {message}");
}

struct RocmModuleHandle {
    handle: HipModule,
    api: Arc<Api>,
}

unsafe impl Send for RocmModuleHandle {}
unsafe impl Sync for RocmModuleHandle {}

impl Drop for RocmModuleHandle {
    fn drop(&mut self) {
        // SAFETY: This handle uniquely owns the module loaded by hipModuleLoadData.
        if let Err(error) = unsafe { self.api.module_unload(self.handle) } {
            tracing_fallback(&format!("failed to unload ROCm module: {error}"));
        }
    }
}

pub struct RocmFunction {
    function: HipFunction,
    module: Arc<RocmModuleHandle>,
}

unsafe impl Send for RocmFunction {}
unsafe impl Sync for RocmFunction {}

impl RocmFunction {
    pub fn launch(
        &self,
        device: &RocmDevice,
        config: LaunchConfig,
        arguments: &mut KernelArgs,
    ) -> Result<()> {
        let mut pointers = arguments.pointers();
        let _module_guard = &self.module;
        // SAFETY: The function belongs to the retained module and arguments encode the kernel ABI.
        unsafe {
            device.inner.api.launch(
                self.function,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                device.inner.stream,
                &mut pointers,
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub fn for_num_elems(elements: u32) -> Self {
        const THREADS: u32 = 256;
        Self {
            grid_dim: (elements.div_ceil(THREADS).max(1), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

enum KernelArgument {
    Usize(usize),
    U32(u32),
    I32(i32),
    F32(f32),
    Pointer(*mut c_void),
}

#[derive(Default)]
pub struct KernelArgs {
    values: Vec<KernelArgument>,
}

impl KernelArgs {
    pub fn push_usize(&mut self, value: usize) {
        self.values.push(KernelArgument::Usize(value));
    }

    pub fn push_u32(&mut self, value: u32) {
        self.values.push(KernelArgument::U32(value));
    }

    pub fn push_i32(&mut self, value: i32) {
        self.values.push(KernelArgument::I32(value));
    }

    pub fn push_f32(&mut self, value: f32) {
        self.values.push(KernelArgument::F32(value));
    }

    pub fn push_pointer<T>(&mut self, value: *mut T) {
        self.values.push(KernelArgument::Pointer(value.cast()));
    }

    fn pointers(&mut self) -> Vec<*mut c_void> {
        self.values
            .iter_mut()
            .map(|value| match value {
                KernelArgument::Usize(value) => std::ptr::from_mut(value).cast(),
                KernelArgument::U32(value) => std::ptr::from_mut(value).cast(),
                KernelArgument::I32(value) => std::ptr::from_mut(value).cast(),
                KernelArgument::F32(value) => std::ptr::from_mut(value).cast(),
                KernelArgument::Pointer(value) => std::ptr::from_mut(value).cast(),
            })
            .collect()
    }
}

struct ModuleStore {
    modules: [Option<Arc<RocmModuleHandle>>; kernels::ALL_IDS.len()],
}

struct MemoryProfileState {
    active: AtomicBool,
    free_before: AtomicUsize,
    minimum_free: AtomicUsize,
}

struct Inner {
    api: Arc<Api>,
    ordinal: usize,
    stream: HipStream,
    blas: Option<HipBlasHandle>,
    modules: RwLock<ModuleStore>,
    memory_profile: MemoryProfileState,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Err(error) = self.api.set_device(self.ordinal) {
            tracing_fallback(&format!(
                "failed to select ROCm device during shutdown: {error}"
            ));
            return;
        }
        if let Err(error) = self.api.stream_synchronize(self.stream) {
            tracing_fallback(&format!(
                "failed to synchronize ROCm stream during shutdown: {error}"
            ));
        }
        if let Some(blas) = self.blas {
            // SAFETY: Inner uniquely owns the hipBLAS handle.
            if let Err(error) = unsafe { self.api.blas_destroy(blas) } {
                tracing_fallback(&format!("failed to destroy hipBLAS handle: {error}"));
            }
        }
        // SAFETY: Inner uniquely owns the stream created during initialization.
        if let Err(error) = unsafe { self.api.stream_destroy(self.stream) } {
            tracing_fallback(&format!("failed to destroy ROCm stream: {error}"));
        }
    }
}

#[derive(Clone)]
pub struct RocmDevice {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "RocmDevice({})", self.inner.ordinal)
    }
}

impl RocmDevice {
    pub fn begin_memory_profile(&self) -> Result<()> {
        self.select()?;
        let (free, _) = self.inner.api.mem_get_info()?;
        self.inner
            .memory_profile
            .free_before
            .store(free, Ordering::Relaxed);
        self.inner
            .memory_profile
            .minimum_free
            .store(free, Ordering::Relaxed);
        self.inner
            .memory_profile
            .active
            .store(true, Ordering::Release);
        Ok(())
    }

    pub fn end_memory_profile(&self) -> Result<(usize, usize)> {
        self.record_memory_usage();
        self.inner
            .memory_profile
            .active
            .store(false, Ordering::Release);
        let free_before = self
            .inner
            .memory_profile
            .free_before
            .load(Ordering::Relaxed);
        let minimum_free = self
            .inner
            .memory_profile
            .minimum_free
            .load(Ordering::Relaxed);
        Ok((free_before, free_before.saturating_sub(minimum_free)))
    }

    fn select(&self) -> Result<()> {
        self.inner.api.set_device(self.inner.ordinal)
    }

    fn record_memory_usage(&self) {
        if !self.inner.memory_profile.active.load(Ordering::Acquire) {
            return;
        }
        if self.select().is_ok() {
            if let Ok((free, _)) = self.inner.api.mem_get_info() {
                self.inner
                    .memory_profile
                    .minimum_free
                    .fetch_min(free, Ordering::Relaxed);
            }
        }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<RocmBuffer<T>> {
        self.select()?;
        let pointer = if len == 0 {
            std::ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            self.inner
                .api
                .malloc(len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                    RocmError::new(format!("ROCm allocation length {len} overflows"))
                })?)?
                .cast()
        };
        let allocation = RocmBuffer {
            pointer,
            len,
            ordinal: self.inner.ordinal,
            api: self.inner.api.clone(),
            marker: PhantomData,
        };
        self.record_memory_usage();
        Ok(allocation)
    }

    pub fn alloc_zeros<T: DeviceRepr>(&self, len: usize) -> Result<RocmBuffer<T>> {
        // SAFETY: The buffer is initialized immediately below before it is observed.
        let allocation = unsafe { self.alloc::<T>(len)? };
        if len > 0 {
            // SAFETY: The allocation is valid for exactly len elements.
            unsafe {
                self.inner.api.memset(
                    allocation.pointer().cast(),
                    0,
                    len * std::mem::size_of::<T>(),
                )
            }?;
        }
        Ok(allocation)
    }

    pub fn clone_htod<T: DeviceRepr>(&self, source: &[T]) -> Result<RocmBuffer<T>> {
        // SAFETY: The allocation is initialized by the following host-to-device copy.
        let allocation = unsafe { self.alloc::<T>(source.len())? };
        if !source.is_empty() {
            // SAFETY: Both slices are valid for the same byte count.
            unsafe {
                self.inner.api.memcpy(
                    allocation.pointer().cast(),
                    source.as_ptr().cast(),
                    std::mem::size_of_val(source),
                    MEMCPY_HOST_TO_DEVICE,
                )
            }?;
        }
        Ok(allocation)
    }

    pub fn clone_dtoh<T: DeviceRepr>(&self, source: &RocmBuffer<T>) -> Result<Vec<T>> {
        self.select()?;
        let mut output = Vec::<T>::with_capacity(source.len());
        if source.len() > 0 {
            // SAFETY: The output has sufficient capacity and is initialized by the device copy.
            unsafe {
                self.inner.api.memcpy(
                    output.as_mut_ptr().cast(),
                    source.pointer().cast(),
                    source.len() * std::mem::size_of::<T>(),
                    MEMCPY_DEVICE_TO_HOST,
                )?;
                output.set_len(source.len());
            }
        }
        Ok(output)
    }

    pub fn memcpy_dtod<T: DeviceRepr>(
        &self,
        source: *const T,
        destination: *mut T,
        elements: usize,
    ) -> Result<()> {
        self.select()?;
        if elements == 0 {
            return Ok(());
        }
        // SAFETY: Callers provide non-overlapping device ranges of `elements` values.
        unsafe {
            self.inner.api.memcpy(
                destination.cast(),
                source.cast(),
                elements * std::mem::size_of::<T>(),
                MEMCPY_DEVICE_TO_DEVICE,
            )
        }
    }

    pub fn get_or_load_func(
        &self,
        function_name: &str,
        module: &kernels::Module,
    ) -> Result<RocmFunction> {
        self.select()?;
        if let Some(module) = self.inner.modules.read().unwrap().modules[module.index()].clone() {
            let function = self
                .inner
                .api
                .module_function(module.handle, function_name)?;
            return Ok(RocmFunction { function, module });
        }

        let mut modules = self.inner.modules.write().unwrap();
        let module = match modules.modules[module.index()].clone() {
            Some(module) => module,
            None => {
                let headers = [
                    ("cuda_utils.cuh", kernels::CUDA_UTILS_HEADER),
                    ("binary_op_macros.cuh", kernels::BINARY_OP_MACROS_HEADER),
                ];
                let code = self
                    .inner
                    .api
                    .compile(module.name(), module.source(), &headers)?;
                let handle = self.inner.api.module_load(&code)?;
                let loaded = Arc::new(RocmModuleHandle {
                    handle,
                    api: self.inner.api.clone(),
                });
                modules.modules[module.index()] = Some(loaded.clone());
                loaded
            }
        };
        let function = self
            .inner
            .api
            .module_function(module.handle, function_name)?;
        Ok(RocmFunction { function, module })
    }

    pub(crate) fn blas(&self) -> Option<HipBlasHandle> {
        self.inner.blas
    }

    pub(crate) fn api(&self) -> &Api {
        &self.inner.api
    }
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(ordinal: usize) -> crate::Result<Self> {
        let api = Arc::new(Api::load()?);
        let count = api.device_count()?;
        if ordinal >= count {
            return Err(RocmError::new(format!(
                "ROCm device ordinal {ordinal} is out of range; found {count} device(s)"
            ))
            .into());
        }
        api.set_device(ordinal)?;
        let stream = api.stream_create()?;
        let blas = match api.blas_create(stream) {
            Ok(blas) => blas,
            Err(error) => {
                // SAFETY: Initialization still uniquely owns this live stream.
                let _ = unsafe { api.stream_destroy(stream) };
                return Err(error.into());
            }
        };
        Ok(Self {
            inner: Arc::new(Inner {
                api,
                ordinal,
                stream,
                blas,
                modules: RwLock::new(ModuleStore {
                    modules: [const { None }; kernels::ALL_IDS.len()],
                }),
                memory_profile: MemoryProfileState {
                    active: AtomicBool::new(false),
                    free_before: AtomicUsize::new(0),
                    minimum_free: AtomicUsize::new(0),
                },
            }),
        })
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Rocm {
            gpu_id: self.inner.ordinal,
        }
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> crate::Result<RocmStorage> {
        let elements = shape.elem_count();
        let slice = match dtype {
            DType::U32 => RocmStorageSlice::U32(self.alloc_zeros(elements)?),
            DType::F32 => RocmStorageSlice::F32(self.alloc_zeros(elements)?),
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> crate::Result<RocmStorage> {
        let elements = shape.elem_count();
        let slice = match dtype {
            // SAFETY: The caller of BackendDevice::alloc_uninit accepts uninitialized storage.
            DType::U32 => RocmStorageSlice::U32(unsafe { self.alloc(elements)? }),
            // SAFETY: The caller of BackendDevice::alloc_uninit accepts uninitialized storage.
            DType::F32 => RocmStorageSlice::F32(unsafe { self.alloc(elements)? }),
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: WithDType>(&self, values: &[T]) -> crate::Result<RocmStorage> {
        let slice = match T::cpu_storage_ref(values) {
            CpuStorageRef::U32(values) => RocmStorageSlice::U32(self.clone_htod(values)?),
            CpuStorageRef::F32(values) => RocmStorageSlice::F32(self.clone_htod(values)?),
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> crate::Result<RocmStorage> {
        let slice = match storage {
            CpuStorage::U32(values) => RocmStorageSlice::U32(self.clone_htod(&values)?),
            CpuStorage::F32(values) => RocmStorageSlice::F32(self.clone_htod(&values)?),
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }
}
