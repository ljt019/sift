use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use super::{Result, RocmError};

pub type HipStream = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipRtcProgram = *mut c_void;
pub type HipBlasHandle = *mut c_void;

pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const HIPBLAS_OP_N: c_int = 111;
pub const HIPBLAS_OP_T: c_int = 112;

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;
const RTLD_LOCAL: c_int = 0;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

struct Library {
    handle: *mut c_void,
}

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    fn open(name: &str, global: bool) -> Result<Self> {
        Self::open_any(&[name], global)
    }

    fn open_any(names: &[&str], global: bool) -> Result<Self> {
        let mut failures = Vec::new();
        for name in names {
            for candidate in library_candidates(name) {
                let path = CString::new(candidate.as_os_str().as_bytes()).map_err(|_| {
                    RocmError::new(format!(
                        "ROCm library path contains a NUL byte: {}",
                        candidate.display()
                    ))
                })?;
                // SAFETY: `path` is NUL terminated and the flags are accepted by glibc's loader.
                let handle = unsafe {
                    dlopen(
                        path.as_ptr(),
                        RTLD_NOW | if global { RTLD_GLOBAL } else { RTLD_LOCAL },
                    )
                };
                if !handle.is_null() {
                    return Ok(Self { handle });
                }
                failures.push(format!("{}: {}", candidate.display(), loader_error()));
            }
        }
        Err(RocmError::new(format!(
            "failed to load {}; searched SIFT_ROCM_LIB_DIR, ROCM_PATH, HIP_PATH, and the system library path: {}",
            names.join(" or "),
            failures.join("; ")
        )))
    }

    unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T> {
        let name = CString::new(name).expect("symbol names do not contain NUL bytes");
        // SAFETY: Calling dlerror clears any previous dynamic-loader error.
        unsafe { dlerror() };
        // SAFETY: The library handle is live and `name` is NUL terminated.
        let pointer = unsafe { dlsym(self.handle, name.as_ptr()) };
        if pointer.is_null() {
            return Err(RocmError::new(format!(
                "failed to load ROCm symbol {}: {}",
                name.to_string_lossy(),
                loader_error()
            )));
        }
        // SAFETY: Callers specify the ABI declared by the corresponding ROCm header.
        Ok(unsafe { std::mem::transmute_copy(&pointer) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: `handle` came from a successful dlopen and is closed once here.
        unsafe { dlclose(self.handle) };
    }
}

fn loader_error() -> String {
    // SAFETY: dlerror returns either null or a process-owned NUL-terminated string.
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dynamic-loader error".to_owned()
    } else {
        // SAFETY: A non-null dlerror result is a valid C string until the next loader call.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

fn library_candidates(name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(paths) = std::env::var_os("SIFT_ROCM_LIB_DIR") {
        candidates.extend(std::env::split_paths(&paths).map(|path| path.join(name)));
    }
    for variable in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            let root = Path::new(&root);
            candidates.push(root.join("lib").join(name));
            candidates.push(root.join("lib64").join(name));
        }
    }
    candidates.push(PathBuf::from(OsStr::new(name)));
    candidates
}

type HipSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type HipStreamCreate = unsafe extern "C" fn(*mut HipStream) -> c_int;
type HipStreamDestroy = unsafe extern "C" fn(HipStream) -> c_int;
type HipStreamSynchronize = unsafe extern "C" fn(HipStream) -> c_int;
type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type HipMemset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
type HipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
type HipModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> c_int;
type HipModuleUnload = unsafe extern "C" fn(HipModule) -> c_int;
type HipModuleGetFunction =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> c_int;
type HipModuleLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

type HipRtcGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type HipRtcCreateProgram = unsafe extern "C" fn(
    *mut HipRtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type HipRtcDestroyProgram = unsafe extern "C" fn(*mut HipRtcProgram) -> c_int;
type HipRtcCompileProgram =
    unsafe extern "C" fn(HipRtcProgram, c_int, *const *const c_char) -> c_int;
type HipRtcGetProgramLogSize = unsafe extern "C" fn(HipRtcProgram, *mut usize) -> c_int;
type HipRtcGetProgramLog = unsafe extern "C" fn(HipRtcProgram, *mut c_char) -> c_int;
type HipRtcGetCodeSize = unsafe extern "C" fn(HipRtcProgram, *mut usize) -> c_int;
type HipRtcGetCode = unsafe extern "C" fn(HipRtcProgram, *mut c_char) -> c_int;

type HipBlasCreate = unsafe extern "C" fn(*mut HipBlasHandle) -> c_int;
type HipBlasDestroy = unsafe extern "C" fn(HipBlasHandle) -> c_int;
type HipBlasSetStream = unsafe extern "C" fn(HipBlasHandle, HipStream) -> c_int;
type HipBlasStatusToString = unsafe extern "C" fn(c_int) -> *const c_char;
type HipBlasSgemmStridedBatched = unsafe extern "C" fn(
    HipBlasHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    i64,
    *const f32,
    c_int,
    i64,
    *const f32,
    *mut f32,
    c_int,
    i64,
    c_int,
) -> c_int;

struct BlasApi {
    _library: Library,
    create: HipBlasCreate,
    destroy: HipBlasDestroy,
    set_stream: HipBlasSetStream,
    status_to_string: HipBlasStatusToString,
    sgemm_strided_batched: HipBlasSgemmStridedBatched,
}

impl BlasApi {
    fn load() -> Result<Self> {
        let library = Library::open_any(
            &["libhipblas.so", "libhipblas.so.3", "libhipblas.so.2"],
            false,
        )?;
        // SAFETY: Each function type mirrors its declaration in the ROCm C headers.
        unsafe {
            Ok(Self {
                create: library.symbol("hipblasCreate")?,
                destroy: library.symbol("hipblasDestroy")?,
                set_stream: library.symbol("hipblasSetStream")?,
                status_to_string: library.symbol("hipblasStatusToString")?,
                sgemm_strided_batched: library.symbol("hipblasSgemmStridedBatched")?,
                _library: library,
            })
        }
    }

    fn check(&self, status: c_int, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: hipblasStatusToString accepts every hipblasStatus_t value.
        let message = unsafe { (self.status_to_string)(status) };
        Err(RocmError::new(format!(
            "{operation} failed with hipBLAS status {status}: {}",
            c_string(message)
        )))
    }
}

pub struct Api {
    _hip: Library,
    _rtc: Library,
    blas: Option<BlasApi>,
    hip_set_device: HipSetDevice,
    hip_get_device_count: HipGetDeviceCount,
    hip_get_error_string: HipGetErrorString,
    hip_stream_create: HipStreamCreate,
    hip_stream_destroy: HipStreamDestroy,
    hip_stream_synchronize: HipStreamSynchronize,
    hip_malloc: HipMalloc,
    hip_free: HipFree,
    hip_memcpy: HipMemcpy,
    hip_memset: HipMemset,
    hip_mem_get_info: HipMemGetInfo,
    hip_module_load_data: HipModuleLoadData,
    hip_module_unload: HipModuleUnload,
    hip_module_get_function: HipModuleGetFunction,
    hip_module_launch_kernel: HipModuleLaunchKernel,
    hiprtc_get_error_string: HipRtcGetErrorString,
    hiprtc_create_program: HipRtcCreateProgram,
    hiprtc_destroy_program: HipRtcDestroyProgram,
    hiprtc_compile_program: HipRtcCompileProgram,
    hiprtc_get_program_log_size: HipRtcGetProgramLogSize,
    hiprtc_get_program_log: HipRtcGetProgramLog,
    hiprtc_get_code_size: HipRtcGetCodeSize,
    hiprtc_get_code: HipRtcGetCode,
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    pub fn load() -> Result<Self> {
        let hip = Library::open("libamdhip64.so", true)?;
        let rtc = Library::open("libhiprtc.so", false)?;
        let blas = if std::env::var("SIFT_ROCM_FORCE_HIPRTC_MATMUL").as_deref() == Ok("1") {
            None
        } else {
            BlasApi::load().ok()
        };
        // SAFETY: Each function type mirrors its declaration in the ROCm 6+ C headers.
        unsafe {
            Ok(Self {
                hip_set_device: hip.symbol("hipSetDevice")?,
                hip_get_device_count: hip.symbol("hipGetDeviceCount")?,
                hip_get_error_string: hip.symbol("hipGetErrorString")?,
                hip_stream_create: hip.symbol("hipStreamCreate")?,
                hip_stream_destroy: hip.symbol("hipStreamDestroy")?,
                hip_stream_synchronize: hip.symbol("hipStreamSynchronize")?,
                hip_malloc: hip.symbol("hipMalloc")?,
                hip_free: hip.symbol("hipFree")?,
                hip_memcpy: hip.symbol("hipMemcpy")?,
                hip_memset: hip.symbol("hipMemset")?,
                hip_mem_get_info: hip.symbol("hipMemGetInfo")?,
                hip_module_load_data: hip.symbol("hipModuleLoadData")?,
                hip_module_unload: hip.symbol("hipModuleUnload")?,
                hip_module_get_function: hip.symbol("hipModuleGetFunction")?,
                hip_module_launch_kernel: hip.symbol("hipModuleLaunchKernel")?,
                hiprtc_get_error_string: rtc.symbol("hiprtcGetErrorString")?,
                hiprtc_create_program: rtc.symbol("hiprtcCreateProgram")?,
                hiprtc_destroy_program: rtc.symbol("hiprtcDestroyProgram")?,
                hiprtc_compile_program: rtc.symbol("hiprtcCompileProgram")?,
                hiprtc_get_program_log_size: rtc.symbol("hiprtcGetProgramLogSize")?,
                hiprtc_get_program_log: rtc.symbol("hiprtcGetProgramLog")?,
                hiprtc_get_code_size: rtc.symbol("hiprtcGetCodeSize")?,
                hiprtc_get_code: rtc.symbol("hiprtcGetCode")?,
                _hip: hip,
                _rtc: rtc,
                blas,
            })
        }
    }

    fn hip_error(&self, status: c_int, operation: &str) -> RocmError {
        // SAFETY: hipGetErrorString accepts every hipError_t value.
        let message = unsafe { (self.hip_get_error_string)(status) };
        RocmError::new(format!(
            "{operation} failed with HIP status {status}: {}",
            c_string(message)
        ))
    }

    fn check_hip(&self, status: c_int, operation: &str) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(self.hip_error(status, operation))
        }
    }

    fn check_rtc(&self, status: c_int, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: hiprtcGetErrorString accepts every hiprtcResult value.
        let message = unsafe { (self.hiprtc_get_error_string)(status) };
        Err(RocmError::new(format!(
            "{operation} failed with HIPRTC status {status}: {}",
            c_string(message)
        )))
    }

    pub fn device_count(&self) -> Result<usize> {
        let mut count = 0;
        // SAFETY: `count` is a valid output pointer.
        self.check_hip(
            unsafe { (self.hip_get_device_count)(&mut count) },
            "hipGetDeviceCount",
        )?;
        Ok(count.max(0) as usize)
    }

    pub fn set_device(&self, ordinal: usize) -> Result<()> {
        let ordinal = c_int::try_from(ordinal)
            .map_err(|_| RocmError::new(format!("ROCm device ordinal {ordinal} is too large")))?;
        // SAFETY: HIP validates the device ordinal.
        self.check_hip(unsafe { (self.hip_set_device)(ordinal) }, "hipSetDevice")
    }

    pub fn stream_create(&self) -> Result<HipStream> {
        let mut stream = std::ptr::null_mut();
        // SAFETY: `stream` is a valid output pointer.
        self.check_hip(
            unsafe { (self.hip_stream_create)(&mut stream) },
            "hipStreamCreate",
        )?;
        Ok(stream)
    }

    pub unsafe fn stream_destroy(&self, stream: HipStream) -> Result<()> {
        // SAFETY: The caller owns `stream` and destroys it once.
        self.check_hip(
            unsafe { (self.hip_stream_destroy)(stream) },
            "hipStreamDestroy",
        )
    }

    pub fn stream_synchronize(&self, stream: HipStream) -> Result<()> {
        // SAFETY: `stream` is live for this call.
        self.check_hip(
            unsafe { (self.hip_stream_synchronize)(stream) },
            "hipStreamSynchronize",
        )
    }

    pub fn malloc(&self, bytes: usize) -> Result<*mut c_void> {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: `pointer` is a valid output pointer and HIP owns the allocation.
        self.check_hip(
            unsafe { (self.hip_malloc)(&mut pointer, bytes) },
            "hipMalloc",
        )?;
        Ok(pointer)
    }

    pub unsafe fn free(&self, pointer: *mut c_void) -> Result<()> {
        // SAFETY: The caller passes a live allocation returned by hipMalloc.
        self.check_hip(unsafe { (self.hip_free)(pointer) }, "hipFree")
    }

    pub unsafe fn memcpy(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> Result<()> {
        // SAFETY: The caller guarantees pointer validity for `bytes` and the selected direction.
        self.check_hip(
            unsafe { (self.hip_memcpy)(destination, source, bytes, kind) },
            "hipMemcpy",
        )
    }

    pub unsafe fn memset(
        &self,
        destination: *mut c_void,
        value: c_int,
        bytes: usize,
    ) -> Result<()> {
        // SAFETY: The caller guarantees destination validity for `bytes`.
        self.check_hip(
            unsafe { (self.hip_memset)(destination, value, bytes) },
            "hipMemset",
        )
    }

    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        let mut free = 0;
        let mut total = 0;
        // SAFETY: Both output pointers are valid.
        self.check_hip(
            unsafe { (self.hip_mem_get_info)(&mut free, &mut total) },
            "hipMemGetInfo",
        )?;
        Ok((free, total))
    }

    pub fn compile(&self, name: &str, source: &str, headers: &[(&str, &str)]) -> Result<Vec<u8>> {
        let name = CString::new(name).expect("kernel names do not contain NUL bytes");
        let source = CString::new(source).expect("kernel sources do not contain NUL bytes");
        let header_sources = headers
            .iter()
            .map(|(_, source)| CString::new(*source).expect("kernel headers do not contain NUL"))
            .collect::<Vec<_>>();
        let header_names = headers
            .iter()
            .map(|(name, _)| CString::new(*name).expect("kernel header names do not contain NUL"))
            .collect::<Vec<_>>();
        let header_source_pointers = header_sources
            .iter()
            .map(|source| source.as_ptr())
            .collect::<Vec<_>>();
        let header_name_pointers = header_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut program = std::ptr::null_mut();
        // SAFETY: All strings and pointer arrays remain live through program creation.
        self.check_rtc(
            unsafe {
                (self.hiprtc_create_program)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    headers.len() as c_int,
                    header_source_pointers.as_ptr(),
                    header_name_pointers.as_ptr(),
                )
            },
            "hiprtcCreateProgram",
        )?;

        let result = self.compile_program(program);
        // SAFETY: `program` was created successfully and is destroyed once here.
        let destroy = unsafe { (self.hiprtc_destroy_program)(&mut program) };
        self.check_rtc(destroy, "hiprtcDestroyProgram")?;
        result
    }

    fn compile_program(&self, program: HipRtcProgram) -> Result<Vec<u8>> {
        let options = [
            CString::new("-O3").unwrap(),
            CString::new("--std=c++17").unwrap(),
        ];
        let option_pointers = options
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: The program and option strings remain live through compilation.
        let status = unsafe {
            (self.hiprtc_compile_program)(
                program,
                option_pointers.len() as c_int,
                option_pointers.as_ptr(),
            )
        };
        if status != 0 {
            let log = self.program_log(program);
            let base = self
                .check_rtc(status, "hiprtcCompileProgram")
                .expect_err("non-success status must produce an error");
            return Err(RocmError::new(format!(
                "{base}\nHIPRTC compiler log:\n{log}"
            )));
        }

        let mut size = 0;
        // SAFETY: `size` is a valid output pointer.
        self.check_rtc(
            unsafe { (self.hiprtc_get_code_size)(program, &mut size) },
            "hiprtcGetCodeSize",
        )?;
        let mut code = vec![0_u8; size];
        // SAFETY: `code` has the exact capacity reported by HIPRTC.
        self.check_rtc(
            unsafe { (self.hiprtc_get_code)(program, code.as_mut_ptr().cast()) },
            "hiprtcGetCode",
        )?;
        Ok(code)
    }

    fn program_log(&self, program: HipRtcProgram) -> String {
        let mut size = 0;
        // SAFETY: `size` is a valid output pointer.
        if unsafe { (self.hiprtc_get_program_log_size)(program, &mut size) } != 0 || size == 0 {
            return "<unavailable>".to_owned();
        }
        let mut log = vec![0_u8; size];
        // SAFETY: `log` has the size reported by HIPRTC.
        if unsafe { (self.hiprtc_get_program_log)(program, log.as_mut_ptr().cast()) } != 0 {
            return "<unavailable>".to_owned();
        }
        String::from_utf8_lossy(&log)
            .trim_end_matches('\0')
            .to_owned()
    }

    pub fn module_load(&self, code: &[u8]) -> Result<HipModule> {
        let mut module = std::ptr::null_mut();
        // SAFETY: `code` contains a HIPRTC-produced code object and stays live through loading.
        self.check_hip(
            unsafe { (self.hip_module_load_data)(&mut module, code.as_ptr().cast()) },
            "hipModuleLoadData",
        )?;
        Ok(module)
    }

    pub unsafe fn module_unload(&self, module: HipModule) -> Result<()> {
        // SAFETY: The caller owns a live module and unloads it once.
        self.check_hip(
            unsafe { (self.hip_module_unload)(module) },
            "hipModuleUnload",
        )
    }

    pub fn module_function(&self, module: HipModule, name: &str) -> Result<HipFunction> {
        let name = CString::new(name).expect("kernel function names do not contain NUL bytes");
        let mut function = std::ptr::null_mut();
        // SAFETY: `module` is live and `name` is a valid C string.
        self.check_hip(
            unsafe { (self.hip_module_get_function)(&mut function, module, name.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(function)
    }

    pub unsafe fn launch(
        &self,
        function: HipFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_memory: u32,
        stream: HipStream,
        arguments: &mut [*mut c_void],
    ) -> Result<()> {
        // SAFETY: The caller guarantees function, stream, dimensions, and argument ABI validity.
        self.check_hip(
            unsafe {
                (self.hip_module_launch_kernel)(
                    function,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_memory,
                    stream,
                    arguments.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )
    }

    pub fn blas_create(&self, stream: HipStream) -> Result<Option<HipBlasHandle>> {
        let Some(blas) = &self.blas else {
            return Ok(None);
        };
        let mut handle = std::ptr::null_mut();
        // SAFETY: `handle` is a valid output pointer.
        blas.check(unsafe { (blas.create)(&mut handle) }, "hipblasCreate")?;
        // SAFETY: `handle` and `stream` are live.
        if let Err(error) = blas.check(
            unsafe { (blas.set_stream)(handle, stream) },
            "hipblasSetStream",
        ) {
            // SAFETY: Initialization still uniquely owns this live handle.
            let _ = unsafe { (blas.destroy)(handle) };
            return Err(error);
        }
        Ok(Some(handle))
    }

    pub unsafe fn blas_destroy(&self, handle: HipBlasHandle) -> Result<()> {
        let blas = self
            .blas
            .as_ref()
            .ok_or_else(|| RocmError::new("hipBLAS is unavailable"))?;
        // SAFETY: The caller owns a live handle and destroys it once.
        blas.check(unsafe { (blas.destroy)(handle) }, "hipblasDestroy")
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm_strided_batched(
        &self,
        handle: HipBlasHandle,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        stride_a: i64,
        b: *const f32,
        ldb: c_int,
        stride_b: i64,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> Result<()> {
        let blas = self
            .blas
            .as_ref()
            .ok_or_else(|| RocmError::new("hipBLAS is unavailable"))?;
        // SAFETY: The caller validates all GEMM dimensions, strides, handles, and allocations.
        blas.check(
            unsafe {
                (blas.sgemm_strided_batched)(
                    handle,
                    trans_a,
                    trans_b,
                    m,
                    n,
                    k,
                    alpha,
                    a,
                    lda,
                    stride_a,
                    b,
                    ldb,
                    stride_b,
                    beta,
                    c,
                    ldc,
                    stride_c,
                    batch_count,
                )
            },
            "hipblasSgemmStridedBatched",
        )
    }
}

fn c_string(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return "<unknown>".to_owned();
    }
    // SAFETY: ROCm error-string APIs return process-owned NUL-terminated strings.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}
