use crate::{DType, DeviceLocation, Shape};

#[derive(thiserror::Error)]
pub enum Error {
    #[error("{msg}, expected {expected:?}, got {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("dtype mismatch in {op}: {lhs:?} vs {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    #[error("{op}: dimension {dim} is out of range for {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: i32,
        op: &'static str,
    },

    #[error("expected rank {expected}, got {got} for {shape:?}")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error("{msg}, expected {expected:?}, got {got:?}")]
    UnexpectedShape {
        msg: String,
        expected: Shape,
        got: Shape,
    },

    #[error("shape mismatch in {op}: {lhs:?} vs {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("device mismatch in {op}: {lhs:?} vs {rhs:?}")]
    DeviceMismatchBinaryOp {
        lhs: DeviceLocation,
        rhs: DeviceLocation,
        op: &'static str,
    },

    #[error("invalid narrow on {shape:?}: dim {dim}, start {start}, len {len}: {msg}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },

    #[error("{op}: index {index} is out of range for dimension size {size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("{op} requires a contiguous tensor")]
    RequiresContiguous { op: &'static str },

    #[error("CUDA support is not enabled")]
    NotCompiledWithCudaSupport,

    #[error("ROCm support is not enabled")]
    NotCompiledWithRocmSupport,

    #[error("tensor {path} was not found")]
    CannotFindTensor { path: String },

    #[error(transparent)]
    Cuda(Box<dyn std::error::Error + Send + Sync>),

    #[error(transparent)]
    Rocm(Box<dyn std::error::Error + Send + Sync>),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("unsupported safetensor dtype {0:?}")]
    UnsupportedSafeTensorDtype(safetensors::Dtype),

    #[error("path {path:?}: {inner}")]
    WithPath {
        inner: Box<Self>,
        path: std::path::PathBuf,
    },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    #[error("{0}")]
    Msg(String),
}

impl std::fmt::Debug for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self}")
    }
}

impl Error {
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        if backtrace.status() == std::backtrace::BacktraceStatus::Captured {
            Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            }
        } else {
            self
        }
    }

    pub fn with_path(self, path: impl AsRef<std::path::Path>) -> Self {
        Self::WithPath {
            inner: Box::new(self),
            path: path.as_ref().to_path_buf(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($($arg)*)).bt())
    };
}
