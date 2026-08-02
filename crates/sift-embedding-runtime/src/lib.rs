mod backend;
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
mod custom_op;
mod device;
#[path = "tensor/display.rs"]
mod display;
#[path = "tensor/dtype.rs"]
mod dtype;
#[cfg(not(feature = "cuda"))]
mod dummy_cuda_backend;
mod error;
#[path = "tensor/layout.rs"]
mod layout;
#[path = "tensor/nditer.rs"]
mod nditer;
pub mod nn;
#[path = "tensor/op.rs"]
mod op;
pub mod safetensors;
#[path = "tensor/shape.rs"]
mod shape;
#[path = "tensor/storage.rs"]
mod storage;
#[path = "tensor/strided_index.rs"]
mod strided_index;
mod tensor;
mod utils;

pub use cpu::{CpuStorage, CpuStorageRef};
pub use custom_op::{CustomOp1, CustomOp3};
pub use device::{Device, DeviceLocation, MemoryProfile};
pub use dtype::{DType, WithDType};
pub use error::{Error, Result};
pub use layout::Layout;
pub use shape::{Shape, D};
pub use storage::Storage;
pub(crate) use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::Tensor;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDevice, CudaStorage};
#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend::{CudaDevice, CudaStorage};

pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}
