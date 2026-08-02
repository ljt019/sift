//! HIP/ROCm implementation of the specialized EmbeddingGemma tensor backend.

use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result as TensorResult, Shape};

mod device;
mod sys;

pub use device::{DeviceRepr, KernelArgs, LaunchConfig, RocmBuffer, RocmDevice};
pub use sift_embedding_kernels as kernels;
use sys::{HIPBLAS_OP_N, HIPBLAS_OP_T};

#[derive(Debug)]
pub struct RocmError {
    message: String,
}

impl RocmError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RocmError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RocmError {}

impl From<RocmError> for crate::Error {
    fn from(error: RocmError) -> Self {
        Self::Rocm(Box::new(error))
    }
}

pub(super) type Result<T> = std::result::Result<T, RocmError>;

#[derive(Debug)]
pub enum RocmStorageSlice {
    U32(RocmBuffer<u32>),
    F32(RocmBuffer<f32>),
}

#[derive(Debug)]
pub struct RocmStorage {
    pub(crate) slice: RocmStorageSlice,
    pub(crate) device: RocmDevice,
}

enum LayoutInfo {
    Buffer(RocmBuffer<usize>),
    Contiguous,
}

impl LayoutInfo {
    fn from_values(device: &RocmDevice, values: Vec<usize>) -> TensorResult<Self> {
        Ok(Self::Buffer(device.clone_htod(&values)?))
    }

    fn from_layout(device: &RocmDevice, layout: &Layout) -> TensorResult<Self> {
        if layout.is_contiguous() {
            Ok(Self::Contiguous)
        } else {
            Self::from_values(device, [layout.dims(), layout.stride()].concat())
        }
    }

    fn pointer(&self) -> *mut usize {
        match self {
            Self::Buffer(buffer) => buffer.pointer(),
            Self::Contiguous => std::ptr::null_mut(),
        }
    }
}

fn unary(
    input: &RocmBuffer<f32>,
    device: &RocmDevice,
    layout: &Layout,
    function_name: &str,
) -> TensorResult<RocmBuffer<f32>> {
    let elements = layout.shape().elem_count();
    let info = LayoutInfo::from_layout(device, layout)?;
    let function = device.get_or_load_func(function_name, &kernels::UNARY)?;
    // SAFETY: The kernel initializes every output element before it is observed.
    let output = unsafe { device.alloc::<f32>(elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_usize(elements);
    arguments.push_usize(layout.dims().len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(input.pointer_at(layout.start_offset())?);
    arguments.push_pointer(output.pointer());
    function.launch(
        device,
        LaunchConfig::for_num_elems(elements as u32),
        &mut arguments,
    )?;
    Ok(output)
}

fn unary_float<U: UnaryOpT>(
    input: &RocmBuffer<f32>,
    device: &RocmDevice,
    layout: &Layout,
) -> TensorResult<RocmBuffer<f32>> {
    unary(input, device, layout, &format!("{}_f32", U::KERNEL))
}

fn binary_float<B: BinaryOpT>(
    left: &RocmBuffer<f32>,
    left_layout: &Layout,
    right: &RocmBuffer<f32>,
    right_layout: &Layout,
    device: &RocmDevice,
) -> TensorResult<RocmBuffer<f32>> {
    let elements = left_layout.shape().elem_count();
    let info = if left_layout.is_contiguous() && right_layout.is_contiguous() {
        LayoutInfo::Contiguous
    } else {
        LayoutInfo::from_values(
            device,
            [
                left_layout.dims(),
                left_layout.stride(),
                right_layout.stride(),
            ]
            .concat(),
        )?
    };
    let function = device.get_or_load_func(&format!("{}_f32", B::KERNEL), &kernels::BINARY)?;
    // SAFETY: The kernel initializes every output element before it is observed.
    let output = unsafe { device.alloc::<f32>(elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_usize(elements);
    arguments.push_usize(left_layout.dims().len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(left.pointer_at(left_layout.start_offset())?);
    arguments.push_pointer(right.pointer_at(right_layout.start_offset())?);
    arguments.push_pointer(output.pointer());
    function.launch(
        device,
        LaunchConfig::for_num_elems(elements as u32),
        &mut arguments,
    )?;
    Ok(output)
}

fn affine(
    input: &RocmBuffer<f32>,
    device: &RocmDevice,
    layout: &Layout,
    multiplier: f32,
    addend: f32,
) -> TensorResult<RocmBuffer<f32>> {
    let elements = layout.shape().elem_count();
    let info = LayoutInfo::from_layout(device, layout)?;
    let function = device.get_or_load_func("affine_f32", &kernels::AFFINE)?;
    // SAFETY: The kernel initializes every output element before it is observed.
    let output = unsafe { device.alloc::<f32>(elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_usize(elements);
    arguments.push_usize(layout.dims().len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(input.pointer_at(layout.start_offset())?);
    arguments.push_pointer(output.pointer());
    arguments.push_f32(multiplier);
    arguments.push_f32(addend);
    function.launch(
        device,
        LaunchConfig::for_num_elems(elements as u32),
        &mut arguments,
    )?;
    Ok(output)
}

fn reduce_sum(
    input: &RocmBuffer<f32>,
    device: &RocmDevice,
    layout: &Layout,
    sum_dimensions: &[usize],
) -> TensorResult<RocmBuffer<f32>> {
    let source_stride = layout.stride();
    let source_dimensions = layout.shape().dims();
    let source_elements: usize = source_dimensions.iter().product();
    let mut dimensions = Vec::new();
    let mut strides = Vec::new();
    let mut output_elements = 1;
    for (dimension, &size) in source_dimensions.iter().enumerate() {
        if !sum_dimensions.contains(&dimension) {
            output_elements *= size;
            dimensions.push(size);
            strides.push(source_stride[dimension]);
        }
    }
    for &dimension in sum_dimensions {
        dimensions.push(source_dimensions[dimension]);
        strides.push(source_stride[dimension]);
    }
    let elements_per_output = source_elements / output_elements;
    let threads = usize::min(1024, elements_per_output).next_power_of_two();
    let info =
        LayoutInfo::from_values(device, [dimensions.as_slice(), strides.as_slice()].concat())?;
    let function = device.get_or_load_func("fast_sum_f32", &kernels::REDUCE)?;
    // SAFETY: The reduction kernel initializes every output element.
    let output = unsafe { device.alloc::<f32>(output_elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_usize(source_elements);
    arguments.push_usize(elements_per_output);
    arguments.push_usize(source_dimensions.len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(input.pointer_at(layout.start_offset())?);
    arguments.push_pointer(output.pointer());
    function.launch(
        device,
        LaunchConfig {
            grid_dim: (output_elements as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        },
        &mut arguments,
    )?;
    Ok(output)
}

fn cast_u32_f32(
    input: &RocmBuffer<u32>,
    device: &RocmDevice,
    layout: &Layout,
) -> TensorResult<RocmBuffer<f32>> {
    let elements = layout.shape().elem_count();
    let info = LayoutInfo::from_layout(device, layout)?;
    let function = device.get_or_load_func("cast_u32_f32", &kernels::CAST)?;
    // SAFETY: The cast kernel initializes every output element.
    let output = unsafe { device.alloc::<f32>(elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_usize(elements);
    arguments.push_usize(layout.dims().len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(input.pointer_at(layout.start_offset())?);
    arguments.push_pointer(output.pointer());
    function.launch(
        device,
        LaunchConfig::for_num_elems(elements as u32),
        &mut arguments,
    )?;
    Ok(output)
}

fn index_select(
    input: &RocmBuffer<f32>,
    ids: &RocmBuffer<u32>,
    device: &RocmDevice,
    input_layout: &Layout,
    ids_layout: &Layout,
    dimension: usize,
) -> TensorResult<RocmBuffer<f32>> {
    let ids_shape = ids_layout.shape();
    let info = LayoutInfo::from_values(device, [ids_shape.dims(), ids_layout.stride()].concat())?;
    let (input_start, _) = input_layout
        .contiguous_offsets()
        .ok_or(crate::Error::RequiresContiguous { op: "index-select" })?;
    let left_size = input_layout.dims()[..dimension].iter().product::<usize>();
    let right_size = input_layout.dims()[dimension + 1..]
        .iter()
        .product::<usize>();
    let source_size = input_layout.dims()[dimension];
    let ids_size = ids_shape.elem_count();
    let output_elements = ids_size * left_size * right_size;
    let function = device.get_or_load_func("is_u32_f32", &kernels::INDEXING)?;
    // SAFETY: The indexing kernel initializes every output element.
    let output = unsafe { device.alloc::<f32>(output_elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_usize(output_elements);
    arguments.push_usize(ids_shape.dims().len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(ids.pointer_at(ids_layout.start_offset())?);
    arguments.push_pointer(input.pointer_at(input_start)?);
    arguments.push_pointer(output.pointer());
    arguments.push_usize(left_size);
    arguments.push_usize(source_size);
    arguments.push_usize(ids_size);
    arguments.push_usize(right_size);
    function.launch(
        device,
        LaunchConfig::for_num_elems(output_elements as u32),
        &mut arguments,
    )?;
    Ok(output)
}

#[derive(Clone, Copy)]
struct GemmConfig {
    trans_a: i32,
    trans_b: i32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    batch: i32,
}

fn gemm_config(
    (batch, rows, columns, inner): (usize, usize, usize, usize),
    left: &Layout,
    right: &Layout,
) -> TensorResult<GemmConfig> {
    let left_stride = left.stride();
    let right_stride = right.stride();
    let right_minor = right_stride[right_stride.len() - 1];
    let right_major = right_stride[right_stride.len() - 2];
    let left_minor = left_stride[left_stride.len() - 1];
    let left_major = left_stride[left_stride.len() - 2];
    let (lda, trans_a) = if (right_minor == 1 || columns == 1)
        && (right_major == columns || inner == 1)
    {
        (columns as i32, HIPBLAS_OP_N)
    } else if (right_minor == inner || columns == 1) && (right_major == 1 || inner == 1) {
        (inner as i32, HIPBLAS_OP_T)
    } else {
        return Err(RocmError::new(format!(
            "ROCm matmul requires contiguous matrix dimensions; left={left:?}, right={right:?}, mnk=({rows}, {columns}, {inner})"
        ))
        .into());
    };
    let (ldb, trans_b) = if (left_minor == 1 || inner == 1) && (left_major == inner || rows == 1) {
        (inner as i32, HIPBLAS_OP_N)
    } else if (left_minor == rows || inner == 1) && (left_major == 1 || rows == 1) {
        (rows as i32, HIPBLAS_OP_T)
    } else {
        return Err(RocmError::new(format!(
            "ROCm matmul requires contiguous matrix dimensions; left={left:?}, right={right:?}, mnk=({rows}, {columns}, {inner})"
        ))
        .into());
    };
    let stride_b = batch_stride(left, rows * inner)?;
    let stride_a = batch_stride(right, columns * inner)?;
    Ok(GemmConfig {
        trans_a,
        trans_b,
        m: columns as i32,
        n: rows as i32,
        k: inner as i32,
        lda,
        ldb,
        ldc: columns as i32,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (rows * columns) as i64,
        batch: batch as i32,
    })
}

fn batch_stride(layout: &Layout, contiguous: usize) -> TensorResult<usize> {
    let strides = &layout.stride()[..layout.stride().len() - 2];
    let dimensions = layout.dims();
    match strides {
        [outer, inner] if *outer == *inner * dimensions[1] => Ok(*inner),
        [_, inner] if dimensions[0] == 1 => Ok(*inner),
        [outer, _] if dimensions[1] == 1 => Ok(*outer),
        [stride] => Ok(*stride),
        [] => Ok(contiguous),
        _ => Err(RocmError::new(format!(
            "ROCm matmul does not support batch strides {:?} for shape {:?}",
            layout.stride(),
            layout.shape()
        ))
        .into()),
    }
}

fn matmul(
    left: &RocmBuffer<f32>,
    right: &RocmBuffer<f32>,
    device: &RocmDevice,
    dimensions: (usize, usize, usize, usize),
    left_layout: &Layout,
    right_layout: &Layout,
) -> TensorResult<RocmBuffer<f32>> {
    let (batch, rows, columns, inner) = dimensions;
    // SAFETY: Either hipBLAS or the HIPRTC kernel initializes every output element.
    let output = unsafe { device.alloc::<f32>(batch * rows * columns)? };
    if batch == 0 || rows == 0 || columns == 0 {
        return Ok(output);
    }
    let left = left.pointer_at(left_layout.start_offset())?;
    let right = right.pointer_at(right_layout.start_offset())?;

    if let (Some(blas), Ok(config)) = (
        device.blas(),
        gemm_config(dimensions, left_layout, right_layout),
    ) {
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        // Keep the CUDA-compatible row-major transformation: hipBLAS sees right as A and left as B.
        unsafe {
            device.api().sgemm_strided_batched(
                blas,
                config.trans_a,
                config.trans_b,
                config.m,
                config.n,
                config.k,
                &alpha,
                right,
                config.lda,
                config.stride_a,
                left,
                config.ldb,
                config.stride_b,
                &beta,
                output.pointer(),
                config.ldc,
                config.stride_c,
                config.batch,
            )
        }?;
        return Ok(output);
    }

    let left_stride = left_layout.stride();
    let right_stride = right_layout.stride();
    let left_batch_stride = batch_stride(left_layout, rows * inner)?;
    let right_batch_stride = batch_stride(right_layout, columns * inner)?;
    let function = device.get_or_load_func("matmul_f32", &kernels::MATMUL)?;
    let mut arguments = KernelArgs::default();
    arguments.push_pointer(left);
    arguments.push_pointer(right);
    arguments.push_pointer(output.pointer());
    arguments.push_usize(rows);
    arguments.push_usize(columns);
    arguments.push_usize(inner);
    arguments.push_usize(left_stride[left_stride.len() - 2]);
    arguments.push_usize(left_stride[left_stride.len() - 1]);
    arguments.push_usize(right_stride[right_stride.len() - 2]);
    arguments.push_usize(right_stride[right_stride.len() - 1]);
    arguments.push_usize(left_batch_stride);
    arguments.push_usize(right_batch_stride);
    let grid_dimension = |value: usize, name: &str| {
        u32::try_from(value).map_err(|_| {
            crate::Error::from(RocmError::new(format!(
                "ROCm matmul {name} dimension {value} exceeds the HIP launch limit"
            )))
        })
    };
    function.launch(
        device,
        LaunchConfig {
            grid_dim: (
                grid_dimension(columns.div_ceil(16).max(1), "column")?,
                grid_dimension(rows.div_ceil(16).max(1), "row")?,
                grid_dimension(batch.max(1), "batch")?,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        &mut arguments,
    )?;
    Ok(output)
}

fn copy_strided<T: DeviceRepr>(
    source: &RocmBuffer<T>,
    destination: &RocmBuffer<T>,
    destination_offset: usize,
    layout: &Layout,
    device: &RocmDevice,
    kernel: &str,
) -> TensorResult<()> {
    let elements = layout.shape().elem_count();
    let source = source.pointer_at(layout.start_offset())?;
    let destination = destination.pointer_at(destination_offset)?;
    if layout.is_contiguous() {
        return Ok(device.memcpy_dtod(source, destination, elements)?);
    }
    let info = LayoutInfo::from_layout(device, layout)?;
    let function = device.get_or_load_func(kernel, &kernels::UNARY)?;
    let mut arguments = KernelArgs::default();
    arguments.push_usize(elements);
    arguments.push_usize(layout.dims().len());
    arguments.push_pointer(info.pointer());
    arguments.push_pointer(source);
    arguments.push_pointer(destination);
    function.launch(
        device,
        LaunchConfig::for_num_elems(elements as u32),
        &mut arguments,
    )?;
    Ok(())
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn dtype(&self) -> DType {
        match self.slice {
            RocmStorageSlice::U32(_) => DType::U32,
            RocmStorageSlice::F32(_) => DType::F32,
        }
    }

    fn device(&self) -> &RocmDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> TensorResult<CpuStorage> {
        match &self.slice {
            RocmStorageSlice::U32(values) => Ok(CpuStorage::U32(self.device.clone_dtoh(values)?)),
            RocmStorageSlice::F32(values) => Ok(CpuStorage::F32(self.device.clone_dtoh(values)?)),
        }
    }

    fn affine(&self, layout: &Layout, multiplier: f64, addend: f64) -> TensorResult<Self> {
        let RocmStorageSlice::F32(input) = &self.slice else {
            return Err(RocmError::new("ROCm affine requires f32 storage").into());
        };
        Ok(Self {
            slice: RocmStorageSlice::F32(affine(
                input,
                &self.device,
                layout,
                multiplier as f32,
                addend as f32,
            )?),
            device: self.device.clone(),
        })
    }

    fn reduce_sum(&self, layout: &Layout, dimensions: &[usize]) -> TensorResult<Self> {
        let RocmStorageSlice::F32(input) = &self.slice else {
            return Err(RocmError::new("ROCm reduction requires f32 storage").into());
        };
        Ok(Self {
            slice: RocmStorageSlice::F32(reduce_sum(input, &self.device, layout, dimensions)?),
            device: self.device.clone(),
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> TensorResult<Self> {
        let RocmStorageSlice::U32(input) = &self.slice else {
            return Err(RocmError::new(format!(
                "ROCm only supports u32-to-f32 casts, got {:?}-to-{dtype:?}",
                self.dtype()
            ))
            .into());
        };
        if dtype != DType::F32 {
            return Err(RocmError::new(format!(
                "ROCm only supports u32-to-f32 casts, got u32-to-{dtype:?}"
            ))
            .into());
        }
        Ok(Self {
            slice: RocmStorageSlice::F32(cast_u32_f32(input, &self.device, layout)?),
            device: self.device.clone(),
        })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> TensorResult<Self> {
        let RocmStorageSlice::F32(input) = &self.slice else {
            return Err(RocmError::new("ROCm unary floating-point operation requires f32").into());
        };
        Ok(Self {
            slice: RocmStorageSlice::F32(unary_float::<U>(input, &self.device, layout)?),
            device: self.device.clone(),
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        right: &Self,
        left_layout: &Layout,
        right_layout: &Layout,
    ) -> TensorResult<Self> {
        let (RocmStorageSlice::F32(left), RocmStorageSlice::F32(right)) =
            (&self.slice, &right.slice)
        else {
            return Err(RocmError::new("ROCm binary operation requires f32 storage").into());
        };
        Ok(Self {
            slice: RocmStorageSlice::F32(binary_float::<B>(
                left,
                left_layout,
                right,
                right_layout,
                &self.device,
            )?),
            device: self.device.clone(),
        })
    }

    fn index_select(
        &self,
        ids: &Self,
        layout: &Layout,
        ids_layout: &Layout,
        dimension: usize,
    ) -> TensorResult<Self> {
        let (RocmStorageSlice::F32(input), RocmStorageSlice::U32(ids)) = (&self.slice, &ids.slice)
        else {
            return Err(RocmError::new("ROCm index-select requires f32 input and u32 ids").into());
        };
        Ok(Self {
            slice: RocmStorageSlice::F32(index_select(
                input,
                ids,
                &self.device,
                layout,
                ids_layout,
                dimension,
            )?),
            device: self.device.clone(),
        })
    }

    fn matmul(
        &self,
        right: &Self,
        dimensions: (usize, usize, usize, usize),
        left_layout: &Layout,
        right_layout: &Layout,
    ) -> TensorResult<Self> {
        let (RocmStorageSlice::F32(left), RocmStorageSlice::F32(right)) =
            (&self.slice, &right.slice)
        else {
            return Err(RocmError::new("ROCm matmul requires f32 storage").into());
        };
        Ok(Self {
            slice: RocmStorageSlice::F32(matmul(
                left,
                right,
                &self.device,
                dimensions,
                left_layout,
                right_layout,
            )?),
            device: self.device.clone(),
        })
    }

    fn copy_strided_src(
        &self,
        destination: &mut Self,
        offset: usize,
        layout: &Layout,
    ) -> TensorResult<()> {
        match (&self.slice, &destination.slice) {
            (RocmStorageSlice::U32(source), RocmStorageSlice::U32(destination)) => copy_strided(
                source,
                destination,
                offset,
                layout,
                &self.device,
                "ucopy_u32",
            ),
            (RocmStorageSlice::F32(source), RocmStorageSlice::F32(destination)) => copy_strided(
                source,
                destination,
                offset,
                layout,
                &self.device,
                "ucopy_f32",
            ),
            _ => Err(RocmError::new("ROCm copy source and destination dtypes differ").into()),
        }
    }
}

pub(crate) fn softmax_fwd(
    storage: &RocmStorage,
    layout: &Layout,
) -> TensorResult<(RocmStorage, Shape)> {
    let RocmStorageSlice::F32(source) = &storage.slice else {
        return Err(RocmError::new("ROCm softmax requires f32 storage").into());
    };
    let (start, _) = layout
        .contiguous_offsets()
        .ok_or(crate::Error::RequiresContiguous { op: "softmax" })?;
    let elements = layout.shape().elem_count();
    let columns = layout.dims()[layout.dims().len() - 1];
    let function = storage
        .device
        .get_or_load_func("softmax_f32", &kernels::REDUCE)?;
    // SAFETY: The softmax kernel initializes every output element.
    let output = unsafe { storage.device.alloc::<f32>(elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_pointer(source.pointer_at(start)?);
    arguments.push_pointer(output.pointer());
    arguments.push_i32(columns as i32);
    function.launch(
        &storage.device,
        LaunchConfig {
            grid_dim: ((elements / columns) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
        &mut arguments,
    )?;
    Ok((
        RocmStorage {
            slice: RocmStorageSlice::F32(output),
            device: storage.device.clone(),
        },
        layout.shape().clone(),
    ))
}

pub(crate) fn rope_fwd(
    source: &RocmStorage,
    source_layout: &Layout,
    cosine: &RocmStorage,
    cosine_layout: &Layout,
    sine: &RocmStorage,
    sine_layout: &Layout,
) -> TensorResult<(RocmStorage, Shape)> {
    let (
        RocmStorageSlice::F32(source_values),
        RocmStorageSlice::F32(cosine_values),
        RocmStorageSlice::F32(sine_values),
    ) = (&source.slice, &cosine.slice, &sine.slice)
    else {
        return Err(RocmError::new("ROCm RoPE requires f32 storage").into());
    };
    let (source_start, _) = source_layout
        .contiguous_offsets()
        .ok_or(crate::Error::RequiresContiguous { op: "RoPE input" })?;
    let (cosine_start, _) = cosine_layout
        .contiguous_offsets()
        .ok_or(crate::Error::RequiresContiguous { op: "RoPE cosine" })?;
    let (sine_start, _) = sine_layout
        .contiguous_offsets()
        .ok_or(crate::Error::RequiresContiguous { op: "RoPE sine" })?;
    let (batch, heads, tokens, dimension) = source_layout.shape().dims4()?;
    let frequency_batch_stride = if cosine_layout.dims().len() == 3 && sine_layout.dims().len() == 3
    {
        (heads * tokens * dimension) as u32
    } else {
        0
    };
    let elements = batch * heads * tokens * dimension;
    let function = source
        .device
        .get_or_load_func("rope_f32", &kernels::REDUCE)?;
    // SAFETY: The RoPE kernel initializes every output element.
    let output = unsafe { source.device.alloc::<f32>(elements)? };
    let mut arguments = KernelArgs::default();
    arguments.push_pointer(source_values.pointer_at(source_start)?);
    arguments.push_pointer(cosine_values.pointer_at(cosine_start)?);
    arguments.push_pointer(sine_values.pointer_at(sine_start)?);
    arguments.push_pointer(output.pointer());
    arguments.push_u32((batch * heads) as u32);
    arguments.push_u32((tokens * dimension) as u32);
    arguments.push_u32(dimension as u32);
    arguments.push_u32(frequency_batch_stride);
    function.launch(
        &source.device,
        LaunchConfig::for_num_elems((elements / 2) as u32),
        &mut arguments,
    )?;
    Ok((
        RocmStorage {
            slice: RocmStorageSlice::F32(output),
            device: source.device.clone(),
        },
        source_layout.shape().clone(),
    ))
}
