use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, UnaryOpT};
use crate::{DType, Error, Layout, Result, Shape, WithDType};

mod utils;
pub use utils::{binary_map_vec, unary_map, unary_map_vec, Map1};

#[derive(Debug, Clone)]
pub enum CpuStorage {
    U32(Vec<u32>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone, Copy)]
pub enum CpuStorageRef<'a> {
    U32(&'a [u32]),
    F32(&'a [f32]),
}

#[derive(Debug, Clone)]
pub struct CpuDevice;

struct ReduceSum<'a> {
    output_shape: &'a Shape,
    dimensions: &'a [usize],
    dimensions_and_stride: Vec<(usize, usize)>,
}

impl Map1 for ReduceSum<'_> {
    fn f<T: WithDType>(&self, source: &[T], layout: &Layout) -> Result<Vec<T>> {
        let mut output = vec![T::zero(); self.output_shape.elem_count()];
        match layout.contiguous_offsets() {
            Some((start, end)) => {
                let source = &source[start..end];
                let trailing_reduction = self
                    .dimensions
                    .iter()
                    .rev()
                    .enumerate()
                    .all(|(offset, &dimension)| dimension == layout.shape().rank() - 1 - offset);
                if trailing_reduction {
                    let reduction_size = self
                        .dimensions_and_stride
                        .iter()
                        .map(|(size, _)| size)
                        .product::<usize>();
                    for (index, value) in output.iter_mut().enumerate() {
                        let start = index * reduction_size;
                        *value = source[start..start + reduction_size]
                            .iter()
                            .copied()
                            .fold(T::zero(), |sum, item| sum + item);
                    }
                    return Ok(output);
                }

                for (source_index, &value) in source.iter().enumerate() {
                    let mut output_index = source_index;
                    for &(size, stride) in &self.dimensions_and_stride {
                        let (before, after) = (output_index / stride, output_index % stride);
                        output_index = (before / size) * stride + after;
                    }
                    output[output_index] += value;
                }
            }
            None => {
                for (source_index, storage_index) in layout.strided_index().enumerate() {
                    let mut output_index = source_index;
                    for &(size, stride) in &self.dimensions_and_stride {
                        let (before, after) = (output_index / stride, output_index % stride);
                        output_index = (before / size) * stride + after;
                    }
                    output[output_index] += source[storage_index];
                }
            }
        }
        Ok(output)
    }
}

struct Affine(f64, f64);

impl Map1 for Affine {
    fn f<T: WithDType>(&self, values: &[T], layout: &Layout) -> Result<Vec<T>> {
        let multiplier = T::from_f64(self.0);
        let addend = T::from_f64(self.1);
        Ok(unary_map(values, layout, |value| {
            value * multiplier + addend
        }))
    }
}

fn index_select<T: WithDType>(
    source: &[T],
    ids: &[u32],
    source_layout: &Layout,
    ids_layout: &Layout,
    dimension: usize,
) -> Result<Vec<T>> {
    let source = match source_layout.contiguous_offsets() {
        Some((start, end)) => &source[start..end],
        None => return Err(Error::RequiresContiguous { op: "index-select" }.bt()),
    };
    let id_count = match ids_layout.dims() {
        [count] => *count,
        dimensions => {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: dimensions.len(),
                shape: ids_layout.shape().clone(),
            }
            .bt())
        }
    };
    let id_stride = ids_layout.stride()[0];
    let mut output_dimensions = source_layout.dims().to_vec();
    let source_dimension = output_dimensions[dimension];
    output_dimensions[dimension] = id_count;
    let left_size = output_dimensions[..dimension].iter().product::<usize>();
    let right_size = output_dimensions[dimension + 1..].iter().product::<usize>();
    let mut output = vec![T::zero(); output_dimensions.iter().product()];

    for left in 0..left_size {
        let source_base = left * right_size * source_dimension;
        let output_base = left * right_size * id_count;
        for id_index in 0..id_count {
            let output_start = output_base + id_index * right_size;
            let index = ids[ids_layout.start_offset() + id_stride * id_index] as usize;
            if index >= source_dimension {
                return Err(Error::InvalidIndex {
                    index,
                    size: source_dimension,
                    op: "index-select",
                }
                .bt());
            }
            let source_start = source_base + index * right_size;
            output[output_start..output_start + right_size]
                .copy_from_slice(&source[source_start..source_start + right_size]);
        }
    }
    Ok(output)
}

fn copy_strided<T: Copy>(source: &[T], output: &mut [T], output_offset: usize, layout: &Layout) {
    let mut output_index = output_offset;
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            output[output_index..output_index + len]
                .copy_from_slice(&source[start_offset..start_offset + len]);
        }
        crate::StridedBlocks::UniformBlocks {
            start_offset,
            block_len,
            count,
            src_stride,
        } => {
            for block in 0..count {
                let source_index = start_offset + block * src_stride;
                output[output_index..output_index + block_len]
                    .copy_from_slice(&source[source_index..source_index + block_len]);
                output_index += block_len;
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            for source_index in block_start_index {
                output[output_index..output_index + block_len]
                    .copy_from_slice(&source[source_index..source_index + block_len]);
                output_index += block_len;
            }
        }
    }
}

fn matmul_f32(
    lhs: &[f32],
    rhs: &[f32],
    (batch, rows, columns, inner): (usize, usize, usize, usize),
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<Vec<f32>> {
    use gemm::{gemm, Parallelism};

    let rank = lhs_layout.shape().rank();
    let lhs_stride = lhs_layout.stride();
    let rhs_stride = rhs_layout.stride();
    let lhs_batch_stride = batch_stride(lhs_layout, rows * inner)?;
    let rhs_batch_stride = batch_stride(rhs_layout, columns * inner)?;
    let lhs = &lhs[lhs_layout.start_offset()..];
    let rhs = &rhs[rhs_layout.start_offset()..];
    let mut output = vec![0.0; batch * rows * columns];
    let parallelism = match crate::utils::get_num_threads() {
        0 | 1 => Parallelism::None,
        count => Parallelism::Rayon(count),
    };

    let (batch, rows, columns, inner) = if rhs_batch_stride == 0 && lhs_batch_stride == rows * inner
    {
        (1, batch * rows, columns, inner)
    } else if lhs_batch_stride == 0 && rhs_batch_stride == columns * inner {
        (1, rows, batch * columns, inner)
    } else {
        (batch, rows, columns, inner)
    };

    for step in 0..batch {
        let lhs = &lhs[step * lhs_batch_stride..];
        let rhs = &rhs[step * rhs_batch_stride..];
        let output = &mut output[step * rows * columns..];
        unsafe {
            gemm(
                rows,
                columns,
                inner,
                output.as_mut_ptr(),
                1,
                columns as isize,
                false,
                lhs.as_ptr(),
                lhs_stride[rank - 1] as isize,
                lhs_stride[rank - 2] as isize,
                rhs.as_ptr(),
                rhs_stride[rank - 1] as isize,
                rhs_stride[rank - 2] as isize,
                0.0,
                1.0,
                false,
                false,
                false,
                parallelism,
            );
        }
    }
    Ok(output)
}

fn batch_stride(layout: &Layout, contiguous_stride: usize) -> Result<usize> {
    let rank = layout.shape().rank();
    Ok(match layout.stride()[..rank - 2] {
        [outer, inner] if outer == inner * layout.dims()[1] => inner,
        [_, inner] if layout.dims()[0] == 1 => inner,
        [outer, _] if layout.dims()[1] == 1 => outer,
        [stride] => stride,
        [] => contiguous_stride,
        _ => return Err(Error::RequiresContiguous { op: "matmul" }.bt()),
    })
}

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::U32(_) => DType::U32,
            Self::F32(_) => DType::F32,
        }
    }

    fn device(&self) -> &CpuDevice {
        &CpuDevice
    }

    fn to_cpu_storage(&self) -> Result<Self> {
        Ok(self.clone())
    }

    fn affine(&self, layout: &Layout, multiplier: f64, addend: f64) -> Result<Self> {
        Affine(multiplier, addend).map(self, layout)
    }

    fn reduce_sum(&self, layout: &Layout, dimensions: &[usize]) -> Result<Self> {
        let source_dimensions = layout.dims();
        let mut output_dimensions = source_dimensions.to_vec();
        for &dimension in dimensions {
            output_dimensions[dimension] = 1;
        }
        let output_shape = Shape::from(output_dimensions);
        let mut dimensions = dimensions.to_vec();
        dimensions.sort_unstable();
        let dimensions_and_stride = dimensions
            .iter()
            .map(|&dimension| {
                (
                    source_dimensions[dimension],
                    source_dimensions[dimension + 1..].iter().product(),
                )
            })
            .collect();
        ReduceSum {
            output_shape: &output_shape,
            dimensions: &dimensions,
            dimensions_and_stride,
        }
        .map(self, layout)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match (self, dtype) {
            (Self::U32(values), DType::U32) => Ok(Self::U32(unary_map(values, layout, |v| v))),
            (Self::U32(values), DType::F32) => {
                Ok(Self::F32(unary_map(values, layout, |v| v as f32)))
            }
            (Self::F32(values), DType::U32) => {
                Ok(Self::U32(unary_map(values, layout, |v| v as u32)))
            }
            (Self::F32(values), DType::F32) => Ok(Self::F32(unary_map(values, layout, |v| v))),
        }
    }

    fn unary_impl<Op: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::F32(values) => Ok(Self::F32(unary_map_vec(
                values,
                layout,
                Op::f32,
                Op::f32_vec,
            ))),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, Op::NAME).bt()),
        }
    }

    fn binary_impl<Op: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(binary_map_vec(
                lhs_layout,
                rhs_layout,
                lhs,
                rhs,
                Op::f32,
                Op::f32_vec,
                Op::f32_scalar_vec,
            ))),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
                op: Op::NAME,
            }
            .bt()),
        }
    }

    fn index_select(
        &self,
        ids: &Self,
        layout: &Layout,
        ids_layout: &Layout,
        dimension: usize,
    ) -> Result<Self> {
        let Self::U32(ids) = ids else {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype(), "index-select").bt());
        };
        match self {
            Self::U32(source) => Ok(Self::U32(index_select(
                source, ids, layout, ids_layout, dimension,
            )?)),
            Self::F32(source) => Ok(Self::F32(index_select(
                source, ids, layout, ids_layout, dimension,
            )?)),
        }
    }

    fn matmul(
        &self,
        rhs: &Self,
        dimensions: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(matmul_f32(
                lhs, rhs, dimensions, lhs_layout, rhs_layout,
            )?)),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "matmul").bt()),
        }
    }

    fn copy_strided_src(&self, output: &mut Self, offset: usize, layout: &Layout) -> Result<()> {
        match (self, output) {
            (Self::U32(source), Self::U32(output)) => copy_strided(source, output, offset, layout),
            (Self::F32(source), Self::F32(output)) => copy_strided(source, output, offset, layout),
            (source, output) => {
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: source.dtype(),
                    rhs: output.dtype(),
                    op: "copy-strided",
                }
                .bt())
            }
        }
        Ok(())
    }
}

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cpu
    }

    fn storage_from_slice<T: WithDType>(&self, values: &[T]) -> Result<CpuStorage> {
        Ok(T::to_cpu_storage(values))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CpuStorage> {
        Ok(storage)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let count = shape.elem_count();
        Ok(match dtype {
            DType::U32 => CpuStorage::U32(vec![0; count]),
            DType::F32 => CpuStorage::F32(vec![0.0; count]),
        })
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let count = shape.elem_count();
        Ok(match dtype {
            DType::U32 => {
                let mut values = Vec::with_capacity(count);
                values.set_len(count);
                CpuStorage::U32(values)
            }
            DType::F32 => {
                let mut values = Vec::with_capacity(count);
                values.set_len(count);
                CpuStorage::F32(values)
            }
        })
    }
}
