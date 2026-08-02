//! Implementation of Backend traits for CUDA device
//!
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, UnaryOpT};
use crate::{builder_arg as barg, CpuStorage, DType, Layout, Result};
pub use cudarc;
use cudarc::cublas::{GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
pub use sift_embedding_kernels as kernels;

mod device;
mod error;
mod utils;
pub use device::CudaDevice;
pub use error::{CudaError, WrapErr};
pub use utils::Map1;
use utils::Map2;

pub enum SlicePtrOrNull<T> {
    Ptr(CudaSlice<T>),
    Null,
}

impl<T: DeviceRepr> SlicePtrOrNull<T> {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        match self {
            SlicePtrOrNull::Ptr(slice) => builder.arg(slice),
            SlicePtrOrNull::Null => builder.arg(&0usize),
        };
    }
}

impl SlicePtrOrNull<usize> {
    pub fn params_from_vec(dev: &CudaDevice, params: Vec<usize>) -> Result<Self> {
        Ok(SlicePtrOrNull::Ptr(dev.clone_htod(&params)?))
    }

    pub fn params_from_layout(dev: &CudaDevice, l: &Layout) -> Result<Self> {
        if l.is_contiguous() {
            Ok(SlicePtrOrNull::Null)
        } else {
            Self::params_from_vec(dev, [l.dims(), l.stride()].concat())
        }
    }
}

#[derive(Debug)]
pub enum CudaStorageSlice {
    U32(CudaSlice<u32>),
    F32(CudaSlice<f32>),
}

struct Affine(f64, f64);
impl Map1 for Affine {
    fn f(&self, src: &CudaSlice<f32>, dev: &CudaDevice, layout: &Layout) -> Result<CudaSlice<f32>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func("affine_f32", &kernels::AFFINE)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<f32>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&out);
        barg!(builder, self.0 as f32);
        barg!(builder, self.1 as f32);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg).w() }?;
        Ok(out)
    }
}

struct FastReduce<'a>(&'a [usize]);
impl Map1 for FastReduce<'_> {
    fn f(&self, src: &CudaSlice<f32>, dev: &CudaDevice, layout: &Layout) -> Result<CudaSlice<f32>> {
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
        let cfg = LaunchConfig {
            // TODO: Maybe use grid_y if the output is too large?
            // TODO: Specialized implementation when reducing on no or all dimensions or when
            // reducing only aggregate a small number of elements together.
            grid_dim: (dst_el as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let ds =
            SlicePtrOrNull::params_from_vec(dev, [dims.as_slice(), stride.as_slice()].concat())?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func("fast_sum_f32", &kernels::REDUCE)?;
        // SAFETY: The kernel initializes every output element.
        let output = unsafe { dev.alloc::<f32>(dst_el)? };
        let mut builder = func.builder();
        barg!(builder, src_el);
        barg!(builder, el_to_sum_per_block);
        barg!(builder, src_dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&output);
        // SAFETY: Kernel arguments match reduce.cu's fast_sum kernel.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(output)
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f(&self, src: &CudaSlice<f32>, dev: &CudaDevice, layout: &Layout) -> Result<CudaSlice<f32>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&format!("{}_f32", U::KERNEL), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let mut out = unsafe { dev.alloc::<f32>(el_count)? };
        let mut builder = func.builder();
        barg!(builder, el_count);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&mut out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

fn slice_ptr<T: DeviceRepr>(v: &CudaSlice<T>, lo: usize) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    let (_, guard) = v.device_ptr(v.stream());
    let (ptr, _) = v.slice(lo..).device_ptr(v.stream());
    (ptr, guard)
}

struct IndexSelect<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map1 for IndexSelect<'_> {
    fn f(&self, src: &CudaSlice<f32>, dev: &CudaDevice, src_l: &Layout) -> Result<CudaSlice<f32>> {
        let ids_l = &self.1;
        let (name, (ids, _guard)) = match &self.0.slice {
            CudaStorageSlice::U32(slice) => ("is_u32", slice_ptr(slice, ids_l.start_offset())),
            _ => Err(CudaError::UnexpectedDType {
                msg: "index_select ids must be u32",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            .w()?,
        };
        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        let ds = SlicePtrOrNull::params_from_vec(dev, [ids_dims, ids_l.stride()].concat())?;
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let src_dim_size = src_l.dims()[self.2];
        let ids_dim_size = ids_shape.elem_count();
        let dst_el = ids_shape.elem_count() * left_size * right_size;
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&format!("{name}_f32"), &kernels::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<f32>(dst_el)? };
        let mut builder = func.builder();
        barg!(builder, dst_el);
        barg!(builder, ids_dims.len());
        ds.builder_arg(&mut builder);
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, left_size);
        barg!(builder, src_dim_size);
        barg!(builder, ids_dim_size);
        barg!(builder, right_size);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f(
        &self,
        lhs: &CudaSlice<f32>,
        lhs_l: &Layout,
        rhs: &CudaSlice<f32>,
        rhs_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<CudaSlice<f32>> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dims_and_strides = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::params_from_vec(dev, [dims, lhs_l.stride(), rhs_l.stride()].concat())?
        };
        let lhs = &lhs.slice(lhs_l.start_offset()..);
        let rhs = &rhs.slice(rhs_l.start_offset()..);
        let func = dev.get_or_load_func(&format!("{}_f32", U::KERNEL), &kernels::BINARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<f32>(elem_count)? };
        let mut builder = func.builder();
        barg!(builder, elem_count);
        barg!(builder, dims.len());
        dims_and_strides.builder_arg(&mut builder);
        builder.arg(lhs);
        builder.arg(rhs);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

fn slice_src_and_dst<'a, T>(
    src: &'a CudaSlice<T>,
    src_l: &Layout,
    dst: &'a mut CudaSlice<T>,
    dst_offset: usize,
) -> (
    cudarc::driver::CudaView<'a, T>,
    cudarc::driver::CudaViewMut<'a, T>,
) {
    let src_offset = src_l.start_offset();
    let to_copy = dst
        .len()
        .saturating_sub(dst_offset)
        .min(src.len().saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}

#[derive(Debug)]
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub device: CudaDevice,
}

fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<StridedBatchedConfig<T>> {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;

    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // The a tensor has dims batching, k, n (rhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The b tensor has dims batching, m, k (lhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

impl BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn dtype(&self) -> DType {
        match self.slice {
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::F32(_) => DType::F32,
        }
    }

    fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype() != DType::U32 || dtype != DType::F32 {
            return Err(CudaError::UnsupportedDtype {
                dtype,
                op: "to_dtype",
            }
            .into());
        }

        let elements = layout.shape().elem_count();
        let config = LaunchConfig::for_num_elems(elements as u32);
        let parameters = SlicePtrOrNull::params_from_layout(&self.device, layout)?;
        let CudaStorageSlice::U32(input) = &self.slice else {
            unreachable!()
        };
        let input = input.slice(layout.start_offset()..);
        let output = unsafe { self.device.alloc::<f32>(elements)? };
        let function = self
            .device
            .get_or_load_func("cast_u32_f32", &kernels::CAST)?;
        let mut builder = function.builder();
        barg!(builder, elements);
        barg!(builder, layout.dims().len());
        parameters.builder_arg(&mut builder);
        builder.arg(&input);
        builder.arg(&output);
        unsafe { builder.launch(config) }.w()?;
        Ok(Self {
            slice: CudaStorageSlice::F32(output),
            device: self.device.clone(),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn reduce_sum(&self, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let slice = FastReduce(sum_dims).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = U::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = B::V.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            CudaStorageSlice::U32(slice) => Ok(CpuStorage::U32(self.device.clone_dtoh(slice)?)),
            CudaStorageSlice::F32(slice) => Ok(CpuStorage::F32(self.device.clone_dtoh(slice)?)),
        }
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    fn matmul(
        &self,
        rhs: &Self,
        (batch, rows, columns, inner): (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        let (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) = (&self.slice, &rhs.slice)
        else {
            return Err(CudaError::UnsupportedDtype {
                dtype: self.dtype(),
                op: "matmul",
            }
            .into());
        };
        let lhs = lhs.slice(lhs_layout.start_offset()..);
        let rhs = rhs.slice(rhs_layout.start_offset()..);
        let config = gemm_config(
            1.0,
            0.0,
            (batch, rows, columns, inner),
            lhs_layout,
            rhs_layout,
        )?;
        let mut output = unsafe { self.device.alloc::<f32>(batch * rows * columns)? };
        unsafe { gemm_strided_batched_f32(&self.device.blas, config, &rhs, &lhs, &mut output) }
            .w()?;
        Ok(Self {
            slice: CudaStorageSlice::F32(output),
            device: self.device.clone(),
        })
    }

    fn copy_strided_src(
        &self,
        destination: &mut Self,
        offset: usize,
        layout: &Layout,
    ) -> Result<()> {
        fn copy<T: DeviceRepr>(
            device: &CudaDevice,
            source: &CudaSlice<T>,
            destination: &mut CudaSlice<T>,
            offset: usize,
            layout: &Layout,
            kernel: &str,
        ) -> Result<()> {
            let elements = layout.shape().elem_count();
            let (source, mut destination) = slice_src_and_dst(source, layout, destination, offset);
            if layout.is_contiguous() {
                return device.memcpy_dtod(&source, &mut destination);
            }
            let config = LaunchConfig::for_num_elems(elements as u32);
            let parameters = SlicePtrOrNull::params_from_layout(device, layout)?;
            let function = device.get_or_load_func(kernel, &kernels::UNARY)?;
            let mut builder = function.builder();
            barg!(builder, elements);
            barg!(builder, layout.dims().len());
            parameters.builder_arg(&mut builder);
            builder.arg(&source);
            builder.arg(&mut destination);
            unsafe { builder.launch(config) }.w()?;
            Ok(())
        }

        match (&self.slice, &mut destination.slice) {
            (CudaStorageSlice::U32(source), CudaStorageSlice::U32(destination)) => copy(
                &self.device,
                source,
                destination,
                offset,
                layout,
                "ucopy_u32",
            ),
            (CudaStorageSlice::F32(source), CudaStorageSlice::F32(destination)) => copy(
                &self.device,
                source,
                destination,
                offset,
                layout,
                "ucopy_f32",
            ),
            _ => Err(CudaError::InternalError("dtype mismatch in copy_strided").into()),
        }
    }
}

unsafe fn gemm_strided_batched_f32(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f32>,
    a: &cudarc::driver::CudaView<f32>,
    b: &cudarc::driver::CudaView<f32>,
    c: &mut CudaSlice<f32>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let compute_type = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let alpha = &cfg.gemm.alpha as *const f32 as *const _;
    let beta = &cfg.gemm.beta as *const f32 as *const _;

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        a as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.lda,
        cfg.stride_a,
        b as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        c as *mut _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
}
