use crate::backend::BackendStorage;
use crate::op;
use crate::{CpuStorage, CudaStorage, CustomOp1, CustomOp3};
use crate::{DType, Device, Error, Layout, Result, Shape};

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead try_clone should be used.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
            Self::Cuda(storage) => storage.dtype(),
        }
    }

    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs_device = self.device();
        let rhs_device = rhs.device();
        let lhs = lhs_device.location();
        let rhs = rhs_device.location();
        if lhs != rhs {
            Err(Error::DeviceMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub(crate) fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn reduce_sum(&self, layout: &Layout, dimensions: &[usize]) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.reduce_sum(layout, dimensions)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.reduce_sum(layout, dimensions)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Self, Shape)> {
        match self {
            Self::Cpu(storage) => {
                let (storage, shape) = c.cpu_fwd(storage, l)?;
                Ok((Self::Cpu(storage), shape))
            }
            Self::Cuda(storage) => {
                let (storage, shape) = c.cuda_fwd(storage, l)?;
                Ok((Self::Cuda(storage), shape))
            }
        }
    }

    pub(crate) fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        match (self, t2, t3) {
            (Self::Cpu(s1), Self::Cpu(s2), Self::Cpu(s3)) => {
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Cpu(s), shape))
            }
            (Self::Cuda(s1), Self::Cuda(s2), Self::Cuda(s3)) => {
                let (s, shape) = c.cuda_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Cuda(s), shape))
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn unary_impl<B: op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn binary_impl<B: op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: lhs.device().location(),
                    rhs: rhs.device().location(),
                    op: B::NAME,
                }
                .bt())
            }
        }
    }

    pub(crate) fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(rhs, "index-select")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_l, rhs_l, d)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_l, rhs_l, d)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "index-select",
            }
            .bt()),
        }
    }

    pub(crate) fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "matmul",
            }
            .bt()),
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => src.copy_strided_src(dst, dst_offset, src_l),
            (Self::Cuda(src), Self::Cuda(dst)) => Ok(src.copy_strided_src(dst, dst_offset, src_l)?),
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy",
            }
            .bt()),
        }
    }
}
