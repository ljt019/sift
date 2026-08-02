//! Tensors are N-dimensional matrixes of elements using a single data type.
#![allow(clippy::redundant_closure_call)]
use crate::backend::BackendStorage;
use crate::shape::{Dim, ShapeWithOneHole};
use crate::{storage::Storage, DType, Device, Error, Layout, Result, Shape};
use std::sync::Arc;

pub struct Tensor_ {
    storage: Arc<Storage>,
    layout: Layout,
    dtype: DType,
    device: Device,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

// Tensors and storages are independently reference counted so it is possible to avoid
// copying the storage for operations that only modify the shape or stride.
#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self
                .storage()
                .unary_impl::<crate::op::$op_name>(self.layout())?;
            Ok(from_storage(storage, shape.clone()))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self.storage().binary_impl::<crate::op::$op_name>(
                &*rhs.storage(),
                self.layout(),
                rhs.layout(),
            )?;
            Ok(from_storage(storage, shape.clone()))
        }
    };
}

macro_rules! broadcast_binary_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let lhs = self;
            let shape = lhs
                .shape()
                .broadcast_shape_binary_op(rhs.shape(), stringify!($fn_name))?;
            let l_broadcast = shape != *lhs.shape();
            let r_broadcast = shape != *rhs.shape();
            match (l_broadcast, r_broadcast) {
                (true, true) => lhs
                    .broadcast_as(&shape)?
                    .$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

pub(crate) fn from_storage<S: Into<Shape>>(storage: Storage, shape: S) -> Tensor {
    let dtype = storage.dtype();
    let device = storage.device();
    let tensor_ = Tensor_ {
        storage: Arc::new(storage),
        layout: Layout::contiguous(shape),
        dtype,
        device,
    };
    Tensor(Arc::new(tensor_))
}

impl Tensor {
    pub fn arange<D: crate::WithDType>(start: D, end: D, device: &Device) -> Result<Self> {
        let mut data = vec![];
        let mut current = start;
        while current < end {
            data.push(current);
            current += D::one();
        }
        let len = data.len();
        Self::from_vec_impl(data, len, device)
    }

    pub(crate) fn from_vec_impl<S: ShapeWithOneHole, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into_shape(data.len())?;
        let storage = device.storage_owned(data)?;
        Ok(from_storage(storage, shape))
    }

    pub fn from_vec<S: ShapeWithOneHole, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_vec_impl(data, shape, device)
    }

    pub fn from_slice<S: ShapeWithOneHole, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into_shape(array.len())?;
        let storage = device.storage_from_slice(array)?;
        Ok(from_storage(storage, shape))
    }

    pub(crate) fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs != rhs {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            }
            .bt())
        } else {
            Ok(lhs)
        }
    }

    binary_op!(add, Add);
    binary_op!(mul, Mul);
    binary_op!(div, Div);
    broadcast_binary_op!(broadcast_add, add);
    broadcast_binary_op!(broadcast_mul, mul);
    broadcast_binary_op!(broadcast_div, div);
    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(sqr, Sqr);
    unary_op!(sqrt, Sqrt);
    unary_op!(gelu, Gelu);
    unary_op!(gelu_erf, GeluErf);

    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage().affine(self.layout(), mul, add)?;
        Ok(from_storage(storage, self.shape()))
    }

    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        let err = |msg| {
            Err::<(), _>(
                Error::NarrowInvalidArgs {
                    shape: self.shape().clone(),
                    dim,
                    start,
                    len,
                    msg,
                }
                .bt(),
            )
        };
        if start > dims[dim] {
            err("start > dim_len")?
        }
        if start.saturating_add(len) > dims[dim] {
            err("start + len > dim_len")?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let layout = self.layout().narrow(dim, start, len)?;
            let tensor_ = Tensor_ {
                storage: self.storage.clone(),
                layout,
                dtype: self.dtype,
                device: self.device.clone(),
            };
            Ok(Tensor(Arc::new(tensor_)))
        }
    }

    pub fn sum_keepdim<D: Dim>(&self, sum_dim: D) -> Result<Self> {
        let sum_dims = [sum_dim.to_index(self.shape(), "sum")?];
        let storage = self.storage().reduce_sum(self.layout(), &sum_dims)?;
        let mut dims = self.dims().to_vec();
        for &sum_dim in sum_dims.iter() {
            dims[sum_dim] = 1
        }
        Ok(from_storage(storage, dims))
    }

    pub fn sum<D: Dim>(&self, sum_dim: D) -> Result<Self> {
        let sum_dims = [sum_dim.to_index(self.shape(), "sum")?];
        let storage = self.storage().reduce_sum(self.layout(), &sum_dims)?;
        let dims = self
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(index, &size)| (!sum_dims.contains(&index)).then_some(size))
            .collect::<Vec<_>>();
        Ok(from_storage(storage, dims))
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        if dim < 2 || b_dims.len() != dim {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .bt())?
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        if c_shape.elem_count() == 0 || k == 0 {
            let storage = self.device().zeros(&c_shape, self.dtype())?;
            return Ok(from_storage(storage, c_shape));
        }
        let batching: usize = a_dims[..dim - 2].iter().product();
        let batching_b: usize = b_dims[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .bt())?
        }

        let storage = self.storage().matmul(
            &rhs.storage(),
            (batching, m, n, k),
            self.layout(),
            rhs.layout(),
        )?;
        Ok(from_storage(storage, c_shape))
    }

    pub fn index_select<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-select")?;
        let indexes_len = match indexes.dims() {
            [l] => *l,
            _ => Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
                op: "index-select",
            }
            .bt())?,
        };
        let storage = self.storage().index_select(
            &indexes.storage(),
            self.layout(),
            indexes.layout(),
            dim,
        )?;
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes_len;
        Ok(from_storage(storage, dims))
    }

    pub fn strided_index(&self) -> crate::StridedIndex<'_> {
        self.layout.strided_index()
    }

    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let data = match self.layout.contiguous_offsets() {
                Some((o1, o2)) => data[o1..o2].to_vec(),
                None => self.strided_index().map(|i| data[i]).collect(),
            };
            Ok::<Vec<_>, Error>(data)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    pub fn flatten_all(&self) -> Result<Tensor> {
        self.reshape(self.elem_count())
    }

    pub fn t(&self) -> Result<Tensor> {
        let rank = self.rank();
        if rank < 2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        self.transpose(rank - 2, rank - 1)
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Tensor> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let tensor_ = Tensor_ {
            storage: self.storage.clone(),
            layout: self.layout.transpose(dim1, dim2)?,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let tensor_ = Tensor_ {
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape)?,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let storage = self.storage().to_dtype(self.layout(), dtype)?;
            Ok(from_storage(storage, shape.clone()))
        }
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = unsafe { self.device().alloc_uninit(shape, self.dtype())? };
            self.storage()
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(storage, shape.clone()))
        }
    }

    pub fn reshape<S: ShapeWithOneHole>(&self, s: S) -> Result<Tensor> {
        let shape = s.into_shape(self.elem_count())?;
        if shape.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            }
            .bt());
        }
        if self.is_contiguous() {
            let tensor_ = Tensor_ {
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                dtype: self.dtype,
                device: self.device.clone(),
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            let mut storage = unsafe { self.device().alloc_uninit(&shape, self.dtype())? };
            self.storage()
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(storage, shape))
        }
    }

    pub fn squeeze<D: Dim>(&self, dim: D) -> Result<Self> {
        // The PyTorch semantics are to return the same tensor if the target dimension
        // does not have a size of 1.
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "squeeze")?;
        if dims[dim] == 1 {
            let mut dims = dims.to_vec();
            let mut strides = self.stride().to_vec();
            dims.remove(dim);
            strides.remove(dim);
            let tensor_ = Tensor_ {
                storage: self.storage.clone(),
                layout: Layout::new(dims.into(), strides, self.layout.start_offset()),
                dtype: self.dtype,
                device: self.device.clone(),
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            Ok(self.clone())
        }
    }

    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        let mut strides = self.stride().to_vec();
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        // Cannot panic because to_index_plus_one already checks dimensions
        dims.insert(dim, 1);
        // Any stride would work here, but we pick one so as to maximize the probability to remain
        // C contiguous.
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        strides.insert(dim, stride);
        let tensor_ = Tensor_ {
            storage: self.storage.clone(),
            layout: Layout::new(dims.into(), strides, self.layout.start_offset()),
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn apply<M: crate::Module>(&self, m: &M) -> Result<Self> {
        m.forward(self)
    }

    pub(crate) fn storage(&self) -> &Storage {
        &self.storage
    }
}

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

bin_trait!(Add, add, |_| 1., |v| v);
bin_trait!(Mul, mul, |v| v, |_| 0.);
bin_trait!(Div, div, |v| 1. / v, |_| 0.);
