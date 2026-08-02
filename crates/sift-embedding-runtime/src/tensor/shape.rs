//! The shape of a tensor is a tuple with the size of each of its dimensions.
#![allow(clippy::redundant_closure_call)]
use crate::{Error, Result};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(d: $tuple) -> Self {
                Self(vec![$(d.$index,)+])
            }
        }
    }
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl Shape {
    /// The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The dimensions as a slice of `usize`.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// The total number of elements, this is the product of all dimension sizes.
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    /// The strides given in number of elements for a contiguous n-dimensional
    /// arrays using this shape.
    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride: Vec<_> = self
            .0
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    /// Returns true if the strides are C contiguous (aka row major).
    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    /// Modifies the shape by adding a list of additional dimensions at the end of the existing
    /// dimensions.
    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }

    /// Check whether the two shapes are compatible for broadcast, and if it is the case return the
    /// broadcasted shape. This is to be used for binary pointwise ops.
    pub fn broadcast_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx {
                1
            } else {
                lhs_dims[lhs_ndims - rev_idx]
            };
            let r_value = if rhs_ndims < rev_idx {
                1
            } else {
                rhs_dims[rhs_ndims - rev_idx]
            };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    op,
                }
                .bt())?
            }
        }
        Ok(Shape::from(bcast_dims))
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        match self.dims() {
            [first, second, third] => Ok((*first, *second, *third)),
            dims => Err(Error::UnexpectedNumberOfDims {
                expected: 3,
                got: dims.len(),
                shape: self.clone(),
            }
            .bt()),
        }
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        match self.dims() {
            [first, second, third, fourth] => Ok((*first, *second, *third, *fourth)),
            dims => Err(Error::UnexpectedNumberOfDims {
                expected: 4,
                got: dims.len(),
                shape: self.clone(),
            }
            .bt()),
        }
    }
}

impl crate::Tensor {
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.shape().dims3()
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        self.shape().dims4()
    }
}

pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(dim)
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(dim)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    Minus1,
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        Error::DimOutOfRange {
            shape: shape.clone(),
            dim: -1,
            op,
        }
        .bt()
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        if rank >= 1 {
            Ok(rank - 1)
        } else {
            Err(self.out_of_range(shape, op))
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, _: &'static str) -> Result<usize> {
        Ok(shape.rank())
    }
}

pub trait ShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape>;
}

impl<S: Into<Shape>> ShapeWithOneHole for S {
    fn into_shape(self, _el_count: usize) -> Result<Shape> {
        Ok(self.into())
    }
}

fn hole_size(el_count: usize, product: usize) -> Result<usize> {
    if product == 0 || !el_count.is_multiple_of(product) {
        crate::bail!("cannot reshape {el_count} elements with fixed product {product}")
    }
    Ok(el_count / product)
}

impl ShapeWithOneHole for (usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (first, second, ()) = self;
        Ok((first, second, hole_size(el_count, first * second)?).into())
    }
}
