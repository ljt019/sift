//! Tensor Layouts including contiguous or sparse strides

use crate::{Error, Result, Shape};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        self.is_contiguous().then(|| {
            let start = self.start_offset;
            (start, start + self.shape.elem_count())
        })
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op: "narrow",
            }
            .bt())?
        }
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            }
            .bt())?
        }
        let mut dims = dims.to_vec();
        dims[dim] = len;
        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2),
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Ok(Self {
            shape: Shape::from(dims),
            stride,
            start_offset: self.start_offset,
        })
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(),
                dst_shape: shape,
            }
            .bt());
        }
        let added_dims = shape.rank() - self.shape().rank();
        let mut stride = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(),
                    dst_shape: shape,
                }
                .bt());
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self {
            shape,
            stride,
            start_offset: self.start_offset,
        })
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex<'_> {
        crate::StridedIndex::from_layout(self)
    }

    pub(crate) fn strided_blocks(&self) -> crate::StridedBlocks<'_> {
        let mut block_len = 1usize;
        let mut contiguous_dims = 0usize; // Counted from the right.
        for (&stride, &dim) in self.stride().iter().zip(self.dims().iter()).rev() {
            // Size-1 dimensions are trivially contiguous regardless of their stride.
            if dim == 1 {
                contiguous_dims += 1;
                continue;
            }
            if stride != block_len {
                break;
            }
            block_len *= dim;
            contiguous_dims += 1;
        }
        let index_dims = self.dims().len() - contiguous_dims;
        match index_dims {
            0 => crate::StridedBlocks::SingleBlock {
                start_offset: self.start_offset,
                len: block_len,
            },
            1 => crate::StridedBlocks::UniformBlocks {
                start_offset: self.start_offset,
                block_len,
                count: self.dims()[0],
                src_stride: self.stride[0],
            },
            _ => {
                let block_start_index = crate::StridedIndex::new(
                    &self.dims()[..index_dims],
                    &self.stride[..index_dims],
                    self.start_offset,
                );
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len,
                }
            }
        }
    }
}
