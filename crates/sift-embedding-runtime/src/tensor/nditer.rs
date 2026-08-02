use std::iter::FusedIterator;

use crate::Layout;

/// Maximum tensor rank supported by [`NdIter`].
pub const MAX_DIMS: usize = 8;

/// Multi-dimensional iterator that walks `N` tensor layouts simultaneously.
///
/// Internally, adjacent dimensions are merged whenever their strides allow it,
/// reducing the number of outer iterations. Each call to [`Iterator::next`] yields
/// `[usize; N]`, one base offset per layout, for one inner slice of `inner_size`
/// elements. Callers iterate that slice using `inner_strides`:
///
/// ```ignore
/// let nd_iter = NdIter::new([lhs_l, rhs_l]);
/// let inner_size = nd_iter.inner_size;
/// let [inner_ls, inner_rs] = nd_iter.inner_strides;
/// for [lhs_off, rhs_off] in nd_iter {
///     for i in 0..inner_size {
///         let lhs = lhs_buf[lhs_off + i * inner_ls];
///         let rhs = rhs_buf[rhs_off + i * inner_rs];
///         ...
///     }
/// }
/// ```
///
/// Note: `inner_strides[n]` can be 1 contiguous, but also 0 (broadcast/scalar) or > 1 (non-contiguous),
/// so callers must not assume contiguous element access within a slice.
pub struct NdIter<const N: usize> {
    pub inner_size: usize,
    /// Per-layout element stride within each inner slice: `element_offset = base + i * stride`.
    pub inner_strides: [usize; N],

    outer_dims: [usize; MAX_DIMS],
    outer_strides: [[usize; MAX_DIMS]; N],
    outer_len: usize,

    offsets: [usize; N],
    coords: [usize; MAX_DIMS],
    remaining: usize,
}

impl<const N: usize> NdIter<N> {
    pub fn new(layouts: [&Layout; N]) -> NdIter<N> {
        let dims = layouts[0].dims();
        debug_assert!(
            dims.len() <= MAX_DIMS,
            "rank {} exceeds MAX_DIMS={}",
            dims.len(),
            MAX_DIMS
        );
        #[cfg(debug_assertions)]
        for l in &layouts {
            debug_assert_eq!(l.dims(), dims);
        }

        let rank = dims.len();
        let mut out_dims = [0usize; MAX_DIMS];
        let mut out_strides = [[0usize; MAX_DIMS]; N];
        let mut out_len;

        if rank == 0 {
            out_dims[0] = 1;
            out_len = 1;
        } else {
            out_dims[0] = dims[0];
            for n in 0..N {
                out_strides[n][0] = layouts[n].stride()[0];
            }
            out_len = 1;

            for (i, d) in dims.iter().enumerate().take(rank).skip(1) {
                let top = out_len - 1;
                let last_d = out_dims[top];

                let (can_merge, use_inner) = if last_d == 1 {
                    (true, true)
                } else if *d == 1 {
                    (true, false)
                } else {
                    let can_merge =
                        (0..N).all(|n| out_strides[n][top] == layouts[n].stride()[i] * d);
                    (can_merge, true)
                };
                if can_merge {
                    out_dims[top] = last_d * d;
                    if use_inner {
                        for n in 0..N {
                            out_strides[n][top] = layouts[n].stride()[i];
                        }
                    }
                } else {
                    out_dims[out_len] = *d;
                    for n in 0..N {
                        out_strides[n][out_len] = layouts[n].stride()[i];
                    }
                    out_len += 1;
                }
            }
        }

        // Update inner strides
        let inner_idx = out_len - 1;
        let inner_size = out_dims[inner_idx];
        let mut inner_strides = [0usize; N];
        for n in 0..N {
            inner_strides[n] = out_strides[n][inner_idx];
        }

        // Update outer dims
        let outer_len = inner_idx;
        let mut outer_dims = [0usize; MAX_DIMS];
        outer_dims[..outer_len].copy_from_slice(&out_dims[..outer_len]);

        // Update outer strides
        let mut outer_strides = [[0usize; MAX_DIMS]; N];
        for n in 0..N {
            outer_strides[n][..outer_len].copy_from_slice(&out_strides[n][..outer_len]);
        }

        // Update offsets
        let mut offsets = [0usize; N];
        for n in 0..N {
            offsets[n] = layouts[n].start_offset();
        }

        // Number of outer blocks (product of all dims except the innermost).
        let remaining = out_dims[..outer_len].iter().product();

        NdIter {
            inner_size,
            inner_strides,
            outer_dims,
            outer_strides,
            outer_len,
            offsets,
            coords: [0; MAX_DIMS],
            remaining,
        }
    }
}

impl<const N: usize> Iterator for NdIter<N> {
    type Item = [usize; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.offsets;
        self.remaining -= 1;

        for k in (0..self.outer_len).rev() {
            self.coords[k] += 1;
            for n in 0..N {
                self.offsets[n] += self.outer_strides[n][k];
            }
            if self.coords[k] < self.outer_dims[k] {
                break;
            }
            self.coords[k] = 0;
            for n in 0..N {
                self.offsets[n] -= self.outer_dims[k] * self.outer_strides[n][k];
            }
        }

        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<const N: usize> ExactSizeIterator for NdIter<N> {}
impl<const N: usize> FusedIterator for NdIter<N> {}
