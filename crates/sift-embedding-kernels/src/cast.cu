#include "cuda_utils.cuh"

extern "C" __global__ void cast_u32_f32(
    const sift_size_t elements,
    const sift_size_t dimensions,
    const sift_size_t *info,
    const sift_u32 *input,
    float *output
) {
    const sift_size_t *shape = info;
    const sift_size_t *strides = info + dimensions;
    const bool contiguous = info == nullptr || is_contiguous(dimensions, shape, strides);
    for (sift_size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += blockDim.x * gridDim.x) {
        const sift_size_t source = contiguous ? index : get_strided_index(index, dimensions, shape, strides);
        output[index] = static_cast<float>(input[source]);
    }
}
