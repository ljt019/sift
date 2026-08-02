#include "cuda_utils.cuh"
#include <stdint.h>

extern "C" __global__ void cast_u32_f32(
    const size_t elements,
    const size_t dimensions,
    const size_t *info,
    const uint32_t *input,
    float *output
) {
    const size_t *shape = info;
    const size_t *strides = info + dimensions;
    const bool contiguous = info == nullptr || is_contiguous(dimensions, shape, strides);
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += blockDim.x * gridDim.x) {
        const size_t source = contiguous ? index : get_strided_index(index, dimensions, shape, strides);
        output[index] = static_cast<float>(input[source]);
    }
}
