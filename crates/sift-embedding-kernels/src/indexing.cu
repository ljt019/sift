#include "cuda_utils.cuh"
#include <assert.h>
#include <stdint.h>

extern "C" __global__ void is_u32_f32(
    const size_t elements,
    const size_t dimensions,
    const size_t *layout,
    const uint32_t *ids,
    const float *input,
    float *output,
    const size_t left_size,
    const size_t source_size,
    const size_t ids_size,
    const size_t right_size
) {
    const size_t *shape = layout;
    const size_t *strides = layout + dimensions;
    const bool contiguous = is_contiguous(dimensions, shape, strides);

    for (size_t destination = blockIdx.x * blockDim.x + threadIdx.x;
         destination < elements;
         destination += blockDim.x * gridDim.x) {
        const size_t left = destination / (ids_size * right_size);
        const size_t id_index = destination / right_size % ids_size;
        const size_t right = destination % right_size;
        const uint32_t id = ids[id_index];

        if (id == UINT32_MAX) {
            output[destination] = 0.0f;
            continue;
        }

        assert(id < source_size);
        const size_t source = left * source_size * right_size + id * right_size + right;
        output[destination] = input[contiguous
            ? source
            : get_strided_index(source, dimensions, shape, strides)];
    }
}
