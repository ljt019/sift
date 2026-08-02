#include "cuda_utils.cuh"

extern "C" __global__ void is_u32_f32(
    const sift_size_t elements,
    const sift_size_t dimensions,
    const sift_size_t *layout,
    const sift_u32 *ids,
    const float *input,
    float *output,
    const sift_size_t left_size,
    const sift_size_t source_size,
    const sift_size_t ids_size,
    const sift_size_t right_size
) {
    const sift_size_t *shape = layout;
    const sift_size_t *strides = layout + dimensions;
    const bool contiguous = is_contiguous(dimensions, shape, strides);

    for (sift_size_t destination = blockIdx.x * blockDim.x + threadIdx.x;
         destination < elements;
         destination += blockDim.x * gridDim.x) {
        const sift_size_t left = destination / (ids_size * right_size);
        const sift_size_t id_index = destination / right_size % ids_size;
        const sift_size_t right = destination % right_size;
        const sift_u32 id = ids[id_index];

        if (id == SIFT_U32_MAX) {
            output[destination] = 0.0f;
            continue;
        }

        SIFT_ASSERT(id < source_size);
        const sift_size_t source = left * source_size * right_size + id * right_size + right;
        output[destination] = input[contiguous
            ? source
            : get_strided_index(source, dimensions, shape, strides)];
    }
}
