#include "cuda_utils.cuh"

#define UNARY_OP(TYPENAME, NAME, EXPR)                                         \
extern "C" __global__ void NAME(                                              \
    const sift_size_t elements, const sift_size_t dimensions,                 \
    const sift_size_t *info,                                                   \
    const TYPENAME *input, TYPENAME *output) {                                 \
    const sift_size_t *shape = info;                                           \
    const sift_size_t *strides = info + dimensions;                            \
    const bool contiguous = info == nullptr ||                                 \
        is_contiguous(dimensions, shape, strides);                             \
    for (sift_size_t index = blockIdx.x * blockDim.x + threadIdx.x;            \
         index < elements; index += blockDim.x * gridDim.x) {                  \
        const sift_size_t source = contiguous ? index :                        \
            get_strided_index(index, dimensions, shape, strides);             \
        const TYPENAME x = input[source];                                      \
        output[index] = EXPR;                                                  \
    }                                                                          \
}

__device__ __forceinline__ float gelu(float value) {
    const float cubic = value * value * value;
    const float inner = value + 0.044715f * cubic;
    return 0.5f * value * (1.0f + sift_tanhf(0.7978845608f * inner));
}

__device__ __forceinline__ float gelu_erf(float value) {
    return 0.5f * value * (1.0f + sift_erff(value * 0.7071067812f));
}

UNARY_OP(sift_u32, ucopy_u32, x)
UNARY_OP(float, ucopy_f32, x)
UNARY_OP(float, usin_f32, sift_sinf(x))
UNARY_OP(float, ucos_f32, sift_cosf(x))
UNARY_OP(float, usqr_f32, x * x)
UNARY_OP(float, usqrt_f32, sift_sqrtf(x))
UNARY_OP(float, ugelu_f32, gelu(x))
UNARY_OP(float, ugelu_erf_f32, gelu_erf(x))
