#pragma once

#if defined(__HIP_PLATFORM_AMD__)
typedef __SIZE_TYPE__ sift_size_t;
typedef unsigned int sift_u32;
#define SIFT_U32_MAX 0xffffffffU
#define SIFT_ASSERT(condition) do { if (!(condition)) __builtin_trap(); } while (0)

__device__ __attribute__((const)) float __ocml_fmax_f32(float, float);
__device__ __attribute__((pure)) float __ocml_exp_f32(float);
__device__ float __ocml_sin_f32(float);
__device__ float __ocml_cos_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_f32(float);
__device__ __attribute__((pure)) float __ocml_tanh_f32(float);
__device__ __attribute__((pure)) float __ocml_erf_f32(float);

__device__ __forceinline__ float sift_fmaxf(float left, float right) {
    return __ocml_fmax_f32(left, right);
}
__device__ __forceinline__ float sift_expf(float value) { return __ocml_exp_f32(value); }
__device__ __forceinline__ float sift_sinf(float value) { return __ocml_sin_f32(value); }
__device__ __forceinline__ float sift_cosf(float value) { return __ocml_cos_f32(value); }
__device__ __forceinline__ float sift_sqrtf(float value) { return __ocml_sqrt_f32(value); }
__device__ __forceinline__ float sift_tanhf(float value) { return __ocml_tanh_f32(value); }
__device__ __forceinline__ float sift_erff(float value) { return __ocml_erf_f32(value); }
__device__ __forceinline__ float sift_infinity() { return __builtin_inff(); }
#else
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef size_t sift_size_t;
typedef uint32_t sift_u32;
#define SIFT_U32_MAX UINT32_MAX
#define SIFT_ASSERT(condition) assert(condition)

__device__ __forceinline__ float sift_fmaxf(float left, float right) {
    return fmaxf(left, right);
}
__device__ __forceinline__ float sift_expf(float value) { return expf(value); }
__device__ __forceinline__ float sift_sinf(float value) { return sinf(value); }
__device__ __forceinline__ float sift_cosf(float value) { return cosf(value); }
__device__ __forceinline__ float sift_sqrtf(float value) { return sqrtf(value); }
__device__ __forceinline__ float sift_tanhf(float value) { return tanhf(value); }
__device__ __forceinline__ float sift_erff(float value) { return erff(value); }
__device__ __forceinline__ float sift_infinity() { return INFINITY; }
#endif

__device__ __forceinline__ bool is_contiguous(
    const sift_size_t dimensions,
    const sift_size_t *shape,
    const sift_size_t *strides
) {
    sift_size_t expected = 1;
    for (sift_size_t offset = 0; offset < dimensions; ++offset) {
        const sift_size_t dimension = dimensions - 1 - offset;
        if (shape[dimension] > 1 && strides[dimension] != expected) return false;
        expected *= shape[dimension];
    }
    return true;
}

__device__ __forceinline__ sift_size_t get_strided_index(
    sift_size_t index,
    const sift_size_t dimensions,
    const sift_size_t *shape,
    const sift_size_t *strides
) {
    sift_size_t strided = 0;
    for (sift_size_t offset = 0; offset < dimensions; ++offset) {
        const sift_size_t dimension = dimensions - 1 - offset;
        strided += (index % shape[dimension]) * strides[dimension];
        index /= shape[dimension];
    }
    return strided;
}

__device__ __forceinline__ float maxg(float left, float right) {
    return sift_fmaxf(left, right);
}
