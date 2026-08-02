#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

__device__ __forceinline__ bool is_contiguous(
    const size_t dimensions,
    const size_t *shape,
    const size_t *strides
) {
    size_t expected = 1;
    for (size_t offset = 0; offset < dimensions; ++offset) {
        const size_t dimension = dimensions - 1 - offset;
        if (shape[dimension] > 1 && strides[dimension] != expected) return false;
        expected *= shape[dimension];
    }
    return true;
}

__device__ __forceinline__ size_t get_strided_index(
    size_t index,
    const size_t dimensions,
    const size_t *shape,
    const size_t *strides
) {
    size_t strided = 0;
    for (size_t offset = 0; offset < dimensions; ++offset) {
        const size_t dimension = dimensions - 1 - offset;
        strided += (index % shape[dimension]) * strides[dimension];
        index /= shape[dimension];
    }
    return strided;
}

__device__ __forceinline__ float maxg(float left, float right) {
    return fmaxf(left, right);
}
