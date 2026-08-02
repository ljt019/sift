#include "cuda_utils.cuh"
#include <stdint.h>

constexpr int REDUCTION_BLOCK_SIZE = 1024;

extern "C" __global__ void fast_sum_f32(
    const size_t source_elements,
    const size_t elements_per_output,
    const size_t dimensions,
    const size_t *info,
    const float *source,
    float *output
) {
    const size_t *shape = info;
    const size_t *strides = info + dimensions;
    __shared__ float partial[REDUCTION_BLOCK_SIZE];
    const size_t thread = threadIdx.x;
    const size_t output_index = blockIdx.x;
    const size_t start = output_index * elements_per_output;
    const size_t end = min(start + elements_per_output, source_elements);

    partial[thread] = 0.0f;
    for (size_t index = start + thread; index < end; index += blockDim.x) {
        partial[thread] += source[get_strided_index(index, dimensions, shape, strides)];
    }
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (thread < stride) partial[thread] += partial[thread + stride];
    }
    if (thread == 0) output[output_index] = partial[0];
}

extern "C" __global__ void softmax_f32(
    const float *source,
    float *output,
    const int columns
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int thread = threadIdx.y;
    float maximum = -INFINITY;
    for (int column = thread; column < columns; column += 32) {
        maximum = maxg(maximum, source[row * columns + column]);
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        maximum = maxg(maximum, __shfl_xor_sync(0xffffffff, maximum, mask, 32));
    }

    float sum = 0.0f;
    for (int column = thread; column < columns; column += 32) {
        const int index = row * columns + column;
        const float value = expf(source[index] - maximum);
        output[index] = value;
        sum += value;
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    for (int column = thread; column < columns; column += 32) {
        output[row * columns + column] /= sum;
    }
}

extern "C" __global__ void rope_f32(
    const float *source,
    const float *cosine,
    const float *sine,
    float *output,
    const uint32_t batch_heads,
    const uint32_t token_dimension,
    const uint32_t dimension,
    const uint32_t frequency_batch_stride
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * index >= batch_heads * token_dimension) return;

    const uint32_t batch_head = index / (token_dimension / 2);
    const uint32_t token_offset = index - (token_dimension / 2) * batch_head;
    const uint32_t token = token_offset / (dimension / 2);
    const uint32_t channel = token_offset - (dimension / 2) * token;
    const uint32_t left = batch_head * token_dimension + token * dimension + channel;
    const uint32_t right = left + dimension / 2;
    uint32_t frequency = token * (dimension / 2) + channel;
    if (frequency_batch_stride > 0) {
        frequency += ((2 * index) / frequency_batch_stride) * (token_dimension / 2);
    }

    const float c = cosine[frequency];
    const float s = sine[frequency];
    output[left] = source[left] * c - source[right] * s;
    output[right] = source[left] * s + source[right] * c;
}
