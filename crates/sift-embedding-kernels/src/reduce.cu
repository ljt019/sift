#include "cuda_utils.cuh"

constexpr int REDUCTION_BLOCK_SIZE = 1024;

extern "C" __global__ void fast_sum_f32(
    const sift_size_t source_elements,
    const sift_size_t elements_per_output,
    const sift_size_t dimensions,
    const sift_size_t *info,
    const float *source,
    float *output
) {
    const sift_size_t *shape = info;
    const sift_size_t *strides = info + dimensions;
    __shared__ float partial[REDUCTION_BLOCK_SIZE];
    const sift_size_t thread = threadIdx.x;
    const sift_size_t output_index = blockIdx.x;
    const sift_size_t start = output_index * elements_per_output;
    const sift_size_t end = min(start + elements_per_output, source_elements);

    partial[thread] = 0.0f;
    for (sift_size_t index = start + thread; index < end; index += blockDim.x) {
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
#if defined(__HIP_PLATFORM_AMD__)
    constexpr int SOFTMAX_BLOCK_SIZE = 256;
    __shared__ float partial[SOFTMAX_BLOCK_SIZE];
    const int row = blockIdx.x;
    const int thread = threadIdx.x;

    float maximum = -sift_infinity();
    for (int column = thread; column < columns; column += blockDim.x) {
        maximum = maxg(maximum, source[row * columns + column]);
    }
    partial[thread] = maximum;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (thread < stride) partial[thread] = maxg(partial[thread], partial[thread + stride]);
    }
    __syncthreads();
    maximum = partial[0];

    float sum = 0.0f;
    for (int column = thread; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        const float value = sift_expf(source[index] - maximum);
        output[index] = value;
        sum += value;
    }
    partial[thread] = sum;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (thread < stride) partial[thread] += partial[thread + stride];
    }
    __syncthreads();
    sum = partial[0];
    for (int column = thread; column < columns; column += blockDim.x) {
        output[row * columns + column] /= sum;
    }
#else
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int thread = threadIdx.y;
    float maximum = -sift_infinity();
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
        const float value = sift_expf(source[index] - maximum);
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
#endif
}

extern "C" __global__ void rope_f32(
    const float *source,
    const float *cosine,
    const float *sine,
    float *output,
    const sift_u32 batch_heads,
    const sift_u32 token_dimension,
    const sift_u32 dimension,
    const sift_u32 frequency_batch_stride
) {
    const sift_u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * index >= batch_heads * token_dimension) return;

    const sift_u32 batch_head = index / (token_dimension / 2);
    const sift_u32 token_offset = index - (token_dimension / 2) * batch_head;
    const sift_u32 token = token_offset / (dimension / 2);
    const sift_u32 channel = token_offset - (dimension / 2) * token;
    const sift_u32 left = batch_head * token_dimension + token * dimension + channel;
    const sift_u32 right = left + dimension / 2;
    sift_u32 frequency = token * (dimension / 2) + channel;
    if (frequency_batch_stride > 0) {
        frequency += ((2 * index) / frequency_batch_stride) * (token_dimension / 2);
    }

    const float c = cosine[frequency];
    const float s = sine[frequency];
    output[left] = source[left] * c - source[right] * s;
    output[right] = source[left] * s + source[right] * c;
}
