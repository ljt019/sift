#include "cuda_utils.cuh"

constexpr int MATMUL_TILE = 16;

extern "C" __global__ void matmul_f32(
    const float *left,
    const float *right,
    float *output,
    const sift_size_t rows,
    const sift_size_t columns,
    const sift_size_t inner,
    const sift_size_t left_row_stride,
    const sift_size_t left_inner_stride,
    const sift_size_t right_inner_stride,
    const sift_size_t right_column_stride,
    const sift_size_t left_batch_stride,
    const sift_size_t right_batch_stride
) {
    __shared__ float left_tile[MATMUL_TILE][MATMUL_TILE];
    __shared__ float right_tile[MATMUL_TILE][MATMUL_TILE];

    const sift_size_t row = blockIdx.y * MATMUL_TILE + threadIdx.y;
    const sift_size_t column = blockIdx.x * MATMUL_TILE + threadIdx.x;
    const sift_size_t batch = blockIdx.z;
    const float *batch_left = left + batch * left_batch_stride;
    const float *batch_right = right + batch * right_batch_stride;
    float sum = 0.0f;

    for (sift_size_t tile = 0; tile < inner; tile += MATMUL_TILE) {
        const sift_size_t left_inner = tile + threadIdx.x;
        const sift_size_t right_inner = tile + threadIdx.y;
        left_tile[threadIdx.y][threadIdx.x] = row < rows && left_inner < inner
            ? batch_left[row * left_row_stride + left_inner * left_inner_stride]
            : 0.0f;
        right_tile[threadIdx.y][threadIdx.x] = right_inner < inner && column < columns
            ? batch_right[right_inner * right_inner_stride + column * right_column_stride]
            : 0.0f;
        __syncthreads();

#pragma unroll
        for (int offset = 0; offset < MATMUL_TILE; ++offset) {
            sum += left_tile[threadIdx.y][offset] * right_tile[offset][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && column < columns) {
        output[(batch * rows + row) * columns + column] = sum;
    }
}
