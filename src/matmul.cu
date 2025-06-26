#include "error.h"
#include "matmul.h"

template<typename T>
__global__ void matmul_kernel(
        const T * __restrict__ A,
        const T * __restrict__ B,
        T * __restrict__ C,
        const size_t N, const size_t K, const size_t M,
        const size_t A_stride, const size_t B_stride, const size_t C_stride)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= M)
        return;

    T c = 0;
    for (size_t k = 0; k < K; ++k)
        c += A[row * A_stride + k] * B[k * B_stride + col];

    C[row * C_stride + col] = c;
}

constexpr size_t TILE_SIZE = 16;

template<typename T>
__global__ void matmul_tiled_kernel(
        const T * __restrict__ A,
        const T * __restrict__ B,
        T * __restrict__ C,
        const size_t N, const size_t K, const size_t M,
        const size_t A_stride, const size_t B_stride, const size_t C_stride)
{
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T A_sub[TILE_SIZE][TILE_SIZE];
    __shared__ T B_sub[TILE_SIZE][TILE_SIZE];
    T c = 0;

    for (size_t t = 0; t <= K / TILE_SIZE; ++t) {

        const size_t i = threadIdx.y;
        const size_t j = threadIdx.x;
        size_t k = 0;

        k = t * TILE_SIZE + j;
        A_sub[i][j] = row < N && k < K ? A[row * A_stride + k] : T(0); // A[row][k]

        k = t * TILE_SIZE + i;
        B_sub[i][j] = col < M && k < K ? B[k * B_stride + col] : T(0); // B[k][col]

        __syncthreads();

        for (size_t k = 0; k < TILE_SIZE; ++k)
            c += A_sub[i][k] * B_sub[k][j];

        __syncthreads();
    }

    if (row < N && col < M)
        C[row * C_stride + col] = c;
}

template<typename T> requires (std::integral<T> || std::floating_point<T>)
void Matmul<T>::matmul(
        const T *dev_A, const T *dev_B, T *dev_C,
        size_t N, size_t K, size_t M,
        size_t A_stride, size_t B_stride, size_t C_stride)
{
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);
    matmul_tiled_kernel<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, N, K, M,
                                                   A_stride, B_stride, C_stride);
    assert_cuda_ok();
    cudaDeviceSynchronize();
}

template struct Matmul<int>;
template struct Matmul<float>;
template struct Matmul<double>;
