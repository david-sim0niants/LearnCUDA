#include "error.h"
#include "matmul.h"

template<typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, size_t N, size_t K, size_t M)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= M)
        return;

    T c = 0;
    for (size_t k = 0; k < K; ++k)
        c += A[row * K + k] * B[k * M + col];

    C[row * M + col] = c;
}

template<typename T> requires (std::integral<T> || std::floating_point<T>)
void Matmul<T>::matmul(
        const T *dev_A, const T *dev_B, T *dev_C,
        size_t N, size_t K, size_t M)
{
    dim3 block_size(16, 16);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);
    matmul_kernel<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, N, K, M);
    assert_cuda_ok();
    cudaDeviceSynchronize();
}

template struct Matmul<int>;
template struct Matmul<float>;
template struct Matmul<double>;
