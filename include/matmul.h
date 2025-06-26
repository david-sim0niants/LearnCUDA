#pragma once

#include <cstddef>
#include <concepts>

/** Template struct wrapper to simplify explicit instantiation of the function. */
template<typename T> requires (std::integral<T> || std::floating_point<T>)
struct Matmul {
    static void matmul(const T *dev_A, const T *dev_B, T *dev_C,
                       std::size_t N, std::size_t K, std::size_t M,
                       std::size_t A_stride, std::size_t B_stride, std::size_t C_stride);
};

/** Matrix multiplication: C = A x B
 * A: [N x K], B: [K x M], C: [N x M] — all in device memory */
template<typename T>
inline void matmul(const T *dev_A, const T *dev_B, T *dev_C,
                   std::size_t N, std::size_t K, std::size_t M)
    requires (std::integral<T> || std::floating_point<T>)
{
    return Matmul<T>::matmul(dev_A, dev_B, dev_C, N, K, M, K, M, M);
}

/** Matrix multiplication: C = A x B
 * A: [N x K], B: [K x M], C: [N x M] — all in device memory */
template<typename T>
inline void matmul(const T *dev_A, const T *dev_B, T *dev_C,
                   std::size_t N, std::size_t K, std::size_t M,
                   std::size_t A_stride, std::size_t B_stride, std::size_t C_stride)
    requires (std::integral<T> || std::floating_point<T>)
{
    return Matmul<T>::matmul(dev_A, dev_B, dev_C, N, K, M, A_stride, B_stride, C_stride);
}
