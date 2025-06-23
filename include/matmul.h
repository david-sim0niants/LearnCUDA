#pragma once

#include <cstddef>
#include <concepts>

template<typename T> requires (std::integral<T> || std::floating_point<T>)
struct Matmul {
    static void matmul(const T *dev_A, const T *dev_B, T *dev_C,
                       std::size_t N, std::size_t K, std::size_t M);
};

template<typename T>
inline void matmul(const T *dev_A, const T *dev_B, T *dev_C,
                   std::size_t N, std::size_t K, std::size_t M)
    requires (std::integral<T> || std::floating_point<T>)
{
    return Matmul<T>::matmul(dev_A, dev_B, dev_C, N, K, M);
}
