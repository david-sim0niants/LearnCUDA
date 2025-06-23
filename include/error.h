#pragma once

#include <stdexcept>

#include <cuda_runtime.h>

template<typename ErrorType = std::runtime_error>
inline void assert_cuda_ok(cudaError_t err = cudaGetLastError())
{
    if (err != cudaSuccess)
        throw ErrorType(cudaGetErrorString(err));
}
