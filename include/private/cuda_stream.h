#pragma once

#include "../cuda_stream.h"
#include <cuda_runtime.h>

inline cudaStream_t current_cuda_stream()
{
    return static_cast<cudaStream_t>(CudaStream::current().native());
}
