#pragma once

#if defined(__CUDACC__)
    #define CUDA_HD __host__ __device__
#else
    #define CUDA_HD
#endif
