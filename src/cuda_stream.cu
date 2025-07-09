#include "cuda_stream.h"
#include "error.h"

cudaStream_t create_stream()
{
    cudaStream_t stream = {};
    assert_cuda_ok(cudaStreamCreate(&stream));
    return stream;
}

CudaStream::CudaStream() : CudaStream(create_stream())
{
}

void CudaStream::synchronize()
{
    assert_cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(native())));
}

CudaStream::~CudaStream() noexcept
{
    cudaStreamDestroy(static_cast<cudaStream_t>(native()));
}

CudaStream CudaStream::default_ (0);
