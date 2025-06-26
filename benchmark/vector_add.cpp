#include "vector_add.h"
#include "device_buffer.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

template<typename T>
void bm_vector_add(benchmark::State& state)
{
    const std::size_t size = state.range(0);

    DeviceBuffer<T> dev_a(size), dev_b(size), dev_c(size);
    cudaMemsetAsync(dev_a.data(), 1, dev_a.size());
    cudaMemsetAsync(dev_b.data(), 2, dev_b.size());
    cudaDeviceSynchronize();

    for (auto _ : state)
        add_vectors(dev_a.data(), dev_b.data(), dev_c.data(), size);
}

BENCHMARK_TEMPLATE(bm_vector_add, int)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
BENCHMARK_TEMPLATE(bm_vector_add, float)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
BENCHMARK_TEMPLATE(bm_vector_add, double)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
