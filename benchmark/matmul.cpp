#include "matmul.h"
#include "device_buffer.h"

#include <tuple>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

namespace {

template<typename T>
void bm_matmul(benchmark::State& state)
{
    const size_t N = state.range(0);
    const size_t K = state.range(1);
    const size_t M = state.range(2);

    DeviceBuffer<T> dev_A(N * K), dev_B(K * M), dev_C(N * M);
    cudaMemsetAsync(dev_A.data(), 1, dev_A.size());
    cudaMemsetAsync(dev_B.data(), 2, dev_B.size());
    cudaDeviceSynchronize();

    for (auto _ : state)
        matmul(dev_A.data(), dev_B.data(), dev_C.data(), N, K, M);

    state.counters["FLOPS/sec"] = benchmark::Counter(N * K * M * 2, benchmark::Counter::kIsRate);
    state.SetComplexityN(K);
}

template<typename T>
void register_benchmarks_range_doubling(const char *name, size_t N, size_t K, size_t M)
{
    while ((N * K + K * M + N * M) * sizeof(T) <= size_t(1 << 30) * 3 / 4) {
        benchmark::RegisterBenchmark(name, bm_matmul<T>)->Args({long(N), long(K), long(M)});
        N *= 2;
        K *= 2;
        M *= 2;
    }
}

template<typename T>
void register_benchmarks_for_type(const char *name)
{
    static constexpr std::tuple<size_t, size_t, size_t> ranges[] = {
        { 1, 1, 1 },
        { 1, 1, 2 },
        { 1, 2, 1 },
        { 1, 2, 2 },
        { 2, 1, 1 },
        { 2, 1, 2 },
        { 2, 2, 1 },
    };

    for (auto [N, K, M] : ranges)
        register_benchmarks_range_doubling<T>(name, N, K, M);
}

int register_benchmarks()
{
    register_benchmarks_for_type<int>("bm_matmul<int>");
    register_benchmarks_for_type<float>("bm_matmul<float>");
    register_benchmarks_for_type<double>("bm_matmul<double>");
    return 0;
}

const auto _ = register_benchmarks();

}
