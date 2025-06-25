#include "matmul.h"
#include "device_buffer.h"

#include "prng.h"
#include "test_config.h"

#include <gtest/gtest.h>

#include <vector>

using TestArg = std::tuple<size_t, size_t, size_t>;

class MatmulTest : public ::testing::TestWithParam<TestArg> {
protected:
    template<typename T>
    void do_basic_check()
    {
        const auto [N, K, M] = GetParam();

        std::vector<T> A (N * K), B(K * M), C(N * M, T{});
        auto range = T(std::sqrt(T(std::numeric_limits<T>::max()) / K));

        for (T& a : A)
            a = prng.gen_uniform<T>(-range, +range);
        for (T& b : B)
            b = prng.gen_uniform<T>(-range, +range);

        DeviceBuffer dev_A(A.data(), A.size()),
                     dev_B(B.data(), B.size()),
                     dev_C(C.data(), C.size());

        matmul(dev_A.data(), dev_B.data(), dev_C.data(), N, K, M);

        dev_C.load_to(C.data());

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                T expected_c = 0;
                for (size_t k = 0; k < K; ++k)
                    expected_c += A[i * K + k] * B[k * M + j];
                T actual_c = C[i * M + j];
                if constexpr (std::is_integral_v<T>)
                    ASSERT_EQ(expected_c, actual_c);
                else
                    ASSERT_NEAR(expected_c, actual_c, 1e-6 * std::abs(expected_c));
            }
        }
    }

    PRNG prng {TestConfig::get_prng_seed()};
};

TEST_P(MatmulTest, BasicCheck)
{
    do_basic_check<int>();
    do_basic_check<double>();
}

INSTANTIATE_TEST_SUITE_P(PerMatrixSizes, MatmulTest,
        ::testing::Values(
            TestArg{4, 3, 4},
            TestArg{10, 20, 30},
            TestArg{100, 300, 200},
            TestArg{1000, 1000, 1000}
        ));
