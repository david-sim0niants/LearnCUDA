#include "vector_add.h"
#include "device_buffer.h"

#include "common.h"
#include "prng.h"
#include "test_config.h"

#include <gtest/gtest.h>

template<typename T>
class VectorAddTest : public ::testing::Test {
protected:
    PRNG prng {TestConfig::get_prng_seed()};
};

using TypeParams = ::testing::Types<int, float, double>;
TYPED_TEST_SUITE(VectorAddTest, TypeParams);

TYPED_TEST(VectorAddTest, BasicCheck)
{
    const size_t size = 1 << 20;

    std::vector<TypeParam> a(size), b(size), c(size, 0);

    for (size_t i = 0; i < size; ++i) {
        a[i] = this->prng.template gen_uniform<TypeParam>() / 2;
        b[i] = this->prng.template gen_uniform<TypeParam>() / 2;
    }

    DeviceBuffer dev_a(a.data(), size), dev_b(b.data(), size), dev_c(c.data(), size);

    add_vectors(dev_a.data(), dev_b.data(), dev_c.data(), size);

    dev_c.load_to(c.data());

    for (int i = 0; i < size; ++i)
        ASSERT_NUM_EQUAL(c[i], a[i] + b[i]);
}
