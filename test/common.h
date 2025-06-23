#pragma once

#include <type_traits>

#include <gtest/gtest.h>

#define TEST_NUM_EQUAL(a, b, asserter) \
{ \
    if constexpr (std::is_integral_v<std::remove_cvref_t<decltype(a)>>) \
        asserter##_EQ(a, b); \
    else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(a)>, double>) \
        asserter##_DOUBLE_EQ(a, b); \
    else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(a)>, float>) \
        asserter##_FLOAT_EQ(a, b); \
    else \
        asserter##_NEAR(a, b, std::remove_cvref_t<decltype(a)>(1e-6)); \
}

#define ASSERT_NUM_EQUAL(a, b) TEST_NUM_EQUAL(a, b, ASSERT)
#define EXPECT_NUM_EQUAL(a, b) TEST_NUM_EQUAL(a, b, EXPECT)
