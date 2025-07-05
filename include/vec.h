#pragma once

#include "common_concepts.h"
#include "cuda_compat.h"

template<Numeric Elem, std::size_t size_>
struct Vec {
    Elem array[size_];

    using Element = Elem;

    CUDA_HD constexpr Vec& operator+=(const Vec& other) noexcept
    {
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            array[i] += other.array[i];
        return *this;
    }

    CUDA_HD constexpr Vec& operator-=(const Vec& other) noexcept
    {
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            array[i] -= other.array[i];
        return *this;
    }

    CUDA_HD constexpr Vec& operator*=(const Vec& other) noexcept
    {
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            array[i] *= other.array[i];
        return *this;
    }

    CUDA_HD constexpr Vec& operator/=(const Vec& other) noexcept
    {
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            array[i] /= other.array[i];
        return *this;
    }

    CUDA_HD constexpr Vec& operator*=(Elem scalar) noexcept
    {
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            array[i] *= scalar;
        return *this;
    }

    CUDA_HD constexpr Vec& operator/=(Elem scalar) noexcept
    {
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            array[i] /= scalar;
        return *this;
    }

    CUDA_HD constexpr Vec operator-() const noexcept
    {
        Vec res;
        #pragma unroll
        for (std::size_t i = 0; i < size_; ++i)
            res.array[i] = -array[i];
        return res;
    }

    CUDA_HD constexpr Elem& operator[](std::size_t i) noexcept
    {
        return array[i];
    }

    CUDA_HD constexpr const Elem& operator[](std::size_t i) const noexcept
    {
        return array[i];
    }

    CUDA_HD constexpr Elem* data() noexcept
    {
        return array;
    }

    CUDA_HD constexpr const Elem* data() const noexcept
    {
        return array;
    }

    CUDA_HD constexpr static std::size_t size() noexcept
    {
        return size_;
    }

    CUDA_HD constexpr Elem& x() noexcept requires (size_ >= 1)
    {
        return array[0];
    }

    CUDA_HD constexpr const Elem& x() const noexcept requires (size_ >= 1)
    {
        return array[0];
    }

    CUDA_HD constexpr Elem& y() noexcept requires (size_ >= 2)
    {
        return array[1];
    }

    CUDA_HD constexpr const Elem& y() const noexcept requires (size_ >= 2)
    {
        return array[1];
    }

    CUDA_HD constexpr Elem& z() noexcept requires (size_ >= 3)
    {
        return array[2];
    }

    CUDA_HD constexpr const Elem& z() const noexcept requires (size_ >= 3)
    {
        return array[2];
    }

    CUDA_HD constexpr Elem& w() noexcept requires (size_ >= 4)
    {
        return array[3];
    }

    CUDA_HD constexpr const Elem& w() const noexcept requires (size_ >= 4)
    {
        return array[3];
    }
};

template<Numeric Elem, std::size_t size>
CUDA_HD constexpr Vec<Elem, size> operator+(Vec<Elem, size> a, const Vec<Elem, size>& b) noexcept
{
    return a += b;
}

template<Numeric Elem, std::size_t size>
CUDA_HD constexpr Vec<Elem, size> operator-(Vec<Elem, size> a, const Vec<Elem, size>& b) noexcept
{
    return a -= b;
}

template<Numeric Elem, std::size_t size>
CUDA_HD constexpr Vec<Elem, size> operator*(Vec<Elem, size> a, const Vec<Elem, size>& b) noexcept
{
    return a *= b;
}

template<Numeric Elem, std::size_t size>
CUDA_HD constexpr Vec<Elem, size> operator/(Vec<Elem, size> a, const Vec<Elem, size>& b) noexcept
{
    return a /= b;
}

template<Numeric Elem, std::size_t size, Numeric Scalar>
CUDA_HD constexpr Vec<Elem, size> operator*(Vec<Elem, size> v, Scalar scalar) noexcept
{
    return v *= scalar;
}

template<Numeric Elem, std::size_t size, Numeric Scalar>
CUDA_HD constexpr Vec<Elem, size> operator*(Scalar scalar, Vec<Elem, size> v) noexcept
{
    return v *= scalar;
}
template<Numeric Elem, std::size_t size, Numeric Scalar>
CUDA_HD constexpr Vec<Elem, size> operator/(Vec<Elem, size> v, Scalar scalar) noexcept
{
    return v /= scalar;
}
