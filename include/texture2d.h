#pragma once

#include "common_type_traits.h"
#include "pixel_type.h"
#include "vec.h"
#include "cuda_compat.h"

#include <cstddef>
#include <tuple>
#include <type_traits>

template<typename Byte>
class Texture2DView_ {
public:
    Texture2DView_() noexcept = default;

    Texture2DView_(Byte* data, std::size_t pitch) noexcept
        : data_(data), pitch_(pitch)
    {
    }

    template<typename OtherByte>
    Texture2DView_(Texture2DView_<OtherByte> other) noexcept
        requires std::is_convertible_v<OtherByte*, Byte*>
        : Texture2DView_(other.data(), other.pitch())
    {
    }

    CUDA_HD inline Byte* data() noexcept
    {
        return data_;
    }

    CUDA_HD inline std::size_t pitch() const noexcept
    {
        return pitch_;
    }

    CUDA_HD inline Byte* row_at(std::size_t i) noexcept
    {
        return data_ + i * pitch_;
    }

    template<typename Pixel>
    CUDA_HD inline auto& at(std::size_t i, std::size_t j) noexcept
    {
        return reinterpret_cast<MatchCV<Byte, Pixel>*>(row_at(i))[j];
    }

    template<typename Channel, int nr_channels>
    CUDA_HD inline auto& at(std::size_t i, std::size_t j, int c) noexcept
    {
        return at<Vec<Channel, nr_channels>>(i, j)[c];
    }

    CUDA_HD inline void swap(Texture2DView_& rhs) noexcept
    {
        auto tmp_data_ = data_;
        data_ = rhs.data_;
        rhs.data_ = tmp_data_;

        auto tmp_pitch_ = pitch_;
        pitch_ = rhs.pitch_;
        rhs.pitch_ = tmp_pitch_;
    }

private:
    Byte* data_ = nullptr;
    std::size_t pitch_ = 0;
};

using Texture2DView = Texture2DView_<std::byte>;
using Texture2DReadOnlyView = Texture2DView_<const std::byte>;

struct Texture2DSize {
    std::size_t width, height;
};

namespace std {
    template <>
    struct tuple_size<Texture2DSize> : std::integral_constant<size_t, 2> {};

    template <>
    struct tuple_element<0, Texture2DSize> { using type = size_t; };

    template <>
    struct tuple_element<1, Texture2DSize> { using type = size_t; };

    template <size_t I>
    CUDA_HD auto get(const Texture2DSize& s)
    {
        if constexpr (I == 0)
            return s.width;
        else if constexpr (I == 1)
            return s.height;
    }
}

class Texture2D {
public:
    Texture2D() = default;

    Texture2D(const Texture2DView& view, const Texture2DSize& size, PixelType pixel_type)
        : view_(view), size_(size), pixel_type_(pixel_type)
    {
    }

    virtual ~Texture2D() = default;

    inline Texture2DView view() noexcept
    {
        return view_;
    }

    inline Texture2DReadOnlyView view() const noexcept
    {
        return view_;
    }

    inline const Texture2DSize& size() const noexcept
    {
        return size_;
    }

    inline std::size_t mem_size() const noexcept
    {
        return size_.height * view_.pitch();
    }

    inline const PixelType& pixel_type() const noexcept
    {
        return pixel_type_;
    }

protected:
    inline void swap(Texture2D& other) noexcept
    {
        std::swap(view_, other.view_);
        std::swap(size_.width, other.size_.width);
        std::swap(size_.height, other.size_.height);
        std::swap(pixel_type_, other.pixel_type_);
    }

private:
    Texture2DView view_;
    Texture2DSize size_ = {0, 0};
    PixelType pixel_type_ = PixelType::NONE;
};
