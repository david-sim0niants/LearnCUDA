#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>

enum class ChannelType : uint8_t {
    NONE = 0,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    FLOAT, DOUBLE
};

template<typename Handler, typename... Args>
constexpr bool handle_channel_type(ChannelType channel_type, Handler&& handler, Args&&... args)
{
    switch (channel_type) {
        using enum ChannelType;
    case I8:
        handler(int8_t(), std::forward<Args>(args)...); break;
    case I16:
        handler(int16_t(), std::forward<Args>(args)...); break;
    case I32:
        handler(int32_t(), std::forward<Args>(args)...); break;
    case I64:
        handler(int64_t(), std::forward<Args>(args)...); break;
    case U8:
        handler(uint8_t(), std::forward<Args>(args)...); break;
    case U16:
        handler(uint16_t(), std::forward<Args>(args)...); break;
    case U32:
        handler(uint32_t(), std::forward<Args>(args)...); break;
    case U64:
        handler(uint64_t(), std::forward<Args>(args)...); break;
    case FLOAT:
        handler(float(), std::forward<Args>(args)...); break;
    case DOUBLE:
        handler(double(), std::forward<Args>(args)...); break;
    default:
        return false;
    }
    return true;
}

constexpr std::size_t get_channel_type_size(ChannelType channel)
{
    switch (channel) {
        using enum ChannelType;
    case I8:
    case U8:
        return 1;
    case I16:
    case U16:
        return 2;
    case I32:
    case U32:
        return 4;
    case I64:
    case U64:
        return 8;
    case FLOAT:
        return sizeof(float);
    case DOUBLE:
        return sizeof(double);
    default:
        return 0;
    }
}

constexpr uint16_t make_pixel_type(ChannelType channel, uint8_t nr_channels)
{
    return (uint16_t(channel) << 8) | uint16_t(nr_channels);
}

enum class PixelType : uint16_t {
    NONE    = make_pixel_type(ChannelType::NONE, 0)

    , I8x1  = make_pixel_type(ChannelType::I8,  1)
    , I16x1 = make_pixel_type(ChannelType::I16, 1)
    , I32x1 = make_pixel_type(ChannelType::I32, 1)
    , I64x1 = make_pixel_type(ChannelType::I64, 1)

    , I8x2  = make_pixel_type(ChannelType::I8,  2)
    , I16x2 = make_pixel_type(ChannelType::I16, 2)
    , I32x2 = make_pixel_type(ChannelType::I32, 2)
    , I64x2 = make_pixel_type(ChannelType::I64, 2)

    , I8x3  = make_pixel_type(ChannelType::I8,  3)
    , I16x3 = make_pixel_type(ChannelType::I16, 3)
    , I32x3 = make_pixel_type(ChannelType::I32, 3)
    , I64x3 = make_pixel_type(ChannelType::I64, 3)

    , I8x4  = make_pixel_type(ChannelType::I8,  4)
    , I16x4 = make_pixel_type(ChannelType::I16, 4)
    , I32x4 = make_pixel_type(ChannelType::I32, 4)
    , I64x4 = make_pixel_type(ChannelType::I64, 4)

    , U8x1  = make_pixel_type(ChannelType::U8,  1)
    , U16x1 = make_pixel_type(ChannelType::U16, 1)
    , U32x1 = make_pixel_type(ChannelType::U32, 1)
    , U64x1 = make_pixel_type(ChannelType::U64, 1)

    , U8x2  = make_pixel_type(ChannelType::U8,  2)
    , U16x2 = make_pixel_type(ChannelType::U16, 2)
    , U32x2 = make_pixel_type(ChannelType::U32, 2)
    , U64x2 = make_pixel_type(ChannelType::U64, 2)

    , U8x3  = make_pixel_type(ChannelType::U8,  3)
    , U16x3 = make_pixel_type(ChannelType::U16, 3)
    , U32x3 = make_pixel_type(ChannelType::U32, 3)
    , U64x3 = make_pixel_type(ChannelType::U64, 3)

    , U8x4  = make_pixel_type(ChannelType::U8,  4)
    , U16x4 = make_pixel_type(ChannelType::U16, 4)
    , U32x4 = make_pixel_type(ChannelType::U32, 4)
    , U64x4 = make_pixel_type(ChannelType::U64, 4)

    , Fx1   = make_pixel_type(ChannelType::FLOAT, 1)
    , Fx2   = make_pixel_type(ChannelType::FLOAT, 2)
    , Fx3   = make_pixel_type(ChannelType::FLOAT, 3)
    , Fx4   = make_pixel_type(ChannelType::FLOAT, 4)

    , Dx1   = make_pixel_type(ChannelType::DOUBLE, 1)
    , Dx2   = make_pixel_type(ChannelType::DOUBLE, 2)
    , Dx3   = make_pixel_type(ChannelType::DOUBLE, 3)
    , Dx4   = make_pixel_type(ChannelType::DOUBLE, 4)
};

constexpr ChannelType get_channel_type(PixelType pixel_type)
{
    return ChannelType(uint16_t(pixel_type) >> 8);
}

constexpr uint8_t get_nr_channels(PixelType pixel_type)
{
    return uint8_t(uint16_t(pixel_type) & 0xFF);
}

constexpr std::size_t get_pixel_type_size(PixelType pixel_type)
{
    return get_channel_type_size(get_channel_type(pixel_type)) * get_nr_channels(pixel_type);
}
