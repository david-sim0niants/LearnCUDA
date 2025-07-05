#include "host_texture2d.h"

#include <cstdlib>
#include <cstring>

static constexpr size_t CACHE_LINE = 64;

static Texture2DView alloc_and_view(const Texture2DSize& size, PixelType pixel_type)
{
    const std::size_t pixel_size = get_pixel_type_size(pixel_type);
    const std::size_t min_pitch = size.width * pixel_size;

    const std::size_t pitch = ((min_pitch + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
    std::byte* data = static_cast<std::byte*>(std::aligned_alloc(CACHE_LINE, size.height * pitch));
    if (nullptr == data)
        throw std::bad_alloc();
    return Texture2DView(data, pitch);
}

HostTexture2D::HostTexture2D(const Texture2DSize& size, PixelType pixel_type)
    : Texture2D(alloc_and_view(size, pixel_type), size, pixel_type)
{
}

HostTexture2D::HostTexture2D(HostTexture2D&& other)
{
    swap(other);
}

HostTexture2D& HostTexture2D::operator=(HostTexture2D&& rhs)
{
    if (&rhs != this)
        swap(rhs);
    return *this;
}

HostTexture2D::~HostTexture2D()
{
    std::free(view().data());
}

void HostTexture2D::fill(int value)
{
    std::memset(view().data(), value, mem_size());
}
