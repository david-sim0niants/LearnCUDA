#include "host_texture2d.h"

#include <cstdlib>
#include <cstring>

static Texture2DView alloc_and_view(const Texture2DSize& size, PixelType pixel_type,
        std::size_t alignment)
{
    const std::size_t pixel_size = get_pixel_type_size(pixel_type);
    const std::size_t min_pitch = size.width * pixel_size;

    const std::size_t pitch = ((min_pitch + alignment - 1) / alignment) * alignment;
    std::byte* data = static_cast<std::byte*>(std::aligned_alloc(alignment, size.height * pitch));
    if (nullptr == data)
        throw std::bad_alloc();
    return Texture2DView(data, pitch);
}

static constexpr size_t CACHE_LINE = 128;

HostTexture2D::HostTexture2D(const Texture2DSize& size, PixelType pixel_type)
    : HostTexture2D(size, pixel_type, CACHE_LINE)
{
}

HostTexture2D::HostTexture2D(const Texture2DSize& size, PixelType pixel_type, std::size_t alignment)
    : Texture2DMoveOnly(alloc_and_view(size, pixel_type, alignment), size, pixel_type)
{
}

HostTexture2D::~HostTexture2D()
{
    std::free(view().data());
}

void HostTexture2D::fill(int value)
{
    std::memset(view().data(), value, mem_size());
}
