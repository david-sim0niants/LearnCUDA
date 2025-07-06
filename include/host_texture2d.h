#pragma once

#include "texture2d.h"

class HostTexture2D : public Texture2DMoveOnly {
public:
    HostTexture2D(const Texture2DSize& size, PixelType pixel_type);
    HostTexture2D(const Texture2DSize& size, PixelType pixel_type, std::size_t alignment);

    HostTexture2D(HostTexture2D&&) = default;
    HostTexture2D& operator=(HostTexture2D&&) = default;

    ~HostTexture2D();

    void fill(int value);
};
