#pragma once

#include "texture2d.h"

class HostTexture2D : public Texture2D {
public:
    HostTexture2D(const Texture2DSize& size, PixelType pixel_type);

    HostTexture2D(const HostTexture2D&) = delete;
    HostTexture2D(HostTexture2D&&);

    HostTexture2D& operator=(const HostTexture2D&) = delete;
    HostTexture2D& operator=(HostTexture2D&&);

    ~HostTexture2D();

    inline void swap(HostTexture2D& other)
    {
        Texture2D::swap(other);
    }

    void fill(int value);
};
