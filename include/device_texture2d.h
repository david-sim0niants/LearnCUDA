#pragma once

#include "texture2d.h"

class DeviceTexture2D : public Texture2DMoveOnly {
public:
    DeviceTexture2D(const Texture2DSize& size, PixelType pixel_type);
    DeviceTexture2D(DeviceTexture2D&&) = default;
    DeviceTexture2D& operator=(DeviceTexture2D&&) = default;
    ~DeviceTexture2D();

    void upload(Texture2DReadOnlyView host_view);
    void download(Texture2DView host_view) const;
    void fill(int value);
};
