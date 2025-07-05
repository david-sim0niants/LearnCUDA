#pragma once

#include "texture2d.h"

class DeviceTexture2D : public Texture2D {
public:
    DeviceTexture2D(const Texture2DSize& size, PixelType pixel_type);

    DeviceTexture2D(const DeviceTexture2D&) = delete;
    DeviceTexture2D(DeviceTexture2D&&);

    DeviceTexture2D& operator=(const DeviceTexture2D&) = delete;
    DeviceTexture2D& operator=(DeviceTexture2D&&);

    ~DeviceTexture2D();

    void swap(DeviceTexture2D& other);

    void upload(Texture2DReadOnlyView host_view);
    void download(Texture2DView host_view) const;
    void fill(int value);
};
