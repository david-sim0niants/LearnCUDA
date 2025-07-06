#pragma once

#include "texture2d.h"

#include <istream>
#include <optional>
#include <ostream>

struct ImageDeleter {
    void operator()(std::byte* data);
};

using Image = Texture2DWithDeleter<ImageDeleter>;

enum class ImageFormat {
    RAW = 0, BMP, PNG, JPG, TGA
};

struct ImageSaveParams {
    ImageFormat format = ImageFormat::RAW;
    int quality = 0;
};

std::optional<Image> load_image(std::istream& is);
bool save_image(const Texture2D& image, std::ostream& os, const ImageSaveParams& params = {});
