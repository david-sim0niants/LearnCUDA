#include "imgio.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <cstddef>
#include <span>
#include <array>

namespace {

int read(void* user, char* data, int size)
{
    std::istream& is = *static_cast<std::istream*>(user);
    is.read(data, size);
    return int(is.gcount());
}

void skip(void* user, int n)
{
    std::istream& is = *static_cast<std::istream*>(user);
    is.seekg(is.tellg() + static_cast<std::streamoff>(n));
}

int eof(void* user)
{
    std::istream& is = *static_cast<std::istream*>(user);
    return is.eof();
}

void write(void* ctx, void *data, int size)
{
    std::ostream& os = *static_cast<std::ostream*>(ctx);
    os.write(static_cast<char*>(data), size);
}

constexpr char BMP_SIGNATURE[] = { 'B', 'M' };
constexpr char PNG_SIGNATURE[] = { '\x89', 'P', 'N', 'G', '\r', '\n', '\x1A', '\n' };
constexpr char JPG_SIGNATURE[] = { '\xFF', '\xD8', '\xFF' };
constexpr size_t MAX_HEADER_LEN = 16;

bool test_image_header(std::istream& is, std::span<const char> expected)
{
    std::streampos initial_pos = is.tellg();
    std::array< char, MAX_HEADER_LEN> actual;
    is.read(actual.data(), expected.size());
    const bool match = is.good() && std::equal(expected.begin(), expected.end(), actual.begin());
    is.clear();
    is.seekg(initial_pos);
    return match;
}

}

void ImageDeleter::operator()(std::byte* data)
{
    stbi_image_free(data);
}

ImageFormat retrieve_image_format(std::istream& is)
{
    using enum ImageFormat;
    if (test_image_header(is, BMP_SIGNATURE))
        return BMP;
    if (test_image_header(is, PNG_SIGNATURE))
        return PNG;
    if (test_image_header(is, JPG_SIGNATURE))
        return JPG;
    return RAW;
}

std::optional<Image> load_image(std::istream& is)
{
    stbi_io_callbacks callbacks = {
        .read = read,
        .skip = skip,
        .eof = eof,
    };

    int width = 0, height = 0, channels = 0;
    stbi_uc* data = stbi_load_from_callbacks(&callbacks, &is, &width, &height, &channels, 0);
    if (! data)
        return std::nullopt;

    Texture2DView view(reinterpret_cast<std::byte*>(data), width * channels * sizeof(stbi_uc));

    return Image(view, Texture2DSize(width, height), make_pixel_type(ChannelType::U8, channels));
}

template<auto write_func, typename... OtherArgs>
inline bool write_image(const Texture2D& image, std::ostream& os, OtherArgs&&... other_args)
{
    return write_func(write, &os, image.size().width, image.size().height,
                      get_nr_channels(image.pixel_type()), image.view().data(),
                      std::forward<OtherArgs>(other_args)...);
}

bool save_image(const Texture2D& image, std::ostream& os, const ImageSaveParams& params)
{
    switch (params.format) {
        using enum ImageFormat;
    case RAW:
        return bool(os.write(reinterpret_cast<const char*>(image.view().data()), image.mem_size()));
    case BMP:
        return write_image<stbi_write_bmp_to_func>(image, os);
    case PNG:
        return write_image<stbi_write_png_to_func>(image, os, image.view().pitch());
    case JPG:
        return write_image<stbi_write_jpg_to_func>(image, os, params.quality);
    default:
        throw std::invalid_argument("invalid image format");
    };
}
