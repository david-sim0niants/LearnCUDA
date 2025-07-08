#include "conv2d.h"
#include "cuda_runtime_api.h"
#include "host_texture2d.h"
#include "device_texture2d.h"
#include "prng.h"
#include "test_config.h"

#include <vector>

#include <gtest/gtest.h>

struct TestArg {
    std::vector<float> conv_kernel;
    Conv2DKernelSize conv_kernel_size;
    size_t width, height;
    unsigned int channels;

    inline auto to_tuple() const
    {
        return std::make_tuple(conv_kernel, conv_kernel_size, width, height, channels);
    }
};

inline void PrintTo(Conv2DKernelSize kernel_size, std::ostream* os)
{
    testing::internal::PrintTo(get_kernel_size_dims(kernel_size), os);
}

inline void PrintTo(const TestArg& arg, std::ostream* os)
{
    testing::internal::PrintTo(arg.to_tuple(), os);
}

class Conv2DTest : public ::testing::TestWithParam<TestArg> {
protected:
    HostTexture2D gen_host_texture(size_t width, size_t height, unsigned int channels)
    {
        HostTexture2D host_image {
            {width, height}, PixelType(make_pixel_type_val(ChannelType::FLOAT, channels))};
        for (size_t i = 0; i < host_image.size().height; ++i)
            for (size_t j = 0; j < host_image.size().width * channels; ++j)
                host_image.view().at<float>(i, j) = prng(0.0F, 1.0F);
        return host_image;
    }

    std::vector<float> gen_kernel(Conv2DKernelSize size)
    {
        auto [width, height] = get_kernel_size_dims(size);
        std::vector<float> kernel (width * height);
        float tot = 0.0F;
        for (float& cell : kernel)
            tot += cell = prng(0.0F, 16.0F);
        for (float& cell : kernel)
            cell /= tot;
        return kernel;
    }

    template<int nr_channels>
    void test_textures(const HostTexture2D& expected, const HostTexture2D& actual)
    {
        for (size_t i = 0; i < actual.size().height; ++i) {
            for (size_t j = 0; j < actual.size().width; ++j) {
                for (int c = 0; c < nr_channels; ++c) {
                    ASSERT_FLOAT_EQ(
                        (expected.view().at<float, nr_channels>(i, j, c)),
                        (actual.view().at<float, nr_channels>(i, j, c))
                    ) << "where (i, j, c) = (" << i << ", " << j << ", " << c << ')';
                }
            }
        }
    }

    PRNG prng {TestConfig::get_prng_seed()};
};

TEST_P(Conv2DTest, BasicCheck)
{
    auto [kernel, kernel_size, width, height, channels] = GetParam();

    auto texture = gen_host_texture(width, height, channels);

    if (kernel.empty())
        kernel = gen_kernel(kernel_size);

    HostTexture2D expected_texture {texture.size(), texture.pixel_type()};

    Conv2DParams params = {
        .width = width,
        .height = height,
        .input_pitch = texture.view().pitch(),
        .output_pitch = expected_texture.view().pitch(),
        .pixel_type = make_pixel_type(ChannelType::FLOAT, channels),
        .kernel_size = kernel_size,
    };

    convolve_2d_host(params, texture.view().data(), kernel.data(), expected_texture.view().data());

    DeviceTexture2D dev_texture {texture.size(), texture.pixel_type()};
    dev_texture.upload(texture.view());

    DeviceTexture2D dev_actual_texture {dev_texture.size(), dev_texture.pixel_type()};
    params.input_pitch = dev_texture.view().pitch();
    params.output_pitch = dev_actual_texture.view().pitch();

    cudaDeviceSynchronize();
    convolve_2d(params, dev_texture.view().data(), kernel.data(), dev_actual_texture.view().data());
    cudaDeviceSynchronize();

    HostTexture2D actual_texture (dev_actual_texture.size(), dev_actual_texture.pixel_type());
    dev_actual_texture.download(actual_texture.view());
    cudaDeviceSynchronize();

    switch (get_nr_channels(actual_texture.pixel_type())) {
    case 1: test_textures<1>(expected_texture, actual_texture); break;
    case 2: test_textures<2>(expected_texture, actual_texture); break;
    case 3: test_textures<3>(expected_texture, actual_texture); break;
    case 4: test_textures<4>(expected_texture, actual_texture); break;
    default: FAIL() << "invalid number of channels"; break;
    }
}

static const TestArg test_args[] = {
    { { 1.f/16, 2.f/16, 1.f/16,
        2.f/16, 4.f/16, 2.f/16,
        1.f/16, 2.f/16, 1.f/16, }, Conv2DKernelSize::K_3x3, 500, 500, 3, },
    { { 1.f/16, 2.f/16, 1.f/16,
        2.f/16, 4.f/16, 2.f/16,
        1.f/16, 2.f/16, 1.f/16, }, Conv2DKernelSize::K_3x3, 800, 600, 3 },
    { {}, Conv2DKernelSize::K_3x5, 200, 200, 4 },
    { {}, Conv2DKernelSize::K_5x3, 200, 200, 4 },
};

INSTANTIATE_TEST_SUITE_P(PerConv2DArgs, Conv2DTest, ::testing::ValuesIn(test_args));
