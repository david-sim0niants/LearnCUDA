#include "gaussian_blur.h"

#include <cmath>

void gen_gaussian_kernel(float* kernel, int width, int height, float sigma_x, float sigma_y)
{
    float sum = 0.0F;

    const int half_w = width / 2;
    const int half_h = height / 2;

    const float pi_coeff = 1.0F / (2 * M_PI * sigma_x * sigma_y);

    for (int y = -half_h; y <= half_h; ++y) {
        for (int x = -half_w; x <= half_w; ++x) {
            const float exponent = -(x*x / (2 * sigma_x*sigma_x) + y*y / (2 * sigma_y*sigma_y));
            const float value = pi_coeff * std::exp(exponent);
            kernel[(y + half_h) * width + (x + half_w)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < width * height; ++i)
        kernel[i] /= sum;
}

void gen_gaussian_kernel(float* kernel, int width, int height)
{
    gen_gaussian_kernel(kernel, width, height, float(width) / 6, float(height) / 6);
}

namespace {

template<int KW, int KH>
float kernel[KW * KH];

template<int KW, int KH>
void gen_gaussian_kernel_static()
{
    gen_gaussian_kernel(kernel<KW, KH>, KW, KH);
}

auto _ = []()
{
    gen_gaussian_kernel_static<1, 3>();
    gen_gaussian_kernel_static<3, 1>();
    gen_gaussian_kernel_static<3, 3>();
    gen_gaussian_kernel_static<3, 5>();
    gen_gaussian_kernel_static<5, 3>();
    gen_gaussian_kernel_static<5, 5>();
    gen_gaussian_kernel_static<7, 7>();
    gen_gaussian_kernel_static<9, 9>();
    return 0;
}();

}

const float* get_kernel_for_size(Conv2DKernelSize size)
{
    switch (size) {
        using enum Conv2DKernelSize;
    case K_1x3:
        return kernel<1, 3>;
    case K_3x1:
        return kernel<3, 1>;
    case K_3x3:
        return kernel<3, 3>;
    case K_3x5:
        return kernel<3, 5>;
    case K_5x3:
        return kernel<5, 3>;
    case K_5x5:
        return kernel<5, 5>;
    case K_7x7:
        return kernel<7, 7>;
    case K_9x9:
        return kernel<9, 9>;
    default:
        return nullptr;
    }
}

static Conv2DParams make_params(
        const Texture2D& input, Conv2DKernelSize kernel_size, Texture2DView output)
{
    return {
        .width = input.size().width,
        .height = input.size().height,
        .input_pitch = input.view().pitch(),
        .output_pitch = output.pitch(),
        .pixel_type = input.pixel_type(),
        .kernel_size = kernel_size,
    };
}

void gaussian_blur(const Texture2D& input, Conv2DKernelSize kernel_size, Texture2DView output)
{
    convolve_2d(make_params(input, kernel_size, output),
            input.view().data(), get_kernel_for_size(kernel_size), output.data());
}

void gaussian_blur_host(const Texture2D& input, Conv2DKernelSize kernel_size, Texture2DView output)
{
    convolve_2d_host(make_params(input, kernel_size, output),
            input.view().data(), get_kernel_for_size(kernel_size), output.data());
}
