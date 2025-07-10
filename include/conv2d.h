#pragma once

#include "pixel_type.h"

enum class Conv2DKernelSize {
    K_1x3,
    K_3x1,
    K_3x3,
    K_3x5,
    K_5x3,
    K_5x5,
    K_7x7,
    K_9x9,
};

constexpr std::pair<int, int> get_kernel_size_dims(Conv2DKernelSize size)
{
    switch (size) {
        using enum Conv2DKernelSize;
    case K_1x3:
        return {1, 3};
    case K_3x1:
        return {3, 1};
    case K_3x3:
        return {3, 3};
    case K_3x5:
        return {3, 5};
    case K_5x3:
        return {5, 3};
    case K_5x5:
        return {5, 5};
    case K_7x7:
        return {7, 7};
    case K_9x9:
        return {9, 9};
    default:
        return {0, 0};
    }
}

struct Conv2DParams {
    std::size_t width, height;
    std::size_t input_pitch, output_pitch;
    PixelType pixel_type;
    Conv2DKernelSize kernel_size;
};

void convolve_2d(const Conv2DParams& params,
        const void* input, const float* kernel, void* output);

void convolve_2d_host(const Conv2DParams& params,
        const void* input, const float* kernel, void* output);
