#pragma once

#include "conv2d.h"
#include "texture2d.h"

void gen_gaussian_kernel(float* kernel, int width, int height, float sigma_x, float sigma_y);
void gen_gaussian_kernel(float* kernel, int width, int height);
const float* get_kernel_for_size(Conv2DKernelSize size);

void gaussian_blur(const Texture2D& input, Conv2DKernelSize kernel_size, Texture2DView output);
void gaussian_blur_host(const Texture2D& input, Conv2DKernelSize kernel_size, Texture2DView output);
