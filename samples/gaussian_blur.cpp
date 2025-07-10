#include "gaussian_blur.h"
#include "conv2d.h"
#include "imgio.h"
#include "device_texture2d.h"
#include "cuda_stream.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string_view>

#include <cuda_runtime.h>

const std::map<std::string_view, Conv2DKernelSize> ksize_per_str = {
    { "1x3", Conv2DKernelSize::K_1x3 },
    { "3x1", Conv2DKernelSize::K_3x1 },
    { "3x3", Conv2DKernelSize::K_3x3 },
    { "3x5", Conv2DKernelSize::K_3x5 },
    { "5x3", Conv2DKernelSize::K_5x3 },
    { "5x5", Conv2DKernelSize::K_5x5 },
    { "7x7", Conv2DKernelSize::K_7x7 },
    { "9x9", Conv2DKernelSize::K_9x9 },
};

void apply_gaussian(Image& img, Conv2DKernelSize kernel_size);
void usage(const char* prog_name);

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Error: insufficient arguments\n";
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char* in_fn = argv[1];
    const char* out_fn = argv[2];

    std::ifstream in_f (in_fn, std::ios::binary);
    if (! in_f) {
        std::cerr << "Error: failed opening input file\n";
        return EXIT_FAILURE;
    }

    Conv2DKernelSize kernel_size = Conv2DKernelSize::K_3x3;
    if (argc > 3) {
        std::string_view ksize_str = argv[3];
        auto it = ksize_per_str.find(ksize_str);
        if (it == ksize_per_str.end()) {
            std::cerr << "Error: invalid kernel size: " << ksize_str << std::endl;
            return EXIT_FAILURE;
        }
        kernel_size = it->second;
    }

    const ImageFormat img_fmt = retrieve_image_format(in_f);
    auto in_img_opt = load_image(in_f);
    if (! in_img_opt) {
        std::cerr << "Error: failed loading input image\n";
        return EXIT_FAILURE;
    }

    Image& img = *in_img_opt;

    std::cout << "Image size: ("
        << img.size().width << ", " << img.size().height << ')' << std::endl;

    apply_gaussian(img, kernel_size);

    std::ofstream out_f (out_fn, std::ios::binary);
    if (! out_f) {
        std::cerr << "Error: failed opening output file\n";
        return EXIT_FAILURE;
    }

    bool succeeded = save_image(img, out_f, { img_fmt, 100 });
    if (not succeeded) {
        std::cerr << "Error: failed saving output image\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void usage(const char* prog_name)
{
    std::cerr << "Usage: " << prog_name << " <INPUT-FILE> <OUTPUT-FILE> [KERNEL_SIZE = 3x3]\n";
    std::cerr << "Supported kernel sizes: ";
    for (auto [k, _] : ksize_per_str)
        std::cerr << k << ' ';
    std::cerr << std::endl;
}

void apply_gaussian(Image& img, Conv2DKernelSize kernel_size)
{
    CudaStream cuda_stream;
    cuda_stream.set_as_current();

    DeviceTexture2D dev_in (img.size(), img.pixel_type());
    dev_in.upload(img.view());

    DeviceTexture2D dev_out (img.size(), img.pixel_type());
    gaussian_blur(dev_in, kernel_size, dev_out.view());
    dev_out.download(img.view());
    cuda_stream.synchronize();
}
