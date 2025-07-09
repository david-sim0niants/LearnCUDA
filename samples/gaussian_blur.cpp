#include "conv2d.h"
#include "imgio.h"
#include "device_texture2d.h"
#include "cuda_stream.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>

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

    const ImageFormat img_fmt = retrieve_image_format(in_f);
    auto in_img_opt = load_image(in_f);
    if (! in_img_opt) {
        std::cerr << "Error: failed loading input image\n";
        return EXIT_FAILURE;
    }

    Image& img = *in_img_opt;

    std::cout << "Loaded input image: ("
        << img.size().width << ", " << img.size().height << ')' << std::endl;

    CudaStream cuda_stream;
    cuda_stream.set_as_current();

    DeviceTexture2D dev_in_img (img.size(), img.pixel_type());
    dev_in_img.upload(img.view());

    DeviceTexture2D dev_out_img (dev_in_img.size(), dev_in_img.pixel_type());

    Conv2DParams params = {
        .width = dev_in_img.size().width,
        .height = dev_in_img.size().height,
        .input_pitch = dev_in_img.view().pitch(),
        .output_pitch = dev_out_img.view().pitch(),
        .pixel_type = dev_out_img.pixel_type(),
        .kernel_size = Conv2DKernelSize::K_3x3,
    };

    float kernel[3 * 3] = {
        1.f/16, 2.f/16, 1.f/16,
        2.f/16, 4.f/16, 2.f/16,
        1.f/16, 2.f/16, 1.f/16,
    };

    convolve_2d(params, dev_in_img.view().data(), kernel, dev_out_img.view().data());

    dev_out_img.download(img.view());
    cuda_stream.synchronize();

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
    std::cerr << "Usage: " << prog_name << " <INPUT-FILE> <OUTPUT-FILE>\n";
}
