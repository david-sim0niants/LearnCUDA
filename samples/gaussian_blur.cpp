#include "conv2d.h"
#include "imgio.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

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

    std::ofstream out_f (out_fn, std::ios::binary);
    if (! out_f) {
        std::cerr << "Error: failed opening output file\n";
        return EXIT_FAILURE;
    }

    auto in_img_opt = load_image(in_f);
    if (! in_img_opt) {
        std::cerr << "Error: failed loading input image\n";
        return EXIT_FAILURE;
    }

    Image& in_img = *in_img_opt;

    std::cout << "Loaded input image: ("
        << in_img.size().width << ", " << in_img.size().height << ')' << std::endl;

    bool succeeded = save_image(in_img, out_f, { ImageFormat::PNG, 100 });
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
