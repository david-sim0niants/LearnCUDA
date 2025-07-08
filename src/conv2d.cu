#include "conv2d.h"

#include "common_type_traits.h"
#include "pixel_type.h"
#include "vec.h"

#include <cstdio>
#include <error.h>
#include <stdexcept>

namespace {

__constant__ float kernel[9*9];

template<int K>
inline __device__ float kernel_at(int i, int j)
{
    return kernel[i * K + j];
}

template<typename T>
inline __device__ T& at(T* data, size_t pitch, size_t i, size_t j)
{
    return reinterpret_cast<T*>(reinterpret_cast<MatchCV<T, std::byte>*>(data) + pitch * i)[j];
}

template<typename Vec, int BLOCK_DIM, int KW, int KH>
__global__ void convolve_2d_kernel(
        const size_t width, const size_t height,
        const size_t input_pitch, const size_t output_pitch,
        const Vec* __restrict__ input,
        Vec* __restrict__ output)
{
    constexpr int SHARED_H = BLOCK_DIM + KH - 1;
    constexpr int SHARED_W = BLOCK_DIM + KW - 1;

    __shared__ Vec input_sub[SHARED_H][SHARED_W]; // input_sub[ty][tx] = input[y - KH/2][x - KW/2]

    const unsigned ty = threadIdx.y;
    const unsigned tx = threadIdx.x;

    const unsigned By = blockIdx.y * BLOCK_DIM;
    const unsigned Bx = blockIdx.x * BLOCK_DIM;

    for (unsigned y = ty; y < SHARED_H; y += BLOCK_DIM) {
        for (unsigned x = tx; x < SHARED_W; x += BLOCK_DIM) {
            if (    KH > 1 && By + y < KH / 2 || KW > 1 && Bx + x < KW / 2
                 || By + y >= KH / 2 + height || Bx + x >= KW / 2 + width
            )
                input_sub[y][x] = Vec();
            else
                input_sub[y][x] = at(input, input_pitch, By + y - KH / 2, Bx + x - KW / 2);
        }
    }

    __syncthreads();

    const unsigned row = By + ty;
    const unsigned col = Bx + tx;

    if (row >= height || col >= width)
        return;

    ::Vec<float, Vec::size()> out = {};
    for (int i = 0; i < KH; ++i)
        for (int j = 0; j < KW; ++j)
            out += input_sub[ty + i][tx + j].as(float()) * kernel_at<KW>(i, j);
    at(output, output_pitch, row, col) = out.as(typename Vec::Element());
}

template<typename Elem, unsigned int channels, int kernel_width, int kernel_height>
void convolve_2d_per_elem_per_channels_per_kernel_size(const Conv2DParams& params,
        const void* input, float* kernel, void* output)
{
    constexpr int BLOCK_DIM = 16;
    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_size((params.width + block_size.x - 1) / block_size.x,
                   (params.height + block_size.y - 1) / block_size.y);

    using Vector = Vec<Elem, channels>;

    cudaMemcpyToSymbol(::kernel, kernel, sizeof(float) * kernel_width * kernel_height);

    convolve_2d_kernel
        <Vector, BLOCK_DIM, kernel_width, kernel_height>
        <<<grid_size, block_size>>>
        (params.width, params.height,
         params.input_pitch, params.output_pitch,
         static_cast<const Vector*>(input),
         static_cast<Vector*>(output));
}

template<typename Elem, unsigned int channels>
void convolve_2d_per_elem_per_channels(const Conv2DParams& params,
        const void* input, float* kernel, void* output)
{
    switch (params.kernel_size) {
        using enum Conv2DKernelSize;
    case K_1x3:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 1, 3>
            (params, input, kernel, output); break;
    case K_3x1:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 3, 1>
            (params, input, kernel, output); break;
    case K_3x3:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 3, 3>
            (params, input, kernel, output); break;
    case K_3x5:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 3, 5>
		    (params, input, kernel, output); break;
    case K_5x3:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 5, 3>
		    (params, input, kernel, output); break;
    case K_5x5:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 5, 5>
		    (params, input, kernel, output); break;
    case K_7x7:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 7, 7>
            (params, input, kernel, output); break;
    case K_9x9:
        convolve_2d_per_elem_per_channels_per_kernel_size<Elem, channels, 9, 9>
            (params, input, kernel, output); break;
    default:
        throw std::invalid_argument("unsupported kernel size");
    }
}

template<typename Elem>
void convolve_2d_per_elem(const Conv2DParams& params,
        const void* input, float* kernel, void* output)
{
    switch (get_nr_channels(params.pixel_type)) {
    case 1:
        convolve_2d_per_elem_per_channels<Elem, 1>(params, input, kernel, output); break;
    case 2:
        convolve_2d_per_elem_per_channels<Elem, 2>(params, input, kernel, output); break;
    case 3:
        convolve_2d_per_elem_per_channels<Elem, 3>(params, input, kernel, output); break;
    case 4:
        convolve_2d_per_elem_per_channels<Elem, 4>(params, input, kernel, output); break;
    default:
        throw std::invalid_argument("invalid number of channels");
    }
}

}

void convolve_2d(const Conv2DParams& params,
        const void* input, float* kernel, void* output)
{
    switch (get_channel_type(params.pixel_type)) {
        using enum ChannelType;
    case U8:
        convolve_2d_per_elem<uint8_t>(params, input, kernel, output); break;
    case FLOAT:
        convolve_2d_per_elem<float>(params, input, kernel, output); break;
    default:
        throw std::invalid_argument("invalid channel type");
    }
    assert_cuda_ok();
}
