#include "conv2d.h"
#include "common_type_traits.h"
#include "vec.h"

#include <stdexcept>

namespace {

template<int K>
inline float kernel_at(float* kernel, int i, int j)
{
    return kernel[i * K + j];
}

template<typename T>
inline T& at(T* data, size_t pitch, size_t i, size_t j)
{
    return reinterpret_cast<T*>(reinterpret_cast<MatchCV<T, std::byte>*>(data) + pitch * i)[j];
}

template<typename Vec, int KW, int KH>
inline Vec convolve_2d_cell(const Conv2DParams& params,
        const Vec* input, float* kernel, size_t i, size_t j)
{
    Vec out {};
    for (unsigned k_i = 0; k_i < KH; ++k_i) {
        for (unsigned k_j = 0; k_j < KW; ++k_j) {
            if (KH > 1 && i + k_i < KH / 2 || KW > 1 && j + k_j < KW / 2)
                continue;

            if (i + k_i >= KH / 2 + params.height || j + k_j >= KW / 2 + params.width)
                continue;

            const size_t u = i + k_i - KH / 2;
            const size_t v = j + k_j - KW / 2;

            const float k_f = kernel_at<KW>(kernel, k_i, k_j);

            out += at(input, params.input_pitch, u, v) * k_f;
        }
    }
    return out;
}

template<typename Vec, int KW, int KH>
void convolve_2d_impl(const Conv2DParams& params,
        const Vec* input, float* kernel, Vec* output)
{
    #pragma omp parallel for
    for (size_t i = 0; i < params.height; ++i) {
        for (size_t j = 0; j < params.width; ++j) {
            at(output, params.output_pitch, i, j) = convolve_2d_cell<Vec, KW, KH>
                                                    (params, input, kernel, i, j);
        }
    }
}

template<typename Elem, unsigned channels, int KW, int KH>
inline void convolve_2d_per_elem_per_channels_per_kernel_size(const Conv2DParams& params,
        const void* input, float* kernel, void* output)
{
    using Vector = Vec<Elem, channels>;
    convolve_2d_impl<Vector, KW, KH>
        (params, static_cast<const Vector*>(input), kernel, static_cast<Vector*>(output));
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

void convolve_2d_host(const Conv2DParams& params,
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
}
