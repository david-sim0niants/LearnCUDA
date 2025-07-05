#include "device_texture2d.h"
#include "error.h"

static Texture2DView alloc_and_view(const Texture2DSize& size, PixelType pixel_type)
{
    std::byte* data;
    std::size_t pitch;
    const std::size_t pixel_size = get_pixel_type_size(pixel_type);
    assert_cuda_ok(cudaMallocPitch(&data, &pitch, size.width * pixel_size, size.height));
    return Texture2DView(data, pitch);
}

DeviceTexture2D::DeviceTexture2D(const Texture2DSize& size, PixelType pixel_type)
    : Texture2D(alloc_and_view(size, pixel_type), size, pixel_type)
{
}

DeviceTexture2D::DeviceTexture2D(DeviceTexture2D&& other)
{
    swap(other);
}

DeviceTexture2D& DeviceTexture2D::operator=(DeviceTexture2D&& rhs)
{
    if (&rhs != this)
        swap(rhs);
    return *this;
}

DeviceTexture2D::~DeviceTexture2D()
{
    cudaFree(view().data());
}

void DeviceTexture2D::swap(DeviceTexture2D& other)
{
    Texture2D::swap(other);
}

void DeviceTexture2D::upload(Texture2DReadOnlyView host_view)
{
    assert_cuda_ok(
        cudaMemcpy2DAsync(
            view().data(), view().pitch(), host_view.data(), host_view.pitch(),
            size().width * get_pixel_type_size(pixel_type()), size().height,
            cudaMemcpyHostToDevice));
}

void DeviceTexture2D::download(Texture2DView host_view) const
{
    assert_cuda_ok(
        cudaMemcpy2DAsync(
            host_view.data(), host_view.pitch(), view().data(), view().pitch(),
            size().width * get_pixel_type_size(pixel_type()), size().height,
            cudaMemcpyDeviceToHost));
}

void DeviceTexture2D::fill(int value)
{
    assert_cuda_ok(
        cudaMemset2DAsync(
            view().data(), view().pitch(), value,
            size().width * get_pixel_type_size(pixel_type()), size().height));
}
