#pragma once

#include "error.h"

#include <cstddef>

#include <cuda_runtime.h>

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_(size)
    {
        assert_cuda_ok(cudaMalloc(&data_, mem_size()));
    }

    explicit DeviceBuffer(const T *host_data, std::size_t size) : DeviceBuffer(size)
    {
        load_from(host_data);
    }

    DeviceBuffer(const DeviceBuffer& rhs) : DeviceBuffer(rhs.size_)
    {
        copy_from(rhs.data_);
    }

    DeviceBuffer(DeviceBuffer&& rhs) noexcept
    {
        swap(rhs);
    }

    DeviceBuffer& operator=(const DeviceBuffer& rhs)
    {
        if (this == &rhs)
            return *this;
        DeviceBuffer copy = rhs;
        swap(copy);
        return *this;
    }

    DeviceBuffer& operator=(DeviceBuffer&& rhs) noexcept
    {
        if (this != &rhs)
            swap(rhs);
        return *this;
    }

    ~DeviceBuffer()
    {
        cudaFree(data_);
    }

    void swap(DeviceBuffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    void load_from(const T *host_data)
    {
        assert_cuda_ok(cudaMemcpy(data_, host_data, mem_size(), cudaMemcpyHostToDevice));
    }

    void load_to(T *host_data) const
    {
        assert_cuda_ok(cudaMemcpy(host_data, data_, mem_size(), cudaMemcpyDeviceToHost));
    }

    void copy_from(const T *device_data)
    {
        assert_cuda_ok(cudaMemcpy(data_, device_data, mem_size(), cudaMemcpyDeviceToDevice));
    }

    void copy_to(T *device_data) const
    {
        assert_cuda_ok(cudaMemcpy(device_data, data_, mem_size(), cudaMemcpyDeviceToDevice));
    }

    inline T *data() noexcept
    {
        return data_;
    }

    inline const T *data() const noexcept
    {
        return data_;
    }

    inline std::size_t size() const noexcept
    {
        return size_;
    }

    inline std::size_t mem_size() const noexcept
    {
        return size_ * sizeof(T);
    }

private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};
