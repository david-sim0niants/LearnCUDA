#pragma once

#include "move_only.h"

class CudaStream {
public:
    using Native = void*;

    CudaStream();
    ~CudaStream() noexcept;

    void synchronize();

    inline void set_as_current() noexcept
    {
        current_ = this;
    }

    inline Native native() const noexcept
    {
        return native_;
    }

    inline static CudaStream& current() noexcept
    {
        return *current_;
    }

    inline static void set_current(CudaStream& stream) noexcept
    {
        current_ = &stream;
    }

    inline static void sync_curr() noexcept
    {
        current().synchronize();
    }

    inline static CudaStream& default_stream() noexcept
    {
        return default_;
    }

private:
    explicit CudaStream(Native native) : native_(native) {}

    MoveOnly<Native{}> native_;

    static CudaStream default_;
    inline static thread_local CudaStream* current_ = &default_;
};
