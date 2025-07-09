#pragma once

#include <utility>

template<auto NULL_HANDLE>
class MoveOnly {
public:
    using Handle = decltype(NULL_HANDLE);

    MoveOnly(Handle handle) noexcept : handle(handle)
    {
    }

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    MoveOnly(MoveOnly&& other) noexcept : MoveOnly(std::move(other.handle))
    {
        other.handle = NULL_HANDLE;
    }

    MoveOnly& operator=(MoveOnly&& rhs) noexcept
    {
        if (this != &rhs) {
            handle = std::move(rhs.handle);
            rhs.handle = NULL_HANDLE;
        }
        return *this;
    }

    inline operator Handle&() noexcept
    {
        return handle;
    }

    inline operator const Handle&() const noexcept
    {
        return handle;
    }

    inline void release() noexcept
    {
        handle = NULL_HANDLE;
    }

private:
    Handle handle;
};
