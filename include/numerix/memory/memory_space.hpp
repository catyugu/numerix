#pragma once

#include <concepts>

#include <numerix/execution/execution_space.hpp>

namespace numerix {

    using Memory = Device::memory_space;
    using HostMemory = HostDevice::memory_space;

    // Kokkos 内存空间的结构契约：至少提供自身别名与静态名字。
    template <class M>
    concept MemorySpace = requires {
        typename M::memory_space;
        { M::name() } -> std::convertible_to<const char*>;
    };

} // namespace numerix
