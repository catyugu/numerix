#pragma once

#include <string_view>

#include <Kokkos_Core.hpp>

namespace numerix {

    // numerix 不重新设计执行模型：Kokkos 就是 numerix 的机器模型，这里只给出 numerix 语义下的名字。
    using Device = Kokkos::DefaultExecutionSpace;
    using HostDevice = Kokkos::DefaultHostExecutionSpace;

    inline std::string_view DeviceName() { return Device::name(); }

} // namespace numerix
