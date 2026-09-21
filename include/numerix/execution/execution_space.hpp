#pragma once

#include <Kokkos_Core.hpp>

namespace numerix {

    // numerix 不重新设计执行模型：Kokkos 就是 numerix 的机器模型，这里只给出 numerix
    // 语义下的名字。刻意不叫 Device：Kokkos::Device<Exec, Mem> 与 KokkosSparse 的
    // Device 模板参数都指"执行空间 + 内存空间"的组合，而这里只是执行空间。
    using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DefaultHostExecutionSpace = Kokkos::DefaultHostExecutionSpace;

} // namespace numerix
