#pragma once

#include <numerix/execution/execution_space.hpp>

namespace numerix {

    // 内存空间别名与执行空间别名成对出现：DefaultMemorySpace 就是 DefaultExecutionSpace
    // 的默认内存空间，而不是一个独立概念。
    using DefaultMemorySpace = typename DefaultExecutionSpace::memory_space;
    using DefaultHostMemorySpace = typename DefaultHostExecutionSpace::memory_space;

} // namespace numerix
