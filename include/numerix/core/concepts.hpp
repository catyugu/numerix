#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include <numerix/core/scalar.hpp>
#include <numerix/execution/execution_space.hpp>

namespace numerix {

    // Kokkos 已经以 C++20 concept 的形式给出了这两个契约，numerix 只把它们纳入自己的词汇表，
    // 不重复定义结构（不猜 `M::name()`、`M::memory_space` 之类的形状）。
    using Kokkos::ExecutionSpace;
    using Kokkos::MemorySpace;

    // Exec 能否访问 Mem 中的内存。非法的组合（设备内存 + 主机执行流）应在模板实例化阶段
    // 被拒绝，而不是运行时出错。SpaceAccessibility 给出的是 enum 而不是 bool，
    // 概念里的原子约束必须是 bool，因此显式转换一次。
    template <class Exec, class Mem>
    concept AccessibleFrom = ExecutionSpace<Exec> && MemorySpace<Mem> && static_cast<bool>(Kokkos::SpaceAccessibility<std::remove_cvref_t<Exec>, std::remove_cvref_t<Mem>>::accessible);

    // rank-1 Kokkos::View，且 Exec 可访问其内存。裸指针不参与契约：设备端指针在主机上
    // 不可解引用，subview/跨步视图也不该用指针表达。
    template <class V, class Exec = DefaultExecutionSpace>
    concept DenseVector = Kokkos::is_view_v<std::remove_cvref_t<V>> && std::remove_cvref_t<V>::rank == 1 && AccessibleFrom<Exec, typename std::remove_cvref_t<V>::memory_space>;

    // 可写变体：输出参数必须是非 const 元素类型。
    template <class V, class Exec = DefaultExecutionSpace>
    concept MutableDenseVector = DenseVector<V, Exec> && !std::is_const_v<typename std::remove_cvref_t<V>::value_type>;

    // View 或容器的标量类型。去掉元素类型上的 const：View<const double*> 的标量仍是 double，
    // "只读"由 MutableDenseVector 单独表达。
    template <class T>
    using ScalarOf = std::remove_cv_t<typename std::remove_cvref_t<T>::value_type>;

} // namespace numerix
