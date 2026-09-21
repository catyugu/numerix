#pragma once

#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

#include <numerix/numerix.hpp>

namespace numerix::test_support {

    using execution_space = DefaultExecutionSpace;
    using index_type = typename Vector<double>::view_type::index_type;

    // 默认 execution-space instance。Kokkos 要求在 initialize() 之后才构造执行空间，
    // 因此按需构造，而不是用全局对象。
    inline DefaultExecutionSpace Exec() { return DefaultExecutionSpace {}; }

    // RangePolicy 的索引类型跟随 View 的 index_type：不把 extent 窄化成 int。
    template <class Exec, class View>
    auto RangeOf(const Exec& exec, const View& view)
    {
        return Kokkos::RangePolicy<Exec, Kokkos::IndexType<typename View::index_type>>(exec, 0, view.extent(0));
    }

    // 读回主机内存：不假设默认内存空间就是主机内存。
    template <class T>
    std::vector<T> ToHost(const Vector<T>& v)
    {
        const auto mirror = Kokkos::create_mirror_view_and_copy(DefaultHostMemorySpace {}, v.View());
        std::vector<T> values(v.Size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = mirror(i);
        }
        return values;
    }

    // v[i] = base + step * i
    inline void FillRamp(Vector<double>& v, double base, double step)
    {
        auto values = v.View();
        Kokkos::parallel_for(
            "test_support::FillRamp", RangeOf(execution_space {}, values),
            KOKKOS_LAMBDA(index_type i) { values(i) = base + step * static_cast<double>(i); });
    }

} // namespace numerix::test_support
