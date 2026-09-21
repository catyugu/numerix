#pragma once

#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

#include <numerix/core/concepts.hpp>
#include <numerix/core/scalar.hpp>
#include <numerix/execution/execution_space.hpp>
#include <numerix/memory/memory_space.hpp>

namespace numerix {

    // 动态向量：owning、move-only 的存储，只负责"拥有内存 + 交出 View"。
    // 刻意不提供拷贝构造：Kokkos::View 的拷贝是引用计数式的句柄共享，与数学向量的直觉冲突，
    // 隐式共享会让 a = b 的行为难以预测。需要复制时显式写 Clone/DeepCopy，
    // 需要别名或 subview 时直接用 Kokkos::View。
    template <Scalar T, MemorySpace Mem = DefaultMemorySpace>
    class Vector {
    public:
        using value_type = T;
        using memory_space = Mem;
        using execution_space = typename Mem::execution_space;
        using size_type = std::size_t;
        using view_type = Kokkos::View<T*, Mem>;
        using const_view_type = typename view_type::const_type;

        explicit Vector(size_type size) : values_("numerix::Vector", size) { }

        // 接管已有 View 的所有权（移动，不复制数据）。调用后源 View 不再持有该 allocation。
        explicit Vector(view_type values) noexcept : values_(std::move(values)) { }

        Vector(const Vector&) = delete;
        Vector& operator=(const Vector&) = delete;
        Vector(Vector&&) noexcept = default;
        Vector& operator=(Vector&&) noexcept = default;
        ~Vector() = default;

        size_type Size() const noexcept { return static_cast<size_type>(values_.extent(0)); }

        T* Data() noexcept { return values_.data(); }
        const T* Data() const noexcept { return const_view_type(values_).data(); }

        // View 是廉价句柄，按值返回即可；const Vector 只能交出 const 数据的 View，
        // 与 Kokkos 的 const View<double*> / View<const double*> 区分一致。
        view_type View() noexcept { return values_; }
        const_view_type View() const noexcept { return values_; }

    private:
        view_type values_;
    };

    // 显式复制：dst = src，在给定的 execution-space instance 上执行。
    template <class Exec, Scalar T, MemorySpace Mem>
        requires AccessibleFrom<Exec, Mem>
    void DeepCopy(const Exec& exec, Vector<T, Mem>& dst, const Vector<T, Mem>& src)
    {
        Kokkos::deep_copy(exec, dst.View(), src.View());
    }

    // 显式克隆：分配新内存并复制内容，结果与源不共享 allocation。
    template <class Exec, Scalar T, MemorySpace Mem>
        requires AccessibleFrom<Exec, Mem>
    Vector<T, Mem> Clone(const Exec& exec, const Vector<T, Mem>& x)
    {
        Vector<T, Mem> copy(x.Size());
        DeepCopy(exec, copy, x);
        return copy;
    }

} // namespace numerix
