#pragma once

#include <cstddef>
#include <utility>

#include <numerix/core/concepts.hpp>
#include <numerix/memory/memory_space.hpp>

namespace numerix {

    // Rows are vector entries; columns are independent vectors. LayoutLeft keeps columns contiguous.
    template <Scalar T, MemorySpace Mem = DefaultMemorySpace>
    class MultiVector {
    public:
        using value_type = T;
        using memory_space = Mem;
        using execution_space = typename Mem::execution_space;
        using size_type = std::size_t;
        using view_type = Kokkos::View<T**, Kokkos::LayoutLeft, Mem>;
        using const_view_type = typename view_type::const_type;

        MultiVector(size_type rows, size_type columns) : values_("numerix::MultiVector", rows, columns) { }
        explicit MultiVector(view_type values) noexcept : values_(std::move(values)) { }

        MultiVector(const MultiVector&) = delete;
        MultiVector& operator=(const MultiVector&) = delete;
        MultiVector(MultiVector&&) noexcept = default;
        MultiVector& operator=(MultiVector&&) noexcept = default;
        ~MultiVector() = default;

        size_type Rows() const noexcept { return values_.extent(0); }
        size_type Columns() const noexcept { return values_.extent(1); }
        view_type View() noexcept { return values_; }
        const_view_type View() const noexcept { return values_; }

    private:
        view_type values_;
    };

    template <class Exec, Scalar T, MemorySpace Mem>
        requires AccessibleFrom<Exec, Mem>
    void DeepCopy(const Exec& exec, MultiVector<T, Mem>& dst, const MultiVector<T, Mem>& src)
    {
        Kokkos::deep_copy(exec, dst.View(), src.View());
    }

    template <class Exec, Scalar T, MemorySpace Mem>
        requires AccessibleFrom<Exec, Mem>
    MultiVector<T, Mem> Clone(const Exec& exec, const MultiVector<T, Mem>& x)
    {
        MultiVector<T, Mem> copy(x.Rows(), x.Columns());
        DeepCopy(exec, copy, x);
        return copy;
    }

} // namespace numerix
