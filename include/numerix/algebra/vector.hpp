#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

#include <numerix/core/assert.hpp>
#include <numerix/core/scalar.hpp>
#include <numerix/memory/memory_space.hpp>

namespace numerix {

    // 动态向量：内部持有 Kokkos::View，不重新定义 View 语义，也不引入任何隐式拷贝。
    template <Scalar T, MemorySpace Mem = Memory>
    class Vector {
    public:
        using value_type = T;
        using memory_space = Mem;
        using execution_space = typename Mem::execution_space;
        using view_type = Kokkos::View<T*, Mem>;

        explicit Vector(std::size_t size) : values_("numerix::Vector", size) { }

        // 包装已有 View：不复制数据，所有权仍归调用方。
        explicit Vector(view_type values) : values_(std::move(values)) { }

        std::size_t Size() const { return static_cast<std::size_t>(values_.extent(0)); }

        T* Data() const { return values_.data(); }

        const view_type& View() const { return values_; }

        void Scale(T alpha);
        void Axpy(T alpha, const Vector& x);
        void CopyFrom(const Vector& other);
        T Dot(const Vector& x) const;
        RealOfT<T> SquaredNorm() const;
        RealOfT<T> Norm() const;

    private:
        view_type values_;
    };

    template <Scalar T, MemorySpace Mem>
    void Vector<T, Mem>::Scale(T alpha)
    {
        const view_type values = values_;
        Kokkos::parallel_for(
            "numerix::Vector::Scale", Kokkos::RangePolicy<execution_space>(0, static_cast<int>(Size())),
            KOKKOS_LAMBDA(int i) { values(i) *= alpha; });
    }

    template <Scalar T, MemorySpace Mem>
    void Vector<T, Mem>::Axpy(T alpha, const Vector& x)
    {
        NUMERIX_ASSERT(x.Size() == Size());
        const view_type values = values_;
        const view_type other = x.values_;
        Kokkos::parallel_for(
            "numerix::Vector::Axpy", Kokkos::RangePolicy<execution_space>(0, static_cast<int>(Size())),
            KOKKOS_LAMBDA(int i) { values(i) += alpha * other(i); });
    }

    template <Scalar T, MemorySpace Mem>
    void Vector<T, Mem>::CopyFrom(const Vector& other)
    {
        NUMERIX_ASSERT(other.Size() == Size());
        Kokkos::deep_copy(values_, other.values_);
    }

    template <Scalar T, MemorySpace Mem>
    T Vector<T, Mem>::Dot(const Vector& x) const
    {
        NUMERIX_ASSERT(x.Size() == Size());
        const view_type values = values_;
        const view_type other = x.values_;
        T sum = T(0);
        Kokkos::parallel_reduce(
            "numerix::Vector::Dot", Kokkos::RangePolicy<execution_space>(0, static_cast<int>(Size())),
            KOKKOS_LAMBDA(int i, T& local) { local += Conj(values(i)) * other(i); }, sum);
        return sum;
    }

    template <Scalar T, MemorySpace Mem>
    RealOfT<T> Vector<T, Mem>::SquaredNorm() const
    {
        if constexpr (Complex<T>) {
            return std::real(Dot(*this));
        }
        else {
            return Dot(*this);
        }
    }

    template <Scalar T, MemorySpace Mem>
    RealOfT<T> Vector<T, Mem>::Norm() const
    {
        return std::sqrt(SquaredNorm());
    }

} // namespace numerix
