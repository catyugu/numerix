#pragma once

#include <cstddef>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <numerix/core/scalar.hpp>

namespace numerix {

    // 栈上定长向量：只用于单元局部计算（几何、材料、局部算子），不做动态分配。
    // 所有操作都是 KOKKOS_INLINE_FUNCTION：StaticVector 最重要的使用场景就是 element/cell
    // kernel，若它不能安全进入 CUDA/HIP/SYCL kernel，这个模块就失去大半意义。
    // 存储用 Kokkos::Array 而不是 std::array：前者是 Kokkos 的设备端可移植版本。
    template <Scalar T, std::size_t N>
    class StaticVector {
    public:
        using value_type = T;
        using size_type = std::size_t;

        KOKKOS_INLINE_FUNCTION
        constexpr StaticVector() = default;

        KOKKOS_INLINE_FUNCTION
        constexpr explicit StaticVector(const Kokkos::Array<T, N>& values) : values_(values) { }

        KOKKOS_INLINE_FUNCTION
        static constexpr size_type Size() noexcept { return N; }

        KOKKOS_INLINE_FUNCTION
        constexpr T* Data() noexcept { return values_.data(); }

        KOKKOS_INLINE_FUNCTION
        constexpr const T* Data() const noexcept { return values_.data(); }

        KOKKOS_INLINE_FUNCTION
        constexpr T& operator[](size_type i) noexcept { return values_[i]; }

        KOKKOS_INLINE_FUNCTION
        constexpr const T& operator[](size_type i) const noexcept { return values_[i]; }

        KOKKOS_INLINE_FUNCTION
        constexpr StaticVector& operator+=(const StaticVector& other) noexcept
        {
            for (size_type i = 0; i < N; ++i) {
                values_[i] += other.values_[i];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        constexpr StaticVector& operator-=(const StaticVector& other) noexcept
        {
            for (size_type i = 0; i < N; ++i) {
                values_[i] -= other.values_[i];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        constexpr StaticVector& operator*=(const T& alpha) noexcept
        {
            for (size_type i = 0; i < N; ++i) {
                values_[i] *= alpha;
            }
            return *this;
        }

        // 只用成员 +=/*= 组合：Kokkos::complex 的自由运算符不是 constexpr，而成员运算符是，
        // 因此这样写可以让实数和复数两种标量都在编译期求值。
        KOKKOS_INLINE_FUNCTION
        constexpr T Dot(const StaticVector& other) const noexcept
        {
            T sum = T(0);
            for (size_type i = 0; i < N; ++i) {
                T term = other.values_[i];
                term *= Conj(values_[i]);
                sum += term;
            }
            return sum;
        }

        KOKKOS_INLINE_FUNCTION
        constexpr RealOfT<T> SquaredNorm() const noexcept
        {
            RealOfT<T> sum = RealOfT<T>(0);
            for (size_type i = 0; i < N; ++i) {
                sum += SquaredMagnitude(values_[i]);
            }
            return sum;
        }

        // Kokkos::sqrt 是设备端函数而非 constexpr，因此范数只能在运行期求值，
        // 平方范数才可以用于常量表达式。
        KOKKOS_INLINE_FUNCTION
        RealOfT<T> Norm() const noexcept { return Kokkos::sqrt(SquaredNorm()); }

    private:
        Kokkos::Array<T, N> values_ {};
    };

    template <Scalar T, std::size_t N>
    KOKKOS_INLINE_FUNCTION constexpr StaticVector<T, N> operator+(StaticVector<T, N> lhs, const StaticVector<T, N>& rhs) noexcept
    {
        return lhs += rhs;
    }

    template <Scalar T, std::size_t N>
    KOKKOS_INLINE_FUNCTION constexpr StaticVector<T, N> operator-(StaticVector<T, N> lhs, const StaticVector<T, N>& rhs) noexcept
    {
        return lhs -= rhs;
    }

    template <Scalar T, std::size_t N>
    KOKKOS_INLINE_FUNCTION constexpr StaticVector<T, N> operator*(StaticVector<T, N> lhs, const T& alpha) noexcept
    {
        return lhs *= alpha;
    }

    template <Scalar T, std::size_t N>
    KOKKOS_INLINE_FUNCTION constexpr StaticVector<T, N> operator*(const T& alpha, StaticVector<T, N> rhs) noexcept
    {
        return rhs *= alpha;
    }

} // namespace numerix
