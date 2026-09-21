#pragma once

#include <cstddef>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

#include <numerix/algebra/static_vector.hpp>

namespace numerix {

    // 栈上定长矩阵：行主序存储，values_[i * C + j] 对应 A(i, j)。
    // 与 StaticVector 一样，所有操作都是 KOKKOS_INLINE_FUNCTION。
    template <Scalar T, std::size_t R, std::size_t C>
    class StaticMatrix {
    public:
        using value_type = T;
        using size_type = std::size_t;

        KOKKOS_INLINE_FUNCTION
        constexpr StaticMatrix() = default;

        KOKKOS_INLINE_FUNCTION
        static constexpr size_type NumRows() noexcept { return R; }

        KOKKOS_INLINE_FUNCTION
        static constexpr size_type NumCols() noexcept { return C; }

        KOKKOS_INLINE_FUNCTION
        constexpr T& operator()(size_type i, size_type j) noexcept { return values_[i * C + j]; }

        KOKKOS_INLINE_FUNCTION
        constexpr const T& operator()(size_type i, size_type j) const noexcept { return values_[i * C + j]; }

        KOKKOS_INLINE_FUNCTION
        static constexpr StaticMatrix Identity() noexcept
            requires(R == C)
        {
            StaticMatrix result;
            for (size_type i = 0; i < R; ++i) {
                result(i, i) = T(1);
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        constexpr T Trace() const noexcept
            requires(R == C)
        {
            T sum = T(0);
            for (size_type i = 0; i < R; ++i) {
                sum += values_[i * C + i];
            }
            return sum;
        }

        KOKKOS_INLINE_FUNCTION
        constexpr StaticMatrix<T, C, R> Transpose() const noexcept
        {
            StaticMatrix<T, C, R> result;
            for (size_type i = 0; i < R; ++i) {
                for (size_type j = 0; j < C; ++j) {
                    result(j, i) = values_[i * C + j];
                }
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        constexpr StaticVector<T, R> Apply(const StaticVector<T, C>& x) const noexcept
        {
            StaticVector<T, R> y;
            for (size_type i = 0; i < R; ++i) {
                T sum = T(0);
                for (size_type j = 0; j < C; ++j) {
                    T term = values_[i * C + j];
                    term *= x[j];
                    sum += term;
                }
                y[i] = sum;
            }
            return y;
        }

    private:
        Kokkos::Array<T, R * C> values_ {};
    };

} // namespace numerix
