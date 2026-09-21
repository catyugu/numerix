#pragma once

#include <array>
#include <cstddef>

#include <numerix/algebra/static_vector.hpp>

namespace numerix {

    // 栈上定长矩阵：行主序存储，values_[i * C + j] 对应 A(i, j)。
    template <Scalar T, std::size_t R, std::size_t C>
    class StaticMatrix {
    public:
        using value_type = T;

        constexpr StaticMatrix() = default;

        static constexpr std::size_t NumRows() { return R; }
        static constexpr std::size_t NumCols() { return C; }

        constexpr T& operator()(std::size_t i, std::size_t j) { return values_[i * C + j]; }
        constexpr const T& operator()(std::size_t i, std::size_t j) const { return values_[i * C + j]; }

        static constexpr StaticMatrix Identity()
            requires(R == C)
        {
            StaticMatrix result;
            for (std::size_t i = 0; i < R; ++i) {
                result(i, i) = T(1);
            }
            return result;
        }

        constexpr T Trace() const
            requires(R == C)
        {
            T sum = T(0);
            for (std::size_t i = 0; i < R; ++i) {
                sum += values_[i * C + i];
            }
            return sum;
        }

        constexpr StaticMatrix<T, C, R> Transpose() const
        {
            StaticMatrix<T, C, R> result;
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t j = 0; j < C; ++j) {
                    result(j, i) = values_[i * C + j];
                }
            }
            return result;
        }

        constexpr StaticVector<T, R> Apply(const StaticVector<T, C>& x) const
        {
            StaticVector<T, R> y;
            for (std::size_t i = 0; i < R; ++i) {
                T sum = T(0);
                for (std::size_t j = 0; j < C; ++j) {
                    sum += values_[i * C + j] * x[j];
                }
                y[i] = sum;
            }
            return y;
        }

    private:
        std::array<T, R * C> values_ {};
    };

} // namespace numerix
