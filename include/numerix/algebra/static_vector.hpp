#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include <numerix/core/scalar.hpp>

namespace numerix {

    // 栈上定长向量：只用于单元局部计算（几何、材料、局部算子），不做动态分配。
    template <Scalar T, std::size_t N>
    class StaticVector {
    public:
        using value_type = T;

        constexpr StaticVector() = default;
        constexpr explicit StaticVector(const std::array<T, N>& values) : values_(values) { }

        static constexpr std::size_t Size() { return N; }

        constexpr T* Data() { return values_.data(); }
        constexpr const T* Data() const { return values_.data(); }

        constexpr T& operator[](std::size_t i) { return values_[i]; }
        constexpr const T& operator[](std::size_t i) const { return values_[i]; }

        constexpr StaticVector& operator+=(const StaticVector& other)
        {
            for (std::size_t i = 0; i < N; ++i) {
                values_[i] += other.values_[i];
            }
            return *this;
        }

        constexpr StaticVector& operator-=(const StaticVector& other)
        {
            for (std::size_t i = 0; i < N; ++i) {
                values_[i] -= other.values_[i];
            }
            return *this;
        }

        constexpr StaticVector& operator*=(const T& alpha)
        {
            for (std::size_t i = 0; i < N; ++i) {
                values_[i] *= alpha;
            }
            return *this;
        }

        constexpr T Dot(const StaticVector& other) const
        {
            T sum = T(0);
            for (std::size_t i = 0; i < N; ++i) {
                sum += Conj(values_[i]) * other.values_[i];
            }
            return sum;
        }

        // std::sqrt 在 C++20 不是 constexpr，因此只有平方范数可用于常量求值。
        constexpr RealOfT<T> SquaredNorm() const
        {
            RealOfT<T> sum = RealOfT<T>(0);
            for (std::size_t i = 0; i < N; ++i) {
                sum += SquaredMagnitude(values_[i]);
            }
            return sum;
        }

        RealOfT<T> Norm() const { return std::sqrt(SquaredNorm()); }

    private:
        std::array<T, N> values_ {};
    };

    template <Scalar T, std::size_t N>
    constexpr StaticVector<T, N> operator+(StaticVector<T, N> lhs, const StaticVector<T, N>& rhs)
    {
        return lhs += rhs;
    }

    template <Scalar T, std::size_t N>
    constexpr StaticVector<T, N> operator-(StaticVector<T, N> lhs, const StaticVector<T, N>& rhs)
    {
        return lhs -= rhs;
    }

    template <Scalar T, std::size_t N>
    constexpr StaticVector<T, N> operator*(StaticVector<T, N> lhs, const T& alpha)
    {
        return lhs *= alpha;
    }

    template <Scalar T, std::size_t N>
    constexpr StaticVector<T, N> operator*(const T& alpha, StaticVector<T, N> rhs)
    {
        return rhs *= alpha;
    }

} // namespace numerix
