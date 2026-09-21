#pragma once

#include <type_traits>

#include <Kokkos_Complex.hpp>
#include <Kokkos_Macros.hpp>

namespace numerix {

    // 一个类型是 numerix 标量，当且仅当它有合法的 ScalarTraits 特化。内核只通过 traits 取
    // 实部类型与共轭/模平方语义，因此 half/bfloat、AD 标量或混合精度可以后加特化，
    // 不需要改动概念体系本身。
    template <class T>
    struct ScalarTraits;

    template <>
    struct ScalarTraits<float> {
        using value_type = float;
        using real_type = float;

        KOKKOS_INLINE_FUNCTION
        static constexpr real_type Conj(value_type x) noexcept { return x; }

        KOKKOS_INLINE_FUNCTION
        static constexpr real_type SquaredMagnitude(value_type x) noexcept { return x * x; }
    };

    template <>
    struct ScalarTraits<double> {
        using value_type = double;
        using real_type = double;

        KOKKOS_INLINE_FUNCTION
        static constexpr real_type Conj(value_type x) noexcept { return x; }

        KOKKOS_INLINE_FUNCTION
        static constexpr real_type SquaredMagnitude(value_type x) noexcept { return x * x; }
    };

    // Kokkos::complex 是 Kokkos 提供的 std::complex 可移植替代品：它能在设备端构造与求值，
    // 因此 numerix 的复数标量一律用它，不用 std::complex。
    template <std::floating_point T>
    struct ScalarTraits<Kokkos::complex<T>> {
        using value_type = Kokkos::complex<T>;
        using real_type = T;

        KOKKOS_INLINE_FUNCTION
        static constexpr value_type Conj(const value_type& x) noexcept { return Kokkos::conj(x); }

        KOKKOS_INLINE_FUNCTION
        static constexpr real_type SquaredMagnitude(const value_type& x) noexcept { return Kokkos::norm(x); }
    };

    template <class T>
    concept Scalar = requires { typename ScalarTraits<std::remove_cvref_t<T>>::real_type; };

    // 复数标量对应的实标量类型（Kokkos::complex<double> 对应 double）。
    template <Scalar T>
    using RealOfT = typename ScalarTraits<std::remove_cvref_t<T>>::real_type;

    // 实标量的共轭是自身。内积依赖它统一实数与复数语义。
    template <Scalar T>
    KOKKOS_INLINE_FUNCTION constexpr T Conj(const T& value) noexcept
    {
        return ScalarTraits<std::remove_cvref_t<T>>::Conj(value);
    }

    // |x|^2：实标量取平方，复数取模平方。
    template <Scalar T>
    KOKKOS_INLINE_FUNCTION constexpr RealOfT<T> SquaredMagnitude(const T& value) noexcept
    {
        return ScalarTraits<std::remove_cvref_t<T>>::SquaredMagnitude(value);
    }

} // namespace numerix
