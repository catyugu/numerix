#pragma once

#include <complex>
#include <type_traits>

namespace numerix {

    // numerix 只支持实数与 std::complex 两类标量，标量类型决定内核的数学语义。
    template <class T>
    struct IsComplex : std::false_type { };

    template <class T>
    struct IsComplex<std::complex<T>> : std::true_type { };

    template <class T>
    inline constexpr bool is_complex_v = IsComplex<std::remove_cvref_t<T>>::value;

    template <class T>
    concept Complex = is_complex_v<T>;

    template <class T>
    concept Scalar = std::is_floating_point_v<std::remove_cvref_t<T>> || Complex<T>;

    // 复数标量对应的实标量类型（std::complex<double> 对应 double）。
    template <Scalar T>
    struct RealOf {
        using type = std::remove_cvref_t<T>;
    };

    template <Complex T>
    struct RealOf<T> {
        using type = typename std::remove_cvref_t<T>::value_type;
    };

    template <Scalar T>
    using RealOfT = typename RealOf<T>::type;

    // 实标量的共轭是自身。内积依赖它统一实数与复数语义。
    template <Scalar T>
    constexpr T Conj(const T& value)
    {
        if constexpr (Complex<T>) {
            return std::conj(value);
        }
        else {
            return value;
        }
    }

    // |x|^2：实标量取平方，复数取模平方。
    template <Scalar T>
    constexpr RealOfT<T> SquaredMagnitude(const T& value)
    {
        if constexpr (Complex<T>) {
            return std::norm(value);
        }
        else {
            return value * value;
        }
    }

} // namespace numerix
